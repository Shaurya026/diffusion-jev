"""Small public inference API for the actual historical checkpoint formats."""

from dataclasses import replace
from pathlib import Path
import threading
import time

import torch
from transformers import AutoTokenizer
from . import schema


def resolve(path, revision=None):
    if Path(path).is_dir():
        return Path(path)
    if revision is None:
        raise ValueError("Remote models require an explicit immutable --revision")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path, revision=revision))


class DecisionEngine:
    def __init__(
        self,
        checkpoint,
        family="modernbert",
        device="cpu",
        base=None,
        vision_base=None,
        revision=None,
    ):
        self.family, self.device = family, device
        self.path = resolve(checkpoint, revision)
        self.lock = threading.RLock()
        self.vision = self.processor = None
        self.vision_base = vision_base
        torch.backends.mha.set_fastpath_enabled(False)
        if family == "gemma":
            if base is None:
                raise ValueError(
                    "Gemma adapters need --base pointing to the pinned local Gemma weights"
                )
            from .gemma.data import LETTERS
            from .gemma.model import load_model

            self.tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
            letters = [self.tok.encode(c, add_special_tokens=False) for c in LETTERS]
            if any(len(v) != 1 for v in letters):
                raise ValueError("Gemma option letters must each encode to one token")
            self.model = load_model([v[0] for v in letters], device, base=base)
            self.model.restore(self.path)
        elif family in ("modernbert", "scout"):
            if family == "scout":
                from .vision.model import load_model
            else:
                from .modernbert.checkpoint import load_model
            self.model, self.tok = load_model(self.path, device=device)
        else:
            raise ValueError("family must be modernbert, scout or gemma")
        self.model.eval()

    def visual_features(self, image):
        if image is None:
            return torch.zeros(1, 49, 768, device=self.device)
        if self.family != "scout":
            raise ValueError("Only Scout supports image conditioning")
        if not self.vision_base:
            raise ValueError(
                "Images require the pinned --vision-base SigLIP checkpoint"
            )
        from transformers import SiglipImageProcessor, SiglipVisionModel
        from PIL import Image

        if self.vision is None:
            self.vision = (
                SiglipVisionModel.from_pretrained(
                    self.vision_base, local_files_only=True
                )
                .to(self.device)
                .eval()
            )
            self.processor = SiglipImageProcessor.from_pretrained(
                self.vision_base, local_files_only=True
            )
        im = (
            Image.open(image).convert("RGB")
            if isinstance(image, (str, Path))
            else image.convert("RGB")
        )
        pixels = self.processor(images=im, return_tensors="pt")["pixel_values"].to(
            self.device
        )
        h = self.vision(pixel_values=pixels).last_hidden_state
        # Match the historical BF16 encoder -> pooled FP16 cache -> FP32 loader path.
        h = torch.nn.functional.adaptive_avg_pool2d(
            h.transpose(1, 2).reshape(1, 768, 14, 14), (7, 7)
        )
        return h.flatten(2).transpose(1, 2).half().float()

    @torch.no_grad()
    def predict(self, row, steps=1, trajectories=1, seed=42, image=None):
        if not 1 <= steps <= 32 or not 1 <= trajectories <= 8:
            raise ValueError("steps must be 1–32 and trajectories 1–8")
        if self.family == "gemma" and trajectories != 1:
            raise ValueError("Gemma uses a deterministic readout; use trajectories=1")
        if image is not None and self.family != "scout":
            raise ValueError("This model is text-only")
        req = schema.request(row)
        k = schema.keys(req["question"])
        with (
            self.lock,
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=str(self.device).startswith("cuda"),
            ),
        ):
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            began = time.perf_counter()
            if self.family == "gemma":
                from .gemma.data import encode, collate

                # A placeholder satisfies the training collator; predict reads no targets.
                item = encode(dict(req, target=[1 / len(k)] * len(k)), self.tok)
                if len(item["ids"]) + 258 > 8192:
                    raise ValueError(
                        "Prompt exceeds the audited 8192-token limit; no truncation"
                    )
                batch = {
                    n: v.to(self.device) if torch.is_tensor(v) else v
                    for n, v in collate([item], self.tok.pad_token_id).items()
                }
                p, c = self.model.predict(batch, steps=steps)
                explanation = None
                confidence_kind = "predicted_target_mass_of_selected_option"
            else:
                from .modernbert.data import build_item, collate, to_device
                from .modernbert.process import sample

                cfg = replace(
                    self.model.cfg,
                    max_len=self.model.cfg.max_len - (49 if image is not None else 0),
                )
                from .validation import modern_context

                modern_context(req, self.tok, cfg)
                item = build_item(
                    dict(req, workflow="demo"),
                    "q",
                    req["question"],
                    self.tok,
                    cfg,
                    self.model.tokens,
                )
                if image is not None:
                    item["ids"] = [self.tok.pad_token_id] * 49 + item["ids"]
                    for key in ("markers", "reason_pos"):
                        item[key] = [v + 49 for v in item[key]]
                    for key in ("decision_pos", "confidence_pos"):
                        item[key] += 49
                batch = to_device(collate([item], self.tok.pad_token_id), self.device)
                if self.family == "scout":
                    batch["visual_features"] = self.visual_features(image)
                    batch["has_image"] = torch.tensor(
                        [image is not None], device=self.device
                    )
                    batch["confidence_supervised"] = torch.tensor(
                        [False], device=self.device
                    )
                draws = [
                    sample(self.model, batch, steps, seed + i)
                    for i in range(trajectories)
                ]
                p = torch.stack([v["probabilities"] for v in draws]).mean(0)
                c = torch.stack([v["schema_confidence"] for v in draws]).mean(0)
                ids = draws[0]["reason_ids"][0].tolist()
                end = self.model.tokens["reason_end"]
                if end in ids:
                    ids = ids[: ids.index(end)]
                explanation = self.tok.decode(ids, skip_special_tokens=True)
                confidence_kind = "legacy_schema_confidence_not_correctness"
            probabilities = p[0, : len(k)].float().cpu().tolist()
            confidence = float(c[0])
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
        result = dict(
            id=req["id"],
            label=k[max(range(len(k)), key=probabilities.__getitem__)],
            probabilities=dict(zip(k, probabilities)),
            schema_confidence=confidence,
            confidence_semantics=confidence_kind,
            generated_explanation=explanation,
            family=self.family,
            steps=steps,
            trajectories=trajectories,
            seed=seed,
            elapsed_seconds=elapsed,
            timing_scope="request including tokenization; model load excluded; first call may warm kernels",
        )
        if req["question"]["type"] == "score":
            result["expected_score_index"] = sum(
                i * p for i, p in enumerate(probabilities)
            )
        return result
