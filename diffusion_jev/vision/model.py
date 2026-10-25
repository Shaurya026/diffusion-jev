"""Frozen visual patches condition the same jointly attended typed diffusion canvas."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from diffusion_jev.modernbert.common import DecisionModel
from diffusion_jev.modernbert.checkpoint import (
    load_model as load_parent,
    save_model as save_base,
)
from diffusion_jev.modernbert.config import DiffusionConfig
from diffusion_jev.modernbert.model import TypedDiffusion, gather
from diffusion_jev.modernbert.process import corrupt as corrupt_base, typed_loss


def corrupt(batch, cfg, tokens):
    noisy, clean, time, masks, conf_bin = corrupt_base(batch, cfg, tokens)
    unsupervised = ~batch["confidence_supervised"]
    noisy[unsupervised, batch["confidence_pos"][unsupervised]] = tokens["mask"]
    masks["confidence"] &= batch["confidence_supervised"][:, None]
    return noisy, clean, time, masks, conf_bin


class ScoutDiffusion(TypedDiffusion):
    def __init__(self, decision_model, cfg, tokens, vision_width=768):
        super().__init__(decision_model, cfg, tokens)
        d = self.encoder.config.hidden_size
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(vision_width),
            nn.Linear(vision_width, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.visual_scale = 0.1

    def forward(self, batch, input_ids, time, training_state=None):
        embeddings = self.encoder.get_input_embeddings()(input_ids)
        visual = (
            self.visual_projector(batch["visual_features"].to(embeddings.dtype))
            * self.visual_scale
        )
        n = min(visual.shape[1], embeddings.shape[1])
        first = torch.where(
            batch["has_image"][:, None, None], visual[:, :n], embeddings[:, :n]
        )
        embeddings = torch.cat([first, embeddings[:, n:]], dim=1)
        h = self.encoder(
            inputs_embeds=embeddings, attention_mask=batch["attention_mask"]
        ).last_hidden_state
        schema = torch.zeros_like(input_ids)
        schema.scatter_(1, batch["reason_pos"], 1)
        schema.scatter_(1, batch["decision_pos"][:, None], 2)
        schema.scatter_(1, batch["confidence_pos"][:, None], 3)
        time_features = torch.stack(
            (time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)), -1
        )
        h = h + self.schema_emb(schema) + self.type_emb(batch["qtype"])[:, None]
        h = h + self.time_emb(time_features.to(h.dtype))[:, None]
        if self.head is not None:
            for layer in self.head.layers:
                if self.head_checkpointing and self.training:
                    h = checkpoint(
                        layer,
                        h,
                        src_key_padding_mask=~batch["attention_mask"],
                        use_reentrant=False,
                    )
                else:
                    h = layer(h, src_key_padding_mask=~batch["attention_mask"])
        decision_h = gather(h, batch["decision_pos"][:, None])
        marker_h = gather(h, batch["marker_pos"])
        logits = (
            self.scorer(marker_h + self.decision_query(decision_h)).squeeze(-1).float()
        )
        logits = logits.masked_fill(~batch["marker_mask"], -1e4)
        confidence_h = gather(h, batch["confidence_pos"][:, None]).squeeze(1)
        output = {
            "decision": logits,
            "confidence": self.confidence_head(confidence_h).float(),
            "reason_hidden": gather(h, batch["reason_pos"]),
        }
        if training_state is not None:
            clean, masks, conf_bin = training_state
            return typed_loss(self, output, batch, clean, time, masks, conf_bin)
        return output


def load_model(path, cfg=None, device="cpu"):
    path = Path(path)
    if (path / "scout_config.json").exists():
        meta = json.loads((path / "diffusion_config.json").read_text())
        saved_cfg = DiffusionConfig(**meta["config"])
        if cfg is not None and vars(cfg) != vars(saved_cfg):
            raise ValueError("checkpoint configuration mismatch")
        encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(path / "encoder"), attn_implementation="sdpa"
        )
        model = ScoutDiffusion(
            DecisionModel(encoder, head_layers=meta["head_layers"]),
            saved_cfg,
            meta["tokens"],
        )
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        tok = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    else:
        parent, tok = load_parent(path)
        parts = SimpleNamespace(
            **{k: getattr(parent, k) for k in ("encoder", "head", "type_emb", "scorer")}
        )
        model = ScoutDiffusion(parts, cfg or parent.cfg, parent.tokens)
        mismatch = model.load_state_dict(parent.state_dict(), strict=False)
        if mismatch.unexpected_keys or any(
            not key.startswith("visual_projector.") for key in mismatch.missing_keys
        ):
            raise ValueError(f"parent mismatch: {mismatch}")
    return model.to(device), tok


def save_model(model, tok, path, manifest):
    save_base(model, tok, path)
    (Path(path) / "scout_config.json").write_text(
        json.dumps(
            {
                "visual_tokens": 49,
                "vision_width": 768,
                "visual_scale": model.visual_scale,
                "vision_encoder": next(
                    s for s in manifest["sources"] if s["name"] == "vision"
                ),
                "requires_frozen_visual_features": True,
            },
            indent=2,
        )
    )
