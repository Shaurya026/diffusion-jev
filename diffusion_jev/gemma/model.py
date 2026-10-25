"""Gemma text weights + LoRA; explicit bidirectional masked categorical denoising."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from transformers import Gemma4Config, Gemma4TextModel

from .data import REASONS


def attention_masks(valid, window):
    """True permits attention. Local attention is symmetric; padding is never a key."""
    length = valid.shape[1]
    idx = torch.arange(length, device=valid.device)
    local = (idx[:, None] - idx[None, :]).abs() < window
    return {
        "full_attention": valid[:, None, None, :],
        "sliding_attention": local[None, None] & valid[:, None, None, :],
    }


class TitanGemmaDiffusion(nn.Module):
    def __init__(self, backbone, letter_ids, rank=32):
        super().__init__()
        self.backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=0.0,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        d = backbone.config.hidden_size
        self.slot_embeddings = nn.Parameter(torch.zeros(REASONS + 2, d))
        nn.init.normal_(self.slot_embeddings, std=0.01)
        self.time_projection = nn.Sequential(
            nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d)
        )
        nn.init.zeros_(self.time_projection[-1].weight)
        nn.init.zeros_(self.time_projection[-1].bias)
        self.confidence_head = nn.Linear(d, 21)
        self.register_buffer("letter_ids", torch.tensor(letter_ids, dtype=torch.long))
        self.checkpoint_backbone = False

    @property
    def text(self):
        return self.backbone.get_base_model()

    def token_logits(self, hidden, token_ids=None):
        weight = self.text.embed_tokens.weight
        if token_ids is not None:
            weight = weight[token_ids]
        logits = F.linear(hidden.to(weight.dtype), weight).float()
        cap = self.text.config.final_logit_softcapping
        return cap * torch.tanh(logits / cap) if cap else logits

    def hidden(self, batch, ids, time):
        inputs = self.text.embed_tokens(ids)
        ple = self.text.get_per_layer_inputs(ids, inputs)
        positions = batch["positions"]
        features = torch.stack((time, time.square()), -1)
        additions = (
            self.slot_embeddings[None] + self.time_projection(features.float())[:, None]
        )
        inputs = inputs.scatter_add(
            1,
            positions[..., None].expand(-1, -1, inputs.shape[-1]),
            additions.to(inputs.dtype),
        )
        masks = attention_masks(batch["valid"], self.text.config.sliding_window)

        def compute(emb, layer_emb):
            return self.backbone(
                inputs_embeds=emb,
                per_layer_inputs=layer_emb,
                attention_mask=masks,
                use_cache=False,
            ).last_hidden_state

        if self.checkpoint_backbone and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(compute, inputs, ple, use_reentrant=False)
        return compute(inputs, ple)

    def readout(self, batch, ids, time):
        hidden = self.hidden(batch, ids, time)
        p = batch["positions"]
        out = hidden.gather(1, p[..., None].expand(-1, -1, hidden.shape[-1]))
        return dict(
            decision=self.token_logits(out[:, REASONS], self.letter_ids).masked_fill(
                ~batch["options"], -1e4
            ),
            confidence=self.confidence_head(out[:, REASONS + 1].float()),
            reasons=out[:, :REASONS],
        )

    def corrupt(self, batch):
        n = len(batch["ids"])
        device = batch["ids"].device
        full = torch.rand(n, device=device) < 0.5
        time = torch.rand(n, device=device) * 0.9 + 0.1
        time[full] = 1.0
        masked = torch.rand((n, REASONS + 2), device=device) < time[:, None]
        # Without a verified rationale there is no clean textual target to reveal.
        masked[:, :REASONS] |= batch["reason_weights"] == 0
        clean = torch.cat(
            (
                batch["reason"],
                self.letter_ids[torch.multinomial(batch["target"], 1)],
                torch.full((n, 1), 4, device=device, dtype=torch.long),
            ),
            -1,
        )
        # Confidence is a readout, always hidden: no correctness label enters the input.
        masked[:, REASONS + 1] = True
        ids = batch["ids"].clone()
        ids.scatter_(1, batch["positions"], clean.masked_fill(masked, 4))
        return ids, time, masked

    def forward(self, batch):
        ids, time, masked = self.corrupt(batch)
        out = self.readout(batch, ids, time)
        logp = out["decision"].log_softmax(-1)
        decision_mask = masked[:, REASONS].float()
        ce = -(batch["target"] * logp).sum(-1)
        choice = (ce * decision_mask).sum() / decision_mask.sum().clamp_min(1)
        probs = logp.exp()
        ordinal = decision_mask * batch["ordinal"]
        ordered_error = (
            (probs.cumsum(-1) - batch["target"].cumsum(-1)).square() * batch["options"]
        ).sum(-1)
        ordered_error /= (batch["options"].sum(-1) - 1).clamp_min(1)
        rps = (ordered_error * ordinal).sum() / ordinal.sum().clamp_min(1)
        eligible = masked.all(-1).float() * batch["teacher_weight"]
        kl_values = (
            batch["teacher"] * (batch["teacher"].clamp_min(1e-12).log() - logp)
        ).sum(-1)
        kd = (kl_values * eligible).sum() / eligible.sum().clamp_min(1)
        # Proper soft correctness target of this model's selected class; no invented confidence labels.
        correctness = (
            batch["target"]
            .gather(1, probs.detach().argmax(-1, keepdim=True))
            .squeeze(-1)
        )
        grid = torch.linspace(0, 1, 21, device=ids.device)
        conf = out["confidence"].softmax(-1) @ grid
        confidence = (
            (conf - correctness).square() * decision_mask
        ).sum() / decision_mask.sum().clamp_min(1)
        weights = batch["reason_weights"] * masked[:, :REASONS]
        selected = weights.flatten().nonzero().flatten()
        reason = out["reasons"].sum() * 0.0
        if len(selected):
            # Uniform token subsampling gives an unbiased estimate of the weighted mean.
            count = len(selected)
            selected = selected[torch.randperm(count, device=ids.device)[:128]]
            hidden = out["reasons"].flatten(0, 1)[selected]
            labels = batch["reason"].flatten()[selected]
            token_loss = []
            for start in range(0, len(selected), 16):
                token_loss.append(
                    F.cross_entropy(
                        self.token_logits(hidden[start : start + 16]),
                        labels[start : start + 16],
                        reduction="none",
                    )
                )
            reason = (
                (torch.cat(token_loss) * weights.flatten()[selected]).sum()
                * count
                / len(selected)
                / weights.sum().clamp_min(1)
            )
        total = choice + 0.5 * kd + 0.5 * rps + 0.1 * confidence + 0.1 * reason
        return dict(
            total=total,
            choice=choice,
            teacher_kl=kd,
            ordinal_rps=rps,
            confidence_brier=confidence,
            reasoning_ce=reason,
        )

    @torch.no_grad()
    def predict(self, batch, steps=1):
        ids = batch["ids"].clone()
        ids.scatter_(1, batch["positions"], 4)
        for step in range(steps):
            time = torch.full((len(ids),), 1 - step / steps, device=ids.device)
            out = self.readout(batch, ids, time)
            if step + 1 < steps:
                # Deterministic nested reveal; choice and confidence stay masked until final readout.
                begin, end = REASONS * step // steps, REASONS * (step + 1) // steps
                hidden = out["reasons"][:, begin:end]
                for start in range(begin, end, 8):
                    stop = min(end, start + 8)
                    predicted = self.token_logits(
                        hidden[:, start - begin : stop - begin]
                    ).argmax(-1)
                    ids.scatter_(1, batch["positions"][:, start:stop], predicted)
        grid = torch.linspace(0, 1, 21, device=ids.device)
        return out["decision"].softmax(-1), out["confidence"].softmax(-1) @ grid

    @torch.no_grad()
    def predict_ar(self, batch):
        """Frozen native causal next-token letter baseline, with the same complete prompt."""
        lens = batch["positions"][:, 0]
        ids = batch["ids"][:, : int(lens.max())]
        valid = torch.arange(ids.shape[1], device=ids.device)[None] < lens[:, None]
        with self.backbone.disable_adapter():
            hidden = self.backbone(
                input_ids=ids, attention_mask=valid, use_cache=False
            ).last_hidden_state
        last = hidden[torch.arange(len(ids), device=ids.device), lens - 1]
        logits = self.token_logits(last, self.letter_ids).masked_fill(
            ~batch["options"], -1e4
        )
        return logits.softmax(-1), logits.softmax(-1).max(-1).values

    def save(self, folder, info):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        state = {
            n: t.detach().cpu().contiguous()
            for n, t in self.state_dict().items()
            if n in trainable
        }
        save_file(state, folder / "adapter.pending.safetensors")
        (folder / "adapter.pending.safetensors").replace(folder / "adapter.safetensors")
        (folder / "config.json").write_text(json.dumps(info, indent=2))

    def restore(self, folder):
        state = load_file(Path(folder) / "adapter.safetensors")
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("adapter parameter mismatch")
        self.load_state_dict(state, strict=False)


def load_model(letter_ids, device, rank=32, base=None):
    if base is None:
        raise ValueError("Pass a local pinned Gemma base directory")
    BASE = str(base)
    cfg = Gemma4Config.from_pretrained(BASE, local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    # Native causal behavior remains intact for baseline; explicit mask dict overrides it for diffusion.
    with init_empty_weights():
        text = Gemma4TextModel(cfg)
    with safe_open(
        str(Path(BASE) / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        prefix = "model.language_model."
        state = {
            k[len(prefix) :]: handle.get_tensor(k)
            for k in handle.keys()
            if k.startswith(prefix)
        }
    # Match the official loader's documented shared-KV exclusions. Older exports
    # contain unused K/V projections for layers that now reuse earlier projections.
    ignored = [
        k
        for k in state
        if any(
            k.startswith(prefix + ".")
            for prefix in text._keys_to_ignore_on_load_unexpected
        )
    ]
    for key in ignored:
        del state[key]
    print(
        json.dumps(
            {
                "loader": "native shared-KV exclusions",
                "ignored_unused_tensors": len(ignored),
            }
        ),
        flush=True,
    )
    text.load_state_dict(state, strict=True, assign=True)
    del state
    text = text.to(device=device, dtype=torch.bfloat16)
    return TitanGemmaDiffusion(text, letter_ids, rank).to(device)
