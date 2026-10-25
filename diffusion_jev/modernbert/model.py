"""Laya encoder + bidirectional head, with one jointly attended diffusion canvas."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def gather(hidden, positions):
    return hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))


class TypedDiffusion(nn.Module):
    def __init__(self, decision_model, cfg, tokens):
        super().__init__()
        self.cfg, self.tokens = cfg, tokens
        # Reuse checkpoint parameters; the unused Laya act_head is intentionally omitted.
        self.encoder = decision_model.encoder
        self.head = decision_model.head
        self.type_emb = decision_model.type_emb
        self.scorer = decision_model.scorer
        d = self.encoder.config.hidden_size
        self.schema_emb = nn.Embedding(4, d)  # context, reasoning, decision, confidence
        self.time_emb = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.decision_query = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.decision_query.weight)
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg.confidence_bins)
        )
        self.text_transform = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.head_checkpointing = False

    def forward(self, batch, input_ids, time, training_state=None):
        h = self.encoder(
            input_ids=input_ids, attention_mask=batch["attention_mask"]
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
            # Loss stays inside DDP.forward so unused-parameter detection sees text projections.
            from .process import typed_loss

            clean, masks, conf_bin = training_state
            return typed_loss(self, output, batch, clean, time, masks, conf_bin)
        return output

    def text_logits(self, hidden):
        logits = F.linear(
            self.text_transform(hidden), self.encoder.get_input_embeddings().weight
        ).float()
        # Reasoning can emit natural text, END, or reason PAD; never a MASK or typed value.
        return logits.index_fill(
            -1, torch.tensor(self.tokens["forbidden_text"], device=logits.device), -1e4
        )
