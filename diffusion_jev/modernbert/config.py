from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    max_len: int = 1024
    head_max_len: int = 256
    reasoning_tokens: int = 256
    reasoning_mode: str = "latent"  # latent | supervised | none
    max_options: int = 64
    confidence_bins: int = 21
    time_eps: float = 0.02
    all_mask_probability: float = 0.25
    reasoning_power: float = 1.0
    decision_power: float = 1.0
    confidence_power: float = 1.0
    reasoning_weight: float = 0.1
    choice_weight: float = 1.0
    noul_weight: float = 1.0
    score_weight: float = 1.0
    rps_weight: float = 0.5
    confidence_weight: float = 0.1
    confidence_mse_weight: float = 1.0
    reasoning_pad_weight: float = 0.05

    def __post_init__(self):
        if self.reasoning_mode not in {"latent", "supervised", "none"}:
            raise ValueError("reasoning_mode must be latent, supervised, or none")
        if self.reasoning_mode == "none":
            self.reasoning_tokens = 0
        if self.reasoning_tokens < 0 or (
            self.reasoning_mode == "supervised" and self.reasoning_tokens < 2
        ):
            raise ValueError("supervised reasoning needs at least two slots")
        if self.max_options < 2 or self.confidence_bins < 2:
            raise ValueError("at least two options and confidence bins are required")
        if self.max_len - self.reasoning_tokens - 2 < self.head_max_len + 4:
            raise ValueError(
                "max_len must leave room for the question, state and output slots"
            )
        if not 0 < self.time_eps < 1 or not 0 <= self.all_mask_probability <= 1:
            raise ValueError("invalid time or endpoint mixture probability")
        if min(self.reasoning_power, self.decision_power, self.confidence_power) <= 0:
            raise ValueError("schedule powers must be positive")
        weights = [v for k, v in vars(self).items() if k.endswith("weight")]
        if min(weights) < 0:
            raise ValueError("loss weights must be nonnegative")


def add_schema_tokens(tok, cfg):
    """Schema tokens are inputs; output heads restrict each field's allowed values."""
    base_vocab = len(tok)
    choices = [f"<|decision_{i}|>" for i in range(cfg.max_options)]
    confidence = [f"<|confidence_{i}|>" for i in range(cfg.confidence_bins)]
    extras = ["<|reason_end|>", "<|reason_pad|>"] + choices + confidence
    tok.add_special_tokens({"additional_special_tokens": extras})
    return {
        "mask": tok.mask_token_id,
        "pad": tok.pad_token_id,
        "reason_end": tok.convert_tokens_to_ids(extras[0]),
        "reason_pad": tok.convert_tokens_to_ids(extras[1]),
        "choices": tok.convert_tokens_to_ids(choices),
        "confidence": tok.convert_tokens_to_ids(confidence),
        "base_vocab": base_vocab,
        "forbidden_text": list(
            set(tok.all_special_ids)
            - {
                tok.convert_tokens_to_ids(extras[0]),
                tok.convert_tokens_to_ids(extras[1]),
            }
        ),
    }
