import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .common import DecisionModel, build_model
from .config import DiffusionConfig
from .model import TypedDiffusion


def from_laya(model_dir, cfg, tokens, vocab_size):
    path = Path(model_dir)
    original = json.loads((path / "rl_agent_config.json").read_text())
    base = build_model(original, encoder_dir=str(path / "encoder"))
    base.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    base.encoder.resize_token_embeddings(vocab_size, mean_resizing=False)
    return TypedDiffusion(base, cfg, tokens)


def save_model(model, tok, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    # Atomic replacement of the weights, avoiding partial rolling checkpoints.
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(path / "model.tmp"),
    )
    (path / "model.tmp").replace(path / "model.safetensors")
    model.encoder.config.save_pretrained(path / "encoder")
    tok.save_pretrained(path / "tokenizer")
    meta = {
        "config": asdict(model.cfg),
        "tokens": model.tokens,
        "head_layers": len(model.head.layers) if model.head is not None else 0,
    }
    (path / "diffusion_config.json").write_text(json.dumps(meta, indent=2))


def load_model(directory, device="cpu"):
    path = Path(directory)
    meta = json.loads((path / "diffusion_config.json").read_text())
    cfg = DiffusionConfig(**meta["config"])
    ecfg = AutoConfig.from_pretrained(path / "encoder", local_files_only=True)
    encoder = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    base = DecisionModel(encoder, head_layers=meta["head_layers"])
    model = TypedDiffusion(base, cfg, meta["tokens"])
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    tok = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    return model.to(device), tok
