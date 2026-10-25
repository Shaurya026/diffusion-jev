import importlib.util
import json
from pathlib import Path
import pytest
import torch
from diffusion_jev.validation import modern_context
from diffusion_jev.modernbert.checkpoint import save_model, load_model


def test_context_guard(tiny):
    req = {
        "id": "test",
        "state": "state " * 100,
        "question": {
            "type": "choice",
            "instructions": "question",
            "criteria": {"a": "low", "b": "high"},
        },
    }
    with pytest.raises(ValueError, match="truncated"):
        modern_context(req, tiny.tok, tiny.cfg)


def test_hub_export_preserves_model_excludes_optimizer(tmp_path, tiny):
    script = Path(__file__).resolve().parents[1] / "scripts/export_checkpoint.py"
    spec = importlib.util.spec_from_file_location("export_checkpoint", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = tmp_path / "source"
    save_model(tiny.model, tiny.tok, source)
    (source / "training.pt").write_text("must not ship")
    card = tmp_path / "card.md"
    card.write_text("# Tiny fixture\n")
    output = tmp_path / "export"
    mod.export(source, output, "modernbert", card)
    assert not (output / "training.pt").exists()
    loaded, _ = load_model(output)
    assert all(
        torch.equal(v, loaded.state_dict()[k])
        for k, v in tiny.model.state_dict().items()
    )
    assert "model.safetensors" in json.loads((output / "SHA256.json").read_text())
