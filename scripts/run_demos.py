"""Run every declared example; preserve successes and failures without cherry-picking."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusion_jev.runtime import DecisionEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", choices=["modernbert", "scout", "gemma"], required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = json.loads(Path(a.paths).read_text())
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    engine = DecisionEngine(
        paths[a.family],
        a.family,
        "cuda",
        paths.get("gemma_base"),
        paths.get("vision_base"),
    )
    examples = json.loads((root / "examples/demo_cases.json").read_text())
    records = []
    for row in examples:
        for steps in [1, 4]:
            result = engine.predict(row, steps=steps)
            records.append(
                {
                    "request": row,
                    "result": result,
                    "expected_label": row["expected_label"],
                    "correct": result["label"] == row["expected_label"],
                }
            )
            (out / "records.json").write_text(json.dumps(records, indent=2))
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "steps": steps,
                        "label": result["label"],
                        "expected": row["expected_label"],
                        "seconds": result["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    weight = Path(paths[a.family]) / (
        "adapter.safetensors" if a.family == "gemma" else "model.safetensors"
    )
    digest = hashlib.sha256()
    with weight.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "family": a.family,
        "checkpoint_sha256": digest.hexdigest(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "seed": 42,
        "examples_sha256": hashlib.sha256(
            (root / "examples/demo_cases.json").read_bytes()
        ).hexdigest(),
        "records": len(records),
        "correct": sum(r["correct"] for r in records),
        "note": "Handwritten explanatory examples; not a benchmark. Every attempted case is retained.",
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2))
    (out / "complete.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
