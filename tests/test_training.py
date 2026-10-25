"""Exercise the advertised CLI lifecycle on a tiny real transformer, without downloads."""

import json
from pathlib import Path
import subprocess
import sys
from diffusion_jev.modernbert.checkpoint import save_model


def test_prepare_train_resume_evaluate(tmp_path, tiny):
    root = Path(__file__).resolve().parents[1]
    initial = tmp_path / "initial"
    save_model(tiny.model, tiny.tok, initial)
    records = []
    for i, split in enumerate(
        [
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "calibration",
            "test",
        ]
    ):
        records.append(
            {
                "id": f"case{i}",
                "state": f"state {i}",
                "group": f"case{i}",
                "split": split,
                "question": {
                    "type": "choice",
                    "instructions": "question",
                    "criteria": {"a": "low", "b": "high"},
                },
                "target": [0.7, 0.3],
                "confidence": 0.4,
                "rationale": "a b" if split == "train" else "must not leak",
            }
        )
    raw = tmp_path / "rows.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in records))

    def run(module, *args):
        return subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    data = tmp_path / "prepared"
    run(
        "diffusion_jev.prepare",
        "--input",
        raw,
        "--output",
        data,
        "--family",
        "modernbert",
        "--base",
        initial,
        "--checkpoint",
        initial,
    )
    out = tmp_path / "run"
    args = [
        "--data",
        data,
        "--output",
        out,
        "--base",
        initial,
        "--checkpoint",
        initial,
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--device",
        "cpu",
    ]
    run("diffusion_jev.train", *args)
    assert json.loads((out / "complete.json").read_text())["updates"] == 2
    run("diffusion_jev.train", *args, "--resume")
    heldout = tmp_path / "test.jsonl"
    heldout.write_text(json.dumps(records[-1]) + "\n")
    evaluated = tmp_path / "evaluation"
    run(
        "diffusion_jev.evaluate",
        "--checkpoint",
        out / "best",
        "--family",
        "modernbert",
        "--input",
        heldout,
        "--output",
        evaluated,
        "--device",
        "cpu",
    )
    report = json.loads((evaluated / "report.json").read_text())
    assert report["n"] == 1 and report["soft_ce"] > 0
    assert (out / "latest/training.pt").is_file()
