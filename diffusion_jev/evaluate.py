"""Evaluate raw requests; ground truth is scored separately and never passed to the engine."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from .runtime import DecisionEngine
from . import schema


def metrics(records):
    if not records:
        raise ValueError("No predictions")
    ce = brier = correct = 0.0
    bins = [[] for _ in range(15)]
    for r in records:
        p, q = r["probabilities"], r["target"]
        top = max(range(len(p)), key=p.__getitem__)
        hit = int(top == max(range(len(q)), key=q.__getitem__))
        ce -= sum(t * math.log(max(v, 1e-12)) for v, t in zip(p, q))
        brier += sum((v - t) ** 2 for v, t in zip(p, q))
        correct += hit
        bins[min(14, int(p[top] * 15))].append((p[top], hit))
    n = len(records)
    ece = sum(abs(sum(v - h for v, h in b)) / n for b in bins if b)
    return {
        "n": n,
        "accuracy": correct / n,
        "soft_ce": ce / n,
        "brier_sum": brier / n,
        "ece_15": ece,
        "label_protocol": "target argmax; ties use first option",
        "calibration": "raw probabilities",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--family", choices=["modernbert", "scout", "gemma"], required=True)
    p.add_argument("--base")
    p.add_argument("--vision-base")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--trajectories", type=int, default=1)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("Output exists; preserve previous evaluation")
    out.mkdir(parents=True)
    engine = DecisionEngine(a.checkpoint, a.family, a.device, a.base, a.vision_base)
    rows = [
        json.loads(line)
        for line in Path(a.input).read_text().splitlines()
        if line.strip()
    ]
    records = []
    with (out / "predictions.jsonl").open("w") as f:
        for row in rows:
            req = schema.request(row)
            result = engine.predict(
                req, a.steps, a.trajectories, image=row.get("image")
            )
            r = {
                "id": req["id"],
                "keys": schema.keys(req["question"]),
                "probabilities": list(result["probabilities"].values()),
                "target": schema.target(row),
                "source": row.get("source", "custom"),
                "result": result,
            }
            records.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    report = metrics(records)
    report["by_source"] = {
        s: metrics([r for r in records if r["source"] == s])
        for s in sorted({r["source"] for r in records})
    }
    report["input_sha256"] = hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
    report["settings"] = {
        k: v
        for k, v in vars(a).items()
        if k not in ("base", "checkpoint", "vision_base")
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "complete.json").write_text(json.dumps({"rows": len(records)}))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
