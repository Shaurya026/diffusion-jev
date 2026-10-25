"""Fetch a pinned public dataset and convert it to the portable JSONL contract."""

import argparse
import hashlib
import json
from pathlib import Path


def unpack(value):
    return json.loads(value) if isinstance(value, str) else value


def main():
    from datasets import load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--revision", default="f2491dda413a9d94afcb30464123b429c857e079")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("Refusing to overwrite an existing dataset")
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w") as f:
        for split in ["train", "test"]:
            data = load_dataset(
                "LocalLLaMA/typed-decisions", "all", split=split, revision=a.revision
            )
            for row in data:
                case = f"{row['workflow']}:{row['id']}"
                u = (
                    int(hashlib.sha256(("42:" + case).encode()).hexdigest()[:12], 16)
                    / 16**12
                )
                assigned = (
                    "test"
                    if split == "test"
                    else "train"
                    if u < 0.8
                    else "validation"
                    if u < 0.9
                    else "calibration"
                )
                qs, gold = unpack(row["questions"]), unpack(row["gold"])
                for qid, q in qs.items():
                    r = {
                        "id": case + ":" + qid,
                        "group": case,
                        "split": assigned,
                        "source": "typed_decisions",
                        "state": unpack(row["state"]),
                        "question": q,
                        "target": gold[qid]["probabilities"],
                        "confidence": gold[qid]["confidence"],
                    }
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    count += 1
    out.with_suffix(".source.json").write_text(
        json.dumps(
            {
                "repo": "LocalLLaMA/typed-decisions",
                "revision": a.revision,
                "rows": count,
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "protocol": "Official test preserved; train groups hashed 80/10/10 into train/validation/calibration. This is a new public recipe, not the historical workflow-stratified split.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
