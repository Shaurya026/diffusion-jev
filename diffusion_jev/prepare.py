"""Validate/split user JSONL by connected groups and prepare portable tokenized data."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from . import schema
from .runtime import resolve


def partition(rows, seed=42):
    """Connect declared groups, identical states and requests before assigning splits."""
    parents = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    links = []
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate row ID")
        seen.add(row["id"])
        schema.target(row)
        state = json.dumps(row["state"], sort_keys=True, ensure_ascii=False)
        keys = [
            "id:" + str(row["id"]),
            "state:" + hashlib.sha256(state.encode()).hexdigest(),
            "request:" + schema.identity(row),
        ]
        if row.get("group"):
            keys.append("group:" + str(row["group"]))
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parents[max(a, b)] = min(a, b)
        links.append(keys[0])
    fixed = {}
    for row, key in zip(rows, links):
        group = find(key)
        if row.get("split"):
            if row["split"] not in ("train", "validation", "calibration", "test"):
                raise ValueError("Invalid split")
            if group in fixed and fixed[group] != row["split"]:
                raise ValueError("Connected group crosses supplied splits")
            fixed[group] = row["split"]
    result = {s: [] for s in ("train", "validation", "calibration", "test")}
    for row, key in zip(rows, links):
        group = find(key)
        u = (
            int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:12], 16)
            / 16**12
        )
        split = fixed.get(
            group,
            "train"
            if u < 0.8
            else "validation"
            if u < 0.9
            else "calibration"
            if u < 0.95
            else "test",
        )
        row = dict(row, group=group, split=split)
        if split != "train":
            row.pop("rationale", None)
            row.pop("teacher", None)
        result[split].append(row)
    return result


def modern_item(row, tok, cfg, tokens):
    from .modernbert.data import build_item
    from .validation import modern_context

    q = row["question"]
    k = schema.keys(q)
    p = schema.target(row)
    modern_context(row, tok, cfg)
    # Reject overlong rationales rather than using build_item's legacy truncation.
    rationale = row.get("rationale")
    if (
        rationale
        and len(tok.encode(rationale, add_special_tokens=False)) >= cfg.reasoning_tokens
    ):
        raise ValueError(f"{row['id']}: rationale exceeds available slots")
    confidence = row.get("confidence")
    gold = {
        "probabilities": dict(zip(k, p)),
        "label": k[max(range(len(p)), key=p.__getitem__)],
        "confidence": confidence if confidence is not None else 0.0,
    }
    item = build_item(
        dict(row, workflow=row.get("source", "custom")),
        "q",
        q,
        tok,
        cfg,
        tokens,
        gold,
        rationale,
    )
    item.update(confidence_supervised=confidence is not None, group=row["group"])
    return item


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", choices=["modernbert", "gemma"], required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--revision")
    p.add_argument("--checkpoint")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError("Output exists; preserve old data and use a new directory")
    raw = [json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
    splits = partition(raw, a.seed)
    if not splits["train"] or not splits["validation"]:
        raise ValueError(
            "Need nonempty train and validation; supply explicit group-safe splits for small datasets"
        )
    base = resolve(a.base, a.revision)
    token_path = Path(a.checkpoint or base) / "tokenizer"
    if not token_path.is_dir():
        token_path = base
    tok = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
    meta = {
        "family": a.family,
        "base": str(a.base),
        "revision": a.revision,
        "seed": a.seed,
        "input_sha256": hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        "max_length": a.max_length,
    }
    if a.family == "modernbert":
        from .modernbert.config import DiffusionConfig, add_schema_tokens

        cfg = DiffusionConfig(max_len=a.max_length, reasoning_mode="supervised")
        if a.checkpoint:
            saved = json.loads(
                (Path(a.checkpoint) / "diffusion_config.json").read_text()
            )
            cfg = DiffusionConfig(**saved["config"])
            tokens = saved["tokens"]
        else:
            tokens = add_schema_tokens(tok, cfg)
        meta.update(config=asdict(cfg), tokens=tokens, max_length=cfg.max_len)
    else:
        from .gemma.data import LETTERS

        ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
        if any(len(v) != 1 for v in ids):
            raise ValueError("Invalid Gemma letter tokenizer")
        meta["letter_ids"] = [v[0] for v in ids]
    out.mkdir(parents=True)
    try:
        for split, rows in splits.items():
            with (out / f"{split}.jsonl").open("w") as f:
                for row in rows:
                    if row.get("image"):
                        raise ValueError(
                            "Portable JSONL trainer is text-only; use documented Scout feature recipe for image data"
                        )
                    if a.family == "modernbert":
                        item = modern_item(row, tok, cfg, tokens)
                    else:
                        from .gemma.data import encode

                        if (
                            row.get("rationale")
                            and len(
                                tok.encode(row["rationale"], add_special_tokens=False)
                            )
                            > 255
                        ):
                            raise ValueError("Gemma rationale exceeds 255 tokens")
                        item = encode(row, tok)
                        if len(item["ids"]) + 258 > a.max_length:
                            raise ValueError("Gemma full context exceeds --max-length")
                    f.write(json.dumps(item) + "\n")
        tok.save_pretrained(out / "tokenizer")
        meta["splits"] = {
            s: {
                "rows": len(rows),
                "sha256": hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest(),
            }
            for s, rows in splits.items()
        }
        (out / "split-registry.json").write_text(
            json.dumps(
                {
                    r["id"]: {"split": s, "group": r["group"]}
                    for s, rs in splits.items()
                    for r in rs
                },
                indent=2,
            )
        )
        (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        (out / "FAILED.txt").write_text(
            "Preparation failed. Fix input and use a fresh output directory. No manifest means not admitted.\n"
        )
        raise
    print(json.dumps(meta["splits"], indent=2))


if __name__ == "__main__":
    main()
