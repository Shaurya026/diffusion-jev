"""Portable Gemma prompt encoding and collation, matching the trained adapter."""

import json

import torch

REASONS = 256

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?"


def keys_for(q):
    return (
        ["false", "true"]
        if q["type"] == "noul"
        else [str(i) for i in range(len(q["criteria"]))]
        if q["type"] == "score"
        else list(q["criteria"])
    )


def prompt(row):
    q = row["question"]
    keys = row.get("keys", keys_for(q))
    descriptions = q.get("criteria", {})
    options = []
    for i, key in enumerate(keys):
        value = (
            descriptions.get(key, key)
            if isinstance(descriptions, dict)
            else descriptions[i]
        )
        options.append(f"{LETTERS[i]}. {key}: {value}")
    state = (
        row["state"]
        if isinstance(row["state"], str)
        else json.dumps(row["state"], ensure_ascii=False)
    )
    return (
        "Solve the query using the state and any demonstrations. Demonstration answer keys refer to their own schemas. "
        "For this query, respond with only the letter of the best option from the list below.\n"
        f"Task type: {q['type']}\nInstructions: {q['instructions']}\nOptions:\n"
        + "\n".join(options)
        + "\nState:\n"
        + state
    )


def encode(row, tok, teacher=None):
    keys = row.get("keys", keys_for(row["question"]))
    if not 2 <= len(keys) <= len(LETTERS):
        raise ValueError("unsupported schema width")
    target = row["target"]
    if isinstance(target, dict):
        target = [target[k] for k in keys]
    if (
        len(target) != len(keys)
        or abs(sum(target) - 1) > 0.01
        or any(x < 0 for x in target)
    ):
        raise ValueError(f"invalid target: {row['id']}")
    target = [v / sum(target) for v in target]
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt(row)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Some v5 tokenizer APIs return BatchEncoding; normalize explicitly.
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    reason = tok.encode(row.get("rationale") or "", add_special_tokens=False)
    if len(reason) >= REASONS:
        reason = []
    reason = reason + [tok.eos_token_id] if reason else []
    metadata = {
        k: row.get(k)
        for k in ("group", "family", "source_task", "shots", "episode_mode", "query_id")
    }
    metadata.update(
        id=row["id"],
        keys=keys,
        target=target,
        qtype=row["question"]["type"],
        source=row.get("source", "episodic"),
        label=max(range(len(target)), key=target.__getitem__),
    )
    return dict(
        metadata,
        ids=ids,
        reason=reason,
        teacher=teacher["probabilities"] if teacher else target,
        teacher_weight=teacher["weight"] if teacher else 0.0,
    )


def collate(rows, pad_id=0, pad_to=None):
    n = len(rows)
    length = pad_to or (
        (max(len(r["ids"]) for r in rows) + REASONS + 2 + 63) // 64 * 64
    )
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    valid = torch.zeros((n, length), dtype=torch.bool)
    positions = torch.empty((n, REASONS + 2), dtype=torch.long)
    reason = torch.full((n, REASONS), 1, dtype=torch.long)
    reason_weights = torch.zeros((n, REASONS))
    target = torch.zeros(n, len(LETTERS))
    teacher = target.clone()
    options = torch.zeros_like(target, dtype=torch.bool)
    for i, row in enumerate(rows):
        p = len(row["ids"])
        if p + REASONS + 2 > length:
            raise ValueError("batch truncation")
        ids[i, :p] = torch.tensor(row["ids"])
        valid[i, : p + REASONS + 2] = True
        positions[i] = torch.arange(p, p + REASONS + 2)
        if row["reason"]:
            r = len(row["reason"])
            reason[i, :r] = torch.tensor(row["reason"])
            reason_weights[i, :r] = 1.0
            reason_weights[i, r:] = 0.02
        k = len(row["target"])
        target[i, :k] = torch.tensor(row["target"])
        teacher[i, :k] = torch.tensor(row["teacher"])
        options[i, :k] = True
    return dict(
        ids=ids,
        valid=valid,
        positions=positions,
        reason=reason,
        reason_weights=reason_weights,
        target=target,
        teacher=teacher,
        options=options,
        teacher_weight=torch.tensor([r["teacher_weight"] for r in rows]),
        ordinal=torch.tensor([r["qtype"] == "score" for r in rows]),
        rows=rows,
    )
