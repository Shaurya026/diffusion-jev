"""Strict typed-decisions parsing, case-level splits, and fixed output canvases."""

import hashlib
import json
import math
import random
from collections import defaultdict

import torch

from .common import QTYPES, build_sequence


def unpack(value, allow_text=False):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if allow_text:
            return value
        raise


def case_key(row):
    return f"{row['workflow']}:{row['id']}"


def state_hash(row):
    state = json.dumps(
        unpack(row["state"], allow_text=True), sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(state.encode()).hexdigest()


def split_cases(rows, seed=42, validation_fraction=0.1, calibration_fraction=0.1):
    """Keep all questions from a case together; reject duplicated states/IDs."""
    if (
        min(validation_fraction, calibration_fraction) <= 0
        or validation_fraction + calibration_fraction >= 1
    ):
        raise ValueError(
            "validation and calibration fractions must be positive and sum to less than one"
        )
    groups = defaultdict(list)
    seen_ids, seen_states = set(), set()
    for row in rows:
        key, digest = case_key(row), state_hash(row)
        if key in seen_ids or digest in seen_states:
            raise ValueError(f"duplicate case or state: {key}")
        seen_ids.add(key)
        seen_states.add(digest)
        groups[row["workflow"]].append(row)
    result = {"train": [], "validation": [], "calibration": []}
    rng = random.Random(seed)
    for workflow in sorted(groups):
        group = sorted(groups[workflow], key=case_key)
        rng.shuffle(group)
        nv = max(1, round(len(group) * validation_fraction))
        nc = max(1, round(len(group) * calibration_fraction))
        if nv + nc >= len(group):
            raise ValueError(f"too few cases in workflow {workflow}")
        result["validation"].extend(group[:nv])
        result["calibration"].extend(group[nv : nv + nc])
        result["train"].extend(group[nv + nc :])
    return result


def option_keys(question):
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice requires a nonempty criteria mapping")
        return list(criteria)
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score requires an ordered criteria list")
        return [str(i) for i in range(len(criteria))]
    if kind == "noul":
        return ["false", "true"]
    raise ValueError(f"unsupported question type: {kind}")


def build_item(row, qid, question, tok, cfg, tokens, gold=None, rationale=None):
    keys = option_keys(question)
    if len(keys) > cfg.max_options:
        raise ValueError(
            f"{case_key(row)}/{qid}: increase max_options to fit {len(keys)} options"
        )
    q = {
        "t": question["type"],
        "ins": question["instructions"],
        "crit": question.get("criteria"),
    }
    if "labels" in question:
        q["labels"] = question["labels"]
    context_len = cfg.max_len - cfg.reasoning_tokens - 2
    ids, markers, stats = build_sequence(
        tok,
        unpack(row["state"], allow_text=True),
        q,
        context_len,
        cfg.head_max_len,
        return_stats=True,
    )
    if len(markers) != len(keys) or stats["options_distinct"] != len(keys):
        raise ValueError(
            f"{case_key(row)}/{qid}: truncated or collapsed options; increase token budgets"
        )
    reason = [tokens["mask"]] * cfg.reasoning_tokens
    reason_weight = [0.0] * cfg.reasoning_tokens
    if cfg.reasoning_mode == "supervised" and rationale is not None:
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be nonempty text")
        # Ordinary text may not inject schema control tokens.
        text = tok(rationale, add_special_tokens=False)["input_ids"]
        if any(t in tok.all_special_ids for t in text):
            raise ValueError("rationale contains reserved tokens")
        text = text[: cfg.reasoning_tokens - 1] + [tokens["reason_end"]]
        reason = text + [tokens["reason_pad"]] * (cfg.reasoning_tokens - len(text))
        reason_weight = [1.0] * len(text) + [cfg.reasoning_pad_weight] * (
            len(reason) - len(text)
        )
    start = len(ids)
    item = {
        "ids": ids + reason + [tokens["mask"], tokens["mask"]],
        "markers": markers,
        "reason_pos": list(range(start, start + cfg.reasoning_tokens)),
        "reason_weight": reason_weight,
        "decision_pos": start + cfg.reasoning_tokens,
        "confidence_pos": start + cfg.reasoning_tokens + 1,
        "qtype": QTYPES[question["type"]],
        "keys": keys,
        "case_id": case_key(row),
        "qid": qid,
        "workflow": row["workflow"],
    }
    if gold is not None:
        probs = gold["probabilities"]
        if set(probs) != set(keys):
            raise ValueError(f"gold/schema key mismatch: {case_key(row)}/{qid}")
        target = [float(probs[k]) for k in keys]
        if any(not math.isfinite(v) or v < 0 for v in target) or sum(target) <= 0:
            raise ValueError("invalid gold probabilities")
        total = sum(target)
        item["target"] = [v / total for v in target]
        confidence = float(gold["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("gold confidence must be finite and between zero and one")
        item["gold_confidence"] = confidence
        label = (
            str(gold["label"]).lower()
            if question["type"] == "noul"
            else str(gold["label"])
        )
        item["label"] = keys.index(label)
        item["gold_score"] = float(
            gold.get("score", sum(i * p for i, p in enumerate(item["target"])))
        )
    return item


def make_items(rows, tok, cfg, tokens, rationales=None):
    items = []
    for row in rows:
        questions, gold = unpack(row["questions"]), unpack(row["gold"])
        if set(questions) != set(gold):
            raise ValueError(f"missing or extra gold question: {case_key(row)}")
        for qid, question in questions.items():
            rationale = (rationales or {}).get((case_key(row), qid))
            items.append(
                build_item(row, qid, question, tok, cfg, tokens, gold[qid], rationale)
            )
    return items


def collate(items, pad_id):
    b, length = len(items), max(len(i["ids"]) for i in items)
    kmax = max(len(i["markers"]) for i in items)
    ids = torch.full((b, length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    markers = torch.zeros((b, kmax), dtype=torch.long)
    valid = torch.zeros((b, kmax), dtype=torch.bool)
    target = torch.zeros((b, kmax))
    for j, item in enumerate(items):
        n, k = len(item["ids"]), len(item["markers"])
        ids[j, :n] = torch.tensor(item["ids"])
        attention[j, :n] = (
            True  # reason_pad is an active output token, NOT batch padding
        )
        markers[j, :k] = torch.tensor(item["markers"])
        valid[j, :k] = True
        if "target" in item:
            target[j, :k] = torch.tensor(item["target"])
    batch = {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": markers,
        "marker_mask": valid,
        "target": target,
    }
    for name in ("reason_pos", "decision_pos", "confidence_pos", "qtype"):
        batch[name] = torch.tensor([i[name] for i in items], dtype=torch.long)
    batch["reason_weight"] = torch.tensor(
        [i["reason_weight"] for i in items], dtype=torch.float32
    )
    batch["gold_confidence"] = torch.tensor(
        [i.get("gold_confidence", 0.0) for i in items]
    )
    return batch


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
