"""One question per record; gold never enters the inference request."""

import hashlib
import json
import math


def keys(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if (
        not isinstance(question.get("instructions"), str)
        or not question["instructions"].strip()
    ):
        raise ValueError("question.instructions must be nonempty")
    if kind == "choice" and isinstance(criteria, dict):
        result = list(criteria)
    elif kind == "score" and isinstance(criteria, list):
        result = [str(i) for i in range(len(criteria))]
    elif kind == "noul":
        result = ["false", "true"]
    else:
        raise ValueError("Use choice + mapping, score + ordered list, or noul")
    if not 2 <= len(result) <= 64 or any(not isinstance(k, str) for k in result):
        raise ValueError("Schema needs 2–64 string keys")
    return result


def request(row):
    """Explicit allowlist prevents targets, explanations and metadata becoming input."""
    keys(row["question"])
    return {
        "id": str(row.get("id", "request")),
        "state": row["state"],
        "question": row["question"],
    }


def identity(row):
    r = request(row)
    r.pop("id")
    return hashlib.sha256(
        json.dumps(r, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def target(row):
    k = keys(row["question"])
    p = row["target"]
    if isinstance(p, dict):
        if set(p) != set(k):
            raise ValueError("target/schema keys differ")
        p = [p[x] for x in k]
    if (
        len(p) != len(k)
        or any(not math.isfinite(v) or v < 0 for v in p)
        or abs(sum(p) - 1) > 0.01
    ):
        raise ValueError(
            "target must be a finite probability distribution summing to one"
        )
    return [float(v / sum(p)) for v in p]
