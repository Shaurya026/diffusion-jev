"""Explicit metric populations; score distributions are included in proper scores."""

from collections import defaultdict

import numpy as np
import torch

from .common import ece_score


def score_records(records):
    if not records:
        raise ValueError("cannot score empty predictions")
    values = defaultdict(list)
    for rec in records:
        p, q = (
            np.asarray(rec["probabilities"], dtype=float),
            np.asarray(rec["target"], dtype=float),
        )
        if (
            p.shape != q.shape
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError("invalid prediction distribution")
        p = np.clip(p, 1e-12, 1)
        p /= p.sum()
        correct = float(p.argmax() == rec["label"])
        values["accuracy"].append(correct)
        values["prediction_confidence"].append(float(p.max()))
        values["soft_accuracy"].append(float(p @ q))
        values["soft_ce"].append(float(-(q * np.log(p)).sum()))
        positive = q > 0
        values["kl"].append(
            float((q[positive] * (np.log(q[positive]) - np.log(p[positive]))).sum())
        )
        values["brier"].append(float(((p - q) ** 2).sum()))
        values["tv"].append(float(np.abs(p - q).sum() / 2))
        values["schema_confidence_mae"].append(
            abs(rec["schema_confidence"] - rec["gold_confidence"])
        )
        if rec["qtype"] == 1:
            error = abs(float(p @ np.arange(len(p))) - rec["gold_score"])
            values["score_mae"].append(error)
            values["score_within_one"].append(float(error <= 1))
            values["score_rps"].append(
                float(((p.cumsum() - q.cumsum()) ** 2).sum() / (len(p) - 1))
            )
    metrics = {
        name: float(np.mean(val))
        for name, val in values.items()
        if name != "prediction_confidence"
    }
    metrics["ece_15"] = ece_score(
        np.asarray(values["prediction_confidence"]), np.asarray(values["accuracy"])
    )
    metrics["decisions"] = len(records)
    metrics["cases"] = len({r["case_id"] for r in records})
    return metrics


def report(records):
    result = {"overall": score_records(records)}
    for field in ("workflow", "qtype", "source"):
        if not all(field in rec for rec in records):
            continue
        groups = defaultdict(list)
        for rec in records:
            groups[str(rec[field])].append(rec)
        result[f"by_{field}"] = {
            key: score_records(group) for key, group in groups.items()
        }
    # Matching the notebook's population, alongside all-types primary metrics.
    choice_noul = [r for r in records if r["qtype"] != 1]
    result["choice_noul_only"] = score_records(choice_noul) if choice_noul else None
    if "by_source" in result:
        result["source_balanced_soft_ce"] = float(
            np.mean([r["soft_ce"] for r in result["by_source"].values()])
        )
    return result


def fit_temperatures(records):
    """Fit per-type log temperature to soft CE, on calibration cases only."""
    temperatures = {}
    for kind in range(3):
        group = [r for r in records if r["qtype"] == kind]
        if not group:
            temperatures[str(kind)] = 1.0
            continue
        kmax = max(len(r["target"]) for r in group)
        logits = torch.full((len(group), kmax), -1e4, dtype=torch.float64)
        target = torch.zeros_like(logits)
        for i, rec in enumerate(group):
            k = len(rec["target"])
            logits[i, :k] = (
                torch.tensor(rec["probabilities"], dtype=torch.float64)
                .clamp_min(1e-12)
                .log()
            )
            target[i, :k] = torch.tensor(rec["target"], dtype=torch.float64)
        # Bounded search is deterministic and avoids unconstrained LBFGS overflow.
        grid = torch.linspace(np.log(0.5), np.log(5.0), 401, dtype=torch.float64).exp()
        loss = (
            -(target[None] * (logits[None] / grid[:, None, None]).log_softmax(-1))
            .sum(-1)
            .mean(-1)
        )
        temperatures[str(kind)] = float(grid[loss.argmin()])
    return temperatures


def apply_temperatures(records, temperatures):
    result = []
    for rec in records:
        z = (
            np.log(np.clip(rec["probabilities"], 1e-12, 1))
            / temperatures[str(rec["qtype"])]
        )
        p = np.exp(z - z.max())
        result.append({**rec, "probabilities": (p / p.sum()).tolist()})
    return result
