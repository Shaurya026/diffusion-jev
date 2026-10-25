import json
from pathlib import Path
import pytest
from diffusion_jev import schema
from diffusion_jev.prepare import partition
from diffusion_jev.evaluate import metrics


def row(i, state=None):
    return {
        "id": str(i),
        "state": state or f"Case {i}",
        "question": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"a": "Alpha", "b": "Beta"},
        },
        "target": [0.8, 0.2],
    }


def test_no_gold_in_inference_contract():
    r = row(1)
    r.update(rationale="answer is A", teacher={"label": "a"}, confidence=1)
    assert set(schema.request(r)) == {"id", "state", "question"}


def test_connected_group_conflict_rejected():
    a, b = row(1, "identical"), row(2, "identical")
    a["split"] = "train"
    b["split"] = "test"
    with pytest.raises(ValueError, match="crosses"):
        partition([a, b])


def test_heldout_rationale_removed_and_stable_groups():
    a, b = row(1), row(2)
    a.update(group="shared", split="test", rationale="A")
    b.update(group="shared")
    s = partition([a, b])
    assert len(s["test"]) == 2 and "rationale" not in s["test"][0]


def test_bad_probabilities_rejected():
    for p in ([float("nan"), 0], [-0.1, 1.1], [0.1, 0.1]):
        r = row(1)
        r["target"] = p
        with pytest.raises(ValueError):
            schema.target(r)


def test_proper_metrics():
    r = metrics([{"probabilities": [0.8, 0.2], "target": [0.8, 0.2]}])
    assert r["accuracy"] == 1 and r["brier_sum"] == 0 and r["soft_ce"] > 0


def test_recorded_demo_complete_and_normalized():
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / "assets/demo_records.json").read_text())
    assert len(records) == 48
    for family in ("modernbert", "scout", "gemma"):
        subset = [r for r in records if r["result"]["family"] == family]
        assert len({(r["request"]["id"], r["result"]["steps"]) for r in subset}) == 16
    assert any(not r["correct"] for r in records)
    for r in records:
        assert abs(sum(r["result"]["probabilities"].values()) - 1) < 1e-5
