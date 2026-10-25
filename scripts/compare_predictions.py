"""Compare result files only after checking IDs, ordered schema keys and target vectors."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_jev.evaluate import metrics


def compare(paths):
    outputs = {}
    reference = None
    for path in paths:
        rows = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        indexed = {r["id"]: r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate prediction ID")
        cohort = {k: (v["keys"], v["target"]) for k, v in indexed.items()}
        if reference is not None and cohort != reference:
            raise ValueError(
                "Unmatched cohorts, option orders or targets; do not pool these tables"
            )
        reference = cohort
        outputs[str(path)] = metrics(rows)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", nargs="+")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    Path(a.output).write_text(json.dumps(compare(a.predictions), indent=2))


if __name__ == "__main__":
    main()
