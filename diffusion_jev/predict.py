import argparse
import json
from pathlib import Path
from .runtime import DecisionEngine


def main():
    p = argparse.ArgumentParser(
        description="Predict one typed decision without gold labels"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--family", choices=["modernbert", "scout", "gemma"], default="modernbert"
    )
    p.add_argument("--base")
    p.add_argument("--vision-base")
    p.add_argument("--revision")
    p.add_argument("--device", default="cpu")
    p.add_argument("--input", required=True)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--trajectories", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image")
    a = p.parse_args()
    engine = DecisionEngine(
        a.checkpoint, a.family, a.device, a.base, a.vision_base, a.revision
    )
    print(
        json.dumps(
            engine.predict(
                json.loads(Path(a.input).read_text()),
                a.steps,
                a.trajectories,
                a.seed,
                a.image,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
