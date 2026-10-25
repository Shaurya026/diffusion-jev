"""Stage a minimal Hub-ready folder locally. This command never uploads anything."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def export(source, output, family, card):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Destination exists")
    required = (
        ["adapter.safetensors"]
        if family == "gemma"
        else ["model.safetensors", "diffusion_config.json", "encoder", "tokenizer"]
    )
    if family == "scout":
        required += ["scout_config.json"]
    if any(not (source / n).exists() for n in required):
        raise ValueError("Incomplete checkpoint")
    output.mkdir(parents=True)
    for name in required:
        p = source / name
        if p.is_dir():
            # Only model/tokenizer assets, not caches or arbitrary neighboring files.
            (output / name).mkdir()
            for f in p.iterdir():
                if f.is_file() and f.suffix in (".json", ".txt", ".model", ".jinja"):
                    shutil.copy2(f, output / name / f.name)
        else:
            shutil.copy2(p, output / name)
    shutil.copy2(card, output / "README.md")
    (output / "diffusion_jev_config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "family": family,
                "library": "diffusion-jev",
                "base_model": "google/gemma-4-E4B-it" if family == "gemma" else None,
                "base_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2"
                if family == "gemma"
                else None,
                "adapter_only": family == "gemma",
            },
            indent=2,
        )
    )
    hashes = {}
    for p in sorted(output.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(b)
            hashes[str(p.relative_to(output))] = digest.hexdigest()
    (output / "SHA256.json").write_text(json.dumps(hashes, indent=2))
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", required=True, choices=["modernbert", "scout", "gemma"])
    p.add_argument("--card", required=True)
    a = p.parse_args()
    print(json.dumps(export(a.source, a.output, a.family, a.card), indent=2))


if __name__ == "__main__":
    main()
