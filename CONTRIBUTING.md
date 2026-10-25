# Contributing

Keep the public path readable: a strict schema, a small model API, and explicit CLI settings.
New architectures should not silently reinterpret old checkpoints or confidence semantics.

Before proposing a change:

```bash
USE_TF=0 OMP_NUM_THREADS=1 pytest -q
ruff check diffusion_jev scripts tests app.py --select E9,F63,F7,F82,F401,F811
python -m compileall -q diffusion_jev scripts tests app.py
```

For training or sampling changes, report matched inputs, checkpoint identity, seed, pass
count, trajectory count and before/after metrics. Preserve gold-input isolation and group
boundaries. Do not select demos or checkpoints using held-out test outcomes and then present
them as an unbiased benchmark. Keep known failures and unresolved reproducibility findings.

Large weights, original private cases, optimizer state, caches and site-specific deployment
files belong outside Git. Add provenance and attribution when adapting upstream code.
New references should identify what was implemented rather than merely list related papers.
