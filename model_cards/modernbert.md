---
language: en
tags: [typed-decisions, diffusion, modernbert, research]
---
# ModernBERT diffusion

Local checkpoint format; no hosted model ID has been assigned. A Laya-initialized ModernBERT
denoiser with 256 explanation positions, a dynamic choice scorer and a confidence readout.
It accepts text evidence and choice/boolean/ordered-score schemas, one question at a time.
The historical context is 1,024 positions including outputs; at most 64 options.

The initial typed-only experiment trained 200 epochs and selected epoch 36 by validation.
The demo's checkpoint is identified by its weight hash in `assets/demo_provenance.json`;
it is a later adapted checkpoint, not a claim to reproduce the typed-only run. The public
training tutorial describes a new reproducible recipe. Historical full training inputs are
not bundled in this release.

Confidence is a separate schema target, not automatically probability of correctness.
Generated explanations are unverified and may disagree with an ensemble answer. Source-specific
results use explicitly documented label protocols; see `docs/RESULTS.md`.

Use `DecisionEngine(path, family="modernbert")`. Export full `model.safetensors`,
`diffusion_config.json`, `encoder/` and `tokenizer/`. Code is Apache-2.0; preserve Laya and
upstream model notices.
