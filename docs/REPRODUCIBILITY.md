# Reproducibility: what is preserved and what can be rerun

This is a self-contained code distribution. It does not import another checkout of `laya`
or depend on an author's absolute filesystem path. Downloaded weights, prepared data and
local run mappings belong under ignored directories. Model weights and raw training cases
are not included in Git.

## Executable paths

| Path | Included | External artifacts needed |
|---|---|---|
| CPU Gradio replay | App, all recorded inputs/outputs, GIF recorder | None beyond Python dependencies |
| Live ModernBERT / Scout inference | Model, sampler, tokenizer loader, visual connector | Selected checkpoint; SigLIP for images |
| Live Gemma inference | Exact adapter math, pinned loader, prompt format | Selected adapter plus pinned Gemma base |
| New text-model training | Preparation, groups, hashes, losses, trainer, resume, eval | Your data or pinned public data; pretrained initialization |
| Historical full visual training | Architecture, loss, counts and protocol | Original grouped mixture/features and historical training runner |
| Exact historical score reproduction | Aggregate reports and explicit cohort protocol | Original data/teacher caches and checkpoint artifacts; unresolved Gemma reproducibility issue |

The compact reference trainer deliberately uses a simple epoch loop and ordinary distributed
sampling. It is not the historical source-balanced sampler, dynamic token-budget scheduler,
multi-GPU evaluator or continuation runner. New runs must be named and reported as new
experiments, not assigned the old benchmark numbers.

## Historical experiment map

| Stage | Training data | Schedule / selection | Distinguishing settings |
|---|---|---|---|
| Typed-only ModernBERT | 4,800 decisions; 3,284 accepted Gemma explanations | 200 epochs; best 36 | 1,024 total slots; encoder/head LR 1e-5/1e-4; batch 32 × two corruptions; 8 steps × 2 trajectories |
| Scout multimodal | Source mixture with text and image conditioning | Macro-source validation CE | 1,536 slots; 49 visual patches; projector LR 1e-4 |
| Gemma LoRA | Task replay and few-shot supervision | Five epochs; selected epoch 4, step 7,064 | Six H100s; 7,168 padded-token budget per GPU; rank 32; LR 1e-4; 256 workspace slots; default 1-pass readout |

The historical checkpoints were trained on broader mixtures than the public recipe supplied
here. Their complete training inputs and verified teacher caches are not part of this release.
Do not present training on the pinned public dataset as exact reproduction of those weights.

## Data rules

1. Split by case/family before rationale generation or selecting few-shot demonstrations.
2. Preserve supplied held-outs and existing group assignments. Quarantine conflicting links.
3. Keep targets, rationales and annotation metadata outside evidence; allowlist inference input.
4. Check target normalization and exact schema-key alignment.
5. Reject context/schema truncation in the portable preparation path; report excluded records.
6. Supervise explanations only where verified targets exist. Missing labels mean missing loss.
7. Select checkpoints on validation, fit calibration on calibration, and reserve test for reporting.

Exact/group checks do not prove semantic deduplication or no pretrained contamination.
Some datasets have source-specific restrictions; retain their attribution and do not give
the combined data a blanket code license. Raw data are not bundled in this release.

## Provenance and tests

`third_party/source-provenance.json` hashes the source files used when extracting the trained
architecture. `assets/demo_provenance.json` hashes the actual demo checkpoints and records
hardware/versions. `requirements-tested.txt` records observed dependency versions.

The tests exercise both real tiny transformer architectures, output masking, future/padding
attention behavior, LoRA/workspace gradients, frozen-base gradients, save/load, group conflicts,
probability validation and a complete CPU prepare/train/resume/evaluate lifecycle.
Tiny CPU training establishes code integration, not convergence or historical benchmark parity.

Only the dependency-matched GPU smoke demonstrates large-checkpoint inference. Bitwise
reproducibility across hardware, batch grouping, attention implementations and software
versions is not promised; the known historical Gemma mismatch is recorded in RESULTS.md.

## Release procedure

Run tests and the export dry run, inspect the staged folder's hashes and model card, then
publish code/weights only to the destinations you choose. This preparation creates no GitHub
remote and performs no Hugging Face upload. Update `CITATION.cff` with the actual repository
address/release identifier when available; do not invent a DOI or paper reference.
