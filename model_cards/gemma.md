---
language: en
base_model: google/gemma-4-E4B-it
tags: [typed-decisions, diffusion, lora, gemma4, research]
---
# Titan — Gemma — typed diffusion adapter

Local release candidate: validation-selected epoch 4 / update 7,064 of the five-epoch run.
Base: `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
This is Gemma 4 E4B, not Gemma 3 4B. Loaded text-plus-adapter parameters: approximately 7.54B
total / 77.0M trainable. Native separate image/audio towers are excluded.

The adapter adds rank-32 LoRA, learned workspace/typed-slot embeddings, time projection and
a confidence head. Explicit noncausal masks allow bidirectional evidence/workspace interaction.
The pretrained token projection scores option letters. One-pass inference is the primary
readout; optional multipass inference reveals workspace slices before the final decision.
It does not return a verified natural-language rationale.

Training uses separate terms for original targets, eligible offline CLEF distributions,
ordinal structure, selected-option confidence and rationale labels. The public training
recipe is a new experiment, not exact reproduction of the historical training mixture.
Maximum historical training sequence length was 4,009.

Text supervision reuses accepted offline Gemma 4 12B explanations and source-provided
training explanations where available. The 12B teacher is distinct from this E4B student;
it is not called at inference. CLEF supplies optional answer distributions rather than
reasoning text. See [trace provenance](../docs/METHOD.md#reasoning-trace-provenance).

Saved source-specific accuracy: 79.10% on 2,000 typed test decisions and 71.63% on 712 few-shot
episodes. CLEF reached 80.34% on those episodes. No overall mixture score is reported.
Identical adapter bytes produced changed predictions across historical evaluations; the
cause remains unresolved. Small saved-run differences are not proven training gains.

Export contains only trainable tensors in `adapter.safetensors`. Download the pinned base
separately and load through `DecisionEngine(path, family="gemma", base=...)`. This is not a
merged Transformers checkpoint and not interchangeable with a plain PEFT adapter. Preserve
the base model's Apache-2.0 attribution and source-specific dataset notices.
