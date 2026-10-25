---
language: en
tags: [typed-decisions, diffusion, multimodal, research]
---
# Scout multimodal diffusion

Local release candidate. Warm-started from an adapted ModernBERT model, with a frozen SigLIP
visual encoder and trainable 768→1,024 connector. A 7×7 grid adds 49 visual positions to the
joint text/reasoning/typed-output canvas; total context is 1,536 positions.

Historical training mixed text decisions and visual examples, including JEV, ScienceQA and
TabFact. This is not a claim that the full training mixture is distributed in the public recipe.
External sources lack the legacy confidence annotation, so that confidence loss is disabled.
Dataset terms remain source-specific: JEV Apache-2.0, TabFact CC-BY-4.0 and ScienceQA CC-BY-SA-4.0
were declared by the inspected cards. This card does not relicense their data.

Text-only and image inputs are supported by `DecisionEngine(path, family="scout", vision_base=...)`.
Use `google/siglip-base-patch16-224`, revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.
The exported denoiser includes the connector but not the frozen visual tower. Dense OCR,
multi-page documents and reliable arbitrary visual reasoning are not established. Rendered
TabFact pages retain their transcription. See separate historical cohorts in results rather
than assigning the Gemma table to this checkpoint.
