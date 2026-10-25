# Model weights — coming soon on Hugging Face

We will release model weights soon on Hugging Face and add the release links here. No weights
have been uploaded yet. These cards describe the planned checkpoint formats and accompany
the local export tool. Titan's Gemma release contains adapter/workspace/readout weights and
requires the separately downloaded base model.

| Card | Export contents | Runtime requirement |
|---|---|---|
| [ModernBERT](modernbert.md) | Full denoiser, encoder config and tokenizer | This package |
| [Scout](scout.md) | Full denoiser including projector, configs and tokenizer | This package; separate pinned SigLIP for images |
| [Gemma](gemma.md) | Trainable adapter/slot/readout tensors only | This package; separate pinned Gemma base |

Export with `scripts/export_checkpoint.py`. It creates a fresh local folder, excludes
optimizer state and writes hashes. The Gemma file is a Titan-specific adapter, not a bare
PEFT `adapter_config.json` package or a full merged language model. Use `DecisionEngine`.
