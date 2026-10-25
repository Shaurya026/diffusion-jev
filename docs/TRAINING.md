# Train a typed-decision model

This page describes the executable portable trainer, not an exact replay of the historical
training mixture. Start with the ModernBERT path for a smaller experiment. Gemma requires
substantially more memory. The supplied presets are starting points; validate batch memory
and select hyperparameters on validation, not test.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[demo,dev]'
pip install datasets
export USE_TF=0 TOKENIZERS_PARALLELISM=false
```

Install a driver-compatible PyTorch wheel before the editable install when needed. GPU smoke
inference was tested with CUDA 12.8 wheels. `requirements-tested.txt` records versions from
the tested local environments. The repository does not require Kubernetes; the owner's
cluster runs used it. Generic job guidance is in `docs/CLUSTER.md`.

## 2. Choose data or fetch the public example

```bash
python scripts/fetch_typed_decisions.py --output data/typed/raw.jsonl
```

This pins `LocalLLaMA/typed-decisions` to revision
`f2491dda413a9d94afcb30464123b429c857e079`. The official test split is kept separate; original
training cases are deterministically hashed into train/validation/calibration. This differs
from the historical workflow-stratified assignment and should get a new experiment name.
The verified pinned recipe produces 4,765 training, 630 validation, 605 calibration and
2,000 test decisions.

For your own data, use one JSON object per line:

```json
{"id":"ticket-001:q1","group":"ticket-001","split":"train","source":"support","state":"The app crashes at launch.","question":{"type":"choice","instructions":"Route this request.","criteria":{"billing":"Charges and refunds","technical":"Software errors"}},"target":{"billing":0.05,"technical":0.95},"rationale":"A crash is a software error."}
```

- `id`: unique decision ID. `group`: originating case/family; share it across questions,
  paraphrases, counterfactual twins and images from the same case.
- `split`: optional `train`, `validation`, `calibration` or `test`. Supplied assignments are
  preserved. Without assignments, connected groups use a stable 80/10/5/5 hash split.
- `question.type`: `choice` with an ordered key-description mapping; `noul` with false/true
  semantics; or `score` with an ordered list of descriptions. Score keys are zero-based.
- `target`: nonnegative probabilities aligned to those keys, summing to one. A hard label can
  be a one-hot vector; this does not create a confidence annotation.
- `confidence`: optional legacy scalar in [0,1], with semantics you must document. Missing
  ModernBERT annotations disable confidence loss. Gemma derives its selected-option target.
- `rationale`: optional verified explanation, used only for training. Never place gold answers,
  rationales or annotation artifacts inside the evidence unless they are genuine supplied
  demonstrations from a permitted training/support pool.

Preparation connects declared groups and identical states/requests. It rejects a component
spanning supplied splits, duplicate IDs, malformed distributions, collapsed options and
budget overflow. It does not establish semantic deduplication or absence of pretraining
contamination. Preserve `split-registry.json` when extending data; supply those assignments
again so additions cannot silently reshuffle old examples.

For rationale supervision, the original typed-decision experiment used offline Gemma 4 12B
explanations filtered for format, length and agreement with the original labels. See
[reasoning-trace provenance](METHOD.md#reasoning-trace-provenance) for the procedure and
coverage. Supply accepted text in `rationale`; this CLI does not generate it. Keep gold
traces out of inference requests, and retain records without accepted explanations for
decision training. The student does not need to call the explanation teacher at inference.

## 3. Prepare ModernBERT inputs

Download the pinned Laya initialization used by the original experiments:

```bash
hf download convaiinnovations/laya \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  --local-dir checkpoints/laya-base

diffusion-jev-prepare --family modernbert --base checkpoints/laya-base \
  --input data/typed/raw.jsonl --output data/typed/prepared --max-length 1024
```

The total budget includes 256 reasoning slots and two typed slots. The inherited Laya encoder
caps each option description at 48 tokens and the schema header at 256; the portable adapter
rejects inputs that would be clipped. Keep questions concise, or change the representation
and evaluate it as a new protocol. Overlong rationales fail preparation rather than being
silently truncated. Raw `manifest.json` is written only after all partitions succeed.

To warm-start one of our ModernBERT checkpoints, also pass `--checkpoint PATH` during
preparation so its tokenizer IDs, output tokens and configuration stay identical.

## 4. Train and select

```bash
diffusion-jev-train --data data/typed/prepared --base checkpoints/laya-base \
  --init-kind laya --output outputs/typed-v1 \
  --epochs 5 --batch-size 2 --lr 1e-5 --head-lr 1e-4 \
  --eval-steps 1 --device cuda
```

For an ordinary Hugging Face encoder directory rather than a Laya checkpoint, use
`--init-kind encoder` (the default). Its tokenizer must define CLS, SEP, MASK and PAD.
For fine-tuning a trained diffusion-jev checkpoint, add `--checkpoint checkpoints/diffusion-jev-modernbert`
and prepare with that same checkpoint.

The trainer uses AdamW, 5% warmup, cosine decay to 10% of the initial rate, gradient clipping,
and BF16 on CUDA. The initial model is a validation incumbent. It saves `best/` by validation
soft cross-entropy and `latest/` plus optimizer/scheduler at every completed epoch. Accuracy
uses the mode of the supplied target distribution, with the first key breaking ties.

Multi-GPU:

```bash
torchrun --standalone --nproc_per_node=2 -m diffusion_jev.train \
  --data data/typed/prepared --base checkpoints/laya-base --init-kind laya \
  --output outputs/typed-ddp --epochs 5 --batch-size 2 --device cuda
```

Batch size is **per GPU**. DDP replicates weights; it does not pool VRAM. The reference trainer
evaluates on rank zero while other ranks wait, so continuous high utilization is not a
guarantee. It is designed for readability, not to reproduce the historical utilization tuning.

Resume an interrupted run from a completed epoch with the **same arguments** plus `--resume`.
The original total `--epochs`, world size and data hashes must match. This is not a command
for extending the schedule; use a new output and warm-start weights for a new experiment.
Model/optimizer state is restored; bitwise identical random masks are not promised.

## 5. Gemma adaptation

```bash
hf download google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --local-dir checkpoints/gemma-base

diffusion-jev-prepare --family gemma --base checkpoints/gemma-base \
  --input data/typed/raw.jsonl --output data/typed/gemma --max-length 8192

diffusion-jev-train --data data/typed/gemma --base checkpoints/gemma-base \
  --output outputs/gemma-v1 --epochs 5 --batch-size 1 --lr 1e-4 \
  --eval-steps 1 --device cuda
```

This trains LoRA, workspace/time parameters and the confidence readout while keeping the base
frozen. Backbone checkpointing is enabled in the portable Gemma trainer. Use `--checkpoint`
to initialize from a Titan Gemma adapter. An ordinary PEFT adapter is not interchangeable:
our saved adapter also includes trainable workspace/readout tensors.

Without an audited offline teacher cache, teacher KL is inactive. This is a supervised-data
recipe, not a claim to reproduce the 66,596-row historical CLEF-distilled run. Gold targets
and rationale labels remain separate from inference inputs.

## 6. Evaluate without training on test

Create `data/typed/test.jsonl` containing only the raw records assigned to test. Keep the
full `question` and `state` fields; prepared token IDs are model-specific training artifacts.

Use the recorded registry, including when your original JSONL had no split column:

```bash
python - <<'PY'
import json
from pathlib import Path
registry = json.loads(Path('data/typed/prepared/split-registry.json').read_text())
with Path('data/typed/test.jsonl').open('w') as output:
    for line in Path('data/typed/raw.jsonl').open():
        row = json.loads(line)
        if registry[row['id']]['split'] == 'test':
            output.write(json.dumps(row) + '\n')
PY
```

```bash
diffusion-jev-evaluate --family modernbert --checkpoint outputs/typed-v1/best \
  --input data/typed/test.jsonl --output outputs/typed-v1/test-1step \
  --steps 1 --device cuda
```

Repeat with a new output directory and `--steps 8 --trajectories 2` to assess the historical
ModernBERT sampler. Gemma uses `--family gemma --base checkpoints/gemma-base`, one trajectory,
and one or four passes. All attempted records must succeed; the command never silently drops
hard cases. Reports include accuracy, soft CE, summed multiclass Brier and 15-bin ECE, with
per-source breakdowns. Probabilities are raw; do not call them calibrated merely because
the API returns numbers between zero and one.

Use `scripts/compare_predictions.py` only on identical IDs, ordered keys and targets. It
refuses mismatched cohorts. A fair comparison also needs matched evidence/support, declared
prompt serialization, inference budget and checkpoint-selection rules.

## 7. Save a shareable model folder later

```bash
python scripts/export_checkpoint.py --family modernbert \
  --source outputs/typed-v1/best --output artifacts/hub/diffusion-jev-modernbert \
  --card model_cards/modernbert.md
```

The exporter includes the required weights/configuration/tokenizer, a card and hashes. It
excludes optimizer state and arbitrary neighboring files. Gemma exports adapters only and
records the separate base revision. No upload or remote repository creation occurs.
