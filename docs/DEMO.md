# Demo and GIF reproduction

## CPU replay

```bash
pip install -e '.[demo,dev]'
python app.py
```

Select a model, case and pass count, then press **Show recorded decision**. The banner clearly
identifies replay. Results come from the exact request/model/pass-count key in
`assets/demo_records.json`; unsupported combinations fail rather than fabricate a response.
Evidence/schema editing is disabled in replay mode.

## Live inference

```bash
python app.py --checkpoint checkpoints/diffusion-jev-modernbert --family modernbert --device cuda
python app.py --checkpoint checkpoints/titan-gemma --family gemma \
  --base checkpoints/gemma-base --device cuda
```

Live mode loads an actual checkpoint, enables input editing and runs `DecisionEngine` for
each submitted request. It does not call a hosted model. The app defaults to loopback and
does not enable a public Gradio share link. Start separate processes/ports for different
live models rather than keeping multiple large models resident accidentally.

The first public UI is a **text decision interface**. Image conditioning is exposed through
the Scout CLI with `--image PATH --vision-base checkpoints/siglip-base-patch16-224`. The
Gemma adapter is text-only. Do not infer native image support from the base model's branding.

## Regenerate inference records

Copy `configs/local-paths.example.json` to an ignored local file and fill in your checkpoint
directories. On a GPU worker:

```bash
python scripts/run_demos.py --paths artifacts/local-paths.json \
  --family gemma --output artifacts/demos-new/gemma
```

Run separately for `modernbert` and `scout`. Each run refuses to overwrite an existing output.
It executes all eight predefined cases at both one and four passes, records every result,
and hashes the checkpoint and example set. Expected labels are used only for reporting;
the engine allowlists state/question/ID and never consumes expected answers.

To update the replay artifact, combine the three `records.json` lists into
`assets/demo_records.json` and their `provenance.json` objects into `assets/demo_provenance.json`.
Retain failures and record any changed example/version hashes. Do not hand-edit predictions
to make a GIF look better.

## Record the browser

```bash
python -m playwright install chromium
python app.py
# In a second terminal:
python scripts/record_demo.py --url http://127.0.0.1:7860
```

The recorder drives the actual interface and takes browser screenshots for invoice review,
support routing and opaque-label few-shot routing. It writes one GIF per model plus a frame
manifest. Each frame remains visible for 2.5 seconds for readability; that duration is not
inference latency. The first request's measured latency may include kernel warmup.

Fresh recorded results: ModernBERT 13/16 matches, Scout 12/16, Gemma 16/16. These illustrative
examples were authored for explanation and are not a generalization benchmark. The ModernBERT
GIF intentionally retains its few-shot-label mismatch.
