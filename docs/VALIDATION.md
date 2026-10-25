# Release validation — 25 October 2026

These checks apply to the standalone source distribution. They do not establish a new
benchmark result or reproduce the historical large-scale training schedules.

| Check | Observed result |
|---|---|
| CPU test suite | 14 tests passed, including decision-to-reasoning gradient checks |
| Training integration | Tiny real transformer: prepare, train two updates, save/select, load resume state, evaluate held-out input |
| Architecture contracts | Decision-only losses reach reasoning-role/slot embeddings in ModernBERT and Gemma; attention, frozen weights and checkpoint round trips checked |
| Release contracts | Context overflow rejected; local checkpoint export reloads identical weights and excludes optimizer state |
| Python checks | Ruff selected error checks and compileall passed |
| Distribution | Renamed diffusion-jev wheel built and installed locally; package imports and all four renamed CLI entry points checked outside the source tree |
| Public dataset | Pinned 8,000-row fetch and complete ModernBERT preparation succeeded |
| GPU inference | One H100 80GB; 48 actual predictions across three checkpoint families; worker job completed successfully |
| Gradio | Replay served locally; browser exercised model/case selection and submission; actual screenshots became two GIFs |
| Source audit | No private absolute paths, credential patterns, broken relative Markdown links or files larger than 10 MB found |

The public recipe has **4,765 train / 630 validation / 605 calibration / 2,000 test**
decisions. [Dataset and split hashes](../results/public_recipe.json) record the exact prepared
outputs. This is a newly defined public split, distinct from the historical experiment.

GPU output hashes and hardware appear in [demo provenance](../assets/demo_provenance.json).
The [browser frame manifest](../assets/recording.json) records the displayed cases and answers.
All 48 outputs remain available, including failures. The tiny example set is not a benchmark.

The CPU host emitted driver/NVML and optional torchao compatibility warnings; CPU tests passed.
Large-model inference ran separately on the H100 worker with the recorded environment.
The checks above were run locally. GitHub Actions status is tracked separately on the
[repository's Actions page](https://github.com/Shaurya026/diffusion-jev/actions).
Multi-GPU reference training, full public-data convergence, live image
conditioning and exact historical score reproduction were not revalidated in this release.

The integration test reloads a completed run's optimizer/scheduler state; it is not a
mid-epoch recovery or crash-consistency test. See [reproducibility scope](REPRODUCIBILITY.md)
and [the unresolved historical Gemma result discrepancy](RESULTS.md).

The reasoning-workspace gradient checks establish a computational connection. They do not
establish an accuracy gain caused by rationale supervision or the 256-slot choice. The
method documentation distinguishes implemented behavior, measured system results and
proposed controlled ablations. Demo GIFs were re-recorded with diffusion-jev branding while
retaining the same model predictions and their provenance.
