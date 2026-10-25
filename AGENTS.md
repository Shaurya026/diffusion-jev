# Working in diffusion-jev

This is the standalone distribution. Do not introduce imports from a sibling Laya checkout
or absolute paths from the author's machine. Preserve compatibility with the documented
checkpoint formats and keep gold fields outside inference inputs.

Run the CPU tests and lint/compile commands in CONTRIBUTING.md. Large GPU checks need an
explicit local checkpoint; never replace model output with a fabricated demo answer.
Keep artifacts/ ignored. No GitHub push or model upload is implied by code maintenance.
Historical aggregate scores have different cohort/label protocols; retain RESULTS.md caveats.
