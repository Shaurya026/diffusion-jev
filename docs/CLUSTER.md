# Running on a GPU cluster

The library and CLIs work on a workstation. Kubernetes is an optional execution mechanism.
Use the same Python environment on the login machine and workers, mount your code/data, and
pass explicit paths. Training code does not assume a username, namespace, hostPath or scheduler.

The owner's original runs activated a shared `vlms` environment and used `kubectl`; those
site-specific paths belong in a private deployment manifest, not in this public source.
`artifacts/demo-job.yaml` in the local working copy is such an ignored private manifest.

Minimal container command for a two-GPU allocation:

```bash
export USE_TF=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=2 -m diffusion_jev.train \
  --data /work/data/prepared --base /work/checkpoints/base \
  --output /work/outputs/run1 --epochs 5 --batch-size 2 --device cuda
```

Request two `nvidia.com/gpu` devices and enough host RAM for model loading and worker data.
Change `--init-kind laya` for a Laya base and add the appropriate learning rates from TRAINING.md.
Begin with actual longest-example forward/backward probes before increasing batch size.
Expandable memory segments helped the historical variable-length Scout workload avoid
fragmentation; they do not guarantee freedom from OOM.

Record `nvidia-smi` compute and occupied VRAM, plus live PyTorch allocation and reservation.
Loading, parsing, validation and checkpoint writes can reduce utilization. The readable
reference loop does not promise both metrics above 85% continuously and adds no filler memory
or background burn kernels. A cluster enforcing a hard instantaneous threshold needs an
appropriate job class or agreed measurement window.
