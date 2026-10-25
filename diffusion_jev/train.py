"""Portable reference trainer: CPU/CUDA or torchrun DDP, validation selection and resume."""

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel, AutoTokenizer
from .runtime import resolve


class Rows(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        with self.path.open("rb") as f:
            f.seek(self.offsets[i])
            return json.loads(f.readline())


def collate(rows, pad, family):
    if family == "gemma":
        from .gemma.data import collate as fn

        return fn(rows, pad)
    from .modernbert.data import collate as fn

    batch = fn(rows, pad)
    batch["confidence_supervised"] = torch.tensor(
        [r.get("confidence_supervised", True) for r in rows]
    )
    return batch


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, family, device, steps):
    model.eval()
    total = correct = count = 0
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
        ):
            if family == "gemma":
                p, _ = model.predict(batch, steps)
            else:
                from .modernbert.process import sample

                p = sample(model, batch, steps=steps, seed=42)["probabilities"]
        target = batch["target"]
        total += float(-(target * p.clamp_min(1e-12).log()).sum())
        correct += int((p.argmax(-1) == target.argmax(-1)).sum())
        count += len(p)
    if not count:
        raise ValueError("Empty evaluation split")
    return {
        "n": count,
        "soft_ce": total / count,
        "accuracy": correct / count,
        "label_protocol": "target argmax; first key breaks ties",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--revision")
    p.add_argument("--checkpoint")
    p.add_argument("--init-kind", choices=["encoder", "laya"], default="encoder")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--eval-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.eval_steps) < 1:
        raise ValueError("epochs/batch/steps must be positive")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}" if a.device == "cuda" else a.device
    if str(device).startswith("cuda"):
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if str(device).startswith("cuda") else "gloo")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    data, out = Path(a.data), Path(a.output)
    meta = json.loads((data / "manifest.json").read_text())
    family = meta["family"]
    for split, entry in meta["splits"].items():
        if (
            hashlib.sha256((data / f"{split}.jsonl").read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError("Prepared data hash mismatch")
    base = resolve(a.base, a.revision)
    tok = AutoTokenizer.from_pretrained(data / "tokenizer", local_files_only=True)
    checkpoint = (
        out / "latest" if a.resume else Path(a.checkpoint) if a.checkpoint else None
    )
    if family == "gemma":
        from .gemma.model import load_model

        model = load_model(meta["letter_ids"], device, base=base)
        model.checkpoint_backbone = True
        if checkpoint:
            model.restore(checkpoint)
    else:
        from .modernbert.config import DiffusionConfig
        from .modernbert.checkpoint import load_model, from_laya
        from .modernbert.common import DecisionModel
        from .modernbert.model import TypedDiffusion

        cfg = DiffusionConfig(**meta["config"])
        if checkpoint:
            model, _ = load_model(checkpoint, device)
            if vars(model.cfg) != vars(cfg) or model.tokens != meta["tokens"]:
                raise ValueError("Prepared data/checkpoint mismatch")
        elif a.init_kind == "laya":
            model = from_laya(base, cfg, meta["tokens"], len(tok)).to(device)
        else:
            encoder = AutoModel.from_pretrained(
                base, local_files_only=True, attn_implementation="sdpa"
            )
            encoder.resize_token_embeddings(len(tok), mean_resizing=False)
            model = TypedDiffusion(
                DecisionModel(encoder, head_layers=2), cfg, meta["tokens"]
            ).to(device)
        model.reasoning_chunk_size = 128
    model_params = [p for p in model.parameters() if p.requires_grad]
    if family == "gemma":
        groups = [{"params": model_params, "lr": a.lr}]
    else:
        groups = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n.startswith("encoder.")
                ],
                "lr": a.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not n.startswith("encoder.")
                ],
                "lr": a.head_lr,
            },
        ]
    optim = torch.optim.AdamW(groups, weight_decay=0.01)
    train, val = Rows(data / "train.jsonl"), Rows(data / "validation.jsonl")
    sampler = DistributedSampler(train, world, rank, seed=a.seed) if world > 1 else None
    fn = partial(collate, pad=tok.pad_token_id, family=family)
    loader = DataLoader(
        train,
        batch_size=a.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=fn,
    )
    vloader = DataLoader(val, batch_size=a.batch_size, collate_fn=fn)
    total_steps = len(loader) * a.epochs
    warmup = max(1, round(0.05 * total_steps))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return (
            0.1
            + 0.9
            * (
                1
                + math.cos(
                    math.pi * min(1, (step - warmup) / max(1, total_steps - warmup))
                )
            )
            / 2
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, schedule)
    args = vars(a).copy()
    args["resume"] = False
    state = {
        "arguments": args,
        "manifest": meta,
        "world": world,
        "epoch": 0,
        "updates": 0,
        "best": float("inf"),
    }
    if a.resume:
        saved = torch.load(
            out / "latest/training.pt", map_location="cpu", weights_only=True
        )
        if any(saved[k] != state[k] for k in ("arguments", "manifest", "world")):
            raise ValueError("Resume needs identical config, data and world size")
        optim.load_state_dict(saved.pop("optimizer"))
        scheduler.load_state_dict(saved.pop("scheduler"))
        state = saved
    elif out.exists():
        raise ValueError("Output exists; choose a fresh output or --resume")
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(
            json.dumps({"arguments": args, "manifest": meta, "world": world}, indent=2)
        )
    if world > 1:
        dist.barrier()
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[local] if str(device).startswith("cuda") else None,
            find_unused_parameters=True,
        )
        if world > 1
        else model
    )

    def save(folder):
        if family == "gemma":
            model.save(
                folder,
                {
                    "family": family,
                    "base": a.base,
                    "revision": a.revision,
                    "epoch": state["epoch"],
                    "step": state["updates"],
                    "letter_ids": meta["letter_ids"],
                },
            )
        else:
            from .modernbert.checkpoint import save_model

            save_model(model, tok, folder)
        (folder / "selection.json").write_text(
            json.dumps(
                {
                    "epoch": state["epoch"],
                    "updates": state["updates"],
                    "metric": "validation.soft_ce",
                    "value": state["best"],
                }
            )
        )

    if not a.resume:
        if rank == 0:
            baseline = validate(model, vloader, family, device, a.eval_steps)
            state["best"] = baseline["soft_ce"]
            save(out / "best")
            (out / "initial-validation.json").write_text(json.dumps(baseline, indent=2))
        if world > 1:
            dist.barrier()
    for epoch in range(state["epoch"], a.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        torch.manual_seed(a.seed + epoch * world + rank)
        wrapped.train()
        for batch in loader:
            batch = move(batch, device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
            ):
                if family == "gemma":
                    losses = wrapped(batch)
                else:
                    from .modernbert.process import corrupt

                    noisy, clean, t, masks, conf = corrupt(
                        batch, model.cfg, model.tokens
                    )
                    unsup = ~batch["confidence_supervised"]
                    noisy[unsup, batch["confidence_pos"][unsup]] = model.tokens["mask"]
                    masks["confidence"] &= batch["confidence_supervised"][:, None]
                    losses = wrapped(batch, noisy, t, (clean, masks, conf))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1, error_if_nonfinite=True)
            optim.step()
            scheduler.step()
            state["updates"] += 1
            if rank == 0 and state["updates"] % 10 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "updates": state["updates"],
                            "loss": float(losses["total"]),
                        }
                    ),
                    flush=True,
                )
        state["epoch"] = epoch + 1
        if rank == 0:
            result = validate(model, vloader, family, device, a.eval_steps)
            if result["soft_ce"] < state["best"]:
                state["best"] = result["soft_ce"]
                save(out / "best")
            save(out / "latest")
            torch.save(
                dict(
                    state,
                    optimizer=optim.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                out / "latest/training.tmp",
            )
            (out / "latest/training.tmp").replace(out / "latest/training.pt")
            with (out / "history.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            epoch=epoch + 1, updates=state["updates"], validation=result
                        )
                    )
                    + "\n"
                )
            print(json.dumps(result), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        (out / "complete.json").write_text(
            json.dumps({"epochs": a.epochs, "updates": state["updates"]})
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
