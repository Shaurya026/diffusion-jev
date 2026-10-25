"""Forward masking, field-specific objectives and absorbing-state reverse sampling."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def repeat_noise_batch(batch, samples):
    """Replicate examples before independently drawing diffusion noise; loss averages all draws."""
    if samples < 1:
        raise ValueError("noise samples must be positive")
    return (
        {k: v.repeat_interleave(samples, dim=0) for k, v in batch.items()}
        if samples > 1
        else batch
    )


def corrupt(batch, cfg, tokens):
    clean = batch["input_ids"].clone()
    b = len(clean)
    rows = torch.arange(b, device=clean.device)
    target = batch["target"]
    if not torch.allclose(target.sum(-1), torch.ones(b, device=clean.device)):
        raise ValueError("training requires normalized targets")
    choice = torch.multinomial(target, 1).squeeze(-1)
    choice_ids = torch.tensor(tokens["choices"], device=clean.device)
    conf_ids = torch.tensor(tokens["confidence"], device=clean.device)
    conf_bin = (batch["gold_confidence"] * (cfg.confidence_bins - 1)).round().long()
    clean[rows, batch["decision_pos"]] = choice_ids[choice]
    clean[rows, batch["confidence_pos"]] = conf_ids[conf_bin]
    time = cfg.time_eps + (1 - cfg.time_eps) * torch.rand(b, device=clean.device)
    endpoint = torch.rand(b, device=clean.device) < cfg.all_mask_probability
    time = torch.where(endpoint, 1.0, time)
    noisy = clean.clone()
    masks = {}
    for name, power in (
        ("reason", cfg.reasoning_power),
        ("decision", cfg.decision_power),
        ("confidence", cfg.confidence_power),
    ):
        pos = batch[f"{name}_pos"]
        if pos.ndim == 1:
            pos = pos[:, None]
        selected = torch.rand(pos.shape, device=clean.device) < time[:, None].pow(power)
        if name == "reason":
            # Even zero-loss PAD tokens must be corrupted: exposing them leaks gold length.
            selected &= batch["reason_weight"].any(-1, keepdim=True)
        noisy.scatter_(
            1, pos, torch.where(selected, tokens["mask"], clean.gather(1, pos))
        )
        masks[name] = selected
    return noisy, clean, time, masks, conf_bin


def typed_loss(model, output, batch, clean, time, masks, conf_bin):
    """Task-weighted denoising objective, NOT an exact joint likelihood bound."""
    cfg = model.cfg
    logp = F.log_softmax(output["decision"], -1)
    p, q = logp.exp(), batch["target"]
    ce = -(q * logp).sum(-1)
    rps = ((p.cumsum(-1) - q.cumsum(-1)).square() * batch["marker_mask"]).sum(-1)
    rps = rps / (batch["marker_mask"].sum(-1) - 1).clamp_min(1)
    decision_mask = masks["decision"].squeeze(-1)
    # For rho_s(t)=t**gamma_s the masked diffusion rate is rho'_s/rho_s=gamma_s/t.
    decision_rate = cfg.decision_power / time
    losses = {}
    for kind, index, weight in (
        ("choice", 0, cfg.choice_weight),
        ("score", 1, cfg.score_weight),
        ("noul", 2, cfg.noul_weight),
    ):
        per_item = ce + (cfg.rps_weight * rps if kind == "score" else 0)
        # Two-class soft CE here is exactly Bernoulli BCE for noul.
        losses[kind] = (
            per_item * decision_mask * (batch["qtype"] == index) * decision_rate
        ).mean() * weight
    conf_logp = F.log_softmax(output["confidence"], -1)
    grid = torch.linspace(0, 1, cfg.confidence_bins, device=p.device)
    estimate = (conf_logp.exp() * grid).sum(-1)
    conf_ce = F.nll_loss(conf_logp, conf_bin, reduction="none")
    conf_mse = (estimate - batch["gold_confidence"]).square()
    losses["confidence"] = (
        (conf_ce + cfg.confidence_mse_weight * conf_mse)
        * masks["confidence"].squeeze(-1)
        * cfg.confidence_power
        / time
    ).mean()
    losses["confidence"] = losses["confidence"] * cfg.confidence_weight
    # Compute vocabulary projections only for supervised masked reasoning tokens, in chunks.
    selected = masks["reason"] & (batch["reason_weight"] > 0)
    h = output["reason_hidden"][selected]
    labels = clean.gather(1, batch["reason_pos"])[selected]
    denom = batch["reason_weight"].sum(-1).clamp_min(1)
    weights = (
        batch["reason_weight"] / denom[:, None] * cfg.reasoning_power / time[:, None]
    )[selected]
    reason_loss = output["reason_hidden"].sum() * 0.0
    chunk_size = getattr(model, "reasoning_chunk_size", 64)
    for start in range(0, len(h), chunk_size):

        def chunk_loss(hidden, targets, token_weights):
            return (
                F.cross_entropy(model.text_logits(hidden), targets, reduction="none")
                * token_weights
            ).sum()

        reason_loss = reason_loss + checkpoint(
            chunk_loss,
            h[start : start + chunk_size],
            labels[start : start + chunk_size],
            weights[start : start + chunk_size],
            use_reentrant=False,
        )
    losses["reasoning"] = reason_loss / len(p) * cfg.reasoning_weight
    losses["total"] = sum(losses.values())
    return losses


@torch.no_grad()
def sample(model, batch, steps=16, seed=42, greedy=False):
    """Ancestral unmasking. Record typed posteriors BEFORE exposing their sampled values.

    A single trajectory's posterior is conditional on its generated partial canvas.
    Average several trajectories' recorded posteriors to estimate decision marginals.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    model.eval()
    cfg, tokens = model.cfg, model.tokens
    ids = batch["input_ids"].clone()
    b, device = len(ids), ids.device
    generator = torch.Generator(device=device).manual_seed(seed)
    positions = {
        name: batch[f"{name}_pos"] for name in ("reason", "decision", "confidence")
    }
    for name, pos in positions.items():
        if pos.ndim == 1:
            pos = pos[:, None]
            positions[name] = pos
        ids.scatter_(
            1, pos, tokens["mask"]
        )  # ignores ALL supplied gold output contents
    undecided = torch.ones(b, dtype=torch.bool, device=device)
    unconfident = undecided.clone()
    saved_p = torch.zeros_like(batch["target"])
    saved_c = torch.zeros(b, device=device)
    grid = torch.linspace(0, 1, cfg.confidence_bins, device=device)
    for step in range(steps):
        t, s = 1 - step / steps, 1 - (step + 1) / steps
        output = model(batch, ids, torch.full((b,), t, device=device))
        for name, power in (
            ("reason", cfg.reasoning_power),
            ("decision", cfg.decision_power),
            ("confidence", cfg.confidence_power),
        ):
            if name == "reason" and cfg.reasoning_mode != "supervised":
                continue  # latent slots are attention workspace, not generated text
            pos = positions[name]
            unknown = ids.gather(1, pos) == tokens["mask"]
            reveal = unknown & (
                torch.rand(pos.shape, device=device, generator=generator)
                < 1 - (s / t) ** power
            )
            if not reveal.any():
                continue
            logits = (
                model.text_logits(output["reason_hidden"][reveal])
                if name == "reason"
                else output[name][reveal[:, 0]]
            )
            probs = logits.softmax(-1)
            value = (
                probs.argmax(-1)
                if greedy
                else torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            )
            if name == "decision":
                sel = reveal[:, 0] & undecided
                saved_p[sel] = probs
                undecided[sel] = False
                value = torch.tensor(tokens["choices"], device=device)[value]
            elif name == "confidence":
                sel = reveal[:, 0] & unconfident
                saved_c[sel] = (probs * grid).sum(-1)
                unconfident[sel] = False
                value = torch.tensor(tokens["confidence"], device=device)[value]
            current = ids.gather(1, pos)
            current[reveal] = value
            ids.scatter_(1, pos, current)
    if undecided.any() or unconfident.any():
        raise RuntimeError("sampler did not finish")
    return {
        "probabilities": saved_p,
        "schema_confidence": saved_c,
        "reason_ids": ids.gather(1, batch["reason_pos"]),
        "input_ids": ids,
    }
