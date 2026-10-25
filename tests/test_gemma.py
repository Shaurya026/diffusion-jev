"""Small real Gemma architecture checks; no downloaded checkpoint or GPU needed."""

import copy
import tempfile

import torch
from transformers import Gemma4TextConfig, Gemma4TextModel

from diffusion_jev.gemma.data import collate
from diffusion_jev.gemma.model import TitanGemmaDiffusion, attention_masks


def test_gemma_invariants():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = Gemma4TextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        hidden_size_per_layer_input=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        sliding_window=32,
        final_logit_softcapping=30.0,
    )
    cfg._attn_implementation = "sdpa"
    model = TitanGemmaDiffusion(Gemma4TextModel(cfg), list(range(10, 74)), rank=2)
    model.eval()
    valid = torch.tensor([[True, True, True, False]])
    masks = attention_masks(valid, 2)
    assert (
        masks["sliding_attention"][0, 0, 0, 1]
        and not masks["sliding_attention"][0, 0, 0, 2]
    )
    assert not masks["full_attention"][0, 0, 0, 3]
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        a = model.backbone(
            input_ids=ids, attention_mask=masks, use_cache=False
        ).last_hidden_state
        ids[0, 2] = 9
        b = model.backbone(
            input_ids=ids, attention_mask=masks, use_cache=False
        ).last_hidden_state
        assert not torch.allclose(a[:, 0], b[:, 0]), "future token invisible"
        ids[0, 3] = 99
        c = model.backbone(
            input_ids=ids, attention_mask=masks, use_cache=False
        ).last_hidden_state
        assert torch.allclose(b[:, :3], c[:, :3], atol=1e-6), "padding leaks"
        causal = model.backbone(
            input_ids=ids, attention_mask=valid, use_cache=False
        ).last_hidden_state
        ids[0, 2] = 7
        causal2 = model.backbone(
            input_ids=ids, attention_mask=valid, use_cache=False
        ).last_hidden_state
        assert torch.allclose(causal[:, 0], causal2[:, 0], atol=1e-6), (
            "baseline not causal"
        )
    row = dict(
        ids=[2, 5, 6],
        target=[0.0, 1.0],
        teacher=[0.1, 0.9],
        teacher_weight=1.0,
        reason=[9, 10, 1],
        qtype="choice",
        id="test",
    )
    batch = collate([row])
    with torch.no_grad():
        pred, _ = model.predict(batch)
        altered = collate([{**row, "target": [1.0, 0.0], "reason": [88, 89, 1]}])
        other, _ = model.predict(altered)
        assert torch.equal(pred, other), "gold labels/rationales enter inference"
    model.train()
    # Choice supervision alone must reach reasoning slots through shared attention.
    # All output inputs are masked; no rationale loss or gold text is involved here.
    fully_masked = batch["ids"].clone()
    fully_masked.scatter_(1, batch["positions"], 4)
    output = model.readout(batch, fully_masked, torch.ones(len(fully_masked)))
    decision_only = (
        -(batch["target"] * output["decision"].log_softmax(-1)).sum(-1).mean()
    )
    decision_only.backward()
    assert model.slot_embeddings.grad[:256].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    torch.manual_seed(88)
    loss = model(batch)["total"]
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_B" in n
    )
    assert model.slot_embeddings.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if n in frozen)
    gradients = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    model.checkpoint_backbone = True
    torch.manual_seed(88)
    check_loss = model(batch)["total"]
    check_loss.backward()
    assert torch.allclose(loss, check_loss, atol=1e-6)
    for n, p in model.named_parameters():
        if n in gradients:
            assert torch.allclose(p.grad, gradients[n], atol=1e-5, rtol=1e-4), n
    with tempfile.TemporaryDirectory() as folder:
        model.save(folder, {"test": True})
        snapshot = copy.deepcopy(model.slot_embeddings.detach())
        with torch.no_grad():
            model.slot_embeddings.add_(1)
        model.restore(folder)
        assert torch.equal(snapshot, model.slot_embeddings)
    print(
        "PASS: future attention, causal baseline, padding isolation, no gold leakage, LoRA gradients, shared-KV checkpoint gradient parity, adapter roundtrip"
    )
