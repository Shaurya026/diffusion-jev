from types import SimpleNamespace
import copy
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from diffusion_jev.modernbert.common import DecisionModel
from diffusion_jev.modernbert.config import DiffusionConfig, add_schema_tokens
from diffusion_jev.modernbert.data import build_item, collate
from diffusion_jev.modernbert.model import TypedDiffusion
from diffusion_jev.modernbert.process import corrupt, sample
from diffusion_jev.modernbert.checkpoint import save_model, load_model


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "a",
        "b",
        "c",
        "yes",
        "no",
        "true",
        "false",
        "low",
        "high",
        "middle",
        "question",
        "choice",
        "noul",
        "score",
        ":",
        "level",
        "0",
        "1",
        "2",
        "state",
    ]
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    cfg = DiffusionConfig(
        max_len=64,
        head_max_len=24,
        reasoning_tokens=8,
        reasoning_mode="supervised",
        max_options=4,
        confidence_bins=5,
        all_mask_probability=1,
    )
    tokens = add_schema_tokens(tok, cfg)
    encoder = BertModel(
        BertConfig(
            vocab_size=len(tok),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    model = TypedDiffusion(
        DecisionModel(encoder, head_layers=1, dropout=0), cfg, tokens
    )
    row = {"workflow": "test", "id": "0", "state": "state"}
    items = []
    specs = [
        ("choice", {"a": "low", "b": "high"}, {"a": 0.7, "b": 0.3}, "a"),
        ("score", ["low", "middle", "high"], {"0": 0.2, "1": 0.6, "2": 0.2}, "1"),
        ("noul", {"false": "no", "true": "yes"}, {"false": 0.1, "true": 0.9}, "true"),
    ]
    for i, (kind, criteria, probs, label) in enumerate(specs):
        q = {"type": kind, "criteria": criteria, "instructions": "question"}
        gold = {"probabilities": probs, "label": label, "confidence": 0.4}
        items.append(build_item(row, str(i), q, tok, cfg, tokens, gold, "a b"))
    return SimpleNamespace(
        model=model,
        cfg=cfg,
        tokens=tokens,
        tok=tok,
        items=items,
        batch=collate(items, tok.pad_token_id),
    )


def test_corruption_and_typed_gradients(tiny):
    batch, model = tiny.batch, tiny.model
    noisy, clean, time, masks, conf_bin = corrupt(batch, tiny.cfg, tiny.tokens)
    assert all(m.all() for m in masks.values())
    assert torch.equal(time, torch.ones(3))
    assert torch.all(conf_bin == 2)  # provided confidence .4, not max(q)
    for j, item in enumerate(tiny.items):
        assert torch.equal(
            noisy[j, : item["reason_pos"][0]], clean[j, : item["reason_pos"][0]]
        )
    losses = model(batch, noisy, time, (clean, masks, conf_bin))
    assert all(torch.isfinite(v) for v in losses.values())
    assert all(
        losses[k] > 0 for k in ("choice", "noul", "score", "confidence", "reasoning")
    )
    losses["total"].backward()
    assert model.schema_emb.weight.grad[1].abs().sum() > 0
    assert model.confidence_head[-1].weight.grad.abs().sum() > 0
    assert model.text_transform[0].weight.grad.abs().sum() > 0
    assert model.encoder.get_input_embeddings().weight.grad.abs().sum() > 0


def test_decision_loss_reaches_reasoning_attention(tiny):
    model = tiny.model
    noisy, clean, time, masks, conf_bin = corrupt(tiny.batch, tiny.cfg, tiny.tokens)
    losses = model(tiny.batch, noisy, time, (clean, masks, conf_bin))
    (losses["choice"] + losses["score"] + losses["noul"]).backward()
    assert model.schema_emb.weight.grad[1].abs().sum() > 0


def test_generation_ignores_gold_and_finishes(tiny):
    first = sample(tiny.model, tiny.batch, steps=4, seed=19)
    changed = copy.deepcopy(tiny.batch)
    changed["target"] = changed["target"].flip(-1)
    changed["gold_confidence"].fill_(1)
    changed["input_ids"].scatter_(1, changed["reason_pos"], tiny.tokens["reason_pad"])
    second = sample(tiny.model, changed, steps=4, seed=19)
    assert torch.equal(first["probabilities"], second["probabilities"])
    assert torch.equal(first["reason_ids"], second["reason_ids"])
    assert torch.allclose(first["probabilities"].sum(-1), torch.ones(3))
    assert (first["probabilities"][~tiny.batch["marker_mask"]] == 0).all()
    assert (first["reason_ids"] != tiny.tokens["mask"]).all()
    assert ((first["schema_confidence"] >= 0) & (first["schema_confidence"] <= 1)).all()


def test_checkpoint_round_trip(tiny, tmp_path):
    tiny.model.eval()
    before = sample(tiny.model, tiny.batch, steps=2)
    save_model(tiny.model, tiny.tok, tmp_path)
    restored, tok = load_model(tmp_path)
    after = sample(restored, tiny.batch, steps=2)
    assert len(tok) == len(tiny.tok)
    assert torch.allclose(before["probabilities"], after["probabilities"], atol=1e-6)
