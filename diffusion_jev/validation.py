"""Input-budget guards shared by training preparation and live inference."""

from .modernbert.common import build_sequence, render_options


def modern_context(row, tok, cfg):
    q = row["question"]
    internal = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
    if "labels" in q:
        internal["labels"] = q["labels"]
    for text in render_options(internal):
        if (
            len(
                tok(" " + text.replace(tok.mask_token, " "), add_special_tokens=False)[
                    "input_ids"
                ]
            )
            > 48
        ):
            raise ValueError(f"{row['id']}: option exceeds the historical 48-token cap")
    full, _ = build_sequence(tok, row["state"], internal, 100000, 100000)
    actual, _ = build_sequence(
        tok,
        row["state"],
        internal,
        cfg.max_len - cfg.reasoning_tokens - 2,
        cfg.head_max_len,
    )
    if full != actual:
        raise ValueError(f"{row['id']}: context/instruction/option would be truncated")
