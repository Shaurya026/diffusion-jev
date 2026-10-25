"""Gradio: honest CPU replay of recorded GPU results, or live local checkpoint inference."""

import argparse
import html
import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent


def build(engine=None):
    examples = json.loads((ROOT / "examples/demo_cases.json").read_text())
    cases = {r["id"]: r for r in examples}
    path = ROOT / "assets/demo_records.json"
    records = json.loads(path.read_text()) if path.exists() else []
    families = sorted({r["result"]["family"] for r in records}) or ["modernbert"]
    live = engine is not None

    def load_example(name):
        row = cases[name]
        return row["state"], json.dumps(row["question"], indent=2)

    def infer(name, family, steps, state, question):
        if live:
            row = {"id": name, "state": state, "question": json.loads(question)}
            result = engine.predict(row, steps=int(steps))
            expected = (
                cases[name]["expected_label"]
                if state == cases[name]["state"]
                and row["question"] == cases[name]["question"]
                else None
            )
        else:
            found = [
                r
                for r in records
                if r["request"]["id"] == name
                and r["result"]["family"] == family
                and r["result"]["steps"] == int(steps)
            ]
            if not found:
                raise gr.Error(
                    "No recorded result for this exact configuration. Use live mode to run a checkpoint."
                )
            result = found[0]["result"]
            expected = found[0]["expected_label"]
        bars = "".join(
            f'<div class="prob-row"><span>{html.escape(k)}</span><div class="track"><div class="fill" style="width:{v * 100:.2f}%"></div></div><strong>{v:.1%}</strong></div>'
            for k, v in result["probabilities"].items()
        )
        match = (
            "Reference not supplied"
            if expected is None
            else f"Reference: {expected} · "
            + ("match" if result["label"] == expected else "mismatch")
        )
        card = f'<div class="result-card"><div class="eyebrow">TYPED DECISION</div><h2>{html.escape(result["label"])}</h2>{bars}<p>{html.escape(match)}</p></div>'
        timing = f"**{result['family']} · {result['steps']} step(s) · {result['elapsed_seconds']:.3f}s measured request time**\n\n{'Live inference' if live else 'Replayed from recorded H100 inference'}. Tokenization included; model load excluded. Single requests are not throughput benchmarks."
        return card, timing, result

    css = """
    .gradio-container {max-width:1180px!important; margin:auto!important; background:#faf8f4!important;}
    .hero {padding:28px 8px 18px}.hero h1{font-size:40px!important;letter-spacing:-1.3px;margin:8px 0!important;}
    .eyebrow{color:#13876e;font-size:12px;font-weight:700;letter-spacing:2px}
    .hero p{color:#626975;font-size:17px;max-width:840px}
    .result-card{padding:24px;border:1px solid #d5e4dd;border-radius:14px;background:#fff;min-height:240px}
    .result-card h2{font-size:36px;color:#13876e;margin:8px 0 24px}
    .prob-row{display:flex;align-items:center;gap:14px;margin:14px 0}.prob-row span{width:85px}.prob-row strong{width:55px}
    .track{flex:1;height:12px;background:#f0ede7;border-radius:8px;overflow:hidden}.fill{height:100%;background:#13876e;border-radius:8px}
    footer{display:none!important}
    """
    with gr.Blocks(
        title="diffusion-jev · Typed Decisions",
        css=css,
        theme=gr.themes.Soft(primary_hue="emerald"),
    ) as demo:
        gr.HTML(
            '<div class="hero"><div class="eyebrow">DIFFUSION-JEV / RESEARCH DEMO</div><h1>Evidence in. Typed decisions out.</h1><p>Explore choice, yes/no and ordered-score decisions with a shared reasoning workspace.</p></div>'
        )
        gr.Markdown(
            "**LIVE LOCAL CHECKPOINT**"
            if live
            else "**RECORDED GPU INFERENCE** · CPU replay of actual H100 results. No model is loaded in replay mode."
        )
        with gr.Row():
            model = gr.Dropdown(
                [engine.family] if live else families,
                value=engine.family if live else families[0],
                label="Model",
                interactive=not live,
                elem_id="model",
            )
            steps = gr.Radio([1, 4], value=1, label="Denoising passes", elem_id="steps")
            example = gr.Dropdown(
                list(cases), value="invoice-review", label="Example", elem_id="example"
            )
        with gr.Row():
            with gr.Column():
                state = gr.Textbox(
                    value=cases["invoice-review"]["state"],
                    label="Evidence",
                    lines=5,
                    interactive=live,
                    elem_id="evidence",
                )
                question = gr.Code(
                    value=json.dumps(cases["invoice-review"]["question"], indent=2),
                    language="json",
                    label="Question + allowed answers",
                    interactive=live,
                )
                run = gr.Button(
                    "Run decision" if live else "Show recorded decision",
                    variant="primary",
                    elem_id="run",
                )
            with gr.Column():
                answer = gr.HTML(
                    '<div class="result-card">Choose an example, then run the decision.</div>',
                    elem_id="answer",
                )
                timing = gr.Markdown(
                    "Model output and measured timing will appear here."
                )
                with gr.Accordion("Raw result and confidence semantics", open=False):
                    raw = gr.JSON()
        gr.Markdown(
            "Examples are handwritten demonstrations, not a benchmark. All attempted cases and failures are retained. Confidence fields differ between model families; generated explanations are unverified. Gemma here is text-only; use the Scout CLI for image inputs."
        )
        example.change(load_example, example, [state, question])
        run.click(
            infer,
            [example, model, steps, state, question],
            [answer, timing, raw],
            concurrency_limit=1,
        )
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument(
        "--family", default="modernbert", choices=["modernbert", "scout", "gemma"]
    )
    p.add_argument("--base")
    p.add_argument("--vision-base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    engine = None
    if a.checkpoint:
        from diffusion_jev.runtime import DecisionEngine

        engine = DecisionEngine(a.checkpoint, a.family, a.device, a.base, a.vision_base)
    build(engine).queue().launch(
        server_name=a.host, server_port=a.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
