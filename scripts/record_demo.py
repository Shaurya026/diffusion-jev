"""Record the actual local replay UI; never synthesize model outputs or timing."""

import argparse
import io
import json
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--output", default="assets")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    cases = {
        row["id"]: row
        for row in json.loads((root / "examples/demo_cases.json").read_text())
    }
    records = json.loads((root / "assets/demo_records.json").read_text())
    labels = {
        (row["result"]["family"], row["request"]["id"], row["result"]["steps"]): row[
            "result"
        ]["label"]
        for row in records
    }
    evidence = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(
            viewport={"width": 1240, "height": 1000}, device_scale_factor=1
        )
        for family in ["gemma", "modernbert"]:
            page.goto(a.url, wait_until="domcontentloaded")
            page.locator("#model input").fill(family)
            page.get_by_role("option", name=family, exact=True).click()
            frames = []
            for name in ["invoice-review", "support-route", "few-shot-routing"]:
                page.locator("#example input").fill(name)
                page.get_by_role("option", name=name, exact=True).click()
                page.wait_for_function(
                    '(expected) => document.querySelector("#evidence textarea").value === expected',
                    arg=cases[name]["state"],
                )
                page.locator("#run").click()
                page.wait_for_function(
                    '(expected) => document.querySelector("#answer h2")?.innerText === expected',
                    arg=labels[family, name, 1],
                )
                shot = page.screenshot(full_page=True)
                im = Image.open(io.BytesIO(shot)).convert("RGB")
                frames.append(im)
                evidence.append(
                    {
                        "family": family,
                        "example": name,
                        "steps": 1,
                        "visible_answer": page.locator("#answer h2").inner_text(),
                    }
                )
            frames[0].save(
                out / f"{family}_demo.gif",
                save_all=True,
                append_images=frames[1:],
                duration=[2500] * len(frames),
                loop=0,
                optimize=False,
            )
            frames[0].save(out / f"{family}_demo.png")
        browser.close()
    (out / "recording.json").write_text(
        json.dumps(
            {
                "mode": "actual browser screenshots of recorded-GPU replay UI",
                "frames": evidence,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
