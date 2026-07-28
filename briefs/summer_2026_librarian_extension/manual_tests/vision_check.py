#!/usr/bin/env python3
"""Manual live probe: does a VLM actually see the image, and see it correctly?

NOT a pytest test — it needs a running backend with a vision-capable model loaded, so it
lives here under `briefs/` rather than in the suite.

"Describe this image" is a weak test: a model that received nothing at all can still emit a
plausible description, and a model that received a *degraded* image can still name the obvious
subject. So the generated test image carries four independent facts — three shapes, each with
its own colour and position, plus a rendered digit — and the check counts how many come back.
Guessing all four is not plausible; guessing "a red circle" is.

The digit matters most. Shape-and-colour can survive a badly downscaled or mis-normalized
image; small rendered text is the first thing to break, so it doubles as a resolution check on
whatever the mmproj pipeline did to the input.

Also asks the model whether it can see images at all. Older Qwen VLMs (3.0-VL era) would insist
they had only been given a *text description*, while going on to describe the image correctly —
the input was a latent from the mmproj all along. Worth knowing which way a given model answers,
because that self-report has been unreliable in a specific, documented direction.

Requires Pillow locally to synthesize the image; the backend can be remote (e.g. through an ssh
tunnel), since only the encoded image travels.

Usage:
    python vision_check.py                       # localhost:1234, first model
    python vision_check.py <base_url> [model]
"""

import base64
import io
import json
import sys
import urllib.error
import urllib.request
from typing import Any

from PIL import Image, ImageDraw, ImageFont

DEFAULT_BASE = "http://localhost:1234"
TIMEOUT = 300

# Four independent facts. Deliberately unrelated to each other, so getting one right says nothing
# about the others, and colours chosen to be unambiguous under any sane colour handling.
FACTS = {"red circle (upper left)": ("red", "circle"),
         "blue square (lower right)": ("blue", "square"),
         "green triangle (centre)": ("green", "triangle"),
         "the digit 7": ("7",)}


def make_image() -> bytes:
    """Synthesize the test image as PNG bytes. Deterministic."""
    size = 512
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 200, 200], fill="red")                       # upper left
    d.rectangle([320, 320, 470, 470], fill="blue")                  # lower right
    d.polygon([(256, 190), (320, 310), (192, 310)], fill="green")   # centre
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 96)
    except OSError:
        font = ImageFont.load_default()
    d.text((380, 60), "7", fill="black", font=font)                 # upper right
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"}
    except Exception as e:  # noqa: BLE001 -- a probe reports failures rather than raising
        return {"_error": f"{type(e).__name__}: {e}"}


def ask_about_image(base: str, model: str, data_url: str, question: str) -> dict[str, str]:
    body = post(base, {"model": model, "max_tokens": 1200, "temperature": 0.0,
                       "messages": [{"role": "user",
                                     "content": [{"type": "text", "text": question},
                                                 {"type": "image_url",
                                                  "image_url": {"url": data_url}}]}]})
    if "_error" in body:
        return {"content": body["_error"], "reasoning": "", "finish": "error"}
    choice = body.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return {"content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "").strip(),
            "finish": choice.get("finish_reason") or "?"}


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    model = sys.argv[2] if len(sys.argv) > 2 else None
    if model is None:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=TIMEOUT) as r:
                ids = [m["id"] for m in json.loads(r.read().decode("utf-8")).get("data", [])]
        except Exception as e:  # noqa: BLE001
            print(f"cannot reach {base}: {type(e).__name__}: {e}")
            return
        if not ids:
            print(f"{base} reports no models loaded")
            return
        model = ids[0]

    png = make_image()
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    print(f"probing {base} with model {model!r}; image is {len(png)} bytes of PNG")
    print(f"ground truth: {', '.join(FACTS)}\n")

    print("[1] Open description — how many of the four facts come back?")
    got = ask_about_image(base, model, data_url,
                          "Describe this image precisely: every shape, its colour, its position, "
                          "and any text or numbers you can read.")
    if got["finish"] == "error":
        print(f"    REJECTED -- {got['content'][:160]}")
        return
    blob = f"{got['content']}\n{got['reasoning']}".lower()
    hits = [label for label, needles in FACTS.items() if all(n in blob for n in needles)]
    print(f"    {len(hits)}/{len(FACTS)} facts present: {hits}")
    print(f"    reply: {' '.join(got['content'].split())[:400]!r}\n")

    print("[2] The digit alone — small rendered text is the first thing a broken pipeline loses.")
    got = ask_about_image(base, model, data_url,
                          "What single digit is written in this image? Answer with the digit only.")
    print(f"    {'CORRECT' if '7' in got['content'] else 'WRONG'}  "
          f"reply: {' '.join(got['content'].split())[:80]!r}\n")

    print("[3] Self-report — older Qwen VLMs claimed they were given a text description instead.")
    got = ask_about_image(base, model, data_url,
                          "Are you seeing an actual image, or were you given a text description of one? "
                          "Answer honestly in one sentence.")
    print(f"    reply: {' '.join(got['content'].split())[:300]!r}")
    print("\ndone")


if __name__ == "__main__":
    main()
