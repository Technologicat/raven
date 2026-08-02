#!/usr/bin/env python3
"""Manual live probe: at realistic scale, does retrieved material still have to sit at the front?

NOT a pytest test — it needs a running backend with a model loaded, so it lives here
under `briefs/` rather than in the suite.

`scaffold._perform_injects` inserts every RAG match at `history.insert(1, ...)`, ahead of the
whole conversation. That placement is deliberate: it dates from Qwen 3.0, which would not engage
with material injected late. It costs a full KV-cache prefix rebuild on every single turn, because
the prompt's prefix changes each time the matches do.

`inject_shapes.py` (probe 4) found the constraint did not reproduce on any of four current models —
but with one short fact in a nearly empty context, which is not the case that motivated the front
placement. The real shape is `docs_num_results = 20` *results*, each a merged span of one to a few
1000-character chunks, where long-context attention is what degrades. This probe measures that
case, and varies the two things that plausibly matter besides position:

  - **where the needle sits inside the block** (first / middle / last), since a lost-in-the-middle
    effect would show up as a depth dependence rather than a placement one;
  - **the role**, `user` against `tool` with a synthetic call — the latter being required for
    Gemma 4, which ignores a bare tool message and confabulates in its place.

The corpus is synthesized here rather than read from disk, so the probe is self-contained and
reproducible on any machine, and no copyrighted abstracts are involved. It is deterministic:
no randomness, so two runs of the same model are comparable.

A **needle-absent control** runs alongside every condition. Without it a model that happens to
guess, or that confabulates a plausible figure, is indistinguishable from one that read the
material — and confabulation on retrieval paths is a demonstrated failure mode here, not a
hypothetical one.

Usage:
    python rag_placement.py                                  # localhost:1234, first model, k=20
    python rag_placement.py <base_url> [model] [n_results] [filter]

The filter selects conditions: a role ("tool+call"), a placement ("before"), or a pair
("tool+call@before"), comma-separated. Omit it to run the whole grid.
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://localhost:1234"
TIMEOUT = 600

CLOSED_THINK = "<think>\n\n</think>\n\n"

SYSTEM_PROMPT = ("You are Aria, a helpful research assistant. Answer the user's questions "
                 "accurately and concisely, using the provided knowledge-base material.")

# The planted fact. The system name is invented, and the figure is specific enough that hitting it
# by chance is not plausible — so an exact match means the material was read, not recalled.
NEEDLE_SYSTEM = "the Vantaa-3 pressurized alkaline stack"
NEEDLE_VALUE = "41.7"
NEEDLE_FACT = (f"Long-duration testing of {NEEDLE_SYSTEM} established a specific energy consumption of "
               f"{NEEDLE_VALUE} kWh per kilogram of hydrogen at a current density of 0.62 A/cm2 and a "
               f"stack temperature of 82 degrees Celsius, sustained across 4,000 hours of operation "
               f"with a measured degradation rate of 3.1 microvolts per hour.")
NEEDLE_QUESTION = (f"According to the knowledge base, what specific energy consumption was measured for "
                   f"{NEEDLE_SYSTEM}?")

# Distractors, deliberately from the same field as the needle and carrying figures of their own, so
# the needle is not the only numeric passage and cannot be found by shape alone.
TOPICS = [
    ("proton-exchange-membrane electrolysis", "iridium-oxide anode catalysts",
     "a loading reduction to 0.35 mg/cm2", "cell voltage rose by 24 mV at 2 A/cm2"),
    ("solid-oxide electrolysis", "scandia-stabilized zirconia electrolytes",
     "operation at 780 degrees Celsius", "area-specific resistance settled near 0.28 ohm cm2"),
    ("steam methane reforming with carbon capture", "nickel-alumina catalyst beds",
     "a steam-to-carbon ratio of 3.4", "capture fractions of 91 percent were sustained"),
    ("autothermal reforming", "rhodium-substituted perovskite supports",
     "an oxygen-to-carbon ratio of 0.52", "methane conversion held above 96 percent"),
    ("biomass gasification", "olivine bed material with calcined dolomite",
     "gasification at 850 degrees Celsius", "tar yields fell to 4.2 g per normal cubic metre"),
    ("chemical looping hydrogen production", "iron-titanium oxygen carriers",
     "twenty redox cycles", "carrier conversion stabilized at 87 percent"),
    ("photocatalytic water splitting", "nitrogen-doped strontium titanate",
     "irradiation at 420 nm", "apparent quantum yield reached 2.8 percent"),
    ("ammonia cracking", "ruthenium on cerium oxide",
     "a gas hourly space velocity of 12,000 per hour", "conversion exceeded 98 percent at 550 degrees"),
    ("anion-exchange-membrane electrolysis", "nickel-iron layered double hydroxides",
     "a potassium hydroxide concentration of 1 molar", "the cell sustained 1.0 A/cm2 at 1.87 V"),
    ("underground hydrogen storage", "depleted sandstone reservoirs",
     "cushion-gas fractions between 30 and 45 percent", "recovery factors of 78 percent were modelled"),
    ("methane pyrolysis", "molten tin-bismuth bubble columns",
     "residence times near 12 seconds", "carbon separation efficiency reached 94 percent"),
    ("alkaline water electrolysis", "Raney-nickel cathode coatings",
     "a zero-gap configuration", "the overpotential dropped by 118 mV"),
    ("sulfur-iodine thermochemical cycling", "Bunsen-reaction phase separation",
     "a peak process temperature of 900 degrees Celsius", "cycle efficiency was estimated at 38 percent"),
    ("hydrogen compression", "metal-hydride thermal compressors",
     "a three-stage arrangement", "delivery pressures of 700 bar were achieved"),
    ("offshore wind-coupled electrolysis", "dynamic load-following control",
     "capacity factors near 0.54", "curtailment losses fell below 6 percent"),
    ("hydrogen liquefaction", "ortho-para conversion catalysts",
     "a mixed-refrigerant precooling stage", "specific work reached 6.4 kWh per kilogram"),
    ("photoelectrochemical tandem cells", "bismuth-vanadate photoanodes",
     "one-sun illumination", "solar-to-hydrogen efficiency reached 7.9 percent"),
    ("fermentative hydrogen production", "mixed anaerobic consortia",
     "an initial pH of 5.5", "yields of 2.1 mol hydrogen per mol glucose were observed"),
    ("hydrogen embrittlement of pipeline steel", "X70 line-pipe specimens",
     "blend fractions up to 20 percent", "fatigue crack growth rates rose by a factor of 3.6"),
    ("techno-economic assessment of green hydrogen", "regional electricity price series",
     "electrolyser capital costs of 1,100 euro per kilowatt", "levelized cost fell to 4.3 euro per kilogram"),
    ("membrane-reactor water-gas shift", "palladium-silver permeable membranes",
     "a sweep-gas ratio of 1.8", "hydrogen recovery reached 89 percent"),
    ("seawater electrolysis", "selective manganese-oxide overlayers",
     "chloride concentrations of 0.5 molar", "no hypochlorite evolution was detected over 500 hours"),
]


# --------------------------------------------------------------------------------
# HybridIR's shape, so that "20 results" here means what it means in Librarian
#
# `chunk_size = 1000` characters with `overlap_fraction = 0.25` gives a sliding window of stride 750.
# A search result is NOT one chunk: `merge_contiguous_spans` seamlessly joins adjacent matched chunks
# from the same document (overlaps removed), and `k` counts results *after* that merge. So a result
# spanning n chunks is about 1000 + (n - 1) * 750 characters. Most results are a single chunk; some
# are runs of two or three. Modelling that distribution matters, because the volume of injected text
# is exactly the variable this probe is about.
SPAN_PATTERN = (1, 1, 2, 1, 1, 3, 1, 2, 1, 1)  # deterministic stand-in for the real spread


def span_chars(n_chunks_in_span: int) -> int:
    return 1000 + (n_chunks_in_span - 1) * 750


def _sentence(index: int, j: int) -> str:
    """One sentence of plausible filler, distinct for every (index, j) pair.

    Distinctness is the point: a long corpus assembled from a handful of repeated sentences is both
    unrealistic and *easier* than the real thing, since the model can answer from any one copy.
    """
    technology, material, condition, finding = TOPICS[index % len(TOPICS)]
    seed = index * 97 + j * 31
    cohort = 40 + seed % 260
    duration = 200 + (seed * 13) % 1800
    delta = 1.2 + (seed % 170) / 10
    temperature = 60 + seed % 380
    pressure = 1 + seed % 60
    templates = (
        f"This study investigates {technology} using {material}, carried out under {condition}.",
        f"Across the measurement series, {finding}.",
        f"Trial {chr(65 + seed % 26)}-{1000 + seed % 9000} ran {cohort} samples over {duration} hours.",
        f"Relative to the control arm the improvement was {delta:.1f} percent.",
        f"Operation at {temperature} degrees Celsius and {pressure} bar was maintained throughout.",
        f"Balance-of-plant losses accounted for {delta:.1f} percent of the total energy penalty.",
        "Sensitivity analysis indicates the outcome is dominated by feedstock quality rather than reactor geometry.",
        f"Degradation over {duration} hours remained within the {delta:.1f} percent envelope specified for pilot operation.",
        f"Comparison against the {technology} baseline shows no statistically significant divergence at {cohort} samples.",
        f"The authors note that accelerated-stress protocols understate thermal cycling effects for {material}.",
    )
    return templates[j % len(templates)]


def passage(index: int, target_chars: int) -> str:
    """Synthesize one retrieved result of roughly `target_chars`, deterministically."""
    parts: list[str] = []
    total = 0
    j = 0
    while total < target_chars:
        s = _sentence(index, j)
        parts.append(s)
        total += len(s) + 1
        j += 1
    return " ".join(parts)


def build_corpus(n_results: int, needle_at: int | None) -> list[str]:
    """Return `n_results` knowledge-base match blocks, with the needle at index `needle_at`.

    `needle_at` of `None` builds the control corpus, which contains no needle at all — the
    condition that separates a model reading the material from one guessing a plausible figure.
    """
    blocks = []
    for i in range(n_results):
        if needle_at is not None and i == needle_at:
            # The needle sits inside a full-size result, not alone in a short one; a conspicuously
            # short passage among long ones would be findable by shape rather than by reading.
            filler = passage(i, span_chars(SPAN_PATTERN[i % len(SPAN_PATTERN)]) - len(NEEDLE_FACT))
            body, source = f"{filler} {NEEDLE_FACT}", "vantaa3_stack_report.txt"
        else:
            body = passage(i, span_chars(SPAN_PATTERN[i % len(SPAN_PATTERN)]))
            source = f"abstract_{i:03d}.txt"
        blocks.append(f"[System information: Knowledge-base match from '{source}'.]\n\n{body}\n-----")
    return blocks


def post(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:160]}"}
    except Exception as e:  # noqa: BLE001 -- a probe reports failures rather than raising
        return {"_error": f"{type(e).__name__}: {e}"}


def material_messages(role: str, blocks: list[str]) -> list[dict]:
    """The retrieved material as it would go on the wire, for one role choice.

    The two tool variants differ in a way that turns out to matter. `tool+call` emits one `tool`
    message per result, all carrying the same `tool_call_id` — which is not what the OpenAI schema
    describes, since a `tool` message answers *one* call. `tool+call-merged` sends the whole result
    set as a single `tool` message, which is schema-correct. Most models tolerate the first; Gemma
    E4B does not, and reads none of the material.
    """
    if role in ("tool+call", "tool+call-merged"):
        call = {"role": "assistant", "content": "",
                "tool_calls": [{"id": "call_rag", "type": "function",
                                "function": {"name": "search_documents",
                                             "arguments": json.dumps({"query": "specific energy consumption"})}}]}
        if role == "tool+call-merged":
            return [call, {"role": "tool", "tool_call_id": "call_rag", "content": "\n\n".join(blocks)}]
        return [call] + [{"role": "tool", "tool_call_id": "call_rag", "content": b} for b in blocks]
    return [{"role": role, "content": b} for b in blocks]


def run_one(base: str, model: str, role: str, where: str, blocks: list[str]) -> dict[str, Any]:
    """One condition: material in `role`, placed at `where`, question asked at the end."""
    # Filler so that "front" and "end" are genuinely different positions relative to the conversation,
    # rather than adjacent messages in a two-message history.
    filler = [{"role": "user", "content": "I'm reviewing the hydrogen production literature this week."},
              {"role": "assistant", "content": "Happy to help — tell me what you need from it."}]
    material = material_messages(role, blocks)
    question = {"role": "user", "content": NEEDLE_QUESTION}

    if where == "front":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *material, *filler, question]
    elif where == "before":
        # Between the conversation and the user's question. Keeps the cache benefit of a late insert —
        # everything ahead of the material is still a stable prefix — while leaving the *last* message
        # the user's question rather than a tool result, which is what invites Qwen 3.6 to answer a
        # `tool+call` block by emitting another tool call instead of an answer.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *filler, *material, question]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *filler, question, *material]

    body = post(base, {"model": model,
                       "messages": messages + [{"role": "assistant", "content": CLOSED_THINK}],
                       "max_tokens": 1500, "temperature": 0.0})
    if "_error" in body:
        return {"error": body["_error"], "prompt_tokens": None, "content": ""}
    choice = body.get("choices", [{}])[0]
    return {"error": None,
            "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
            "finish": choice.get("finish_reason") or "?",
            "content": (choice.get("message", {}).get("content") or "").strip()}


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    model = sys.argv[2] if len(sys.argv) > 2 else None
    n_results = int(sys.argv[3]) if len(sys.argv) > 3 else 20

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

    depths = {"first": 0, "middle": n_results // 2, "last": n_results - 1}
    print(f"probing {base} with model {model!r}, {n_results} knowledge-base results")
    print(f"needle: {NEEDLE_VALUE} kWh/kg for {NEEDLE_SYSTEM}\n")

    # Optional filter, so a single condition can be re-run without paying for the whole grid. A token
    # matches a role ("tool+call"), a placement ("before"), or a specific pair ("tool+call@before").
    wanted = {t.strip() for t in sys.argv[4].split(",")} if len(sys.argv) > 4 else None

    for role in ("user", "tool+call", "tool+call-merged"):
        for where in ("front", "before", "end"):
            if wanted is not None and not (wanted & {role, where, f"{role}@{where}"}):
                continue
            print(f"  --- material as {role}, placed at the {where} ---")
            for depth_label, depth in depths.items():
                got = run_one(base, model, role, where, build_corpus(n_results, depth))
                if got["error"]:
                    print(f"    needle {depth_label:<7} REJECTED -- {got['error'][:90]}")
                    continue
                found = NEEDLE_VALUE in got["content"]
                tokens = got["prompt_tokens"]
                # `finish` is load-bearing: a model whose reasoning arrives inline in `content`
                # (Gemma E4B does this) can be cut off mid-thought, and the empty-handed result
                # is indistinguishable from a genuine miss without it.
                cut = "" if got["finish"] == "stop" else f" [finish={got['finish']}]"
                print(f"    needle {depth_label:<7} {'FOUND ' if found else 'MISSED'} "
                      f"(prompt {tokens} tok){cut} {' '.join(got['content'].split())[:90]!r}")

            # Needle absent: anything containing the figure now is confabulation, and would have
            # scored as a hit above.
            got = run_one(base, model, role, where, build_corpus(n_results, None))
            if got["error"]:
                print(f"    control         REJECTED -- {got['error'][:90]}")
            else:
                leaked = NEEDLE_VALUE in got["content"]
                print(f"    control         {'CONFABULATED -- hits above are unreliable' if leaked else 'clean (no needle invented)'}"
                      f" {' '.join(got['content'].split())[:70]!r}")
            print()
    print("done")


if __name__ == "__main__":
    main()
