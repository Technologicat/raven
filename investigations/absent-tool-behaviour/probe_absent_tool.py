#!/usr/bin/env python
"""What does a model do when asked for the time and given no clock tool?

Samples one model over one prompt and classifies each reply. Talks to an OpenAI-compatible backend
directly rather than through `raven.librarian`, for the sampler: temperature, `min_p` and the seed vary per
request here, where `agent.turn` takes them from `llm_settings` and this study is about what happens across
a spread of samples.

`agent.turn(..., use_character_card=False, tools_enabled=False)` would in fact present the model just as
bare — no character card, no system injects, and Raven's shipped `system_prompt` is empty — so the layer is
not what is being avoided. Worth knowing before assuming a probe has to go around Librarian to ask a
question about a model.

    python probe_absent_tool.py --model qwen3.5-9b --prompt plain -n 24 --out samples.json

`--prompt tool-mention` asks the model to use a tool that is not offered; `--prompt plain` just asks the
question. The difference between the two is the point of the study, so run both against any model you add.

No tools are ever sent. That is the experiment.
"""

import argparse
import collections
import json
import re
import sys
import urllib.request

PROMPTS = {
    "plain": "What is the current date and time?",
    "tool-mention": "What is the current time? Use your tool to check, then say it.",
}

# A date in any of the shapes these models reach for, and a clock time. Deliberately loose: a false positive
# reads as "it answered" and is visible in the transcript, where a missed one would silently inflate the
# refusal count, which is the number the study turns on.
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}"
                     r"|(?:January|February|March|April|May|June|July|August|September|October|November"
                     r"|December)\s+\d{1,2},?\s+\d{4})", re.I)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?")
# Tool syntax written out as prose, with no tool in the request to have prompted it.
TOOL_PROSE_RE = re.compile(r"\[CALL\b|<tool_call>|<tool_use>|<function_call>|\bI'll use my\b", re.I)
REFUSAL_RE = re.compile(r"don't have access|do not have access|can't access|cannot access|no access to"
                        r"|can't provide|cannot provide|don't know the current|real-time", re.I)


def sample(backend_url, model, prompt, temperature, min_p, max_tokens, seed):
    """One completion. Returns the raw response body."""
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature,
               "min_p": min_p,
               "max_tokens": max_tokens,
               "seed": seed}
    request = urllib.request.Request(f"{backend_url}/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def classify(body):
    """Sort one reply into a bucket, and pull out whatever date and time it stated.

    Buckets, in the order they are tested — the order matters, since a reply can satisfy more than one:

    `truncated`: the budget ran out before any reply was emitted. On a reasoning model this is the loop
                 case, and the reasoning trace is where it is visible.
    `answered`:  stated a date or a time, which without a tool means it invented one.
    `tool-prose`: wrote tool-call syntax into the reply text, with no tool offered to prompt it.
    `refused`:   said it has no access to real-time data.
    `other`:     none of the above; read the transcript.
    """
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    dates = DATE_RE.findall(content)
    times = TIME_RE.findall(content)

    if choice["finish_reason"] == "length" and not content.strip():
        bucket = "truncated"
    elif dates or times:
        bucket = "answered"
    elif TOOL_PROSE_RE.search(content):
        bucket = "tool-prose"
    elif REFUSAL_RE.search(content):
        bucket = "refused"
    else:
        bucket = "other"

    usage = body.get("usage", {})
    return {"bucket": bucket,
            "date": dates[0].strip() if dates else None,
            "time": times[0].strip() if times else None,
            "finish_reason": choice["finish_reason"],
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            "content": content,
            "reasoning_tail": " ".join((message.get("reasoning_content") or "").split())[-400:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-url", default="http://localhost:1234",
                        help="OpenAI-compatible backend (default: %(default)s)")
    parser.add_argument("--model", required=True, help="model id, as the backend lists it")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="plain")
    parser.add_argument("-n", "--samples", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed0", type=int, default=1000, help="first seed; sample i uses seed0 + i")
    parser.add_argument("--out", default=None, help="write the full records here as JSON")
    args = parser.parse_args()

    prompt = PROMPTS[args.prompt]
    print(f"# model={args.model} prompt={args.prompt!r} n={args.samples} "
          f"T={args.temperature} min_p={args.min_p} max_tokens={args.max_tokens}", flush=True)
    print(f"# {prompt!r}\n", flush=True)

    records, buckets = [], collections.Counter()
    for i in range(args.samples):
        record = classify(sample(args.backend_url, args.model, prompt,
                                 args.temperature, args.min_p, args.max_tokens, args.seed0 + i))
        record["seed"] = args.seed0 + i
        records.append(record)
        buckets[record["bucket"]] += 1
        shown = " ".join(record["content"].split())[:96] or "(empty)"
        print(f"[{i:2d}] {record['bucket']:<10} {shown}", flush=True)

    print(f"\n=== {args.samples} samples: {args.model}, prompt={args.prompt} ===")
    for bucket, n in buckets.most_common():
        print(f"{n:3d}  {bucket}")
    stated = [(r["date"], r["time"]) for r in records if r["bucket"] == "answered"]
    if stated:
        print("--- what the answering samples claimed ---")
        for date, time in stated:
            print(f"     {date or '(no date)'}  {time or '(no time)'}")
        distinct = len({d for d, _ in stated if d})
        print(f"     distinct dates: {distinct} of {len([d for d, _ in stated if d])} stating one")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "prompt_key": args.prompt, "prompt": prompt,
                       "temperature": args.temperature, "min_p": args.min_p,
                       "max_tokens": args.max_tokens, "samples": records}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
