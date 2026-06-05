"""V4 probe: does LM Studio render a Gemma history with reasoning_content + tool_calls?

Mirrors raven's resend payload shape (llmclient.invoke). The decisive question is whether
LM Studio's minja can render Gemma's chat_template for a tool-call-bearing assistant turn
(last night it threw "Cannot call something that is not a function: got UndefinedValue").

Run with LM Studio up and a Gemma model loaded. Usage:
    python /tmp/v4_gemma_probe.py [base_url]
"""
import json
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1234"


def discover_loaded():
    """Return (model_id, record) for the loaded model via LM Studio's native endpoint."""
    data = requests.get(f"{BASE}/api/v0/models", timeout=10).json().get("data", [])
    loaded = [m for m in data if m.get("state") == "loaded"]
    if not loaded:
        print(f"No model loaded. Available: {[m.get('id') for m in data]}")
        sys.exit(1)
    rec = loaded[0]
    return rec.get("id"), rec


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

# A resend history exactly as raven would rebuild it on a tool-continuation turn:
# the assistant turn carries BOTH reasoning_content (sibling field) AND structured tool_calls,
# followed by the tool result. This is the shape that broke minja last night.
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
    {"role": "user", "content": "What's the weather in Paris right now?"},
    {"role": "assistant",
     "content": "",
     "reasoning_content": "The user is asking for the current weather in Paris. "
                          "I should call get_weather with city=Paris.",
     "tool_calls": [{
         "id": "call_0",
         "type": "function",
         "index": "0",
         "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
     }]},
    {"role": "tool",
     "tool_call_id": "call_0",
     "content": json.dumps({"temp_c": 14, "conditions": "light rain"})},
]


def main():
    model_id, rec = discover_loaded()
    print(f"Loaded model: {model_id}")
    print(f"  arch={rec.get('arch')} quant={rec.get('quantization')} "
          f"ctx={rec.get('loaded_context_length')}")
    print(f"LM Studio: {BASE}\n")

    data = {
        "stream": True,
        "model": model_id,
        "messages": MESSAGES,
        "tools": TOOLS,
        "stream_options": {"include_usage": True},
        "max_tokens": 512,
    }

    resp = requests.post(f"{BASE}/v1/chat/completions", json=data, stream=True, timeout=120)
    print(f"HTTP {resp.status_code}\n")

    content_buf = []
    reasoning_buf = []
    tool_calls = []
    error_msg = None
    finish_reason = None
    saw_done = False

    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8")
        if line.startswith("event:"):
            if "error" in line:
                error_msg = error_msg or "(event: error)"
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            print(f"  !! non-JSON data line: {payload!r}")
            continue
        if "error" in obj:
            err = obj["error"]
            error_msg = err.get("message") if isinstance(err, dict) else str(err)
            continue
        choices = obj.get("choices") or []
        if not choices:
            if obj.get("usage"):
                print(f"  usage: {obj['usage']}")
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_buf.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_buf.append(delta["reasoning_content"])
        for tc in delta.get("tool_calls") or []:
            tool_calls.append(tc)
        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]

    print("=" * 60)
    if error_msg:
        print(f"❌ TEMPLATE / RENDER ERROR: {error_msg}")
        print("   -> minja still cannot render this GGUF's template for a")
        print("      tool-call + reasoning_content history. Next: override")
        print("      with Google's official chat_template.jinja.")
    else:
        print("✅ RENDERED OK — no minja error on reasoning_content + tool_calls.")
        print(f"   finish_reason={finish_reason}  [DONE]={saw_done}")
        if reasoning_buf:
            print(f"\n   reasoning_content ({len(''.join(reasoning_buf))} chars):")
            print("   " + "".join(reasoning_buf)[:400].replace("\n", "\n   "))
        if content_buf:
            print(f"\n   content ({len(''.join(content_buf))} chars):")
            print("   " + "".join(content_buf)[:400].replace("\n", "\n   "))
        if tool_calls:
            print(f"\n   tool_calls: {json.dumps(tool_calls)[:300]}")
        if not content_buf and not reasoning_buf and not tool_calls:
            print("   (empty output — rendered but model produced nothing)")
    print("=" * 60)


if __name__ == "__main__":
    main()
