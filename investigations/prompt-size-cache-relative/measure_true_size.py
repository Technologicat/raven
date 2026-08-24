"""How big is the prompt really, when the backend's own answer is the thing under suspicion?

Three measurements, none of which trusts `usage["prompt_tokens"]` about the prompt Raven actually sends:

  1. **Same bytes?** Serialize the branch as-is and with a nonce prepended, and compare character counts.
     `probe_prompt_size.py` reads wildly different figures for those two requests; this says whether the
     difference is in what was sent or in what was counted.

  2. **Chars per token, on content the backend cannot have cached.** Slices of the attachment text, each
     behind a fresh UUID, sent as a plain single-message prompt. A unique prompt has no cache to hit, so
     the figure is about the text. Several sizes, because the ratio is not constant: prose and reference
     lists tokenize very differently, so a small slice does not extrapolate.

  3. **The served model's own tokenizer, offline.** A GGUF carries its vocabulary and merges, so the machine
     running a llama.cpp-family backend already holds the exact answer — no backend, no cache, no network.
     This is the measurement from outside the instrument, and it is what overturned the earlier conclusion.

**Reads a chat datastore, and (1) and (2) send its contents to the backend you name.** Measurement (3) sends
nothing. Point it at a scratch chat unless you are re-measuring the original finding; the default is
Librarian's configured datastore, which is what made the numbers in the README.

Usage:
    python measure_true_size.py [--backend-url URL] [--gguf PATH] [--datastore PATH] [--head NODE_ID]
"""

import argparse
import json
import pathlib
import uuid
from typing import Dict, List, Optional

import requests

from raven.librarian import chattree, chatutil, llmclient
from raven.librarian import config as librarian_config


def wire_characters(settings, messages: List[Dict], datastore) -> tuple:
    """Return `(total_characters, longest_text_part)` for `messages` as they would go on the wire."""
    wire = llmclient.serialize_history_for_wire(settings, messages, continue_=False, datastore=datastore)
    texts = [part["text"] for message in wire for part in message["content"] if part.get("type") == "text"]
    return sum(len(text) for text in texts), max(texts, key=len)


def ask_plain(backend_url: str, model: str, text: str) -> int:
    """Send `text` as a lone user message and return the backend's `prompt_tokens`."""
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1}
    response = requests.post(f"{backend_url}/v1/chat/completions", json=body, timeout=600)
    return response.json()["usage"]["prompt_tokens"]


def count_offline(gguf_path: pathlib.Path, text: str) -> Optional[int]:
    """Count tokens in `text` with the tokenizer embedded in `gguf_path`. `None` if it cannot be read."""
    try:
        import gguf  # noqa: PLC0415 -- optional, and only this branch needs it
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex  # noqa: PLC0415
    except ImportError as exc:
        print(f"  (offline count unavailable: {exc})")
        return None

    reader = gguf.GGUFReader(str(gguf_path))

    def field(key):
        return reader.fields[key].contents()

    vocabulary = {token: index for index, token in enumerate(field("tokenizer.ggml.tokens"))}
    merges = [tuple(merge.split(" ", 1)) for merge in field("tokenizer.ggml.merges")]
    print(f"  GGUF tokenizer: {len(vocabulary)} tokens, {len(merges)} merges, "
          f"pre={field('tokenizer.ggml.pre')!r}, class={field('tokenizer.ggml.model')!r}")

    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=merges, fuse_unk=False, byte_fallback=False))
    # The GPT-2-family byte-level BPE that `tokenizer.ggml.model = 'gpt2'` names: split on the family's
    # regex, then byte-encode each piece. The class is shared; the vocabulary above is the model's own.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                                   r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
                             behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a branch's true prompt size without trusting the backend.")
    parser.add_argument("--backend-url", default="http://localhost:1234",
                        help="OpenAI-compatible endpoint to ask (default: LM Studio's on localhost)")
    parser.add_argument("--gguf", default=None,
                        help="the .gguf the backend is serving, for the offline count (skipped if omitted)")
    parser.add_argument("--datastore", default=None,
                        help="chat datastore to read (default: Librarian's configured one)")
    parser.add_argument("--head", default=None,
                        help="node id to measure (default: the HEAD in Librarian's state file)")
    args = parser.parse_args()

    head = args.head or json.loads(pathlib.Path(librarian_config.llm_state_file).read_text())["HEAD"]
    datastore = chattree.PersistentForest(args.datastore or librarian_config.llm_datastore_file, autosave=False)
    settings = llmclient.setup(backend_url=args.backend_url)
    history = chatutil.linearize_chat(datastore=datastore, node_id=head)

    # 1. Do the two requests `probe_prompt_size.py` compares actually carry the same text?
    busted = [dict(message) for message in history]
    first = dict(busted[0])
    first["content"] = [chatutil.text_content_part(f"[{uuid.uuid4()}]")] + list(first["content"])
    busted[0] = first

    as_is_characters, corpus = wire_characters(settings, history, datastore)
    busted_characters, _ = wire_characters(settings, busted, datastore)
    print("same bytes?")
    print(f"  as-is        {as_is_characters} characters")
    print(f"  nonce ahead  {busted_characters} characters  (difference: {busted_characters - as_is_characters})")
    print(f"  attachment text: {len(corpus)} characters")

    # 2. What does this corpus cost per character, on copies nothing can have cached?
    print("chars per token, on unique (uncacheable) slices:")
    for size in (10000, 50000, 100000, 200000, len(corpus)):
        if size > len(corpus):
            continue
        tokens = ask_plain(args.backend_url, settings.model_id or settings.model, f"[{uuid.uuid4()}] " + corpus[:size])
        print(f"  {size:7d} chars -> {tokens:6d} tokens  ({size / tokens:.2f} chars/token)")

    # 3. The answer that never leaves the machine.
    if args.gguf is not None:
        print("offline, from the served model's own tokenizer:")
        tokens = count_offline(pathlib.Path(args.gguf), corpus)
        if tokens is not None:
            print(f"  {len(corpus)} chars -> {tokens} tokens  ({len(corpus) / tokens:.2f} chars/token)")

    estimate, _is_exact = llmclient.count_branch_tokens(settings, datastore, head)
    print(f"Raven's local estimate for the whole branch: {estimate} tokens")


if __name__ == "__main__":
    main()
