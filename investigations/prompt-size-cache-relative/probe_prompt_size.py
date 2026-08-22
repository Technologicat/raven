"""Is a backend's reported `prompt_tokens` the size of the prompt, or only of the part it had to process?

Raven upgrades its context-fill readout from a local estimate (`~X%`) to the backend's own figure (`X%`) by
sending the real prompt with `max_tokens=1` and reading `usage["prompt_tokens"]`. That is exact on a cold
backend and, on LM Studio, an order of magnitude short on a warm one — because `prefill`'s *other* purpose is
to warm the KV cache, so Raven creates the condition itself.

Sends four requests for one chat branch and prints what comes back:

  1. the branch as it is — whatever the cache state happens to be
  2. the same again — a repeat is necessarily warm
  3. the branch with attachments left unresolved, as a size reference for the conversation alone
  4. the branch with a nonce at the very front, which busts any prefix cache and so reports the whole prompt

**Reads a chat datastore and sends its contents to a local LLM backend.** Point it at a scratch chat unless
you are re-measuring the original finding; the default is Librarian's configured datastore, which is what
made the numbers in the README.

Usage:
    python probe_prompt_size.py [--backend-url URL] [--datastore PATH] [--head NODE_ID]
"""

import argparse
import json
import pathlib
import uuid

from raven.librarian import chattree, chatutil, llmclient
from raven.librarian import config as librarian_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether a backend's prompt_tokens counts the whole prompt.")
    parser.add_argument("--backend-url", default="http://localhost:1234",
                        help="OpenAI-compatible endpoint to ask (default: LM Studio's on localhost)")
    parser.add_argument("--datastore", default=None,
                        help="chat datastore to read (default: Librarian's configured one)")
    parser.add_argument("--head", default=None,
                        help="node id to measure (default: the HEAD in Librarian's state file)")
    args = parser.parse_args()

    datastore_path = args.datastore or librarian_config.llm_datastore_file
    head = args.head
    if head is None:
        head = json.loads(pathlib.Path(librarian_config.llm_state_file).read_text())["HEAD"]

    datastore = chattree.PersistentForest(datastore_path, autosave=False)
    settings = llmclient.setup(backend_url=args.backend_url)
    tool_names = llmclient.maybe_tool_names_for_turn(settings, documents_available=False, internet_available=True)
    history = chatutil.linearize_chat(datastore=datastore, node_id=head)

    n_documents = sum(1 for node_id in datastore.linearize_up(head)
                      for part in (datastore.get_payload(node_id)["message"].get("content") or [])
                      if isinstance(part, dict) and part.get("type") == "text_file")
    estimate, _is_exact = llmclient.count_branch_tokens(settings, datastore, head)
    wire = llmclient.serialize_history_for_wire(settings, history, continue_=False, datastore=datastore)
    wire_characters = sum(len(part["text"]) for message in wire for part in message["content"]
                          if part.get("type") == "text")

    print(f"branch:                    {n_documents} attached document(s)")
    print(f"local estimate:            {estimate} tokens")
    print(f"characters sent on the wire: {wire_characters}")

    def ask(label: str, messages, resolve_attachments: bool = True) -> None:
        out = llmclient.prefill(settings, messages, tools_enabled=True, tool_names=tool_names,
                                datastore=(datastore if resolve_attachments else None))
        reported = (out.usage or {}).get("prompt_tokens") if out is not None else None
        print(f"{label:34s} prompt_tokens = {reported}")

    ask("as-is (cache state unknown)", history)
    ask("repeat (necessarily warm)", history)
    ask("attachments not resolved", history, resolve_attachments=False)

    # A nonce at the very front changes the first token, so no prefix of the cached sequence matches and the
    # backend has to process the whole prompt — which is what makes this the whole-prompt figure.
    busted = [dict(message) for message in history]
    first = dict(busted[0])
    first["content"] = [chatutil.text_content_part(f"[{uuid.uuid4()}]")] + list(first["content"])
    busted[0] = first
    ask("cache-busted (whole prompt)", busted)


if __name__ == "__main__":
    main()
