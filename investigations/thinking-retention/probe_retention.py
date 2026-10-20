"""Does the loaded model's chat template keep earlier turns' thinking, and does Librarian's rule say so too?

Builds a chat of three exchanges whose answers carry long thinking traces, and asks the backend for the
prompt size the way Librarian's context prefill does: `scaffold.build_prefill_prompt`, then
`llmclient.prefill`. Two answers come out of that, and the point is that they agree:

  - **Measured.** The same prompt sent a second time with the traces removed. If the backend counts the two
    alike, its template dropped the traces.
  - **The rule.** `llmclient.template_strips_thinking`, given the backend's figure and Librarian's own count
    of the messages with traces and without.

The local count is taken twice: before the background tokenizer load finishes, when it is the
tokens-per-character estimate, and after, when it is exact. The difference is the estimate's error on
this text, which is one sentence repeated, so read it as a data point about the estimate rather than as
its size in general.

Usage: python investigations/thinking-retention/probe_retention.py [--backend-url URL]
"""

import argparse
import copy
import threading

from raven.librarian import chattree, chatutil, llmclient, scaffold
from raven.librarian import config as librarian_config

THOUGHT = "Let me weigh the evidence for each option in turn before I commit to an answer. " * 40


def build_chat(settings):
    """Three exchanges with a long trace on each answer, then a fourth question. Returns `(forest, head)`."""
    forest = chattree.Forest()

    def add(role, text, parent, **kw):
        message = chatutil.create_chat_message(llm_settings=settings, role=role, text=text, **kw)
        return forest.create_node({"message": message, "general_metadata": {}}, parent_id=parent)
    head = add("system", "You are a helpful assistant.", None)
    for i in range(3):
        head = add("user", f"Question {i}: what is {i} + {i}?", head)
        head = add("assistant", f"It is {2 * i}.", head, reasoning_content=THOUGHT)
    return forest, add("user", "And 3 + 3?", head)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-url", default=librarian_config.llm_backend_url)
    opts = parser.parse_args()

    settings = llmclient.setup(opts.backend_url, quiet=True)
    settings.retain_thinking = True  # the case under test: traces sent and counted
    print(f"model: {settings.model}")

    forest, head = build_chat(settings)
    history = [forest.get_payload(node_id)["message"] for node_id in forest.linearize_up(head)]
    prompt = scaffold.build_prefill_prompt(settings, history)
    without = copy.deepcopy(prompt)
    for message in without:
        message.pop("reasoning_content", None)

    def reported(messages):
        return llmclient.prefill(settings, messages, tools_enabled=False, calibrate=False).usage["prompt_tokens"]
    sent_with, sent_without = reported(prompt), reported(without)

    def local_counts():
        with_traces, is_exact = llmclient.count_branch_tokens(settings, forest, head)
        without_traces, _ = llmclient.count_branch_tokens(settings, forest, head, retain_thinking=False)
        return with_traces, without_traces, is_exact
    estimated = local_counts()
    # The tokenizer loads on a named thread, which ends having either set `settings.tokenizer` or given up.
    for thread in threading.enumerate():
        if thread.name == "llmclient tokenizer load":
            thread.join()
    with_traces, without_traces, is_exact = local_counts()

    measured_strips = abs(sent_with - sent_without) < 0.1 * (with_traces - without_traces)
    rule_strips = llmclient.template_strips_thinking(sent_with, with_traces, without_traces)
    rule_strips_on_estimate = llmclient.template_strips_thinking(sent_with, estimated[0], estimated[1])
    print(f"backend: {sent_with} tokens with the traces sent, {sent_without} without")
    print(f"local count, estimate: {estimated[0]} with, {estimated[1]} without")
    print(f"local count, {'exact' if is_exact else 'still an estimate'}: {with_traces} with, {without_traces} without")
    print(f"measured: template {'strips' if measured_strips else 'keeps'}; "
          f"rule: {'strips' if rule_strips else 'keeps'} (on the estimate: {'strips' if rule_strips_on_estimate else 'keeps'})")
    print("AGREE" if measured_strips == rule_strips == rule_strips_on_estimate else "DISAGREE")


if __name__ == "__main__":
    main()
