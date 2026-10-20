"""Render a GGUF's own chat template on the messages Librarian sends, and say whether earlier thinking survives.

This is the template *as shipped*, rendered locally with Jinja: what it does on its own, before a runtime
sets any of its variables. Compare with `probe_retention.py`, which asks the backend, and the difference is
what the runtime adds. On LM Studio that difference was the whole answer for Qwen 3.6 (see `README.md`).

Usage: python investigations/thinking-retention/render_template.py PATH.gguf
"""

import argparse

import jinja2
from gguf import GGUFReader
from unpythonic.env import env

from raven.librarian import chatutil, llmclient
from raven.librarian import config as librarian_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", help="the model file whose `tokenizer.chat_template` to render")
    opts = parser.parse_args()

    field = GGUFReader(opts.gguf).fields["tokenizer.chat_template"]
    template = bytes(field.parts[field.data[0]]).decode("utf-8")

    librarian_config.llm_tokenizer_path = None  # this needs no counting, and so no tokenizer
    settings = llmclient.configure(model_info=env(label="m", model_id="m", context_length=8192, is_vlm=None, loaded=True),
                                   backend_flavor="lmstudio", backend_url="http://unused", quiet=True)
    history = [chatutil.create_chat_message(llm_settings=settings, role="system", text="You are helpful."),
               chatutil.create_chat_message(llm_settings=settings, role="user", text="Q1"),
               chatutil.create_chat_message(llm_settings=settings, role="assistant", text="A1",
                                            reasoning_content="EARLIER-THOUGHT"),
               chatutil.create_chat_message(llm_settings=settings, role="user", text="Q2")]
    wire = llmclient.serialize_history_for_wire(settings, history, continue_=False)

    jinja = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    jinja.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(RuntimeError(message))
    for label, messages, variables in (("as Librarian sends it", wire, {}),
                                       ("with preserve_thinking=True", wire, {"preserve_thinking": True})):
        rendered = jinja.from_string(template).render(messages=messages, add_generation_prompt=True, **variables)
        print(f"{label}: earlier thinking rendered: {'EARLIER-THOUGHT' in rendered}")


if __name__ == "__main__":
    main()
