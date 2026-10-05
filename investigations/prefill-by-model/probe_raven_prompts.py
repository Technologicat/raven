"""Which part of a Librarian turn's prompt stops the backend from reusing the prefill's cached prefix?

`probe_extend.py` showed the backend reusing an extended prompt with plain messages, while in Librarian the
turn after a prefill was processed whole, though the prefill sends exactly the prefix the turn begins with.
So this builds both with Raven's own code (`scaffold.build_prefill_prompt`, `scaffold.build_turn_prompt`),
sends them as the app does (`llmclient.prefill`), and times the turn after a cold prefill, for variants of
the turn that each leave something out.

Each trial puts a fresh nonce at the head of the system text, so nothing is cached from the trial before.
The branch is a system message (the character card, as `llmclient.setup` resolves it) and the greeting,
the shape of a new chat. Timings are the request's wall time, the reply being capped at one token.

    python investigations/prefill-by-model/probe_raven_prompts.py http://host:1234 [REPEATS]
"""

import sys
import uuid

from raven.librarian import chatutil, llmclient, scaffold


def main() -> None:
    backend_url = sys.argv[1].rstrip("/")
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    settings = llmclient.setup(backend_url=backend_url, quiet=True)

    def message(role: str, text: str) -> dict:
        return chatutil.create_chat_message(llm_settings=settings, role=role, text=text,
                                            add_persona=(role != "system"))

    def send(history: list[dict]) -> float:
        out = llmclient.prefill(settings, history, tools_enabled=True, calibrate=False, purpose="cache probe")
        assert out is not None, "the backend did not answer"
        return out.dt

    def no_tools_context():
        return scaffold.make_tool_context(llm_settings=settings, retriever=None)

    for trial in range(repeats):
        for variant in ("full", "no clock", "bare", "ends in the reply", "ends at the last user message"):
            branch = [message("system", f"Session {uuid.uuid4().hex}.\n\n{settings.character_card}"),
                      message("assistant", settings.greeting)]
            if variant.startswith("ends"):  # a longer branch, whose HEAD is a reply to a user message
                branch += [message("user", "Hello! Tell me briefly what you are."),
                           message("assistant", "A research assistant, running locally. I search your documents and the web.")]
            user = message("user", "What can you do?")
            prefix = scaffold.build_prefill_prompt(settings, branch[:-1] if variant == "ends at the last user message" else branch)
            if variant == "bare":
                turn = scaffold.build_prefill_prompt(settings, branch + [user])
            else:
                turn = scaffold.build_turn_prompt(llm_settings=settings, history=branch + [user],
                                                  docs_query=None, docs_matches=[], tool_context=no_tools_context(),
                                                  tools_enabled=(variant in ("full", "ends in the reply",
                                                                             "ends at the last user message")))
            assert turn[:len(prefix)] == prefix, f"{variant}: the turn does not begin with the prefix"
            roles = [m["role"] + ("+calls" if m.get("tool_calls") else "") for m in turn[len(prefix):]]
            cold, after = send(prefix), send(turn)
            print(f"trial {trial} {variant:30s} prefix {cold:5.2f} s, turn after it {after:5.2f} s   appended {roles}", flush=True)


if __name__ == "__main__":
    main()
