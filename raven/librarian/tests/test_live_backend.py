"""Checks a live OpenAI-compatible LLM backend against what Librarian actually needs from one.

The rest of the suite mocks the backend — `test_llmclient.py` monkeypatches the client, and nothing else
sends a request anywhere. That is the right default, since a unit test should not need a GPU and a loaded
model. It does leave one thing uncovered: whether the backend still *behaves* the way `llmclient` and
`agent` assume. A tool-calling regression in the inference engine — a changed chat template, a reworked
tool parser — breaks Librarian at runtime and nothing here would notice.

So these are the assertions a mock cannot make. They are deliberately few, and they run against whatever
backend is configured rather than a particular one.

**Which backend** comes from `librarian_config.llm_backend_url`, overridden by `--backend-url`. That is the
same spelling every console script uses, and it is what makes these portable: the configured URL is already
a per-machine value, so on a machine whose backend is local it points at localhost and on one talking to
another host it points there, with nothing to change here. The flag covers the case where the two disagree
for a day.

**What is asserted is shape, never wording.** The model on the other end differs between machines and over
time, so "the reply mentions Helsinki" would be a test of the model. What Librarian depends on is that a
turn comes back with text at all, that a tool call is well-formed enough to execute, and that its result
makes it back into the reply.

**They are opt-in, and the reason is the connection rather than the cost.** `--run-llm` uses the configured
backend and `--backend-url URL` names another; either opts in. Left on by default they would have a CI
runner open a socket to whatever the committed URL points at, on a machine nobody here controls and where
nobody can say what is listening on that port. Having opted in, they still skip when nothing answers, since
a backend that is merely switched off is not a failure.

A skipped test looks exactly like a passing one, so every skip here names the URL it tried and the flag that
changes it. A skip should be diagnosable without opening this file.
"""

import pytest

llmclient = pytest.importorskip("raven.librarian.llmclient",
                                reason="librarian LLM client stack not installed")
agent = pytest.importorskip("raven.librarian.agent",
                            reason="librarian agent stack not installed")

from raven.librarian import config as librarian_config  # noqa: E402 -- after importorskip by design

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def backend_url(request):
    """The backend these tests talk to: `--backend-url` if given, else whatever is configured."""
    return request.config.getoption("--backend-url") or librarian_config.llm_backend_url


@pytest.fixture(scope="module")
def llm_settings(backend_url):
    """`llmclient` settings for a backend that is up and has a model loaded, or a skip saying which is not.

    Module-scoped: `setup` costs a couple of round trips, and none of these tests modify what it returns.
    """
    if not llmclient.test_connection(backend_url, quiet=True):
        pytest.skip(f"no LLM backend answering at {backend_url} — start one, or point these somewhere else "
                    f"with: pytest --backend-url http://host:port")
    settings = llmclient.setup(backend_url, quiet=True)
    status = llmclient.backend_status(settings)
    if status is not llmclient.backend_ready:
        pytest.skip(f"backend at {backend_url} is not ready: "
                    f"{llmclient.describe_backend_status(status, backend_url)}")
    return settings


def test_a_ready_backend_names_the_model_it_loaded(llm_settings):
    """The identity Librarian puts on the character card, and the first thing a broken connection loses."""
    assert llm_settings.model, "a ready backend reported no model identity"


def test_a_scripted_turn_comes_back_with_a_reply(llm_settings):
    """The plain path: ask for text, get text.

    Tools are off, so this fails only if generation itself is broken — which separates "the backend cannot
    talk" from "the backend cannot call tools" when both tests go red at once.
    """
    record = agent.turn(llm_settings,
                        user_message_text="Reply with the single word: ready.",
                        tools_enabled=False,
                        use_character_card=False)

    assert record.reply.strip(), f"the turn produced no reply text; messages={record.messages}"
    assert record.rounds == 0, f"tools were disabled, so no tool round should have been taken: {record.rounds}"
    assert record.generation is not None, "a completed reply carries generation metadata"
    assert record.generation.get("model"), "generation metadata names no model"


def test_the_model_calls_a_tool_and_the_result_reaches_the_reply(llm_settings):
    """The path a changed tool parser breaks, end to end: asked, called, executed, answered.

    `get_current_time` is the only tool needing nothing outside the process — the others want the documents
    index or the internet — so this exercises Librarian's whole tool loop while staying hermetic. Documents
    and internet are switched off explicitly rather than left to config, so a machine that enables them by
    default does not turn this into a web search.

    **`use_character_card=False` is what makes the call necessary, and is not tidiness.** With the card on,
    `scaffold.build_turn_prompt` stages the current time as a synthetic `get_current_time` exchange before
    the user's message — the clock inject is gated on `tools_enabled and use_character_card` — so the model
    would already have the answer and would be right not to call anything. Turning the card off is what
    leaves the tool as the only route to the time.

    What is asserted stops at *a* call having been made and answered. Which tool a model reaches for is its
    own judgement, and a model that also consults the clock twice is not wrong.

    **The assertion is on the round, never on the reply text, and that is load-bearing.** Asked the same
    question with tools switched off, a model does not fail in one recognizable way. Sampled 24 times at
    `T=1, min_p=0.02` on qwen3.5-9b (2026-08-24): 20 refused for want of real-time data, 4 invented a
    date — all four different, scattered over 2023 and 2024 — several emitted tool syntax as plain prose
    (`[CALL current_time]`, a `<tool_call>` blob), and 8 spent the entire token budget reasoning in a loop
    and returned nothing at all. A greedy run produced a fluent, specific, invented "3:42 PM on Wednesday,
    June 5, 2024".

    Any assertion on what the reply *says* would therefore be checking the model rather than the tool
    loop, and would pass or fail for reasons unrelated to whether tool calling works.
    """
    record = agent.turn(llm_settings,
                        user_message_text="What is the current time? Use your tool to check, then say it.",
                        tools_enabled=True,
                        internet_enabled=False,
                        docs_enabled=False,
                        use_character_card=False)

    assert record.rounds >= 1, (f"the model took no tool round, so nothing here tests the tool path; "
                                f"reply was {record.reply!r}")
    assert "get_current_time" in record.tool_calls, (f"expected a `get_current_time` call, got "
                                                     f"{dict(record.tool_calls)}")
    assert record.reply.strip(), ("the tool was called but the turn ended without a reply, so the result "
                                  "never came back to the user")
