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

from raven.librarian import chatutil  # noqa: E402 -- after importorskip by design
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


# Every character a model might put between the thousands. The spaces are given as code points rather than
# pasted: four of them are indistinguishable on screen, and a reformat or a whitespace trim would silently
# turn one into another while the diff still looked clean.
_THOUSANDS_SEPARATORS = (",", ".", "_") + tuple(chr(cp) for cp in (0x0020,   # SPACE
                                                                   0x00A0,   # NO-BREAK SPACE
                                                                   0x202F,   # NARROW NO-BREAK SPACE, the SI-style separator
                                                                   0x2009))  # THIN SPACE


def test_the_model_calls_the_calculator_and_the_arithmetic_comes_back_right(llm_settings):
    """The one tool whose result can be checked without checking the model.

    `calculate` is the second tool needing nothing outside the process, and it differs from
    `get_current_time` in the way that matters here: the right answer is knowable from this side. So this
    asserts on the reply's *content*, where the tests above deliberately do not.

    **That is still shape, not wording, because the call is asserted separately.** The tool was called; the
    digits are how we know its result was carried through to the user rather than dropped between the tool
    node and the reply. No claim is made about what the model would have answered unaided — it may well
    multiply five digits by four correctly in its head, and this would pass either way, which is fine: what
    is under test is the path, not the arithmetic.

    Separators are stripped first. Whether a model writes `443339232` or `443,339,232` is presentation, and
    a European writer's `443.339.232` is the same choice made with a different character.
    """
    record = agent.turn(llm_settings,
                        user_message_text="What is 48273 times 9184? Use your calculator tool, then say "
                                          "the result.",
                        tools_enabled=True,
                        internet_enabled=False,
                        docs_enabled=False,
                        use_character_card=False)

    assert "calculate" in record.tool_calls, (f"expected a `calculate` call, got {dict(record.tool_calls)}; "
                                              f"reply was {record.reply!r}")
    plain = record.reply
    for separator in _THOUSANDS_SEPARATORS:
        plain = plain.replace(separator, "")
    assert str(48273 * 9184) in plain, (f"the tool was called but its answer did not reach the reply: "
                                        f"{record.reply!r}")


def test_the_thinking_toggle_actually_switches_reasoning_off(llm_settings):
    """`reasoning_effort: "none"` is a request the backend has to honour, and only it can say whether it does.

    This is the one thing a mock cannot check about the *Thinking* toggle. `test_llmclient.py` pins that the
    field reaches the wire; whether the field then does anything is a property of the backend and of the
    model's chat template, and neither is under our control. A backend that quietly ignores it leaves the
    toggle looking connected and doing nothing.

    Both directions are asked because the off-case alone proves nothing: a model that never reasons produces
    zero reasoning tokens whatever is sent. The on-case is the control, and when it comes back empty the
    backend is serving a non-thinking model — which is not a failure, so it skips.

    Asserted on the reasoning trace's presence, never on its content or the answer's. What the model thinks
    about 17 times 23 is its own business; that it thought at all, or did not, is ours.
    """
    question = [{"role": "user", "content": [chatutil.text_content_part("What is 17 times 23?")]}]

    thought = llmclient.invoke(llm_settings, question, tools_enabled=False)
    if not (thought.data.get("reasoning_content") or "").strip():
        pytest.skip("the loaded model did not reason even when allowed to, so there is nothing here to "
                    "switch off — point these at a thinking model to exercise the toggle")

    straight = llmclient.invoke(llm_settings, question, tools_enabled=False, thinking_enabled=False)
    assert not (straight.data.get("reasoning_content") or "").strip(), (
        f"the backend went on reasoning with `reasoning_effort: \"none\"` sent, so the toggle does not "
        f"reach this backend: {straight.data['reasoning_content']!r}")
    assert chatutil.content_to_text(straight.data["content"]).strip(), (
        "reasoning was switched off and the reply came back empty, so the model was silenced rather than "
        "asked to answer directly")
