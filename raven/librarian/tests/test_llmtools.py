"""The `calculate` tool: what it evaluates, and what it refuses.

The point of the tool is that a model doing arithmetic in its head gets it wrong in front of an audience,
so the answers have to be right — and the refusals have to be *usable*, because the model reads them and
decides what to do next.

Nothing here needs a backend, a retriever or a GUI: `calculate` is a pure function of its argument.
"""

import math

import pytest

from raven.librarian import llmtools


# --------------------------------------------------------------------------------
# What it computes

@pytest.mark.parametrize("expression, expected", [
    ("2 + 2", "4"),
    ("(1 + 5) / 7", str(6 / 7)),
    ("2 ** 10", "1024"),
    ("sqrt(2)", str(math.sqrt(2))),
    ("log(e)", "1.0"),
    ("round(pi, 4)", "3.1416"),
    ("max(3, 7, 5)", "7"),
    ("factorial(10)", "3628800"),
])
def test_it_evaluates(expression, expected):
    assert llmtools.calculate(expression) == f"{expression} = {expected}"


def test_the_answer_carries_the_question():
    """The result goes into the chat as a tool message the model reads back later, possibly several rounds
    on. Bare `4` is ambiguous by then; `2 + 2 = 4` still says what was asked."""
    assert llmtools.calculate("2 + 2").startswith("2 + 2 = ")


def test_big_integers_stay_exact():
    """The reason to call a tool at all rather than let the model estimate."""
    assert llmtools.calculate("2 ** 100") == f"2 ** 100 = {2 ** 100}"


# --------------------------------------------------------------------------------
# What it refuses
#
# Each of these is a way out of an expression evaluator, or a way to hang one. They are checked here rather
# than assumed from `simpleeval`'s documentation, because the guarantee this tool makes to the user is that
# an expression written by a *language model* cannot reach anything.

@pytest.mark.parametrize("expression", [
    "__import__('os').system('true')",   # the classic escape
    "(1).__class__.__bases__",           # attribute access, the usual route out of a sandbox
    "open('/etc/passwd')",               # a name that is simply not there
    "x = 1",                             # a statement, not an expression
    "[i for i in range(10)]",            # comprehensions are not enabled
    "9 ** 9 ** 9",                       # resource exhaustion by exponent
    "'a' * 10 ** 10",                    # ...and by repetition
    "1 / 0",                             # ordinary arithmetic failure
    "sqrt(-1)",                          # ...and a domain error
    "math.sqrt(2)",                      # the module prefix; `math` is not a name here, so it never
                                         # reaches the attribute access above
    "",                                  # nothing at all
])
def test_it_declines(expression):
    result = llmtools.calculate(expression)
    assert result.startswith("That is not an expression this calculator can evaluate")


def test_a_refusal_says_what_to_do_instead():
    """The model acts on this string. A bare "error" would leave it to guess whether to retry, rephrase, or
    give up — and a model that guesses "retry" loops until the round cap."""
    result = llmtools.calculate("x = 1")
    assert "single expression" in result
    assert "yourself" in result, "the model should be told it may simply answer without the tool"


def test_the_guidance_names_the_bare_spelling_of_the_math_functions():
    """Both places the model reads about the expression language have to rule out the `math.` prefix.

    Measured against qwen3.6-35b-a3b on 2026-08-26, asking for a circle's area minus its circumference:
    with the earlier wording ("functions from the math module are available") 4 of 8 samples wrote
    `math.pi`, took the refusal, and spent a second round recovering — twice by substituting a truncated
    decimal literal for `pi`. Saying instead that the names are bare took it to 0 of 8.

    So this asserts the contrast is spelled out, not merely that the tool refuses the prefix. The refusal
    is what the model reads *after* wasting a round; the spec is what stops the round being spent.
    """
    for guidance in (llmtools.CANONICAL_BAD_EXPRESSION,
                     next(spec["function"]["parameters"]["properties"]["expression"]["description"]
                          for spec in llmtools.TOOLS if spec["function"]["name"] == "calculate")):
        assert "math.sqrt(2)" in guidance, f"the wrong spelling is not shown: {guidance}"
        assert "'sqrt(2)'" in guidance, f"the right spelling is not shown: {guidance}"


def test_the_random_functions_are_not_available():
    """`simpleeval` offers `rand` and `randint` by default, and they are deliberately dropped.

    A tool called `calculate` that can return a different answer to the same question is a trap: nothing
    in the transcript marks the answer as arbitrary, so neither the model nor the reader can tell.
    """
    assert llmtools.calculate("rand()").startswith("That is not an expression")
    assert llmtools.calculate("randint(1, 6)").startswith("That is not an expression")


# --------------------------------------------------------------------------------
# The registry
#
# A tool is a schema, a function and a gating decision that all have to agree; these are the mismatches
# that show up at runtime as the model calling something that does not exist.

def test_the_tool_is_registered_and_specified():
    assert "calculate" in llmtools.TOOL_ENTRYPOINTS
    assert any(spec["function"]["name"] == "calculate" for spec in llmtools.TOOLS)


def test_arithmetic_needs_no_permission():
    """Ungated, like `get_current_time`: neither switch claims to govern it, because it reaches nothing.

    A calculator behind the *Internet* toggle would be wrong in a way that costs the demo — arithmetic
    would silently stop working whenever someone turned the network off.
    """
    assert "calculate" not in llmtools.NETWORK_TOOL_NAMES
    assert "calculate" not in llmtools.DOCUMENT_TOOL_NAMES


def test_every_specified_tool_has_an_entrypoint():
    """Not specific to the calculator, and cheap to assert while adding one."""
    specified = {spec["function"]["name"] for spec in llmtools.TOOLS}
    assert specified == set(llmtools.TOOL_ENTRYPOINTS)
