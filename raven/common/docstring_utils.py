"""Parse Raven-style Python docstrings for per-parameter help text.

Raven's house docstring style marks parameters with single-backticked
names followed by a colon — e.g.::

    `strength`: How much noise to apply. 0 is no noise, 1 replaces
                the input with noise.

This module's `extract_param_help` pulls out the description block
for a named parameter, dedents it, and returns a string ready for
rendering with `DearPyGui_Markdown`.

Also included: `rst_inline_to_markdown`, a small best-effort converter
for the subset of RST syntax that appears in Raven docstrings (mainly
double-backtick literals → single-backtick inline code).

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["extract_summary", "extract_param_help", "strip_param_header", "rst_inline_to_markdown"]

import inspect
import re
import textwrap
from typing import Optional


# Matches the start of a Raven-style parameter block. The minimal form is
# ``name`: ``, but some params add a modifier phrase between the backticked
# name and the colon — e.g. ```ntsc_chroma` (``"NTSC"`` only): ``` in the
# postprocessor. Also supports joined headers like ```hold_min`, `hold_max`: ```
# where several params share one description — `extract_param_help` collects
# every backticked name appearing before the colon.
_PARAM_HEADER_RE = re.compile(r"^`(\w+)`[^:\n]*:", re.MULTILINE)
_BACKTICKED_NAME_RE = re.compile(r"`(\w+)`")


def extract_summary(docstring: Optional[str]) -> Optional[str]:
    """Pull out the preamble — everything before the first `` `name`: `` parameter header.

    This is the "what does this thing do" description that typically precedes the
    parameter list in a Raven-style docstring. Returns `None` if `docstring` is
    empty, or an empty-after-stripping string would result.

    If the docstring has no parameter headers at all, the whole (cleandoc'd) text
    is returned as the summary — it's all preamble.
    """
    if not docstring:
        return None
    doc = inspect.cleandoc(docstring)
    m = _PARAM_HEADER_RE.search(doc)
    summary = (doc[:m.start()] if m is not None else doc).strip()
    return summary or None


def extract_param_help(docstring: Optional[str], param_name: str) -> Optional[str]:
    """Pull out the `param_name`'s description block from a Raven-style docstring.

    Returns the description as a single string (with internal newlines
    preserved and continuation lines dedented), or `None` if `docstring`
    is empty or `param_name` isn't documented.

    The description runs from the line starting with `` `param_name`: ``
    up to (but not including) the next line starting with a new `` `name`: ``
    header at column 0, or to end-of-docstring if this is the last one.

    Joined headers: multiple params can share one description block by
    listing them together before the colon, e.g.
    ``\\`hold_min\\`, \\`hold_max\\`: in frames``. Looking up *any* of the
    listed names returns the shared block.
    """
    if not docstring:
        return None
    doc = inspect.cleandoc(docstring)

    matches = list(_PARAM_HEADER_RE.finditer(doc))
    for i, m in enumerate(matches):
        # Collect every backticked name from the header line (up to the colon).
        # Catches both the minimal ``name`: `` form and joined headers like
        # ``name1`, `name2`: ``.
        line_end = doc.find("\n", m.start())
        if line_end == -1:
            line_end = len(doc)
        header_line = doc[m.start():line_end]
        colon_idx = header_line.find(":")
        header_before_colon = header_line[:colon_idx] if colon_idx >= 0 else header_line
        names = _BACKTICKED_NAME_RE.findall(header_before_colon)

        if param_name not in names:
            continue

        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        block = doc[start:end].rstrip()
        # First line is the header; continuation lines carry the original code
        # indentation from the source. Dedent the continuation so rendering
        # doesn't propagate that.
        lines = block.splitlines()
        if len(lines) > 1:
            cont = textwrap.dedent("\n".join(lines[1:]))
            block = lines[0] + "\n" + cont
        return block
    return None


def strip_param_header(text: str) -> str:
    """Remove the leading `` `name`: `` (or `` `name1`, `name2`: ``) prefix from a param help block.

    Useful for UI that shows the parameter name separately from the description text —
    the header becomes redundant, and in some tooltip renderers the leading backtick-code
    at the start of a block interacts oddly with downstream markdown (e.g. bulleted lists
    later in the same block). Strips the first line's header; leaves the rest untouched.
    """
    return re.sub(r"^`\w+`(?:,\s*`\w+`)*[^:\n]*:\s*", "", text, count=1)


def rst_inline_to_markdown(text: str) -> str:
    """Convert the RST inline markup we actually use in Raven docstrings to Markdown.

    Best-effort, covers the common cases:

    - `` ``literal`` `` (RST inline literal, double backticks) →
      `` `literal` `` (Markdown inline code).

    Raven's single-backticked ``\\`name\\``` form is already valid Markdown
    inline code, so it passes through untouched.
    """
    return re.sub(r"``([^`]+)``", r"`\1`", text)
