#!/usr/bin/env python
"""Generate the postprocessor filter manual from the filter docstrings.

`Postprocessor.get_filters()` already ships each filter's defaults, parameter ranges and docstring --
it was built so the settings editor could render per-parameter help without importing the filter
implementations. This walks the same data and writes it out as Markdown, so a reader who is not
sitting in front of the app has somewhere to look that is not the source.

The docstrings are the single source of truth. The output file is generated in full and carries a
header saying so; edit the docstrings in `raven/common/video/postprocessor.py` and re-run this.

    python scripts/gen_postprocessor_manual.py            # write the file
    python scripts/gen_postprocessor_manual.py --check     # exit 1 if it is out of date

`--check` is currently a release-time step (Raven's `CLAUDE.md`, workflow rule 9). It *could* run in CI
and does not yet: the postprocessor needs torch, which `requirements-ci.txt` lacks — but the workflows
install the torch trio from PyTorch's CPU wheel index in a line of their own, on every OS in the matrix,
so the dependency is there. CI would be the better trigger, catching a stale file on the push that
staled it rather than at the next release.
"""

import argparse
import pathlib
import re
import sys

from raven.common import docstring_utils

__all__ = ["build", "main"]

# `raven.common.video.postprocessor` is imported inside `main`, not here. It pulls in torch, which the
# rest of this module does not need -- so importing the module to test its rendering costs nothing, and
# those tests can run in CI, where torch is deliberately absent.

# The contents listing has to agree with `check_doc_links.py` about what a heading anchors as, so it
# uses that checker's own rule rather than a second copy of it. Underscores survive slugification and
# hyphens do not appear -- `## `analog_vhs_noise`` anchors as `analog_vhs_noise` -- which is exactly
# the kind of thing a reimplementation gets wrong and a shared function cannot.
#
# The repository root goes on the path so that the spelling is `scripts.check_doc_links` whether this
# runs as a script (where `sys.path[0]` is `scripts/`) or is imported by the tests (where it is the
# root). The two spellings would otherwise load two copies of the same module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.check_doc_links import slugify  # noqa: E402 -- needs the path entry above

# Where the manual lives: beside the code it is generated from, linked from `raven/avatar/README.md`.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "raven" / "common" / "video" / "postprocessor-filters.md"

# `get_filters` reports a parameter's range as `[min, max]`, as a list of allowed strings, or as a
# one-element list holding a `!`-prefixed hint to the GUI. The hints are the settings editor's, and
# this renders rather than obeys them -- except `!ignore` on `name`, which see below.
GUI_HINTS = {"!RGB": "an RGB colour"}

# Every filter's `name` is the same cache key with the same meaning, so the manual says it once in the
# preamble instead of repeating it per filter. Anything *else* carrying `!ignore` is reported rather
# than dropped: the hint means "hide from the GUI", which is not by itself a reason to hide it here.
UNIVERSAL_PARAM = "name"

PREAMBLE = """\
{n_filters} filters, applied in the order you list them. This page is the reference; \
[`raven/avatar/README.md`](../../avatar/README.md#the-postprocessor) is the shorter catalogue, and says \
how a chain is put together.

**The same material is in the app**, beside live controls: run `raven-avatar-settings-editor` and each \
filter carries its own help there, which is the better place to read it while watching the effect on the \
character. This page is for reading away from the app.

**The order below is the order the filters run in**, which traces a signal from the room through the \
camera and the recording and the transport to the display. A chain that follows it reads as one imagined \
apparatus rather than as a pile of effects.

**Every filter takes a `name`**, which is a cache key rather than a setting. It matters only when the same \
filter appears more than once in one chain, and each instance then needs its own. It is omitted from the \
parameter lists below, being the same in all of them.
"""


def _one_line(text):
    """Collapse a docstring block's own wrapping into a single line.

    The fleet's Markdown is unwrapped -- one line per bullet, however long -- so that a file wrapped in
    places and not in others cannot happen. Re-wrapping here would also put the line breaks somewhere
    different every time a sentence changed length, which makes a regenerated file's diff unreadable.
    """
    return re.sub(r"\s+", " ", text).strip()


def _render_range(param_range):
    """Describe a parameter's permitted values in words, or return None if there is nothing to say."""
    if not param_range:
        return None
    if len(param_range) == 1 and isinstance(param_range[0], str) and param_range[0].startswith("!"):
        return GUI_HINTS.get(param_range[0])  # None for an unrecognized hint: say nothing rather than guess
    if all(isinstance(v, str) for v in param_range):
        return "one of " + ", ".join(f"`{v}`" for v in param_range)
    if len(param_range) == 2:
        lo, hi = param_range
        return f"range `{lo}` to `{hi}`"
    return None


def _split_tag(summary):
    """Separate a leading `[static]` / `[dynamic]` marker from the rest of a filter's summary.

    Every filter carries one (checked across all 21), and it answers a question the prose does not:
    whether the filter animates on its own or renders the same way every frame.
    """
    m = re.match(r"^\[(static|dynamic)\]\s*(.*)$", summary.strip(), flags=re.DOTALL)
    if not m:
        return None, summary.strip()
    return m.group(1), m.group(2).strip()


def build(filters, anomalies):
    """Return the whole manual as a string, appending anything surprising to `anomalies`.

    `filters` is what `Postprocessor.get_filters()` returns: `[(name, {defaults, ranges, docstring}), ...]`.
    """
    out = ["<!-- Generated by scripts/gen_postprocessor_manual.py from the filter docstrings in",
           "     raven/common/video/postprocessor.py. Do not edit this file; edit those and re-run. -->",
           "",
           "# Postprocessor filters",
           "",
           PREAMBLE.format(n_filters=len(filters)),
           "**Contents:**",
           ""]

    for name, _ in filters:
        out.append(f"- [`{name}`](#{slugify(f'`{name}`')})")
    out.append("")

    for name, info in filters:
        summary = docstring_utils.extract_summary(info["docstring"])
        if not summary:
            anomalies.append(f"{name}: no summary in its docstring")
            summary = ""
        kind, body = _split_tag(summary)
        if kind is None:
            anomalies.append(f"{name}: summary carries no [static]/[dynamic] marker")

        out.append(f"## `{name}`")
        out.append("")
        lead = f"*{kind.capitalize()}.* " if kind else ""
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        for i, para in enumerate(paragraphs):
            prefix = lead if i == 0 else ""
            out.append(prefix + docstring_utils.rst_inline_to_markdown(_one_line(para)))
            out.append("")
        if not paragraphs and lead:
            out.extend([lead.strip(), ""])

        for param, default in info["defaults"].items():
            param_range = info["ranges"].get(param)
            if param == UNIVERSAL_PARAM:
                continue  # said once in the preamble
            if param_range and len(param_range) == 1 and isinstance(param_range[0], str) and param_range[0] == "!ignore":
                anomalies.append(f"{name}.{param}: hidden from the GUI but is not `{UNIVERSAL_PARAM}`, so it is documented here")

            help_text = docstring_utils.extract_param_help(info["docstring"], param)
            if not help_text:
                anomalies.append(f"{name}.{param}: no help text in the docstring")

            described = _render_range(param_range)
            facts = f"default `{default!r}`" if not isinstance(default, str) else f"default `{default}`"
            if described:
                facts += f", {described}"
            out.append(f"- **`{param}`** — {facts}")
            if help_text:
                rendered = docstring_utils.rst_inline_to_markdown(_one_line(docstring_utils.strip_param_header(help_text)))
                out.append(f"  {rendered}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the file on disk is not what would be generated")
    args = parser.parse_args()

    from raven.common.video.postprocessor import Postprocessor  # local: drags in torch, see the note at the imports

    filters = Postprocessor.get_filters()
    anomalies = []
    generated = build(filters, anomalies)

    for line in anomalies:
        print(f"note: {line}", file=sys.stderr)

    rel = OUTPUT_PATH.relative_to(REPO_ROOT)
    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"FAIL: {rel} does not exist. Run this script without --check.", file=sys.stderr)
            return 1
        if OUTPUT_PATH.read_text(encoding="utf-8") != generated:
            print(f"FAIL: {rel} is out of date. Run this script without --check.", file=sys.stderr)
            return 1
        print(f"OK: {rel} matches the filter docstrings.")
        return 0

    OUTPUT_PATH.write_text(generated, encoding="utf-8")
    n_params = sum(len(info["defaults"]) for _, info in filters)
    print(f"Wrote {rel}: {len(filters)} filters, {n_params} parameters, "
          f"{len(generated.split())} words.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
