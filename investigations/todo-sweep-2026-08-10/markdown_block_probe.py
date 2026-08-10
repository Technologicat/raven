#!/usr/bin/env python
"""Why block-level Markdown never renders in Librarian's chat view.

Answers a question three deferred items asked separately: ATX headings show their literal `#`
markers, fenced code blocks do not render, and tables do not render. The items attribute all three
to the vendored renderer. Two of the three attributions are wrong.

Run from the repo root:

    python investigations/todo-sweep-2026-08-10/markdown_block_probe.py

What it establishes, in order:

1. The vendored renderer *does* handle headings and preformatted blocks. `parser.py` maps `h1`-`h6`
   and `pre` to entities, `__init__.py` maps those entities to font attributes, and the round trip
   below produces a `MessageEntityH3` from real Markdown. Only `table` is genuinely absent.

2. `chat_controller._render_text` wraps each paragraph as `<font color='...'>{text}</font>` before
   handing it to the renderer. That single wrapper is what loses the headings: with the open tag on
   the same line as the content, the whole thing is an ordinary paragraph containing inline raw
   HTML, and ATX headings are block-level constructs that cannot occur inside a paragraph. Inline
   constructs (`**bold**`, `*italic*`, `` `code` ``) are unaffected, which is why everything else in
   the chat view looks right and only the block constructs vanish.

3. `chat_controller._render_text_paragraphs` splits the message on *single* newlines and renders each
   line as its own call, so a construct spanning lines - a fenced code block, a table - has no way to
   form even before the wrapper is applied.

So the chat view has two independent barriers in front of block-level Markdown, and the renderer is
behind both of them. Fixing the renderer alone would change nothing.
"""

import mistletoe

from raven.vendor.DearPyGui_Markdown import parser


def show(label: str, source: str) -> str:
    html = mistletoe.markdown(source)
    print(f"  {label:34s} -> {html!r}")
    return html


def main() -> None:
    color = "(255,255,255)"  # any colour; the wrapper's presence is what matters, not its value

    print("1. The renderer's own capability, via a real parse")
    html = mistletoe.markdown("### A heading\n\nBody text with `code`.\n")
    p = parser._HTMLToParser()
    p.feed(html)
    print(f"  entities from bare Markdown       -> {[type(e).__name__ for e in p.entities]}")
    p = parser._HTMLToParser()
    p.feed(f"<font color='{color}'>{html}</font>")
    print(f"  entities with the font wrapper    -> {[type(e).__name__ for e in p.entities]}")
    print("  (H3 survives here because the wrapper is applied to *rendered HTML*, not to Markdown)")

    print("\n2. What the chat view actually does: wrap the Markdown, then render")
    show("bare Markdown", "### A heading")
    show("as the chat view sends it", f"<font color='{color}'>### A heading</font>")
    show("inline constructs, same wrapper", f"<font color='{color}'>**bold** and *italic*</font>")

    print("\n3. Multi-line constructs, one line per render call")
    fenced = "```python\nx = 1\n```"
    show("fenced block, whole", fenced)
    for line in fenced.split("\n"):
        show(f"  line {line!r}", f"<font color='{color}'>{line.strip()}</font>")


if __name__ == "__main__":
    main()
