"""Whether `scripts/check_doc_links.py` would notice a document that is actually broken.

The first of these for a `scripts/` checker. The others are untested, which is filed in
`TODO_DEFERRED.md`; this module is the pattern they should follow.

**Every case here feeds the checker a document broken on purpose.** A checker that has only ever seen a
correct README cannot show that it would report an incorrect one — and this one shipped a first draft
whose slug function reported two *working* links as broken, which no amount of running it against a good
document would have revealed.

The cases are the ones from `pyan/tests/test_docs.py`, which the checker's logic is copied from, plus the
`&` case that the copy was written to get right.
"""

from scripts.check_doc_links import dangling_links, slugify, toc_problems

# A document whose table of contents and headings agree, used as the baseline that every other case
# perturbs. If this one ever reports a problem, the perturbed cases prove nothing.
GOOD = """\
- [Alpha](#alpha)
- [Beta](#beta)

# Alpha

# Beta
""".splitlines()


def test_checker_accepts_a_consistent_document():
    assert toc_problems(GOOD) == []


def test_checker_notices_a_heading_absent_from_the_toc():
    lines = [line for line in GOOD if line != "- [Beta](#beta)"]
    assert any("heading missing from TOC" in p for p in toc_problems(lines))


def test_checker_notices_a_toc_entry_with_no_heading():
    lines = GOOD + ["", "Stray: [Gamma](#gamma)"]
    lines.insert(2, "- [Gamma](#gamma)")
    assert any("points at no heading" in p for p in toc_problems(lines))


def test_checker_notices_a_toc_out_of_order():
    lines = ["- [Beta](#beta)", "- [Alpha](#alpha)", "", "# Alpha", "", "# Beta"]
    assert "TOC is not in document order" in toc_problems(lines)


def test_checker_notices_a_dangling_prose_link():
    assert dangling_links(GOOD + ["", "See [Delta](#delta) for more."]) == ["delta"]


def test_checker_ignores_shell_comments_in_fenced_blocks():
    lines = GOOD + ["", "```bash", "# Generate DOT, then render it", "```"]
    assert toc_problems(lines) == []


def test_punctuation_is_dropped_before_spaces_become_hyphens():
    """GitHub removes `&` and leaves the spaces that flanked it, so the slug has a double hyphen.

    The regression this pins: a slug function that collapses whitespace yields `install-run`, which
    matches no real anchor, so every link to such a heading is reported broken. Raven's README has two
    headings of this shape, and both were reported as failures before the rule was checked against
    GitHub's actual output.
    """
    assert slugify("Install & run") == "install--run"
    assert slugify("Pin vsync on multi-monitor setups (NVIDIA + X11)") == "pin-vsync-on-multi-monitor-setups-nvidia--x11"


def test_a_double_hyphen_heading_is_matched_by_its_own_link():
    """The end-to-end form of the case above: the link resolves, so nothing is reported."""
    lines = ["- [Install & run](#install--run)", "", "# Install & run"]
    assert toc_problems(lines) == []
    assert dangling_links(lines) == []


def test_a_link_quoted_in_an_inline_code_span_is_not_a_link():
    """Prose *about* link syntax names no heading, and a brief explaining the notation is not rot.

    Found by running the checker over every tracked document rather than the README it was written
    for: `briefs/librarian-extension/05_librarian-lorebook-brief.md` describes `[text](#anchor)` as the
    thing it walks, and both examples were reported as broken references.
    """
    quoted = ["# Real", "", "The notation `[term](#term-anchor)` is what it follows."]
    assert dangling_links(quoted) == []


def test_an_unquoted_link_is_still_reported():
    """The control for the case above: skipping code spans must not skip everything.

    Without this, a bug that dropped every line would pass the quoted-link test and report nothing
    ever again, which looks exactly like a document with no problems.
    """
    plain = ["# Real", "", "The notation [term](#term-anchor) is what it follows."]
    assert dangling_links(plain) == ["term-anchor"]


def test_completeness_is_reported_separately_from_the_other_toc_faults():
    """`check_file` filters this one out per-document, so it has to be distinguishable by prefix.

    Component READMEs index their top-level sections and stop; demanding completeness of them reports
    the author's intent as a fault. The prefix match is what lets `toc_problems` stay identical to
    pyan's copy.
    """
    from scripts.check_doc_links import TOC_COMPLETENESS_PREFIX

    partial = ["- [Alpha](#alpha)", "", "# Alpha", "", "## Buried"]
    problems = toc_problems(partial)
    assert [p for p in problems if p.startswith(TOC_COMPLETENESS_PREFIX)] == [
        f"{TOC_COMPLETENESS_PREFIX}buried"]
    assert [p for p in problems if not p.startswith(TOC_COMPLETENESS_PREFIX)] == []
