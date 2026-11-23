"""Tests for `raven.librarian.lorebook`: parsing, matching, and choosing what fires."""

import pathlib
import textwrap
import time

import pytest
from unpythonic.env import env

from raven.librarian import llmclient
from raven.librarian import llmtools
from raven.librarian import lorebook


def _parse(text: str, **kwargs) -> list[lorebook.Entry]:
    return lorebook.parse(textwrap.dedent(text), name="glossary", **kwargs)

def _titles(entries) -> list[str]:
    return [entry.title for entry in entries]

def _by_id(entries) -> dict[str, lorebook.Entry]:
    return {entry.id: entry for entry in entries}

def _chars(text: str) -> int:
    """A stand-in token counter: one token per character, so sizes are easy to reason about."""
    return len(text)


GLOSSARY = """\
    # Field Guide

    A glossary of terms, with a letter index.

    [A](#A) [B](#B)

    ---

    # A

    ## Attention rent

    **Meaning:** What an open item costs. See *[hydra tax](#hydra-tax)*.

    ## Approach memory
    Remembering how, not what.

    ---

    # H

    ## Hydra tax

    Closing one item opens two. Paid as *[attention rent](#attention-rent)*.

    ### Example

    A backlog.
    """


class TestParse:
    def test_a_glossary_splits_at_its_entries_not_its_letters(self):
        # One titled h1 with prose, letter h1s holding only h2s and a rule: the h2s are the entries.
        entries = _parse(GLOSSARY)
        assert _titles(entries) == ["Attention rent", "Approach memory", "Hydra tax"]

    def test_a_deeper_header_is_part_of_its_entry(self):
        hydra = _by_id(_parse(GLOSSARY))["hydra-tax"]
        assert "### Example" in hydra.content and hydra.content.endswith("A backlog.")

    def test_a_trailing_rule_is_not_part_of_the_entry(self):
        assert not _by_id(_parse(GLOSSARY))["approach-memory"].content.endswith("---")

    def test_several_h1s_with_prose_are_the_entries(self):
        entries = _parse("""\
            # Fron
            A front, of a kind.
            ## Etymology
            From front.

            # Fup
            Not a typo.
            """)
        assert _titles(entries) == ["Fron", "Fup"]
        assert "## Etymology" in entries[0].content

    def test_a_lone_h1_with_nothing_deeper_is_a_single_entry(self):
        assert _titles(_parse("# Spork\n\nA fork that is also a spoon.\n")) == ["Spork"]

    def test_h2s_with_no_h1_are_the_entries(self):
        assert _titles(_parse("## Jazz\nImprovised.\n\n## Spork\nA utensil.\n")) == ["Jazz", "Spork"]

    def test_a_file_with_no_headers_is_one_entry_named_for_the_file(self):
        entries = lorebook.parse("Only prose, no headers.", name="duolect")
        assert _titles(entries) == ["duolect"] and entries[0].content == "Only prose, no headers."

    def test_an_explicit_entry_level_overrides_the_rule(self):
        # The single-entry file that also has subsections reads, by the rule, as a collection of its subsections.
        text = "# Fron\nA front, of a kind.\n## Etymology\nFrom front.\n"
        assert _titles(_parse(text)) == ["Etymology"]
        assert _titles(_parse(text, entry_level=1)) == ["Fron"]

    def test_a_header_in_fenced_code_is_not_a_header(self):
        entries = _parse("""\
            ## Shell
            Example:
            ```
            # not a header
            ```
            ## Spork
            A utensil.
            """)
        assert _titles(entries) == ["Shell", "Spork"]
        assert "# not a header" in entries[0].content

    def test_parentheses_in_a_header_are_aliases(self):
        entries = _by_id(_parse("## Fup (fuppable, fupped)\nNot a typo.\n\n## Digital nature reserve (for prose basilisks)\nA refuge.\n"))
        assert entries["fup"].aliases == ("fuppable", "fupped")
        reserve = entries["digital-nature-reserve"]
        assert reserve.title == "Digital nature reserve" and reserve.aliases == ("for prose basilisks",)
        # A link may use the anchor of the header as written, parentheses and all.
        assert "digital-nature-reserve-for-prose-basilisks" in reserve.anchors

    def test_closing_hashes_are_not_part_of_the_title(self):
        assert _titles(_parse("## Jazz ##\nImprovised.\n")) == ["Jazz"]

    def test_crossrefs_are_the_anchor_links(self):
        assert _by_id(_parse(GLOSSARY))["attention-rent"].crossrefs == ("hydra-tax",)

    def test_entries_are_numbered_in_order(self):
        entries = _parse(GLOSSARY, first_order=10)
        assert [entry.order for entry in entries] == [10, 11, 12]


class TestSlugify:
    @pytest.mark.parametrize("title, slug", [("Hydra tax", "hydra-tax"),
                                             ("Chekhov's landmine", "chekhovs-landmine"),
                                             ("Build thrice, cherrypick", "build-thrice-cherrypick"),
                                             ("co-", "co-"),
                                             ("EQ curve", "eq-curve")])
    def test_it_matches_githubs_anchors(self, title, slug):
        assert lorebook.slugify(title) == slug


class TestLoad:
    def test_every_markdown_file_but_the_readme_in_name_order(self, tmp_path):
        (tmp_path / "b.md").write_text("## Spork\nA utensil.\n", encoding="utf-8")
        (tmp_path / "a.md").write_text("## Jazz\nImprovised.\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# Put your lorebook entries here\n\nHow to write them.\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("## Ignored\nNot Markdown.\n", encoding="utf-8")
        entries, problems = lorebook.load(tmp_path)
        assert list(entries) == ["jazz", "spork"] and not problems
        assert [entry.order for entry in entries.values()] == [0, 1]

    def test_two_entries_with_one_title_keep_the_first(self, tmp_path):
        (tmp_path / "a.md").write_text("## Jazz\nImprovised.\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("## Jazz\nA band.\n", encoding="utf-8")
        entries, problems = lorebook.load(tmp_path)
        assert entries["jazz"].content == "Improvised." and "b.md" in problems

    def test_an_entry_level_per_file(self, tmp_path):
        (tmp_path / "fron.md").write_text("# Fron\nA front, of a kind.\n## Etymology\nFrom front.\n", encoding="utf-8")
        assert list(lorebook.load(tmp_path)[0]) == ["etymology"]
        assert list(lorebook.load(tmp_path, entry_levels={"fron.md": 1})[0]) == ["fron"]

    def test_a_missing_directory_is_an_empty_lorebook(self, tmp_path):
        assert lorebook.load(tmp_path / "nope") == ({}, {})

    def test_a_path_may_be_str_or_path(self, tmp_path):
        (tmp_path / "a.md").write_text("## Jazz\nImprovised.\n", encoding="utf-8")
        assert lorebook.load(str(tmp_path)) == lorebook.load(tmp_path)


class TestFindMatches:
    @pytest.fixture
    def entries(self):
        return _by_id(_parse("""\
            ## Hydra tax
            Closing one opens two.
            ## EQ curve
            A shape.
            ## True Name
            What a thing is called.
            ## Same bath, different shampoo
            A rerun.
            ## co-
            A prefix.
            ## Chekhov's landmine
            A trap set early.
            ## Cat
            An animal.
            ## Fup (fuppable)
            Not a typo.
            """))

    def _ids(self, entries, text):
        return set(lorebook.find_matches(entries, text))

    def test_a_term_is_matched_as_words(self, entries):
        assert self._ids(entries, "The cat sat.") == {"cat"}
        assert self._ids(entries, "Concatenate the files.") == set()

    def test_a_titles_first_letter_does_not_make_it_exact(self, entries):
        assert self._ids(entries, "we pay a hydra tax on it") == {"hydra-tax"}
        assert self._ids(entries, "Hydra tax again") == {"hydra-tax"}

    def test_an_uppercase_letter_past_the_first_does(self, entries):
        assert self._ids(entries, "the EQ curve") == {"eq-curve"}
        assert self._ids(entries, "the eq curve") == set()
        assert self._ids(entries, "its True Name") == {"true-name"}
        assert self._ids(entries, "its true Name") == {"true-name"}  # `True` is forgiving, being first
        assert self._ids(entries, "its true name") == set()

    def test_the_words_may_come_in_any_order_next_to_each_other(self, entries):
        assert self._ids(entries, "different bath, same shampoo") == {"same-bath-different-shampoo"}
        assert self._ids(entries, "same bath, quite a different shampoo") == set()

    def test_a_suspended_hyphen_is_part_of_the_word(self, entries):
        assert self._ids(entries, "the co- prefix") == {"co-"}
        assert self._ids(entries, "a co-author") == set()
        assert self._ids(entries, "CO poisoning") == set()

    def test_curly_apostrophes_match_straight_ones(self, entries):
        assert self._ids(entries, "a Chekhov’s landmine") == {"chekhovs-landmine"}

    def test_an_alias_fires_its_entry(self, entries):
        assert self._ids(entries, "is that fuppable?") == {"fup"}

    def test_the_wording_is_the_text_as_written_at_the_last_match(self, entries):
        match = lorebook.find_matches(entries, "a hydra tax, then a different bath, same shampoo")["same-bath-different-shampoo"]
        assert match.wording == "different bath, same shampoo"
        assert lorebook.find_matches(entries, "Hydra tax here, hydra tax there")["hydra-tax"].wording == "hydra tax"

    def test_a_compound_of_a_terms_words_fires_it(self, entries):
        assert self._ids(entries, "another hydra-tax situation") == {"hydra-tax"}
        assert lorebook.find_matches(entries, "another hydra-tax situation")["hydra-tax"].wording == "hydra-tax"

    def test_a_word_joined_by_an_apostrophe_is_read_as_its_parts_too(self, entries):
        assert self._ids(entries, "the cat's toy") == {"cat"}, "a possessive, with no form listed"
        names = _by_id(_parse("## Bordeaux\nA city.\n"))
        assert self._ids(names, "Bordeaux'n viinit") == {"bordeaux"}, "Finnish inflects such a name with an apostrophe"

    def test_a_term_with_a_hyphen_of_its_own_still_matches_the_word_whole(self):
        entries = _by_id(_parse("## Aria-worthy design\nClearing the bar.\n"))
        assert self._ids(entries, "an aria-worthy design") == {"aria-worthy-design"}
        assert self._ids(entries, "an aria worthy design") == set(), "its hyphen is part of the term"

    def test_a_colon_suffix_needs_nothing_of_its_own(self, entries):
        # Finnish inflects an abbreviation or a foreign name with a colon and an ending on its last word. The
        # ending is read as a word of its own, after the term, so the term's words still stand together.
        assert self._ids(entries, "EQ curve:n muoto") == {"eq-curve"}


class TestExpansion:
    @pytest.fixture
    def entries(self):
        return _by_id(_parse("""\
            ## Hydra tax
            Closing one opens two.
            ## Kaupunki
            A town.
            ## Kerbal Space Program
            A game.
            ## EQ
            Equalization.
            """))

    @pytest.fixture
    def expansions(self):
        return {"tax": lorebook.clean_expansion("tax", forms=["taxes", "taxed"], stems=[]),
                "kaupunki": lorebook.clean_expansion("kaupunki", forms=[], stems=["kaupung", "kaupunk"]),
                "Program": lorebook.clean_expansion("Program", forms=[], stems=["Program"]),
                "EQ": lorebook.clean_expansion("EQ", forms=["EQs"], stems=[])}

    def _ids(self, entries, text, expansions):
        return set(lorebook.find_matches(entries, text, expansions=expansions))

    def test_without_them_an_inflected_term_does_not_fire(self, entries):
        assert self._ids(entries, "two hydra taxes", None) == set()

    def test_a_form_fires_its_term_and_in_any_order(self, entries, expansions):
        assert self._ids(entries, "two hydra taxes", expansions) == {"hydra-tax"}
        assert self._ids(entries, "the taxes hydra levies", expansions) == {"hydra-tax"}
        assert self._ids(entries, "HYDRA TAXED", expansions) == {"hydra-tax"}, "a lowercase key word matches in any case"

    def test_a_stem_fires_with_letters_after_it(self, entries, expansions):
        assert self._ids(entries, "Asun kaupungissa.", expansions) == {"kaupunki"}
        assert self._ids(entries, "kaupunkien", expansions) == {"kaupunki"}
        assert self._ids(entries, "kaupung2ssa", expansions) == set(), "only letters may follow a stem"

    def test_an_exact_word_takes_only_lowercase_after_a_stem(self, entries, expansions):
        assert self._ids(entries, "Kerbal Space Programissa", expansions) == {"kerbal-space-program"}
        assert self._ids(entries, "Kerbal Space ProgramISSA", expansions) == set()

    def test_an_exact_words_forms_match_as_written(self, entries, expansions):
        assert self._ids(entries, "two EQs", expansions) == {"eq"}
        assert self._ids(entries, "two eqs", expansions) == set()

    def test_a_lookup_finds_a_term_by_its_forms(self, entries, expansions):
        found = lorebook.lookup(entries, "hydra taxes", expansions=expansions)
        assert found.exact and found.maybe_entry.id == "hydra-tax"
        assert not lorebook.lookup(entries, "hydra taxes").exact, "without them, a near miss at best"

    def test_only_a_terms_last_word_matches_by_its_forms(self):
        entries = _by_id(_parse("## Prose basilisks\nText that harms.\n"))
        expansions = {"prose": lorebook.clean_expansion("prose", forms=["proses"], stems=["prosess"]),
                      "basilisks": lorebook.clean_expansion("basilisks", forms=[], stems=["basilisk"])}
        assert set(lorebook.find_matches(entries, "prose basiliskeista", expansions=expansions)) == {"prose-basilisks"}
        assert set(lorebook.find_matches(entries, "proses basilisks", expansions=expansions)) == set()
        assert set(lorebook.find_matches(entries, "prosessi basilisks", expansions=expansions)) == set()
        assert not lorebook.lookup(entries, "proses basilisks", expansions=expansions).exact
        # The rule is about the word's place in the term, not in the text, which may put the words in any order.
        assert set(lorebook.find_matches(entries, "basiliskeista prose", expansions=expansions)) == {"prose-basilisks"}

    def test_an_inflected_compound_fires_and_is_found(self, entries, expansions):
        assert self._ids(entries, "let's not get hydra-taxed", expansions) == {"hydra-tax"}
        found = lorebook.lookup(entries, "hydra-taxed", expansions=expansions)
        assert found.exact and found.maybe_entry.id == "hydra-tax"

    def test_select_passes_them_on(self, entries, expansions):
        firings = lorebook.select(entries, "two hydra taxes", count_tokens=_chars, budget=1000, max_depth=0,
                                  expansions=expansions)
        assert [firing.entry.id for firing in firings] == ["hydra-tax"]
        assert lorebook.select(entries, "two hydra taxes", count_tokens=_chars, budget=1000, max_depth=0) == []


class TestCleanExpansion:
    def test_a_stem_keeps_all_but_the_last_two_letters_of_its_word(self):
        assert lorebook.clean_expansion("kaupunki", forms=[], stems=["kaupun", "kaupu"]).stems == ("kaupun",)

    def test_and_never_fewer_than_five(self):
        assert lorebook.clean_expansion("hydra", forms=[], stems=["hydra", "hydr"]).stems == ("hydra",)
        assert lorebook.clean_expansion("lint", forms=[], stems=["lint"]).stems == (), "even the whole word"
        assert lorebook.clean_expansion("fup", forms=[], stems=["fup", "f"]).stems == ()

    def test_a_stem_that_wanders_off_the_word_is_dropped(self):
        assert lorebook.clean_expansion("duolect", forms=[], stems=["duolekt", "dialekt"]).stems == ("duolekt",)

    def test_the_word_itself_and_malformed_forms_are_dropped(self):
        cleaned = lorebook.clean_expansion("fup", forms=["fup", "Fups", "fups", "fupped up", ""], stems=[])
        assert cleaned.forms == ("fups",), "lowercased, deduplicated, and one word each"

    def test_an_exact_words_case_is_kept(self):
        assert lorebook.clean_expansion("EQ", forms=["EQs", "EQ"], stems=[]).forms == ("EQs",)


class TestSelect:
    @pytest.fixture
    def entries(self):
        more = """
            ## Jazz
            Improvised, the way an approach memory is.

            ## Spork
            A utensil.
            """
        entries = _by_id(_parse(textwrap.dedent(GLOSSARY) + textwrap.dedent(more)))  # each dedented on its own
        assert {"jazz", "spork"} <= set(entries), "the fixture's own entries did not parse"
        return entries

    def _select(self, entries, text, **kwargs):
        options = dict(count_tokens=_chars, budget=10_000, max_depth=2)
        options.update(kwargs)
        return [(firing.entry.id, firing.depth) for firing in lorebook.select(entries, text, **options)]

    def test_a_term_fires_and_its_links_follow(self, entries):
        # Attention rent links to hydra tax, which links back: the cycle ends because nothing fires twice.
        assert self._select(entries, "the attention rent is high") == [("attention-rent", 0), ("hydra-tax", 1)]

    def test_an_entry_without_links_leads_to_the_terms_in_its_content(self, entries):
        assert self._select(entries, "some jazz") == [("jazz", 0), ("approach-memory", 1)]

    def test_depth_is_bounded(self, entries):
        assert self._select(entries, "the attention rent is high", max_depth=0) == [("attention-rent", 0)]

    def test_shallower_first_and_in_order_within_a_depth(self, entries):
        assert self._select(entries, "spork, jazz and attention rent") == \
            [("attention-rent", 0), ("jazz", 0), ("spork", 0), ("approach-memory", 1), ("hydra-tax", 1)]

    def test_what_does_not_fit_is_skipped_and_what_does_is_taken(self, entries):
        sizes = {entry_id: _chars(f"{entry.title}\n\n{entry.content}") for entry_id, entry in entries.items()}
        budget = sizes["jazz"] + sizes["spork"]  # enough for those two, not for attention rent, which comes first
        assert sizes["attention-rent"] > sizes["spork"], "the fixture needs the skipped entry to be the larger one"
        chosen = self._select(entries, "spork, jazz and attention rent", budget=budget)
        assert chosen == [("jazz", 0), ("spork", 0)], "a skipped entry leads nowhere, and later ones that fit are taken"

    def test_already_fired_entries_neither_fire_nor_lead_anywhere(self, entries):
        assert self._select(entries, "the attention rent is high", already_fired={"attention-rent"}) == []
        assert self._select(entries, "the attention rent is high", already_fired={"hydra-tax"}) == [("attention-rent", 0)]

    def test_the_wording_and_the_link_are_recorded(self, entries):
        firings = lorebook.select(entries, "Attention rent!", count_tokens=_chars, budget=10_000, max_depth=1)
        assert firings[0].maybe_wording == "Attention rent" and firings[0].maybe_via is None
        assert firings[1].maybe_wording is None and firings[1].maybe_via == "attention-rent"

    def test_nothing_in_the_text_fires_nothing(self, entries):
        assert self._select(entries, "nothing relevant here") == []


class TestLookup:
    @pytest.fixture
    def entries(self):
        return _by_id(_parse("""\
            ## Hydra tax
            Closing one opens two.
            ## Hydra test
            A test with heads.
            ## Same bath, different shampoo
            A rerun.
            ## Fup (fuppable)
            Not a typo.
            ## Spork
            A utensil.
            ## Miroconnection
            A link between two minds, seen in a mirror.
            ## Endognosis
            What a system knows about itself from inside; introspection names the act, this the result.
            """))

    def test_an_unrelated_word_is_no_near_miss_however_alike_it_looks(self, entries):
        # A model searching by meaning asked for "introspection", and difflib's default cutoff took it for this.
        found = lorebook.lookup(entries, "introspection")
        assert found.maybe_entry is None and "Miroconnection" not in found.candidates

    def test_an_entry_whose_content_mentions_the_term_is_named(self, entries):
        found = lorebook.lookup(entries, "Introspection")
        assert found.mentioned_in == ("Endognosis",)
        answer = lorebook.answer(entries, "Introspection")
        assert lorebook.CANONICAL_MENTIONED_IN.format(titles="'Endognosis'") in answer
        assert answer.startswith(lorebook.CANONICAL_NO_ENTRY.format(term="Introspection"))

    def test_the_words_of_a_key_in_any_case_and_order(self, entries):
        found = lorebook.lookup(entries, "SHAMPOO different same bath")
        assert found.exact and found.maybe_entry.id == "same-bath-different-shampoo"

    def test_an_alias_is_a_key(self, entries):
        assert lorebook.lookup(entries, "fuppable").maybe_entry.id == "fup"

    def test_one_near_miss_is_taken_as_meant(self, entries):
        found = lorebook.lookup(entries, "sprok")
        assert not found.exact and found.maybe_entry.id == "spork"

    def test_several_near_misses_are_offered_by_title(self, entries):
        found = lorebook.lookup(entries, "hydra te")
        assert found.maybe_entry is None and set(found.candidates) == {"Hydra tax", "Hydra test"}

    def test_nothing_close_is_not_defined(self, entries):
        found = lorebook.lookup(entries, "photosynthesis")
        assert found.maybe_entry is None and found.candidates == ()

    def test_the_answers_the_model_reads(self, entries):
        assert lorebook.answer(entries, "spork") == "Spork\n\nA utensil."
        assert lorebook.answer(entries, "sprok").startswith(lorebook.CANONICAL_NEAR_MISS.format(term="sprok", title="Spork"))
        assert "'Hydra tax'" in lorebook.answer(entries, "hydra te")
        assert lorebook.answer(entries, "photosynthesis") == lorebook.CANONICAL_NOT_DEFINED.format(term="photosynthesis")


class TestLorebook:
    @pytest.fixture
    def settings(self):
        model_info = env(label="m", model_id="m", context_length=4096, is_vlm=False, loaded=True)
        return llmclient.configure(model_info=model_info, backend_flavor="generic", backend_url="http://x", quiet=True)

    @pytest.fixture
    def folder(self, tmp_path):
        folder = tmp_path / "lorebook"
        folder.mkdir()
        (folder / "glossary.md").write_text("## Spork\nA utensil.\n\n## Fup\nNot a typo.\n", encoding="utf-8")
        return folder

    @staticmethod
    def _wait_until(predicate, timeout=10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_the_entries_and_the_tool_are_in_the_settings(self, settings, folder):
        with lorebook.loaded(settings, directory=folder) as book:
            assert list(book.entries) == ["spork", "fup"]
            assert list(settings.lorebook_entries) == ["spork", "fup"]
            assert settings.tool_entrypoints[llmtools.LOREBOOK_TOOL_NAME](term="spork") == "Spork\n\nA utensil."
            assert llmtools.LOREBOOK_TOOL_NAME in {spec["function"]["name"] for spec in settings.tools}
        assert llmtools.LOREBOOK_TOOL_NAME not in settings.tool_entrypoints, "leaving the block removes the tool"
        assert settings.lorebook_entries == {}

    def test_an_empty_lorebook_offers_no_tool(self, settings, tmp_path):
        with lorebook.loaded(settings, directory=tmp_path / "nope") as book:
            assert book.entries == {} and llmtools.LOREBOOK_TOOL_NAME not in settings.tool_entrypoints

    def test_watching_creates_a_missing_folder_whose_readme_is_no_entry(self, settings, tmp_path):
        folder = tmp_path / "lorebook"
        with lorebook.loaded(settings, directory=folder):
            assert not folder.exists(), "a script neither creates nor follows the folder"
        book = lorebook.Lorebook(settings, directory=folder)
        book.start(watch=True)
        try:
            assert (folder / "README.md").read_text(encoding="utf-8").startswith("# Put your lorebook entries here")
            assert book.entries == {} and not book.problems
        finally:
            book.shutdown()

    def test_an_entry_added_while_watching_is_offered_and_announced(self, settings, folder):
        calls = []
        book = lorebook.Lorebook(settings, directory=folder, on_changed=lambda: calls.append(1))
        book.start(watch=True)
        try:
            assert llmtools.LOREBOOK_TOOL_NAME in settings.tool_entrypoints
            (folder / "jazz.md").write_text("# Jazz\nImprovised.\n", encoding="utf-8")
            assert self._wait_until(lambda: "jazz" in settings.lorebook_entries)
            assert self._wait_until(lambda: calls)
            assert settings.tool_entrypoints[llmtools.LOREBOOK_TOOL_NAME](term="jazz") == "Jazz\n\nImprovised."
        finally:
            book.shutdown()

    def test_an_edited_definition_is_read_again_without_an_announcement(self, settings, folder):
        calls = []
        book = lorebook.Lorebook(settings, directory=folder, on_changed=lambda: calls.append(1))
        book.start(watch=True)
        try:
            (folder / "glossary.md").write_text("## Spork\nA fork that is also a spoon.\n\n## Fup\nNot a typo.\n", encoding="utf-8")
            assert self._wait_until(lambda: settings.lorebook_entries["spork"].content == "A fork that is also a spoon.")
            assert calls == [], "the same entries, so nothing for a frontend to update"
        finally:
            book.shutdown()

    def test_a_script_does_not_follow_the_folder(self, settings, folder):
        with lorebook.loaded(settings, directory=folder):
            (folder / "jazz.md").write_text("# Jazz\nImprovised.\n", encoding="utf-8")
            time.sleep(0.5)
            assert "jazz" not in settings.lorebook_entries

    def test_the_gate_withholds_the_tool(self, settings, folder):
        with lorebook.loaded(settings, directory=folder):
            gated = dict(documents_available=True, internet_available=True, mcp_available=True, skills_available=True)
            assert llmtools.maybe_tool_names_for_turn(settings, **gated, lorebook_available=True) is None
            offered = llmtools.maybe_tool_names_for_turn(settings, **gated, lorebook_available=False)
            assert llmtools.LOREBOOK_TOOL_NAME not in offered and "websearch" in offered

    def test_the_tool_is_costed_in_its_own_group_while_it_is_offered(self, settings, folder):
        assert llmclient.tool_cost_by_group(settings)["lorebook"] == 0, "no lorebook, so nothing to cost"
        with lorebook.loaded(settings, directory=folder):
            assert llmclient.tool_cost_by_group(settings)["lorebook"] > 0

    def test_a_path_may_be_str_or_path(self, settings, folder):
        with lorebook.loaded(settings, directory=str(folder)) as book:
            assert list(book.entries) == ["spork", "fup"]


@pytest.mark.skipif(not (pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md").exists(),
                    reason="the real glossary is on the maintainer's machines only")
def test_the_real_glossary_parses_and_its_links_resolve():
    path = pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md"
    entries = lorebook.parse(path.read_text(encoding="utf-8"), name="glossary", source=path)
    assert len(entries) > 50, "the letter sections or the title were taken for entries, or the entries were missed"
    anchors = {anchor for entry in entries for anchor in entry.anchors}
    assert {crossref for entry in entries for crossref in entry.crossrefs} <= anchors


@pytest.mark.skipif(not (pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md").exists(),
                    reason="the real glossary is on the maintainer's machines only")
def test_on_the_real_glossary_a_typo_finds_its_entry_and_a_search_by_meaning_finds_none():
    """The near-miss cutoff against real data: typos of every title, and the terms a model tried by meaning."""
    path = pathlib.Path.home() / "Documents/koodit/substrate-independent/glossary.md"
    entries = _by_id(lorebook.parse(path.read_text(encoding="utf-8"), name="glossary", source=path))
    wrong = []
    for entry in entries.values():
        if len(entry.title) >= 4:
            typo = entry.title[:len(entry.title) // 2] + entry.title[len(entry.title) // 2 + 1:]
            found = lorebook.lookup(entries, typo)
            if found.maybe_entry is not None and found.maybe_entry.id != entry.id:
                wrong.append((typo, found.maybe_entry.title))
            elif found.maybe_entry is None and entry.title not in found.candidates:
                wrong.append((typo, None))
    assert wrong == [], "a one-letter typo of a title missed its entry or found another"
    # Asked by Flash Next looking for "endognosis" by meaning, on 2026-11-19.
    for query in ["introspection", "interoception", "metacognition", "meta-attention", "proprioception"]:
        assert lorebook.lookup(entries, query).maybe_entry is None, f"{query!r} was taken for an entry it is not"
