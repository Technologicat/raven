"""Tests for the boolean expression parser and query builder."""

import pytest

from raven.papers.query import (
    BinOp,
    Term,
    Token,
    TT,
    node_to_query,
    parse_query,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_single_word(self):
        assert tokenize("hello") == [Token(TT.WORD, "hello")]

    def test_quoted_phrase(self):
        assert tokenize('"hello world"') == [Token(TT.PHRASE, "hello world")]

    def test_operators(self):
        tokens = tokenize("a AND b OR c ANDNOT d")
        types = [t.type for t in tokens]
        assert types == [TT.WORD, TT.AND, TT.WORD, TT.OR, TT.WORD, TT.ANDNOT, TT.WORD]

    def test_operators_case_insensitive(self):
        tokens = tokenize("a and b or c andnot d")
        types = [t.type for t in tokens]
        assert types == [TT.WORD, TT.AND, TT.WORD, TT.OR, TT.WORD, TT.ANDNOT, TT.WORD]

    def test_parens(self):
        tokens = tokenize("(a OR b)")
        types = [t.type for t in tokens]
        assert types == [TT.LPAREN, TT.WORD, TT.OR, TT.WORD, TT.RPAREN]

    def test_keyword_as_prefix_not_matched(self):
        """'ANDROID' should not be split into AND + ROID."""
        tokens = tokenize("ANDROID")
        assert tokens == [Token(TT.WORD, "ANDROID")]

    def test_keyword_as_prefix_not_matched_or(self):
        """'ORACLE' should not be split into OR + ACLE."""
        tokens = tokenize("ORACLE")
        assert tokens == [Token(TT.WORD, "ORACLE")]

    def test_unterminated_quote(self):
        with pytest.raises(ValueError, match="Unterminated"):
            tokenize('"hello')

    def test_blank_lines_ignored(self):
        tokens = tokenize('a\n\nAND\n\nb')
        types = [t.type for t in tokens]
        assert types == [TT.WORD, TT.AND, TT.WORD]

    def test_mixed_whitespace(self):
        tokens = tokenize('  a \t AND \n b  ')
        types = [t.type for t in tokens]
        assert types == [TT.WORD, TT.AND, TT.WORD]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParseQuery:
    def test_single_word(self):
        ast = parse_query("hello")
        assert ast == Term("hello", is_phrase=False)

    def test_single_phrase(self):
        ast = parse_query('"hello world"')
        assert ast == Term("hello world", is_phrase=True)

    def test_and(self):
        ast = parse_query("a AND b")
        assert ast == BinOp("AND", Term("a"), Term("b"))

    def test_or(self):
        ast = parse_query("a OR b")
        assert ast == BinOp("OR", Term("a"), Term("b"))

    def test_andnot(self):
        ast = parse_query("a ANDNOT b")
        assert ast == BinOp("ANDNOT", Term("a"), Term("b"))

    def test_precedence_and_binds_tighter_than_or(self):
        # a OR b AND c  →  a OR (b AND c)
        ast = parse_query("a OR b AND c")
        assert ast == BinOp("OR", Term("a"), BinOp("AND", Term("b"), Term("c")))

    def test_parens_override_precedence(self):
        # (a OR b) AND c  →  (a OR b) AND c
        ast = parse_query("(a OR b) AND c")
        assert ast == BinOp("AND", BinOp("OR", Term("a"), Term("b")), Term("c"))

    def test_nested_parens(self):
        ast = parse_query("((a))")
        assert ast == Term("a")

    def test_complex_expression(self):
        ast = parse_query('("quantum computing" OR "quantum algorithm") AND "error correction" ANDNOT classical')
        expected = BinOp(
            "ANDNOT",
            BinOp(
                "AND",
                BinOp(
                    "OR",
                    Term("quantum computing", is_phrase=True),
                    Term("quantum algorithm", is_phrase=True),
                ),
                Term("error correction", is_phrase=True),
            ),
            Term("classical"),
        )
        assert ast == expected

    def test_left_associativity(self):
        # a AND b AND c  →  (a AND b) AND c
        ast = parse_query("a AND b AND c")
        assert ast == BinOp("AND", BinOp("AND", Term("a"), Term("b")), Term("c"))

    def test_empty_query(self):
        with pytest.raises(ValueError, match="Empty query"):
            parse_query("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="Empty query"):
            parse_query("   \n\n  ")

    def test_missing_closing_paren(self):
        with pytest.raises(ValueError, match="closing parenthesis"):
            parse_query("(a AND b")

    def test_unexpected_token(self):
        with pytest.raises(ValueError, match="Unexpected"):
            parse_query("AND a")

    def test_multiline_query(self):
        """Query spread across lines, as would come from a file."""
        text = """
        ("large language model" OR LLM)
        AND
        "artificial intelligence"
        """
        ast = parse_query(text)
        expected = BinOp(
            "AND",
            BinOp("OR", Term("large language model", is_phrase=True), Term("LLM")),
            Term("artificial intelligence", is_phrase=True),
        )
        assert ast == expected


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

class TestNodeToQuery:
    def test_bare_word(self):
        q = node_to_query(Term("quantum"))
        assert q == "(ti:quantum OR abs:quantum)"

    def test_phrase(self):
        q = node_to_query(Term("quantum computing", is_phrase=True))
        assert q == '(ti:"quantum computing" OR abs:"quantum computing")'

    def test_and(self):
        tree = BinOp("AND", Term("a"), Term("b"))
        q = node_to_query(tree)
        assert q == "((ti:a OR abs:a) AND (ti:b OR abs:b))"

    def test_or(self):
        tree = BinOp("OR", Term("a"), Term("b"))
        q = node_to_query(tree)
        assert q == "((ti:a OR abs:a) OR (ti:b OR abs:b))"

    def test_andnot(self):
        tree = BinOp("ANDNOT", Term("a"), Term("b"))
        q = node_to_query(tree)
        assert q == "((ti:a OR abs:a) ANDNOT (ti:b OR abs:b))"

    def test_complex(self):
        # ("quantum computing" OR QC) AND "error correction"
        tree = BinOp(
            "AND",
            BinOp(
                "OR",
                Term("quantum computing", is_phrase=True),
                Term("QC"),
            ),
            Term("error correction", is_phrase=True),
        )
        q = node_to_query(tree)
        expected = (
            '(((ti:"quantum computing" OR abs:"quantum computing") '
            "OR (ti:QC OR abs:QC)) "
            'AND (ti:"error correction" OR abs:"error correction"))'
        )
        assert q == expected
