"""Boolean expression parser for arXiv search queries.

Supports AND, OR, ANDNOT operators, parenthesized grouping,
quoted phrases, and bare words.

Precedence (lowest to highest): OR, AND/ANDNOT, atom.

The query builder converts a parsed AST into an arXiv API ``search_query``
string, expanding each leaf term to search both title and abstract.
"""

from __future__ import annotations

__all__ = [
    "Term",
    "BinOp",
    "Node",
    "TT",
    "Token",
    "tokenize",
    "parse_query",
    "node_to_query",
]

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Term:
    """Leaf node — a single word or a quoted phrase."""
    value: str
    is_phrase: bool = False


@dataclass
class BinOp:
    """Binary boolean operator node."""
    op: str          # "AND", "OR", "ANDNOT"
    left: Node
    right: Node


Node = Union[Term, BinOp]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TT(Enum):
    """Token types."""
    LPAREN = auto()
    RPAREN = auto()
    AND = auto()
    OR = auto()
    ANDNOT = auto()
    WORD = auto()
    PHRASE = auto()


@dataclass
class Token:
    type: TT
    value: str


# Keyword patterns, longest first so ANDNOT is tried before AND.
_KEYWORDS = [
    ("ANDNOT", TT.ANDNOT),
    ("AND", TT.AND),
    ("OR", TT.OR),
]


def tokenize(text: str) -> list[Token]:
    """Tokenize a boolean search expression into a list of `Token` objects."""
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        # Whitespace
        if ch.isspace():
            i += 1
            continue

        # Parentheses
        if ch == "(":
            tokens.append(Token(TT.LPAREN, "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TT.RPAREN, ")"))
            i += 1
            continue

        # Quoted phrase
        if ch == '"':
            end = text.find('"', i + 1)
            if end == -1:
                raise ValueError(f"Unterminated quoted phrase starting at position {i}")
            tokens.append(Token(TT.PHRASE, text[i + 1:end]))
            i = end + 1
            continue

        # Try keywords (case-insensitive, must be followed by non-alnum)
        matched = False
        for kw, tt in _KEYWORDS:
            kw_len = len(kw)
            if text[i:i + kw_len].upper() == kw:
                after = i + kw_len
                if after >= n or not text[after].isalnum():
                    tokens.append(Token(tt, kw))
                    i = after
                    matched = True
                    break
        if matched:
            continue

        # Bare word — runs until whitespace or parenthesis
        j = i
        while j < n and not text[j].isspace() and text[j] not in '()"':
            j += 1
        tokens.append(Token(TT.WORD, text[i:j]))
        i = j

    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self._or_expr()
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            raise ValueError(f"Unexpected token at position {self.pos}: {tok.value!r}")
        return node

    def _or_expr(self) -> Node:
        left = self._and_expr()
        while (tok := self.peek()) and tok.type == TT.OR:
            self.consume()
            right = self._and_expr()
            left = BinOp("OR", left, right)
        return left

    def _and_expr(self) -> Node:
        left = self._atom()
        while (tok := self.peek()) and tok.type in (TT.AND, TT.ANDNOT):
            op_tok = self.consume()
            right = self._atom()
            left = BinOp(op_tok.value, left, right)
        return left

    def _atom(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if tok.type == TT.LPAREN:
            self.consume()
            node = self._or_expr()
            closing = self.peek()
            if closing is None or closing.type != TT.RPAREN:
                raise ValueError("Expected closing parenthesis")
            self.consume()
            return node
        if tok.type == TT.PHRASE:
            self.consume()
            return Term(tok.value, is_phrase=True)
        if tok.type == TT.WORD:
            self.consume()
            return Term(tok.value, is_phrase=False)
        raise ValueError(f"Unexpected token: {tok.value!r}")


def parse_query(text: str) -> Node:
    """Parse a boolean search expression into an AST."""
    tokens = tokenize(text.strip())
    if not tokens:
        raise ValueError("Empty query")
    return _Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Query builder — AST → arXiv API search_query string
# ---------------------------------------------------------------------------

def node_to_query(node: Node) -> str:
    """Convert a parsed AST to an arXiv ``search_query`` string.

    Each leaf term is expanded to search both title and abstract:
    ``term`` becomes ``(ti:term OR abs:term)``.
    """
    if isinstance(node, Term):
        if node.is_phrase or " " in node.value:
            escaped = f'"{node.value}"'
        else:
            escaped = node.value
        return f"(ti:{escaped} OR abs:{escaped})"

    if isinstance(node, BinOp):
        left = node_to_query(node.left)
        right = node_to_query(node.right)
        return f"({left} {node.op} {right})"

    raise TypeError(f"Unexpected node type: {type(node)}")
