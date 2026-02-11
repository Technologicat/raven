"""XDot format parser.

This module parses DOT graph files in xdot format (GraphViz output with
xdot drawing attributes) and produces Graph objects for visualization.

Adapted from xdottir (https://github.com/Technologicat/xdottir).
"""

__all__ = ["parse_xdot", "XDotParser", "ParseError"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import colorsys
import re
import sys
from typing import Dict, List, Optional, Tuple

from .constants import Color, Point, X11_COLORS, BREWER_COLORS
from .graph import (
    Pen, Shape, TextShape, EllipseShape, PolygonShape, LineShape, BezierShape,
    Node, Edge, Graph
)


# --------------------------------------------------------------------------------
# Utilities

class ParseError(Exception):
    """Error during DOT parsing."""

    def __init__(self, msg: Optional[str] = None, filename: Optional[str] = None,
                 line: Optional[int] = None, col: Optional[int] = None):
        self.msg = msg
        self.filename = filename
        self.line = line
        self.col = col

    def __str__(self) -> str:
        parts = [str(p) for p in (self.filename, self.line, self.col, self.msg) if p is not None]
        return ":".join(parts)


# --------------------------------------------------------------------------------
# Generic scanner/lexer/parser.

# Token types.
# Only these two are generic; input-data-specific ones are defined after the generic classes.
EOF = -1
SKIP = -2

class Token:
    """A lexical token."""

    def __init__(self, type_: int, text: str, line: int, col: int):
        self.type = type_
        self.text = text
        self.line = line
        self.col = col


class Scanner:
    """Stateless lexical scanner."""

    tokens: List[Tuple[int, str, bool]] = []  # language spec for how to tokenize; set in derived class; [(type_, regex, do_test_for_literal), ...]
    symbols: Dict[str, int] = {}  # language spec for operators, parens, etc.; set in derived class; {literal_text: type_, ...}
    literals: Dict[str, int] = {}  # language spec for reserved words; set in derived class; {literal_text: type_, ...}
    ignorecase: bool = False

    def __init__(self):
        flags = re.DOTALL
        if self.ignorecase:
            flags |= re.IGNORECASE
        self.tokens_re = re.compile(
            "|".join([f"({regexp})" for type_, regexp, test_lit in self.tokens]),
            flags
        )

    def next(self, buf: str, pos: int) -> Tuple[int, str, int]:
        """Return value is (token_type, text, end_pos)."""
        if pos >= len(buf):
            return EOF, "", pos
        mo = self.tokens_re.match(buf, pos)
        if mo:
            text = mo.group()
            type_, regexp, test_lit = self.tokens[mo.lastindex - 1]
            pos = mo.end()
            if test_lit:
                type_ = self.literals.get(text, type_)
            return type_, text, pos
        else:
            c = buf[pos]
            return self.symbols.get(c, None), c, pos + 1


class Lexer:
    """Stateful lexer that produces tokens from input."""

    scanner: Optional[Scanner] = None
    tabsize: int = 8  # ancient *nix default; keeping it because this is for dot/xdot, old formats with a strong *nix tradition.
    newline_re = re.compile(r"\r\n?|\n")  # TODO: Is there any difference in performance to the flattened r"\n|\r\n|\r"?

    def __init__(self, buf: str = "", pos: int = 0, filename: Optional[str] = None):
        """
        buf: the data to lex.
        pos: offset (in characters, i.e. Unicode codepoints) in `buf` where to start lexing. Default is to lex the whole buffer.
        filename: used in error messages.
        """
        self.buf = buf
        self.pos = pos
        self.line = 1
        self.col = 1
        self.filename = filename

    def next(self) -> Token:
        while True:
            pos = self.pos
            line = self.line
            col = self.col

            type_, text, endpos = self.scanner.next(self.buf, pos)
            assert pos + len(text) == endpos
            self._consume(text)
            type_, text = self._filter(type_, text)
            self.pos = endpos

            if type_ == SKIP:
                continue
            elif type_ is None:
                msg = "unexpected char "
                if " " <= text <= "~":  # printable ASCII character? (0x20 - 0x7E)
                    msg += f"'{text}'"
                else:
                    msg += f"0x{ord(text):X}"
                raise ParseError(msg, self.filename, line, col)
            else:
                break
        return Token(type_=type_, text=text, line=line, col=col)

    def _consume(self, text: str) -> None:
        """Update line/column tracking."""
        pos = 0
        for mo in self.newline_re.finditer(text, pos):
            self.line += 1
            self.col = 1
            pos = mo.end()

        while True:
            tabpos = text.find("\t", pos)
            if tabpos == -1:
                break
            self.col += tabpos - pos
            # Round up to next tab stop (here the role of the middle "+1" is to advance to the next stop)
            self.col = ((self.col - 1) // self.tabsize + 1) * self.tabsize + 1
            pos = tabpos + 1
        self.col += len(text) - pos

    def _filter(self, type_: int, text: str) -> Tuple[int, str]:
        """Filter/transform tokens (override in subclass).

        Return value must be (new_type_, new_text).
        Default is passthrough (no filtering).

        Here `text` is a token. If you want to skip it,
        set `new_type = SKIP` in your return value.
        """
        return type_, text


class Parser:
    """Base recursive-descent parser."""

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.lookahead = self.lexer.next()

    def match(self, type_: int) -> None:
        """Match current lookahead token against `type_`. If no match, raise `ParseError`."""
        if self.lookahead.type != type_:
            raise ParseError(
                msg=f"unexpected token {self.lookahead.text!r}",
                filename=self.lexer.filename,
                line=self.lookahead.line,
                col=self.lookahead.col
            )

    def skip(self, type_: int) -> None:
        """Skip tokens until a token of `type_` is encountered."""
        while self.lookahead.type != type_:
            self.consume()

    def consume(self) -> Token:
        """Consume current token, advance the lexer by one token, and make that the new lookahead token."""
        token = self.lookahead
        self.lookahead = self.lexer.next()
        return token


# --------------------------------------------------------------------------------
# The DOT (Graphviz) scanner/lexer/parser, building on the generic ones.

# Token types specific to dot/xdot.
ID = 0
STR_ID = 1
HTML_ID = 2
EDGE_OP = 3

LSQUARE = 4
RSQUARE = 5
LCURLY = 6
RCURLY = 7
COMMA = 8
COLON = 9
SEMI = 10
EQUAL = 11
PLUS = 12

STRICT = 13
GRAPH = 14
DIGRAPH = 15
NODE = 16
EDGE = 17
SUBGRAPH = 18

class DotScanner(Scanner):
    """Scanner for DOT (Graphviz) graph format."""

    tokens = [
        # Whitespace and comments
        (SKIP,
         r"[ \t\f\r\n\v]+|"
         r"//[^\r\n]*|"
         r"/\*.*?\*/|"
         r"#[^\r\n]*",
         False),

        # Alphanumeric IDs
        (ID, r"[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*", True),

        # Numeric IDs
        (ID, r"-?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)", False),

        # String IDs
        (STR_ID, r'"[^"\\]*(?:\\.[^"\\]*)*"', False),

        # HTML IDs
        (HTML_ID, r"<[^<>]*(?:<[^<>]*>[^<>]*)*>", False),

        # Edge operators
        (EDGE_OP, r"-[>-]", False),
    ]

    symbols = {
        "[": LSQUARE,
        "]": RSQUARE,
        "{": LCURLY,
        "}": RCURLY,
        ",": COMMA,
        ":": COLON,
        ";": SEMI,
        "=": EQUAL,
        "+": PLUS,
    }

    literals = {
        "strict": STRICT,
        "graph": GRAPH,
        "digraph": DIGRAPH,
        "node": NODE,
        "edge": EDGE,
        "subgraph": SUBGRAPH,
    }

    ignorecase = True


class DotLexer(Lexer):
    """Lexer for DOT (Graphviz) graph format."""

    scanner = DotScanner()

    def _filter(self, type_: int, text: str) -> Tuple[int, str]:
        """Filter STR_ID and HTML_ID tokens, e.g. by stripping the quotes."""
        if type_ == STR_ID:
            text = text[1:-1]  # Strip quotes

            # Line continuations
            text = text.replace("\\\r\n", "")
            text = text.replace("\\\r", "")
            text = text.replace("\\\n", "")

            # Escaped quotes
            text = text.replace('\\"', '"')

            type_ = ID

        elif type_ == HTML_ID:
            text = text[1:-1]  # Strip angle brackets
            type_ = ID

        return type_, text


class DotParser(Parser):
    """Parser for DOT (Graphviz) graph format."""

    def __init__(self, lexer: Lexer):
        super().__init__(lexer)
        self.graph_attrs: Dict[str, str] = {}
        self.node_attrs: Dict[str, str] = {}
        self.edge_attrs: Dict[str, str] = {}

    def parse(self) -> None:
        self.parse_graph()
        self.match(EOF)

    def parse_graph(self) -> None:
        if self.lookahead.type == STRICT:
            self.consume()
        self.skip(LCURLY)
        self.consume()
        while self.lookahead.type != RCURLY:
            self.parse_stmt()
        self.consume()

    def parse_subgraph(self) -> Optional[str]:
        id_ = None
        if self.lookahead.type == SUBGRAPH:
            self.consume()
            if self.lookahead.type == ID:
                id_ = self.lookahead.text
                self.consume()
        if self.lookahead.type == LCURLY:
            self.consume()
            while self.lookahead.type != RCURLY:
                self.parse_stmt()
            self.consume()
        return id_

    def parse_stmt(self) -> None:
        if self.lookahead.type == GRAPH:
            self.consume()
            attrs = self.parse_attrs()
            self.graph_attrs.update(attrs)
            self.handle_graph(attrs)
        elif self.lookahead.type == NODE:
            self.consume()
            self.node_attrs.update(self.parse_attrs())
        elif self.lookahead.type == EDGE:
            self.consume()
            self.edge_attrs.update(self.parse_attrs())
        elif self.lookahead.type in (SUBGRAPH, LCURLY):
            self.parse_subgraph()
        else:
            id_ = self.parse_node_id()
            if self.lookahead.type == EDGE_OP:
                self.consume()
                node_ids = [id_, self.parse_node_id()]
                while self.lookahead.type == EDGE_OP:
                    self.consume()
                    node_ids.append(self.parse_node_id())
                attrs = self.parse_attrs()
                for i in range(len(node_ids) - 1):
                    self.handle_edge(node_ids[i], node_ids[i + 1], attrs)
            elif self.lookahead.type == EQUAL:
                self.consume()
                self.parse_id()
            else:
                attrs = self.parse_attrs()
                self.handle_node(id_, attrs)
        if self.lookahead.type == SEMI:
            self.consume()

    def parse_attrs(self) -> Dict[str, str]:
        attrs = {}
        while self.lookahead.type == LSQUARE:
            self.consume()
            while self.lookahead.type != RSQUARE:
                name, value = self.parse_attr()
                attrs[name] = value
                if self.lookahead.type == COMMA:
                    self.consume()
            self.consume()
        return attrs

    def parse_attr(self) -> Tuple[str, str]:
        name = self.parse_id()
        if self.lookahead.type == EQUAL:
            self.consume()
            value = self.parse_id()
        else:
            value = "true"
        return name, value

    def parse_node_id(self) -> str:
        node_id = self.parse_id()
        if self.lookahead.type == COLON:
            self.consume()
            self.parse_id()  # port
            if self.lookahead.type == COLON:
                self.consume()
                self.parse_id()  # compass_pt
        return node_id

    def parse_id(self) -> str:
        self.match(ID)
        id_ = self.lookahead.text
        self.consume()
        return id_

    def handle_graph(self, attrs: Dict[str, str]) -> None:
        """Override in subclass."""
        pass

    def handle_node(self, id_: str, attrs: Dict[str, str]) -> None:
        """Override in subclass."""
        pass

    def handle_edge(self, src_id: str, dst_id: str, attrs: Dict[str, str]) -> None:
        """Override in subclass."""
        pass


# --------------------------------------------------------------------------------
# The xdot (DOT with xdot drawing attributes) parser.

class XDotAttrParser:
    """Parser for xdot drawing attributes."""

    def __init__(self, parent_parser: "XDotParser", buf: str):
        self.parser = parent_parser
        self.buf = buf
        self.pos: int = 0
        self.pen: Pen = Pen()
        self.shapes: List[Shape] = []

    def __bool__(self) -> bool:
        return self.pos < len(self.buf)

    def read_code(self) -> str:
        pos = self.buf.find(" ", self.pos)
        if pos == -1:
            pos = len(self.buf)
        res = self.buf[self.pos:pos]
        self.pos = pos + 1
        while self.pos < len(self.buf) and self.buf[self.pos].isspace():
            self.pos += 1
        return res

    def read_number(self) -> int:
        return int(self.read_float())

    def read_float(self) -> float:
        return float(self.read_code())

    def read_point(self) -> Point:
        x = self.read_number()
        y = self.read_number()
        return self.transform(x, y)

    def read_text(self) -> str:
        num = self.read_number()
        pos = self.buf.find("-", self.pos) + 1
        self.pos = pos + num
        res = self.buf[pos:self.pos]
        while self.pos < len(self.buf) and self.buf[self.pos].isspace():
            self.pos += 1
        return res

    def read_polygon(self) -> List[Point]:
        n = self.read_number()
        points = []
        for _ in range(n):
            x, y = self.read_point()
            points.append((x, y))
        return points

    def read_color(self) -> Optional[Color]:
        c = self.read_text()
        if not c:
            return None

        c1 = c[:1]
        if c1 == "#":
            # Hex format: #RRGGBB or #RRGGBBAA
            hex2float = lambda h: float(int(h, 16) / 255.0)
            try:
                r = hex2float(c[1:3])
                g = hex2float(c[3:5])
                b = hex2float(c[5:7])
                try:
                    a = hex2float(c[7:9])
                except (IndexError, ValueError):
                    a = 1.0
                return r, g, b, a
            except (ValueError, IndexError):
                return None
        elif c1.isdigit() or c1 == ".":
            # HSV format: "H,S,V" or "H S V"
            try:
                h, s, v = map(float, c.replace(",", " ").split())
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                return r, g, b, 1.0
            except ValueError:
                return None
        else:
            return self._lookup_color(c)

    def _lookup_color(self, c: str) -> Optional[Color]:
        """Look up a named color.

        X11 and ColorBrewer color names are supported.
        In case of name conflict, X11 takes precedence.
        """
        # Try X11/CSS color names
        color = X11_COLORS.get(c.lower())
        if color is not None:
            return color

        # Try ColorBrewer format: /scheme/index
        try:
            dummy, scheme, index = c.split("/")
            r, g, b = BREWER_COLORS[scheme][int(index)]
            s = 1.0 / 255.0
            return r * s, g * s, b * s, 1.0
        except (ValueError, KeyError, IndexError):
            pass

        # Unknown color
        logger.warning(f"xdot parser: unknown color '{c}'")
        return None

    def parse(self) -> List[Shape]:
        while self:
            op = self.read_code()
            if op == "c":
                color = self.read_color()
                if color is not None:
                    self.pen.color = color
            elif op == "C":
                color = self.read_color()
                if color is not None:
                    self.pen.fillcolor = color
            elif op == "S":
                style = self.read_text()
                if style.startswith("setlinewidth("):
                    lw = style.split("(")[1].split(")")[0]
                    self.pen.linewidth = float(lw)
                elif style == "solid":
                    self.pen.dash = ()
                elif style == "dashed":
                    self.pen.dash = (6,)
                elif style == "dotted":
                    self.pen.dash = (2, 4)
            elif op == "F":
                size = self.read_float()
                self.pen.fontsize = size
                # Font family - skip for now.
                #
                # DPG requires registering font families from a TTF file with `dpg.font`.
                # Raven currently does that in `raven.common.gui.utils`, function `bootup`.
                # So we shouldn't load any fonts here. Hence we can change the size, but
                # not the font family.
                #
                # TODO: implement font loading later?
                logger.warning("XDotAttrParser.parse: font families are currently not supported; setting font size only.")
                # name = self.read_text()
                # self.pen.fontname = name
            elif op == "T":
                x, y = self.read_point()
                j = self.read_number()
                w = self.read_number()
                t = self.read_text()
                self.shapes.append(TextShape(self.pen, x, y, j, w, t))
            elif op == "E":
                x0, y0 = self.read_point()
                w = self.read_number()
                h = self.read_number()
                # Filled ellipse: draw fill first, then outline
                self.shapes.append(EllipseShape(self.pen, x0, y0, w, h, filled=True))
                self.shapes.append(EllipseShape(self.pen, x0, y0, w, h, filled=False))
            elif op == "e":
                x0, y0 = self.read_point()
                w = self.read_number()
                h = self.read_number()
                self.shapes.append(EllipseShape(self.pen, x0, y0, w, h, filled=False))
            elif op == "L":
                points = self.read_polygon()
                self.shapes.append(LineShape(self.pen, points))
            elif op == "B":
                points = self.read_polygon()
                self.shapes.append(BezierShape(self.pen, points, filled=False))
            elif op == "b":
                points = self.read_polygon()
                self.shapes.append(BezierShape(self.pen, points, filled=True))
                self.shapes.append(BezierShape(self.pen, points, filled=False))
            elif op == "P":
                points = self.read_polygon()
                self.shapes.append(PolygonShape(self.pen, points, filled=True))
                self.shapes.append(PolygonShape(self.pen, points, filled=False))
            elif op == "p":
                points = self.read_polygon()
                self.shapes.append(PolygonShape(self.pen, points, filled=False))
            elif op == "I":
                # Image shapes - skip for now (DPG doesn't have direct image support in drawlists)
                # TODO: Implement image loading for DPG
                logger.warning("XDotAttrParser.parse: image shapes are currently not supported; skipping.")
                # x0, y0 = self.read_point()
                # w = self.read_number()
                # h = self.read_number()
                # path = self.read_text()
            else:
                if op:
                    print(f"xdot parser: unknown opcode '{op}'", file=sys.stderr)
                break

        return self.shapes

    def transform(self, x: float, y: float) -> Point:
        return self.parser.transform(x, y)


class XDotParser(DotParser):
    """Parser for xdot format (DOT with xdot drawing attributes)."""

    def __init__(self, xdotcode: str):
        lexer = DotLexer(buf=xdotcode)
        super().__init__(lexer)

        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.shapes: List = []
        self.node_by_name: Dict[str, Node] = {}
        self.top_graph: bool = True

        # Transform parameters
        self.xoffset: float = 0.0
        self.yoffset: float = 0.0
        self.xscale: float = 1.0
        self.yscale: float = -1.0  # Flip Y axis (GraphViz uses Y-up)
        self.width: float = 1.0
        self.height: float = 1.0

    def handle_graph(self, attrs: Dict[str, str]) -> None:
        if self.top_graph:
            try:  # bounding box
                bb = attrs["bb"]
            except KeyError:
                return
            if not bb:
                return

            xmin, ymin, xmax, ymax = map(float, bb.split(","))

            self.xoffset = -xmin
            self.yoffset = -ymax
            self.xscale = 1.0
            self.yscale = -1.0

            self.width = max(xmax - xmin, 1)
            self.height = max(ymax - ymin, 1)

            self.top_graph = False

        # Parse drawing attributes
        for attr in ("_draw_", "_ldraw_", "_hdraw_", "_tdraw_", "_hldraw_", "_tldraw_"):
            if attr in attrs:
                parser = XDotAttrParser(self, attrs[attr])
                self.shapes.extend(parser.parse())

    def handle_node(self, id_: str, attrs: Dict[str, str]) -> None:
        try:
            pos = attrs["pos"]
        except KeyError:
            return

        x, y = self._parse_node_pos(pos)
        w = float(attrs.get("width", 0)) * 72  # Convert inches to points
        h = float(attrs.get("height", 0)) * 72

        shapes = []
        for attr in ("_draw_", "_ldraw_"):
            if attr in attrs:
                parser = XDotAttrParser(self, attrs[attr])
                shapes.extend(parser.parse())

        url = attrs.get("URL", None)
        node = Node(x, y, w, h, shapes, url, internal_name=id_)
        self.node_by_name[id_] = node
        if shapes:
            self.nodes.append(node)

    def handle_edge(self, src_id: str, dst_id: str, attrs: Dict[str, str]) -> None:
        try:
            pos = attrs["pos"]
        except KeyError:
            return

        points = self._parse_edge_pos(pos)
        shapes = []
        for attr in ("_draw_", "_ldraw_", "_hdraw_", "_tdraw_", "_hldraw_", "_tldraw_"):
            if attr in attrs:
                parser = XDotAttrParser(self, attrs[attr])
                shapes.extend(parser.parse())

        if shapes:
            src = self.node_by_name.get(src_id)
            dst = self.node_by_name.get(dst_id)
            if src is not None and dst is not None:
                self.edges.append(Edge(src, dst, points, shapes))

    def parse(self) -> Graph:
        super().parse()
        return Graph(self.width, self.height, self.shapes, self.nodes, self.edges)

    def _parse_node_pos(self, pos: str) -> Point:
        x, y = pos.split(",")
        return self.transform(float(x), float(y))

    def _parse_edge_pos(self, pos: str) -> List[Point]:
        points = []
        for entry in pos.split(" "):
            fields = entry.split(",")
            try:
                x, y = fields
            except ValueError:
                # Skip start/end markers (e, s prefixes)
                continue
            else:
                points.append(self.transform(float(x), float(y)))
        return points

    def transform(self, x: float, y: float) -> Point:
        x = (x + self.xoffset) * self.xscale
        y = (y + self.yoffset) * self.yscale
        return x, y


# --------------------------------------------------------------------------------
# Main entry point

def parse_xdot(xdotcode: str) -> Graph:
    """Parse xdot format code and return a Graph object.

    `xdotcode`: The xdot-format graph as a string.
                This should be the output of GraphViz with xdot output format.

    Returns a Graph object containing all nodes, edges, and shapes.
    """
    parser = XDotParser(xdotcode)
    return parser.parse()
