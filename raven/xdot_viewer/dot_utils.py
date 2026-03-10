"""Utilities for manipulating DOT/xdot source text.

Pure-text operations that don't need the GUI stack.
"""

from ..common.gui.xdotwidget.parser import (DotScanner, ID, STR_ID, HTML_ID,
                                             LSQUARE, RSQUARE, EQUAL, COMMA, SEMI, EOF, SKIP)

# Layout attributes to strip from xdot code before re-layout.
# These are all attributes that GraphViz layout engines produce;
# stripping them gives clean DOT that any engine can process from scratch.
_XDOT_LAYOUT_ATTRS = frozenset({
    "bb", "pos", "lp", "rects", "_background",
    "_draw_", "_ldraw_", "_hdraw_", "_tdraw_", "_hldraw_", "_tldraw_",
})

_dot_scanner = DotScanner()


def _strip_xdot_layout_attrs(xdotcode: str) -> str:
    """Remove layout-related attributes from xdot/dot source.

    Uses the existing DotScanner to correctly handle quoted strings,
    HTML labels, and comments. Works from the original buffer (not
    reconstructed token text) to preserve quoting and formatting.
    """
    buf = xdotcode
    pos = 0
    bracket_depth = 0

    # Byte ranges to delete: list of (start, end) tuples.
    deletions = []

    # State machine for tracking `name = value` inside attribute lists.
    # States: None (idle), "saw_name" (name token matched a layout attr),
    #         "saw_equal" (saw `=` after a matching name).
    attr_state = None
    attr_start = 0  # start position of the name token

    while True:
        tok_type, tok_text, end_pos = _dot_scanner.next(buf, pos)
        tok_start = end_pos - len(tok_text)

        if tok_type == EOF:
            break

        if tok_type == SKIP:
            pos = end_pos
            continue

        if tok_type == LSQUARE:
            bracket_depth += 1
            attr_state = None
            pos = end_pos
            continue

        if tok_type == RSQUARE:
            bracket_depth = max(0, bracket_depth - 1)
            attr_state = None
            pos = end_pos
            continue

        if bracket_depth > 0:
            # Inside an attribute list — run the name=value state machine.
            if attr_state is None:
                # Expecting an attribute name.
                if tok_type in (ID, STR_ID, HTML_ID):
                    # Strip quotes/brackets for name comparison (scanner
                    # returns raw text, unlike the lexer's _filter).
                    name = tok_text
                    if tok_type == STR_ID and len(name) >= 2:
                        name = name[1:-1]
                    elif tok_type == HTML_ID and len(name) >= 2:
                        name = name[1:-1]
                    if name.lower() in _XDOT_LAYOUT_ATTRS:
                        attr_state = "saw_name"
                        attr_start = tok_start
                    # else: not a layout attr, ignore
            elif attr_state == "saw_name":
                if tok_type == EQUAL:
                    attr_state = "saw_equal"
                else:
                    # Not a `name = value` pair (e.g. bare attribute flag).
                    attr_state = None
            elif attr_state == "saw_equal":
                # This token is the value — mark the whole name=value for deletion.
                delete_end = end_pos
                # Also consume any trailing separator (comma, semicolon, whitespace).
                probe_pos = delete_end
                while probe_pos < len(buf) and buf[probe_pos] in " \t\r\n":
                    probe_pos += 1
                if probe_pos < len(buf) and buf[probe_pos] in ",;":
                    probe_pos += 1
                    # Eat whitespace after the separator too.
                    while probe_pos < len(buf) and buf[probe_pos] in " \t\r\n":
                        probe_pos += 1
                delete_end = probe_pos
                # Also consume leading whitespace before the attribute name.
                while attr_start > 0 and buf[attr_start - 1] in " \t":
                    attr_start -= 1
                deletions.append((attr_start, delete_end))
                attr_state = None
        else:
            attr_state = None

        pos = end_pos

    if not deletions:
        return buf

    # Build output by copying everything except deletion ranges.
    parts = []
    prev = 0
    for start, end in deletions:
        if start > prev:
            parts.append(buf[prev:start])
        prev = end
    if prev < len(buf):
        parts.append(buf[prev:])
    return "".join(parts)
