"""Decode HTML character entities: `&amp;`, `&eacute;`, `&#8217;`, `&#x2019;` and the rest.

The table is the standard library's `html.entities.html5`, all 2125 names; what lives here is the rule for
turning one entity into text, which is where decoding can go wrong. `raven.papers.bibtex` builds its
BibTeX-escaping decoder on `resolve`, and `decode` is the plain-text form.
"""

__all__ = ["ENTITY_PATTERN", "resolve", "decode"]

import html.entities
import re
import unicodedata

# An HTML character entity; group 1 is its name, or `#` and a decimal or hex code.
#
# The name is bounded and must carry its semicolon. HTML5 also defines about sixty entities *without* one
# — `&copy`, `&sect`, `&times`, `&not` — and honouring those would decode any `&` that happens to be
# followed by one of those words, including where it is the start of a longer word: `see the &copyright
# notice` would read `see the ©right notice`, and `&section 5` would read `§ion 5`. With the semicolon
# required there is nothing to guess about.
ENTITY_PATTERN = re.compile(r"&(\#\d{1,7}|\#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")

# Unicode categories whose characters must not be written out as themselves. HTML5 names a good number of
# these — `&zwj;`, `&lrm;`, `&NoBreak;`, `&#10;`.
#
#   - **Format characters (Cf)** are dropped. A zero-width joiner or a directional mark carries no
#     information a bibliography record needs, and leaves nothing on screen to explain the text.
#   - **Controls and the line and paragraph separators (Cc, Zl, Zp)** become an ordinary space, and that
#     is a correctness rule rather than a tidiness one: a newline arriving mid-record moves every line
#     after it, and `raven.papers.fixbib` reports faults by line number in the user's own file.
#
# **Space separators (Zs) are the caller's choice**, via `fold_spaces`. In a file they are kept as
# themselves: a no-break space looks like a space, but not breaking the line there is the entire point of
# the character, and the source asked for it. Text headed for analysis wants them folded, because a
# tokenizer *should* see a word boundary there.
_DROPPED_CATEGORIES = frozenset(["Cf"])
_SPACED_CATEGORIES = frozenset(["Cc", "Zl", "Zp"])


def resolve(name: str, fold_spaces: bool = False) -> str | None:
    """The text the entity `name` stands for, or `None` if it names nothing.

    `name`: what `ENTITY_PATTERN` captures — `amp`, `eacute`, `#8217`, `#x2019` — without the `&` and `;`.
    `fold_spaces`: if `True`, a space separator such as a no-break space comes out as an ordinary space.

    A format character comes out as `""`, and a control or a line or paragraph separator as `" "`.
    """
    if name.startswith("#"):
        try:
            code = int(name[2:], 16) if name[1:2].lower() == "x" else int(name[1:])
        except ValueError:
            return None
        if not 0 < code < 0x110000:
            return None
        character = chr(code)
    else:
        character = html.entities.html5.get(name + ";")
        if character is None:
            return None

    # Per character, because ninety-odd names stand for two code points — mostly a symbol and a
    # combining mark (`&NotEqualTilde;`), but `&ThickSpace;` is two spaces.
    return "".join(_as_text(c, fold_spaces) for c in character)


def _as_text(character: str, fold_spaces: bool) -> str:
    """One code point as it should be written out, by the category rules above."""
    category = unicodedata.category(character)
    if category in _DROPPED_CATEGORIES:
        return ""
    if category in _SPACED_CATEGORIES or (fold_spaces and category == "Zs"):
        return " "
    return character


def decode(text: str, fold_spaces: bool = True, spare: str = "") -> str:
    """Decode the HTML character entities in the plain text `text`.

    `fold_spaces`: as in `resolve`, and on by default, plain text being mostly read by something that
                   splits it into words.
    `spare`: characters not to decode into. An entity standing for any of them is left as it is, for a
             later pass — `spare="<>&"` decodes everything that cannot turn into markup or into another
             entity.

    One pass, so `&amp;lt;` becomes the literal `&lt;` its author wrote rather than `<`. An entity naming
    nothing — a stray `&foo;` — is left as it is.
    """
    def replace(match: re.Match) -> str:
        maybe_text = resolve(match.group(1), fold_spaces=fold_spaces)
        if maybe_text is None or any(c in spare for c in maybe_text):
            return match.group(0)
        return maybe_text
    return ENTITY_PATTERN.sub(replace, text)
