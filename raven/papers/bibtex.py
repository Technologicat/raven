"""Read and write BibTeX.

Writing: `entries_to_bibtex` converts arXiv Atom feed entries, used by `raven.papers.search` to format
arXiv API search results.

Reading: `parse_file` and `parse_string` wrap `bibtexparser` with the one middleware chain Raven reads
BibTeX through, so every consumer gets the same normalization. A caller that will *write* the library back
out wants `split_names=False`, and `write_string` either way — see below.

Reading what a parser refused: `header_key`, `field_value`, `unbalanced_field_names` and
`brace_repair_candidates` work on the surface syntax, for saying something useful about a record
`bibtexparser` has already declined. They are not a second parser and must not become one.

Repairing: `repair_record`, `repair_duplicate_field_keys` and `repair_name_lists` each rescue one record
`bibtexparser` refused, for the three reasons a real-world file gives it — braces that do not balance, a
field name that occurs twice, and authors listed APA-style rather than joined with `and`. Both take the record's text and hand back text, so a caller may parse the result or
write it to a file; `raven.papers.fixbib` does the latter.

`decode_html_entities`, `relocate_rights_notices` and `recover_dois` are text-to-text as well, and fix what
is *inside* records a parser is perfectly happy with: HTML a database left in the field values, a
publisher's rights notice appended to an abstract that is metadata in the wrong field, and a DOI written
only into the citation key or the `url`.
"""

from __future__ import annotations

__all__ = ["parse_file", "parse_string", "write_string",

           "entry_field", "readable_text", "VENUE_FIELDS", "entry_venue",
           "DOI_PLACES", "find_doi", "entry_doi",

           "header_key", "field_value", "unbalanced_field_names", "brace_repair_candidates",

           "repair_record", "repair_duplicate_field_keys", "repair_name_lists", "repair_failed_block",
           "read_repeated_keys",
           "decode_html_entities",
           "relocate_rights_notices",
           "recover_dois",

           "entries_to_bibtex"]

import collections
import itertools
import pathlib
import re
from typing import NamedTuple

import bibtexparser
from bibtexparser.model import DuplicateBlockKeyBlock, DuplicateFieldKeyBlock, Entry, Field, MiddlewareErrorBlock
from bibtexparser import Library

from unpythonic import box, unbox

from ..common import utils as common_utils
from ..common.text import boilerplate
from ..common.text import entities

from . import identifiers
from . import utils as papers_utils
from .utils import bibtex_escape


# The fields `SeparateCoAuthors` and `SplitNameParts` transform, read off the middleware rather than
# written out, so that this follows upstream if the set ever changes. Currently
# `("author", "editor", "translator")`.
_NAME_FIELDS = bibtexparser.middlewares.SeparateCoAuthors().name_fields


def _reader_middleware(split_names: bool) -> list:
    """The middleware chain Raven reads BibTeX through, as a fresh list per call.

    `split_names`: whether to include the `SeparateCoAuthors` + `SplitNameParts` step (see below).

    The chain is:

      - `NormalizeFieldKeys` because the key case is not dependable - a Web of Science export writes
        `Title = {...}`, the BibTeX literature writes `title = {...}`.
      - `SeparateCoAuthors` then `SplitNameParts`, in that order. Between them they turn one `author`
        string into name parts that survive "Ludwig van Beethoven", "Petra Johanna Lagerkvist",
        "Holm Dahl, Aksel" and "Beeblebrox, IV, Zaphod" — a particle, two given names, a compound
        surname the comma is needed to mark, and a suffix. Omitted when `split_names` is false, leaving
        `author` and `editor` the strings the input data had.

    Fresh instances rather than a shared module-level list, because a middleware is free to carry
    per-parse state.
    """
    chain = [bibtexparser.middlewares.NormalizeFieldKeys()]
    if split_names:
        chain += [bibtexparser.middlewares.SeparateCoAuthors(),
                  bibtexparser.middlewares.SplitNameParts()]
    return chain


def parse_file(filename: str | pathlib.Path, split_names: bool = True) -> Library:
    """Parse the BibTeX file `filename`, returning a `bibtexparser` `Library`.

    `split_names`: whether to break `author` and `editor` into name parts. See `parse_string`.

    Raises whatever `bibtexparser` raises; a caller that wants to treat unparseable input as a normal
    outcome should catch it. Note a `Library` can also come back *partly* parsed - unreadable records
    land in `library.failed_blocks` rather than raising, so a successful return is not a promise that
    every record was understood.
    """
    return bibtexparser.parse_file(str(filename), append_middleware=_reader_middleware(split_names))


def parse_string(text: str, split_names: bool = True) -> Library:
    """Parse `text` as BibTeX, returning a `bibtexparser` `Library`.

    `split_names`: whether to break `author` and `editor` into name parts. True, the default, is what a
                   *consumer* wants: one `author` string becomes a list of `NameParts`, so a caller can
                   ask for the first author's surname without writing a name parser. False leaves both
                   fields as the strings the file had, which is what a caller that will write the library
                   back out wants — `raven.papers.deduplicate` is the one in-tree.

    `parse_file`, which see, for the error behaviour - it is the same.
    """
    return bibtexparser.parse_string(text, append_middleware=_reader_middleware(split_names))


def write_string(library: Library) -> str:
    """Serialize a `bibtexparser` `Library` back to BibTeX text.

    Use this rather than `bibtexparser.write_string`, whatever the library was read with: it undoes the
    name splitting when the library carries it and does nothing when it does not, so there is no pairing
    for a caller to get right.

    Field *values* survive the round trip byte for byte — inner `{LaTeX}` groups, `\\%` escapes, and
    non-ASCII alike. The *layout* does not, and is not meant to: indentation becomes a tab, a quoted value
    becomes a braced one, and a bare value gains braces. Callers rewriting a bibliography get a
    normalized file, which is the point of writing one out at all.
    """
    # `bibtexparser.write_string` renders a value it does not recognize with `repr()`, so writing a
    # split-name library through it silently produces `author = {[NameParts(first=['Jane'], ...)]}` —
    # a valid-looking BibTeX file with every author field destroyed, and no warning anywhere.
    #
    # Asked of the library rather than taken as an argument, because the two ways of getting the pairing
    # wrong are not equally survivable: merging an unsplit library raises `ValueError` and stops, while
    # *not* merging a split one writes the mangled file and exits 0.
    #
    # The question is specifically "is a **name** field holding a list of `NameParts`", which is the one
    # thing the split does and the only thing there is to undo. `_NAME_FIELDS` comes from the middleware
    # itself rather than being written out here, so it cannot drift from the fields upstream actually
    # transforms.
    names_are_split = any(isinstance(field.value, list)
                          for entry in library.entries
                          for field in entry.fields
                          if field.key in _NAME_FIELDS)
    if not names_are_split:
        return bibtexparser.write_string(library)
    return bibtexparser.write_string(library,
                                     prepend_middleware=[bibtexparser.middlewares.MergeNameParts(),
                                                         bibtexparser.middlewares.MergeCoAuthors()])


# --------------------------------------------------------------------------------
# Reading a parsed record as a person, or a prompt, should see it

def entry_field(entry: Entry, field: str) -> str:
    """The value of `field`, stripped, or the empty string when the record has no such field.

    Whitespace counts as absent: a database export that writes `abstract = { }` has told us nothing, and a
    caller asking for records that carry an abstract does not want that one.
    """
    fields = entry.fields_dict
    if field not in fields:
        return ""
    return str(fields[field].value or "").strip()


def readable_text(text: str) -> str:
    """A field value as a person should see it: markup resolved, whitespace collapsed.

    Stripping the case-preservation braces by hand gets the common case right and the interesting cases
    wrong. `{\\o}nly` becomes `\\only` rather than `ønly`, because the braces are what terminate the
    LaTeX command, and an accent written `Tr{\\c e}bicki` loses the grouping that says where its argument
    ends. `unicodize_basic_markup` runs the ligature and accent passes before it strips, which is the
    whole reason it is longer than a pair of `replace` calls.
    """
    return common_utils.normalize_whitespace(common_utils.unicodize_basic_markup(text))


# Where a record says it appeared, in the order a BibTeX record is likely to carry it.
VENUE_FIELDS = ("journal", "booktitle", "series", "publisher")


def entry_venue(entry: Entry) -> str:
    """Where the record says it appeared, readable, from the first of `VENUE_FIELDS` it carries; `""` if none.

    A reader recognizes a venue faster than a citekey, and for a judge it is often the strongest evidence
    a short title has beside it.
    """
    for field in VENUE_FIELDS:
        text = entry_field(entry, field)
        if text:
            return readable_text(text)
    return ""


_DOI_IN_TEXT = re.compile(r"10\.\d{4,}/\S+", re.IGNORECASE)

#: Where `entry_doi` looks, in order. The key is a place too: a hand-built bibliography sometimes uses the
#: DOI as its citation key, in whatever spelling it was pasted in (`https://doi.org/...`, `doi: ...`).
DOI_PLACES = ("doi", "url", "key")


def find_doi(text: str) -> str | None:
    """The first DOI written anywhere in `text`, as it is spelled there, or `None` if there is none.

    Whatever surrounds it — a resolver URL, a `doi:` label, a trailing `. PMID: 123` — is left behind, and so
    is trailing sentence punctuation. The case is kept; `papers.utils.normalize_doi` makes the comparison key.
    """
    match = _DOI_IN_TEXT.search(text)
    if match is None:
        return None
    doi = match.group(0).rstrip(".,;:")
    return doi if papers_utils.normalize_doi(doi) is not None else None


def entry_doi(entry: Entry) -> tuple[str, str] | None:
    """`(place, doi)` for the first of `DOI_PLACES` holding a DOI, as `find_doi` spells it; `None` if none does."""
    for place in DOI_PLACES:
        maybe_doi = find_doi(entry.key if place == "key" else entry_field(entry, place))
        if maybe_doi is not None:
            return place, maybe_doi
    return None


# --------------------------------------------------------------------------------
# Reading BibTeX by its surface syntax, for when the parser has refused
#
# These are not a second BibTeX parser and must not grow into one. They exist for the case where
# `bibtexparser` has already declined a record - a stray brace in a field value ends the value early and
# takes the rest of the record with it - and something useful can still be said about the wreckage. Three
# callers, wanting three different things from the same shape:
#
#   - `papers.burstbib`, naming a record from its header line, which it does before parsing anything.
#   - `librarian.chatutil`, salvaging a title so a document still gets a legible label.
#   - `visualizer.importer`, naming the fields whose braces look wrong, so the user can go and fix the data.

def header_key(headerline: str) -> str:
    """Get the record key out of a BibTeX header line: `@article{WOS:000123,` -> `WOS:000123`.

    Returns `""` if `headerline` has no key to give. Verbatim - see `papers.burstbib.get_slug` for the
    variant sanitized for use as a filename.
    """
    start = headerline.find("{")
    end = headerline.find(",", start + 1)  # a key cannot contain a comma, so the first one ends it
    return headerline[start + 1:end] if -1 < start < end else ""

# One `Key = ` at the start of a line, which is the shape every BibTeX writer in practice emits.
_BIBTEX_FIELD_LINE = re.compile(r"^\s*([A-Za-z][\w-]*)\s*=")

def field_value(text: str, key: str) -> str:
    """Get the value of the BibTeX field `key` from `text`, by pattern. `""` if there is none to find.

    Case-insensitive, because the key case is not dependable: a Web of Science export writes `Title = {...}`,
    the BibTeX literature writes `title = {...}`. Reads only a value that opens and closes on its own line,
    which is what the writers emit and what keeps this from turning into a parser.
    """
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*\{(.*)\}\s*,?\s*$",
                         re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip("{} ") if match else ""

def unbalanced_field_names(text: str) -> list[str]:
    """Names of the fields in `text` whose own line opens more braces than it closes.

    A shortlist of suspects rather than a verdict: a field value that legitimately spans several lines (an
    `Affiliation` listing one author per line) is unbalanced line by line too. It is still a far shorter
    list than the record, which is the difference between "this file has a broken record somewhere" and
    something the user can act on.
    """
    names = []
    for line in text.splitlines():
        match = _BIBTEX_FIELD_LINE.match(line)
        if match and line.count("{") != line.count("}"):
            names.append(match.group(1))
    return names

def brace_repair_candidates(text: str, max_candidates: int = 200) -> list[str]:
    """Propose repairs for a BibTeX record `text` whose braces do not balance. Most promising first.

    Each candidate escapes some of the record's braces as `\\{` / `\\}`; the text is otherwise identical,
    character for character. Returns `[]` when there is nothing sensible to propose.

    **These are guesses, and the caller decides by parsing them.** Deciding *which* brace is the stray one
    needs to know where each field value begins and ends, which needs a parser — and the record in hand is
    precisely the one no parser will read, so that route closes. The way out is to stop trying to be right
    and start being checkable: propose from surface syntax, then let `bibtexparser` adjudicate. The
    proposals only have to *contain* the truth, not identify it, and a wrong guess costs nothing because it
    fails to parse and is discarded. Callers should take the first candidate that yields an entry.

    **Escaping, never completion.** An unmatched brace in a field value is nearly always a literal one that
    belongs to the text: mathematics reaching a BibTeX file through a PDF extractor arrives as things like
    `{0 <= rho <= 1`, set-builder notation whose closing brace the extractor dropped. `\\{` is what BibTeX
    means by a literal brace in running text, so escaping states what the data actually is, and no character
    is added or removed. Supplying a missing partner instead would invent a grouping the author never wrote,
    and since nothing says *where* that partner belonged, the invention is unconstrained. So a record that
    lost a field value's *terminator*, rather than gaining a stray literal, is not repairable here — every
    candidate will fail the caller's check, which is the correct outcome and not a defect.

    Two cheap facts do the pruning, and between them the real cases collapse to a single candidate:

    - **The sign of the imbalance says which brace to look for.** A surplus of `{` means a stray opener, so
      only openers are eligible; a surplus of `}`, only closers.
    - **Some braces are structurally certain and never eligible** — the `@type{` on the header line, the one
      opening each `Key = {` field line, and the record's own closing `}` on its own line. Those are the
      ones a naive scan would blame first, and escaping any of them destroys the record.
    """
    lines = text.splitlines(keepends=True)
    surplus = text.count("{") - text.count("}")
    wanted = "{" if surplus > 0 else "}"
    if surplus == 0:
        return []

    # Structurally certain braces, which no candidate may touch.
    reserved = set()
    for i, line in enumerate(lines):
        match = _BIBTEX_FIELD_LINE.match(line)
        if match:
            col = line.find("{", match.end())
            if col != -1:
                reserved.add((i, col))
        elif line.lstrip().startswith("@"):
            col = line.find("{")
            if col != -1:
                reserved.add((i, col))
        elif line.strip() == "}":
            reserved.add((i, line.index("}")))

    eligible = [(i, j) for i, line in enumerate(lines) for j, ch in enumerate(line)
                if ch == wanted and (i, j) not in reserved and not line[:j].endswith("\\")]
    if len(eligible) < abs(surplus):
        return []
    # A stray literal sits *inside* a value, so it falls after that value's opening brace and before its
    # terminator. Which way to order the guesses therefore depends on what is being hunted: a surplus
    # opener is more likely the later of the candidates, a surplus closer the earlier one. Ordering only
    # decides which guess the oracle sees first, but where more than one would parse, first is what wins.
    ordered = sorted(eligible, reverse=(wanted == "{"))
    combinations = list(itertools.islice(itertools.combinations(ordered, abs(surplus)), max_candidates + 1))
    if len(combinations) > max_candidates:  # too tangled to guess at; let the caller report it instead
        return []

    candidates = []
    for chosen in combinations:
        repaired = list(lines)
        for i, j in sorted(chosen, reverse=True):  # from the end, so earlier positions stay valid
            repaired[i] = repaired[i][:j] + "\\" + repaired[i][j:]
        candidates.append("".join(repaired))
    return candidates


def repair_record(raw: str) -> str | None:
    """Repair one BibTeX record `raw` that failed to parse. Returns the repaired text, or `None`.

    The repair is for a record whose braces do not balance, which is what a stray `{` or `}` in a field
    value does — mathematics arriving through a PDF extractor is the usual source. Only the offending
    braces are escaped; the text is otherwise identical, character for character.

    **This is where guessing becomes safe.** `brace_repair_candidates` proposes repairs
    from surface syntax, having no way to know which is right — deciding that needs to know where each
    field value begins and ends, which needs a parser, and a record needing repair is precisely the one no
    parser will read. So the proposals are not trusted: each is parsed in turn, and the first that yields
    an entry wins. The parser is the oracle. A wrong proposal fails to parse and costs nothing, which is
    what lets the proposing side stay a heuristic.

    **The oracle is asked a structural question**, with the name splitting off, because a record can carry
    two unrelated faults and repairing one is not failing at the other. Whether the result then reads
    under Raven's full chain is a separate question, and the caller's — `raven.papers.fixbib` asks it, so
    that a record whose braces are now balanced and whose *author* BibTeX cannot express is reported for
    the author rather than for the braces it no longer has a problem with.

    Returning `None` means no proposal parsed, and the caller should treat the record as it did before —
    a record that lost a field value's *terminator*, rather than gaining a stray literal, lands here, and
    reporting it is more honest than inventing the missing brace.
    """
    # A rejected candidate makes `bibtexparser` log a warning, which is noise -- it is the mechanism
    # working, not a problem. Silencing it would mean setting a level on a logger this function does not
    # own, and `importer._parse_input_files` runs on a background thread, so two overlapping parses could restore
    # each other's level and leave the parser muted for the rest of the process. Not worth it for the
    # traffic involved: candidates are generated only for a record that has already failed, and the
    # pruning gets that down to one candidate on real data. An application that wants quiet can set the
    # level itself, which is a decision an application gets to make and a library does not.
    for candidate in brace_repair_candidates(raw):
        try:
            library = parse_string(candidate, split_names=False)
        except Exception:  # noqa: BLE001 -- a repair that breaks the parser is just a failed repair
            continue
        if library.entries:
            return candidate
    return None


# The two delimiter pairs a field value can wear. A value in neither is bare, and ends at the separator.
_FIELD_DELIMITERS = {"{": "}", '"': '"'}
_FIELD_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_+:.-]*")


def _skip_whitespace(raw: str, i: int) -> int:
    while i < len(raw) and raw[i].isspace():
        i += 1
    return i


def _scan_value(raw: str, i: int) -> int | None:
    """Return the offset one past the field value beginning at `i`, or `None` if it never ends.

    Handles the three shapes a value comes in — brace-delimited, quote-delimited, and bare — and the `#`
    concatenation that may join several of them. A backslash escapes the character after it, so the `\\}`
    that `repair_record` writes does not close a value.
    """
    while True:
        delimiter = _FIELD_DELIMITERS.get(raw[i] if i < len(raw) else "")
        if delimiter is None:  # bare value: a number, or a string macro name
            while i < len(raw) and raw[i] not in ",}#":
                i += 1
        elif delimiter == '"':
            i += 1
            while i < len(raw) and raw[i] != '"':
                i += 2 if raw[i] == "\\" else 1
            if i >= len(raw):
                return None
            i += 1
        else:
            depth = 0
            while i < len(raw):
                if raw[i] == "\\":
                    i += 2
                    continue
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            else:
                return None

        after = _skip_whitespace(raw, i)
        if after < len(raw) and raw[after] == "#":  # concatenation: another part follows
            i = _skip_whitespace(raw, after + 1)
            continue
        return i


class _FieldSpan(NamedTuple):
    """Where one field of a BibTeX record sits in its text. Offsets are into the record's `raw`.

    None of them includes the comma that separates one field from the next, so a span can be cut out, or
    its value replaced, without disturbing the punctuation around it.
    """
    key: str            # lowercased
    name_start: int
    name_end: int
    value_start: int
    value_end: int


def _field_spans(raw: str) -> list[_FieldSpan] | None:
    """Locate the fields of the single BibTeX record `raw`. `None` if it does not scan.

    Fields come back in the order they appear, as `_FieldSpan`, which see.

    This is a scanner rather than a parse, because every caller here has a record `bibtexparser` has
    already refused; the point is to locate structure in text that is not going to become an `Entry`.
    """
    open_brace = raw.find("{")
    if open_brace == -1:
        return None
    i = raw.find(",", open_brace)  # past the entry key, which cannot itself contain a comma
    if i == -1:
        return None
    i += 1

    spans = []
    while True:
        i = _skip_whitespace(raw, i)
        while i < len(raw) and raw[i] == ",":  # tolerate a trailing or doubled separator
            i = _skip_whitespace(raw, i + 1)
        if i >= len(raw) or raw[i] == "}":
            return spans

        name = _FIELD_NAME_PATTERN.match(raw, i)
        if name is None:
            return None
        after_name = _skip_whitespace(raw, name.end())
        if after_name >= len(raw) or raw[after_name] != "=":
            return None

        value_start = _skip_whitespace(raw, after_name + 1)
        value_end = _scan_value(raw, value_start)
        if value_end is None:
            return None
        spans.append(_FieldSpan(name.group().lower(), name.start(), name.end(), value_start, value_end))
        i = value_end


def _undelimit(value: str) -> str:
    """Strip one layer of `{}` or `""` from a field value, leaving anything else alone.

    Anything else includes a bare value, and a `#` concatenation of several delimited parts — which opens
    with a delimiter and closes with its mate while being no single value at all. Stripping that pair
    would move the value's own end inwards; returning it untouched merely leaves literal braces in the
    merged text, which is the harmless direction to be wrong in.
    """
    if len(value) < 2 or _FIELD_DELIMITERS.get(value[0]) != value[-1]:
        return value
    if value[0] == '"':
        return value[1:-1]
    depth, i = 0, 0
    while i < len(value):
        if value[i] == "\\":
            i += 2
            continue
        if value[i] == "{":
            depth += 1
        elif value[i] == "}":
            depth -= 1
            if depth == 0:  # the opening brace's mate: only the last character makes this one value
                return value[1:-1] if i == len(value) - 1 else value
        i += 1
    return value


def _widen_cut(raw: str, start: int, end: int) -> tuple[int, int]:
    """Grow the span `[start, end)` to swallow the punctuation and blank line that deleting it would leave.

    Takes a following comma, and — when the field had a line to itself — the indentation before it and the
    newline after, so that removing a field does not leave an empty line where it stood.
    """
    if end < len(raw) and raw[end] == ",":
        end += 1
    line_start = raw.rfind("\n", 0, start) + 1
    if not raw[line_start:start].strip():
        start = line_start
        while end < len(raw) and raw[end] in " \t":
            end += 1
        if end < len(raw) and raw[end] == "\n":
            end += 1
    return start, end


def repair_duplicate_field_keys(raw: str, maybe_duplicate_keys: set[str] | None = None) -> str | None:
    """Repair one BibTeX record `raw` that names the same field twice. Returns the text, or `None`.

    `maybe_duplicate_keys`: the field names to merge, if the caller already knows them —
                            `bibtexparser`'s `DuplicateFieldKeyBlock` reports them as `.duplicate_keys`.
                            Leave it `None` to merge every field name that repeats.

    The repeats are folded into the first occurrence, their values joined by newlines, and the later
    fields removed. Everything else in the record is left byte for byte as it was.

    Merging rather than keeping one is what makes this safe to do unsupervised: a record with two `annote`
    fields has two different notes in it, and picking one would delete somebody's data quietly. Joining
    them keeps every character, in a field that standard BibTeX tools can read, and leaves the result
    plainly inspectable by whoever wants to split it again.

    Returning `None` means the record does not scan, names no field twice, or does not read back as an
    entry once merged. As in `repair_record`, that last check is structural — a record that still fails
    for an unrelated reason has still had its repeated fields merged, and saying which fault remains is
    the caller's job.
    """
    spans = _field_spans(raw)
    if not spans:
        return None

    positions = collections.defaultdict(list)
    for index, span in enumerate(spans):
        positions[span.key].append(index)
    repeated = {key: indices for key, indices in positions.items() if len(indices) > 1}
    if maybe_duplicate_keys is not None:
        wanted = {key.lower() for key in maybe_duplicate_keys}
        repeated = {key: indices for key, indices in repeated.items() if key in wanted}
    if not repeated:
        return None

    edits = []  # (start, end, replacement), disjoint, applied in one pass below
    for indices in repeated.values():
        values = [_undelimit(raw[spans[i].value_start:spans[i].value_end]) for i in indices]
        keep = spans[indices[0]]
        edits.append((keep.value_start, keep.value_end, "{" + "\n".join(values) + "}"))
        for i in indices[1:]:
            edits.append((*_widen_cut(raw, spans[i].name_start, spans[i].value_end), ""))

    pieces, cursor = [], 0
    for start, end, replacement in sorted(edits):
        pieces.append(raw[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(raw[cursor:])
    candidate = "".join(pieces)

    # The parser is the oracle, exactly as in `repair_record`: a repair that does not read back as an
    # entry is a failed repair, whatever the scanner made of the text.
    try:
        library = parse_string(candidate, split_names=False)
    except Exception:  # noqa: BLE001 -- a repair that breaks the parser is just a failed repair
        return None
    return candidate if library.entries else None


_NAME_SUFFIXES = {"jr.", "sr.", "jr", "sr", "ii", "iii", "iv"}


def _is_initials(token: str) -> bool:
    """Whether `token` is one person's initials as APA writes them: `C.`, `C. G. L.`, `L.-A.`, `Ch.`."""
    pieces = [piece for piece in re.split(r"[\s-]+", token) if piece]
    return bool(pieces) and all(re.fullmatch(r"[^\W\d_]{1,2}\.", piece) and piece[0].isupper()
                                for piece in pieces)


def _apa_names_to_bibtex(value: str) -> str | None:
    """Rewrite an APA-style name list as BibTeX's `and`-separated one. `None` unless it is one, unambiguously.

    APA writes `Surname, I., Surname, I., & Surname, I.`, so every comma separates something and BibTeX,
    which allows a name at most two, cannot tell where one name ends. The list is accepted only when its
    comma-separated parts pair up exactly as a surname followed by initials, optionally with a suffix, so
    a list missing a comma (`Costa, R. Viegas, C.`) is left for a person to read. Tolerated on the way:
    the `&` before the last name, a trailing comma, a comma typed for the space inside initials
    (`G.,L.`), and APA's `...` for omitted authors, which becomes BibTeX's `and others`.
    """
    text = value.strip()
    if re.search(r"\sand\s", text):  # already BibTeX's form, or something else entirely
        return None
    text = re.sub(r"\s*&\s*", ", ", text)
    others = bool(re.search(r"\.\.\.|…", text))
    text = re.sub(r"\.\.\.|…", ",", text)
    text = re.sub(r",(?=[^\W\d_]\.)", " ", text)  # `G.,L.` -> `G. L.`: a comma with an initial right after it
    tokens = [token.strip() for token in text.split(",") if token.strip()]

    names, i = [], 0
    while i < len(tokens):
        surname = tokens[i]
        if _is_initials(surname) or surname.casefold() in _NAME_SUFFIXES or i + 1 >= len(tokens):
            return None
        initials, suffix = tokens[i + 1], None
        words = initials.split()
        if len(words) > 1 and words[-1].casefold() in _NAME_SUFFIXES:  # `E. R. Jr.`
            initials, suffix = " ".join(words[:-1]), words[-1]
        if not _is_initials(initials):
            return None
        i += 2
        if suffix is None and i < len(tokens) and tokens[i].casefold() in _NAME_SUFFIXES:  # `E. R., Jr.`
            suffix = tokens[i]
            i += 1
        names.append(f"{surname}, {suffix}, {initials}" if suffix else f"{surname}, {initials}")
    if not names:
        return None
    return " and ".join(names + (["others"] if others else []))


def _strip_stray_commas(value: str) -> str | None:
    """Drop the trailing comma from each name of an `and`-separated list. `None` if there was none.

    `Tan, C. G. L., and Voo, K.` is a list in BibTeX's own form with one comma too many, and a comma at the
    end of a name carries nothing, so removing it changes no name.
    """
    names = re.split(r"\s+and\s+", value.strip())
    if len(names) < 2:
        return None
    stripped = [name.rstrip().rstrip(",").rstrip() for name in names]
    return " and ".join(stripped) if stripped != [name.rstrip() for name in names] else None


def repair_name_lists(raw: str) -> str | None:
    """Repair one BibTeX record `raw` whose name fields BibTeX cannot split. Returns the text, or `None`.

    Two faults, both written by hand: an APA-style list (see `_apa_names_to_bibtex`), and an `and`-separated
    list with a stray comma at the end of a name (see `_strip_stray_commas`). Each of `author`, `editor` and
    `translator` is rewritten when one of the two applies, and everything else in the record is left byte
    for byte as it was. An APA list that does not pair up unambiguously is left alone, which is what stops
    this from guessing where a name ends.

    Returning `None` means nothing was rewritten, or the rewrite does not read back as an entry with names
    split, which is the check this repair exists to pass.
    """
    spans = _field_spans(raw)
    if not spans:
        return None
    edits = []
    for span in spans:
        if span.key not in _NAME_FIELDS:
            continue
        value = _undelimit(raw[span.value_start:span.value_end])
        maybe_names = _apa_names_to_bibtex(value) or _strip_stray_commas(value)
        if maybe_names is not None:
            edits.append((span.value_start, span.value_end, "{" + maybe_names + "}"))
    if not edits:
        return None

    pieces, cursor = [], 0
    for start, end, replacement in edits:
        pieces.append(raw[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(raw[cursor:])
    candidate = "".join(pieces)
    try:
        library = parse_string(candidate)
    except Exception:  # noqa: BLE001 -- a repair that breaks the parser is just a failed repair
        return None
    return candidate if library.entries else None


def repair_failed_block(failed_block) -> str | None:
    """Repair one record `bibtexparser` refused, choosing the repair by what is wrong with it.

    `failed_block`: one of a parsed library's `failed_blocks`. A `DuplicateFieldKeyBlock` gets
                    `repair_duplicate_field_keys`, told which field names the parser saw repeated; a
                    `MiddlewareErrorBlock`, whose syntax was fine and whose names could not be split, gets
                    `repair_name_lists`; any other block gets `repair_record`.

    Returns the repaired text, or `None` if the record stays unreadable.
    """
    if isinstance(failed_block, DuplicateFieldKeyBlock):
        return repair_duplicate_field_keys(failed_block.raw, failed_block.duplicate_keys)
    if isinstance(failed_block, MiddlewareErrorBlock):
        return repair_name_lists(failed_block.raw)
    return repair_record(failed_block.raw)


def read_repeated_keys(library: Library, split_names: bool = True) -> list[tuple[str, Entry | None]]:
    """Read the records `library` refused only for repeating an earlier record's key, each under a new one.

    `library`: as `parse_string` or `parse_file` returned it. Not modified.
    `split_names`: as the library was parsed, so that a record read here matches the others.

    The new key is the old one with `-2` appended, or the next free number. Returns one
    `(repeated_key, maybe_entry)` per such record, in input order, `maybe_entry` being `None` where the
    record does not read even on its own.
    """
    # Read again from the block's text under the new key, rather than taken from the block's
    # `ignore_error_block`, which skipped the middleware: its field names are as the file wrote them, so
    # `Title` never reads as `title`.
    taken = {entry.key for entry in library.entries}
    out = []
    for block in library.failed_blocks:
        if not isinstance(block, DuplicateBlockKeyBlock):
            continue
        n = 2
        while f"{block.key}-{n}" in taken:
            n += 1
        new_key = f"{block.key}-{n}"
        reread = parse_string(block.raw.replace(block.key, new_key, 1), split_names=split_names)
        if len(reread.entries) == 1 and reread.entries[0].key == new_key:
            taken.add(new_key)
            out.append((block.key, reread.entries[0]))
        else:
            out.append((block.key, None))
    return out


# An HTML character entity, together with whatever backslashes precede it. The backslash run is captured
# because a `.bib` file carrying HTML almost always carries it *escaped* — `Q\&amp;A`, not `Q&amp;A` —
# since the `&` was escaped for BibTeX on the way in, and the replacement has to know whether the `&` it
# is consuming was already spoken for. The entity itself is `entities.ENTITY_PATTERN`, which see for why
# the semicolon is required.
_HTML_ENTITY_PATTERN = re.compile(r"(\\*)" + entities.ENTITY_PATTERN.pattern)


def _decode_one_entity(match: re.Match, counter: box) -> str:
    """One `_HTML_ENTITY_PATTERN` match as BibTeX text, or unchanged if it names nothing.

    `counter`: a `box` the decode count is accumulated into. `re.sub` gives the replacement no way to
               report other than by mutating something it was handed.
    """
    backslashes, name = match.group(1), match.group(2)
    # Space separators stay as themselves in a file: a no-break space is typography the source asked for.
    maybe_text = entities.resolve(name, fold_spaces=False)
    if maybe_text is None:
        return match.group(0)

    # An odd run means the last backslash was escaping the entity's own `&`, which is syntax rather than
    # content and goes with it. An even run is literal escaped backslashes, which stay.
    kept = backslashes[:-1] if len(backslashes) % 2 else backslashes
    counter << unbox(counter) + 1
    return kept + bibtex_escape(maybe_text)


def decode_html_entities(source: str) -> tuple[str, int]:
    """Decode the HTML character entities in BibTeX text. Returns `(text, how many were decoded)`.

    A database that exports HTML into a `.bib` leaves entities in the field values, where they are not
    content: a title meaning `Q&A` arrives as `Q\\&amp;A` and reads that way in every tool downstream —
    a citation, a word cloud, a typeset bibliography.

    The result is BibTeX rather than plain text, so a decoded character that BibTeX reserves is escaped
    on the way out: `&amp;` becomes `\\&`, and the file stays as readable as it was. `&nbsp;` becomes a
    real no-break space, since not breaking the line there is what the source asked for.

    Everything outside an entity is left byte for byte as it was, this being a substitution over the text
    rather than a parse and rewrite. An entity naming nothing — a stray `&foo;` — is left alone too, and
    does not count: the number returned is how many characters were *decoded*, which is what a caller
    reports to a user, and `re.subn` would have counted it as a substitution.
    """
    counter = box(0)
    return _HTML_ENTITY_PATTERN.sub(lambda match: _decode_one_entity(match, counter), source), unbox(counter)


def _braces_balance(text: str) -> bool:
    """Whether `text` could stand alone inside a `{...}` field value."""
    depth, i = 0, 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def _relocate_in_record(raw: str, field: str) -> str | None:
    """Move the rights notice out of one record's `abstract` into `field`. `None` if nothing to do."""
    spans = _field_spans(raw)
    if not spans:
        return None
    by_key = {span.key: span for span in spans}
    abstract = by_key.get("abstract")
    if abstract is None:
        return None

    body, maybe_notice = boilerplate.split_rights_notice(_undelimit(raw[abstract.value_start:abstract.value_end]))
    if maybe_notice is None or not body:
        return None
    # A notice sits at the tail, so cutting there normally leaves both halves whole — but a value like
    # `{Body {group (c) 2024} more}` would split a brace pair and produce two field values that are not
    # values. Cheap to check, and the alternative is writing a file that does not parse.
    if not (_braces_balance(body) and _braces_balance(maybe_notice)):
        return None

    # A record can carry a notice in the target field *and* another in its abstract, and both name a
    # source the record came from. Joined rather than either clobbered or refused, the way
    # `repair_duplicate_field_keys` joins repeated fields and `raven.papers.deduplicate` unions this
    # same field across a merge — one rule for combining, wherever the collision turns up.
    existing = by_key.get(field)
    if existing is not None:
        kept = _undelimit(raw[existing.value_start:existing.value_end])
        if maybe_notice in kept.split("\n"):
            return None  # already recorded; moving it again would only duplicate it
        combined = f"{kept}\n{maybe_notice}"
        if not _braces_balance(combined):
            return None
        # Two edits, applied right-to-left so the earlier offsets stay valid.
        first, second = sorted([(abstract.value_start, abstract.value_end, "{" + body + "}"),
                                (existing.value_start, existing.value_end, "{" + combined + "}")])
        candidate = (raw[:first[0]] + first[2] + raw[first[1]:second[0]] + second[2] + raw[second[1]:])
        expected_notice = combined
    else:
        # Match the record's own indentation, so the new field looks like the ones around it.
        line_start = raw.rfind("\n", 0, abstract.name_start) + 1
        indent = raw[line_start:abstract.name_start]
        candidate = (raw[:abstract.value_start] + "{" + body + "}"
                     + f",\n{indent}{field} = {{{maybe_notice}}}"
                     + raw[abstract.value_end:])
        expected_notice = maybe_notice

    # The parser is the oracle here as everywhere else in this module, and it is checked against the
    # values rather than merely against "an entry came back": a repair that parses into the wrong text
    # is not a repair.
    try:
        library = parse_string(candidate, split_names=False)
    except Exception:  # noqa: BLE001 -- an edit that breaks the parser is just a failed edit
        return None
    if not library.entries:
        return None
    entry = library.entries[0]
    try:
        if entry["abstract"] != body or entry[field] != expected_notice:
            return None
    except KeyError:
        return None
    return candidate


def relocate_rights_notices(source: str, field: str = "copyright") -> tuple[str, int]:
    """Move each publisher's rights notice out of an `abstract` into a field of its own.

    Returns `(text, how many were moved)`.

    A rights notice appended to an abstract is metadata in the wrong field: it is not what the paper says,
    and anything reading the abstract as prose has to cope with it — it is why a publisher's name turns up
    in a word cloud. Moved rather than deleted, because it does say something the record otherwise does
    not, and in a bibliography merged from several exports it is often the only thing identifying which
    export a record came from.

    `field`: where to put it. `copyright` by default: absent from real-world exports, so it collides with
             nothing, and not typeset by standard BibTeX styles, so it cannot appear in a reference list.

    A record that already has that field keeps what is there and gains the moved notice on a line below
    it — both name a source the record came from, and the same joining rule applies wherever these
    collide. A notice already recorded there is not moved twice. A record whose braces would not survive
    the split is left alone. Everything outside a moved notice is byte for byte as it was.
    """
    library = parse_string(source, split_names=False)
    pieces, cursor, moved = [], 0, 0
    for entry in library.entries:
        raw = entry.raw
        start = source.find(raw, cursor)
        if start == -1:  # should not happen; skip rather than corrupt the file on a guess
            continue
        candidate = _relocate_in_record(raw, field)
        if candidate is None:
            continue
        pieces.append(source[cursor:start])
        pieces.append(candidate)
        cursor = start + len(raw)
        moved += 1
    pieces.append(source[cursor:])
    return "".join(pieces), moved


def _add_doi_field(raw: str, entry: Entry, doi: str) -> str | None:
    """`raw` with a `doi` field as its first, or `None` where the parser does not read it back exactly so."""
    header_end = raw.find("\n")
    if header_end == -1:
        return None
    following = raw[header_end + 1:]
    indent = following[:len(following) - len(following.lstrip(" \t"))]
    candidate = raw[:header_end + 1] + f"{indent}doi = {{{doi}}},\n" + following
    try:
        library = parse_string(candidate, split_names=False)
    except Exception:  # noqa: BLE001 -- an edit that breaks the parser is just a failed edit
        return None
    if len(library.entries) != 1:
        return None
    new = library.entries[0]
    expected = {field.key: field.value for field in entry.fields} | {"doi": doi}
    if new.key != entry.key or {field.key: field.value for field in new.fields} != expected:
        return None
    return candidate


def recover_dois(source: str) -> tuple[str, int]:
    """Give each record that has no `doi` field the DOI it carries somewhere else.

    Returns `(text, how many were added)`.

    The DOI is looked for in `url` and then in the citation key (see `entry_doi`), and written as
    `find_doi` spells it, as the record's first field. Nothing is removed, the key included. A record with a
    `doi` field already is left alone, even an empty one: that field is somebody's, and the value it ought
    to hold is not this function's to decide. Everything outside an added line is byte for byte as it was.
    """
    library = parse_string(source, split_names=False)
    pieces, cursor, added = [], 0, 0
    for entry in library.entries:
        if "doi" in entry.fields_dict:
            continue
        maybe_found = entry_doi(entry)
        if maybe_found is None:
            continue
        raw = entry.raw
        start = source.find(raw, cursor)
        if start == -1:  # should not happen; skip rather than corrupt the file on a guess
            continue
        candidate = _add_doi_field(raw, entry, maybe_found[1])
        if candidate is None:
            continue
        pieces.append(source[cursor:start])
        pieces.append(candidate)
        cursor = start + len(raw)
        added += 1
    pieces.append(source[cursor:])
    return "".join(pieces), added


def _entry_arxiv_id(entry, keep_version: bool) -> str:
    """The arXiv ID from a feed entry's URL, e.g. ``http://arxiv.org/abs/2103.12345v2``.

    `keep_version`: whether to retain the ``vN`` suffix. Dropping it is right for a search, where the
                    result is "this paper" and the version is incidental. Retaining it is right when the
                    *request* named a version, since then the version is part of what was asked for and
                    two versions of one paper must not collapse into one entry.
    """
    arxiv_id = entry.id.split("/abs/")[-1]
    return arxiv_id if keep_version else identifiers.strip_version(arxiv_id)


def _make_key(entry, keep_version: bool = False) -> str:
    """Generate a BibTeX key from an arXiv feed entry.

    Format: ``LastName_YYYY_arXivID`` — guaranteed unique by the arXiv ID.
    """
    arxiv_id = _entry_arxiv_id(entry, keep_version)
    # Old-style IDs contain a slash (hep-ex/0307015) — replace for BibTeX safety
    arxiv_id = arxiv_id.replace("/", "_")

    # First author's last name
    authors = entry.get("authors", [])
    if authors:
        name = authors[0].get("name", "Unknown")
        last_name = name.split()[-1]
        last_name = re.sub(r"[^a-zA-Z]", "", last_name)
    else:
        last_name = "Unknown"

    year = entry.published[:4]
    return f"{last_name}_{year}_{arxiv_id}"


def _clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def _deduplicate_by_key(entries: list, keep_versions: bool) -> list:
    """Entries with unique BibTeX keys, keeping the highest arXiv version of each, in first-seen order.

    Order is preserved so that output stays comparable across runs; a later, higher version replaces an
    earlier one in place rather than moving to the end.
    """
    chosen: dict[str, tuple[int, int, object]] = {}  # key -> (version, position, entry)
    for position, entry in enumerate(entries):
        key = _make_key(entry, keep_versions)
        _base, version = identifiers.split_version(_entry_arxiv_id(entry, keep_version=True))
        previous = chosen.get(key)
        if previous is None:
            chosen[key] = (version, position, entry)
        elif version > previous[0]:
            chosen[key] = (version, previous[1], entry)  # keep the earlier slot, take the better entry
    return [entry for _version, _position, entry in sorted(chosen.values(), key=lambda t: t[1])]


def entries_to_bibtex(entries: list, keep_versions: bool = False) -> str:
    """Convert a list of feedparser arXiv entries to a BibTeX string.

    `keep_versions`: whether ``eprint`` and the entry key retain the ``vN`` suffix. Defaults to dropping
                     it, which is what a search wants — a search result is a *paper*, and pinning it to
                     whichever version was current that day would be noise. Set it when the caller
                     requested specific versions and needs to know which one it got back.

    Entries that collapse onto the same key keep only the highest-versioned one. That only arises when
    versions are being stripped and the input names two versions of one paper — which is the requested
    behaviour ("one entry per paper"), so it is resolved rather than reported. Emitting both would not
    merely produce a duplicate: `bibtexparser` turns a repeated key into a failed block and then raises
    while writing it, so the whole bibliography would be lost to an error from inside the library.
    """
    library = Library()

    for entry in _deduplicate_by_key(entries, keep_versions):
        arxiv_id = _entry_arxiv_id(entry, keep_versions)

        year = entry.published[:4]
        authors = " and ".join(a.get("name", "") for a in entry.get("authors", []))
        title = _clean_whitespace(entry.get("title", ""))
        abstract = _clean_whitespace(entry.get("summary", ""))

        fields = [
            Field("author", authors),
            Field("title", title),
            Field("year", year),
            Field("eprint", arxiv_id),
            Field("archiveprefix", "arXiv"),
            Field("abstract", abstract),
        ]

        # Primary category
        primary = entry.get("arxiv_primary_category", {})
        if term := primary.get("term"):
            fields.append(Field("primaryclass", term))

        # DOI (may be absent)
        if doi := entry.get("arxiv_doi"):
            fields.append(Field("doi", doi))

        # Journal reference (may be absent)
        if journal_ref := entry.get("arxiv_journal_ref"):
            fields.append(Field("journal", journal_ref))

        key = _make_key(entry, keep_versions)
        library.add(Entry("article", key, fields=fields))

    return bibtexparser.write_string(library)
