"""Read and write BibTeX.

Writing: `entries_to_bibtex` converts arXiv Atom feed entries, used by `raven.papers.search` to format
arXiv API search results.

Reading: `parse_file` and `parse_string` wrap `bibtexparser` with the one middleware chain Raven reads
BibTeX through, so every consumer gets the same normalization. A caller that will *write* the library back
out wants `split_names=False`, and `write_string` either way — see below.

Repairing: `repair_record` and `repair_duplicate_field_keys` each rescue one record `bibtexparser`
refused, for the two reasons a real-world export gives it — braces that do not balance, and a field name
that occurs twice. Both take the record's text and hand back text, so a caller may parse the result or
write it to a file; `raven.papers.fixbib` does the latter. `decode_html_entities` is text-to-text as
well, and fixes what is *inside* records a parser is perfectly happy with: HTML a database left in the
field values.
"""

from __future__ import annotations

__all__ = ["parse_file", "parse_string", "write_string",
           "repair_record", "repair_duplicate_field_keys", "decode_html_entities",

           "entries_to_bibtex"]

import collections
import html.entities
import pathlib
import re
import unicodedata

import bibtexparser
from bibtexparser.model import Entry, Field
from bibtexparser import Library

from ..common import utils as common_utils

from . import identifiers
from .utils import bibtex_escape


def _reader_middleware(split_names: bool) -> list:
    """The middleware chain Raven reads BibTeX through, as a fresh list per call.

    Each link earns its place:

      - `NormalizeFieldKeys` because the key case is not dependable - a Web of Science export writes
        `Title = {...}`, the BibTeX literature writes `title = {...}`.
      - `SeparateCoAuthors` then `SplitNameParts`, in that order, because the second raises without the
        first. Between them they turn one `author` string into name parts that survive "Ludwig van
        Beethoven", "Brinch Hansen, Per" and "Beeblebrox, IV, Zaphod". Omitted when `split_names` is
        false, leaving `author` and `editor` the strings the file had.

    Fresh instances rather than a shared module-level list, because a middleware is free to carry
    per-parse state and sharing one across concurrent parses would be a bug that only shows up under
    load.
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
    # a valid-looking BibTeX file with every author field destroyed, and no warning anywhere. Hence a
    # writer in this module at all, rather than a note telling callers to remember the inverse chain.
    #
    # Detected from the data rather than taken as an argument, because the two ways of getting it wrong
    # are not equally survivable: merging an unsplit library raises `ValueError` and stops, while *not*
    # merging a split one writes the mangled file and exits 0.
    needs_merge = any(isinstance(field.value, list)
                      for entry in library.entries
                      for field in entry.fields)
    if not needs_merge:
        return bibtexparser.write_string(library)
    return bibtexparser.write_string(library,
                                     prepend_middleware=[bibtexparser.middlewares.MergeNameParts(),
                                                         bibtexparser.middlewares.MergeCoAuthors()])


def repair_record(raw: str) -> str | None:
    """Repair one BibTeX record `raw` that failed to parse. Returns the repaired text, or `None`.

    The repair is for a record whose braces do not balance, which is what a stray `{` or `}` in a field
    value does — mathematics arriving through a PDF extractor is the usual source. Only the offending
    braces are escaped; the text is otherwise identical, character for character.

    **This is where guessing becomes safe.** `common_utils.bibtex_brace_repair_candidates` proposes repairs
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
    for candidate in common_utils.bibtex_brace_repair_candidates(raw):
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


def _field_spans(raw: str) -> list[tuple[str, int, int, int, int]] | None:
    """Locate the fields of the single BibTeX record `raw`. `None` if it does not scan.

    Each field is reported as `(lowercased key, name start, name end, value start, value end)`, in the
    order they appear. Offsets are into `raw`, and none of them includes the comma that separates one
    field from the next — so a span can be cut out, or its value replaced, without disturbing the
    punctuation around it.

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
        spans.append((name.group().lower(), name.start(), name.end(), value_start, value_end))
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
        positions[span[0]].append(index)
    repeated = {key: indices for key, indices in positions.items() if len(indices) > 1}
    if maybe_duplicate_keys is not None:
        wanted = {key.lower() for key in maybe_duplicate_keys}
        repeated = {key: indices for key, indices in repeated.items() if key in wanted}
    if not repeated:
        return None

    edits = []  # (start, end, replacement), disjoint, applied in one pass below
    for indices in repeated.values():
        values = [_undelimit(raw[spans[i][3]:spans[i][4]]) for i in indices]
        keep = spans[indices[0]]
        edits.append((keep[3], keep[4], "{" + "\n".join(values) + "}"))
        for i in indices[1:]:
            edits.append((*_widen_cut(raw, spans[i][1], spans[i][4]), ""))

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


# An HTML character entity, together with whatever backslashes precede it. The backslash run is captured
# because a `.bib` file carrying HTML almost always carries it *escaped* — `Q\&amp;A`, not `Q&amp;A` —
# since the `&` was escaped for BibTeX on the way in, and the replacement has to know whether the `&` it
# is consuming was already spoken for.
#
# The name is bounded and must carry its semicolon: HTML5 also defines a handful of entities without one,
# and honouring those here would rewrite `AT&T` into `AT&T` via `&T` at the first opportunity.
_HTML_ENTITY_PATTERN = re.compile(r"(\\*)&(\#\d{1,7}|\#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")

# Unicode categories whose characters must not be written into a `.bib` as themselves: format characters
# and controls (Cf, Cc), and the line and paragraph separators (Zl, Zp). HTML5 names a good number of
# these — `&zwj;`, `&lrm;`, `&NoBreak;`, `&#10;` — and decoding one to itself would put something in the
# file that nobody reading the file can see. A Cf is dropped; the rest become an ordinary space.
#
# A separator has to go for a second reason, which is what makes this a correctness rule rather than a
# tidiness one: a newline arriving mid-record moves every line after it, and `raven.papers.fixbib`
# reports faults by line number in the user's own file.
_INVISIBLE_CATEGORIES = frozenset(["Cf", "Cc", "Zl", "Zp"])


def _decode_one_entity(match: re.Match, counter: list) -> str:
    """One `_HTML_ENTITY_PATTERN` match as BibTeX text, or unchanged if it names nothing.

    `counter`: a one-element list the decode count is accumulated into. A closure over an integer would
               need a `nonlocal`, and `re.sub` gives the replacement no other way to report.
    """
    backslashes, name = match.group(1), match.group(2)
    if name.startswith("#"):
        try:
            code = int(name[2:], 16) if name[1:2].lower() == "x" else int(name[1:])
        except ValueError:
            return match.group(0)
        character = chr(code) if 0 < code < 0x110000 else None
    else:
        character = html.entities.html5.get(name + ";")
    if character is None:
        return match.group(0)

    category = unicodedata.category(character)
    if category == "Cf":
        character = ""
    elif category in _INVISIBLE_CATEGORIES or (character.isspace() and character != " "):
        # A no-break space is the common one, and it is the worst kind of wrong: it looks exactly like a
        # space, so a title carrying one reads correctly and stops splitting into the words it contains.
        character = " "

    # An odd run means the last backslash was escaping the entity's own `&`, which is syntax rather than
    # content and goes with it. An even run is literal escaped backslashes, which stay.
    kept = backslashes[:-1] if len(backslashes) % 2 else backslashes
    counter[0] += 1
    return kept + bibtex_escape(character)


def decode_html_entities(source: str) -> tuple[str, int]:
    """Decode the HTML character entities in BibTeX text. Returns `(text, how many were decoded)`.

    A database that exports HTML into a `.bib` leaves entities in the field values, where they are not
    content: a title meaning `Q&A` arrives as `Q\\&amp;A` and reads that way in every tool downstream —
    a citation, a word cloud, a typeset bibliography.

    The result is BibTeX rather than plain text, so a decoded character that BibTeX reserves is escaped
    on the way out: `&amp;` becomes `\\&`, and the file stays as readable as it was. `&nbsp;` becomes an
    ordinary space, since a non-breaking one is invisible and stops being a word boundary.

    Everything outside an entity is left byte for byte as it was, this being a substitution over the text
    rather than a parse and rewrite. An entity naming nothing — a stray `&foo;` — is left alone too, and
    does not count: the number returned is how many characters were *decoded*, which is what a caller
    reports to a user, and `re.subn` would have counted it as a substitution.
    """
    counter = [0]
    return _HTML_ENTITY_PATTERN.sub(lambda match: _decode_one_entity(match, counter), source), counter[0]


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
