"""Merge the duplicate records in a bibliography assembled from several databases.

A literature search run across Scopus, Web of Science, ProQuest, Springer and arXiv, then concatenated,
holds every paper once per database that indexes it — each copy in that database's own dialect, with a
different subset of the fields filled in. This tool finds those copies and merges them into one record.

    raven-deduplicate search.bib -o deduped.bib --audit audit.tsv

## What decides that two records are the same paper

Two deterministic keys, unioned transitively, so a record matching one twin by DOI and another by title
brings all three together:

  - **The DOI**, normalized. Equality here is conclusive.
  - **The title**, normalized to the point where two databases' spellings of one title agree.

**DOI equality is evidence; DOI inequality is not.** A paper carrying two different DOIs is common and
usually means something ordinary — a preprint beside its published version, a Zenodo deposit beside the
journal's own, an en-dash where the other has a double hyphen. So two records that match on title **are
merged even when their DOIs differ**, and the audit row lists every DOI in the cluster so the disagreement
is visible to whoever wants to check it.

**The output is meant to be citable, and the input is not.** A concatenation of database exports is not a
document anyone would cite from — it is unusable as it stands, which is why this tool exists — so there is
no provenance in its bytes worth preserving. What is worth preserving is *which records were merged into
which*, and that is what the audit is for.

So the values are read through `raven-fixbib`'s repair before anything else happens: braces escaped,
repeated fields merged, HTML character entities decoded. `Ä` stays `Ä` and LaTeX braces stay where the
source put them, because none of that is broken; `Q\\&amp;A` does not, because it is.

**Normalization is a separate thing and never reaches the output.** Normalized titles and stripped
abstracts exist to decide *which* value to keep. The value kept is then whichever copy won, as `fixbib`
would leave it — so a merge chooses among the records in front of it rather than composing a new one.

`--judge` adds an opt-in LLM pass over what the deterministic keys could not settle — near-miss titles
that no exact key joined. It needs a backend, so it is off by default and everything above works without
one.

## What it does with the records it merges

The surviving record is the most complete copy, preferring the version of record over a preprint or
repository deposit; every field it lacks is filled from a twin that has one. Nothing is dropped silently:
each merge writes a row to the audit TSV naming what was merged away, which key matched, and every value
that differed from the one kept.

**The `.bib` is what you came for**, and the audit is what lets you stand behind it — a scoping review has
to report how many duplicates it removed and answer for the number. Two outputs, and the bibliography is
the one with lasting value; the audit is a record of due diligence.

## Reading

Input is read through `raven.papers.fixbib`'s repair, so a record that `bibtexparser` refuses — one naming
`annote` three times, most often — is still seen, and the HTML a database left in the field values is
decoded. The input file is not modified; use `raven-fixbib` to repair the file itself.
"""

from __future__ import annotations

__all__ = ["normalize_doi", "normalize_title", "is_generic_title",

           "Record", "read_records",

           "Cluster", "cluster_records",

           "fuzzy_candidates", "settled_by_rule", "conflicting_clusters", "judge_batch", "judge_pairs",

           "AUDIT_COLUMNS", "AuditRow", "merge_cluster", "deduplicate", "write_audit",

           "main"]

import argparse
import collections
import dataclasses
import difflib
import json
import logging
import pathlib
import re
import sys
import unicodedata

from bibtexparser import Library
from bibtexparser.middlewares import names
from bibtexparser.model import Entry, Field

from .. import __version__

from ..common import text as textutil

from . import bibtex
from . import config as papers_config
from . import fixbib

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Normalizing a record into match keys
#
# Everything below is a key, never a value: nothing normalized here reaches the output. The regexes are
# implementation and stay in this module; the lists and thresholds a user might reasonably turn are in
# `raven.papers.config`.

# The seven Unicode dashes, folded to ASCII `-` in a DOI. Publishers' exports disagree about which one a
# DOI containing a hyphen should use, and two records whose DOIs differ by an en-dash are one paper.
_DASHES = "‐‑‒–—―−"
_DASH_TABLE = str.maketrans({dash: "-" for dash in _DASHES})

# Everything a database might put in front of the DOI itself.
_DOI_PREFIX_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*|info:doi/)+", re.IGNORECASE)

# A Springer living-reference-work chapter carries its version in the DOI: `..._12-1`, `..._12-2`. The
# suffix is a documented convention rather than a guess, which is why a rule reads it instead of a model.
_CHAPTER_VERSION_PATTERN = re.compile(r"_(\d+)-(\d+)$")

# Markup a database wraps title fragments in. Dropped before the title is reduced, so that a record
# writing `<i>Plasmodium</i>` matches the one writing `Plasmodium` rather than gaining an `i`.
_TAG_PATTERN = re.compile(r"</?[a-z][a-z0-9]{0,9}\s*/?>", re.IGNORECASE)

# `&amp;` and `&#8217;` are how one exporter writes what another writes literally. Resolved to the
# character before reduction, so the two spellings normalize alike.
_ENTITY_PATTERN = re.compile(r"&(#\d{1,6}|#x[0-9a-f]{1,5}|[a-z]{2,8});", re.IGNORECASE)
_NAMED_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " ",
                   "ndash": "-", "mdash": "-", "rsquo": "'", "lsquo": "'", "rdquo": '"', "ldquo": '"'}


def normalize_doi(maybe_raw: str | None) -> str | None:
    """The comparison key for a DOI, or `None` if the value is not one.

    Lowercased, stripped of whatever resolver prefix the exporting database put in front of it, with the
    Unicode dashes folded to ASCII and trailing sentence punctuation removed.

    Returns `None` for anything that does not look like a DOI, which a `doi` field regularly holds — an
    empty string, `n/a`, a publisher's landing-page URL. Those must not become a match key: they are
    equal to each other across unrelated records, and would merge papers that have nothing to do with
    one another.
    """
    if not maybe_raw:
        return None
    value = _DOI_PREFIX_PATTERN.sub("", str(maybe_raw).strip().strip("{}").strip())
    value = value.translate(_DASH_TABLE).lower()
    value = "".join(value.split())  # a DOI has no internal whitespace; a line-wrapped export has some
    value = value.rstrip(".,;:")
    # A DOI is `10.`, a registrant code of four or more digits, a slash, and a non-empty suffix
    # (ISO 26324). No upper bound on the digits, there being none in the standard — the corpus this was
    # built against uses four and five. Anything else in a `doi` field is something other than a DOI,
    # whatever the field is called.
    if not re.match(r"^10\.\d{4,}/\S+$", value):
        return None
    return value


def normalize_title(maybe_raw: str | None) -> str | None:
    """The comparison key for a title, or `None` if there is nothing left of it.

    Reduced hard, because the same title reaches this function in as many spellings as there are
    databases: markup and character entities resolved, compatibility-decomposed, combining marks dropped,
    then everything that is not a letter or a digit removed. `Peer-Reviewed AI: A Study` and
    `Peer reviewed AI - a study` both become `peerreviewedaiastudy`.

    Dropping the spaces along with the punctuation is what makes that work, and it is why this is a key
    and not a display value — the result is not readable and is not meant to be.
    """
    if not maybe_raw:
        return None
    text = str(maybe_raw)
    text = _TAG_PATTERN.sub(" ", text)
    text = _ENTITY_PATTERN.sub(_resolve_entity, text)
    # NFKD splits an accented letter into base + combining mark, so dropping the marks leaves the base:
    # `Ä` and `A` agree, as do `é` and the TeX `\'{e}` once the backslash and braces go. Compatibility
    # decomposition also flattens the ligatures and full-width forms that PDF-derived records carry.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text or None


def _resolve_entity(match: re.Match) -> str:
    """One HTML character entity as its character, or a space if it names nothing known."""
    body = match.group(1)
    if body.startswith("#"):
        try:
            code = int(body[2:], 16) if body[1:2].lower() == "x" else int(body[1:])
        except ValueError:
            return " "
        return chr(code) if 0 < code < 0x110000 else " "
    return _NAMED_ENTITIES.get(body.lower(), " ")


def is_generic_title(maybe_normalized: str | None) -> bool:
    """Whether a normalized title names a genre rather than a work — `Editorial`, `Book Review`.

    A record whose title is generic is still deduplicated. What changes is how much its title is allowed
    to prove on its own: see `_title_edge_holds`.
    """
    return bool(maybe_normalized) and maybe_normalized in papers_config.generic_titles


# --------------------------------------------------------------------------------
# Reading a bibliography into records

@dataclasses.dataclass(frozen=True)
class Record:
    """One bibliography record, with the derived values clustering and merging read.

    Frozen and a dataclass rather than this codebase's usual `env`, for `fixbib.RepairReport`'s reason:
    these are counted, sorted and grouped, and a mistyped attribute should fail rather than arrive as
    `None` and quietly change a tally.

    Fields:

    `index`: Position in the input, counting every input file as one stream. Decides output order, and
             breaks every tie in base selection, so a run is reproducible.
    `entry`: The `bibtexparser` `Entry`, read without name splitting so that it can be written back out.
    `key`: The record's BibTeX key.
    `doi`: Normalized DOI, or `None`. A match key.
    `title`: Normalized title, or `None` where the record has none. A match key, subject to
             `_title_edge_holds`.
    `display_title`: The title as the file has it, for the audit and for the judge's prompt.
    `year`: Publication year as an integer, or `None`. Blocks the fuzzy pass; never a match key on its
            own, since two databases can disagree about the year of one paper.
    `surname`: Normalized first-author surname, or `None`. Blocks the fuzzy pass.
    `chapter_version`: Springer living-reference chapter version from the DOI, or `None`.
    `is_preprint`: Whether the DOI names a preprint server or a general repository.
    """
    index: int
    entry: Entry
    key: str
    doi: str | None
    title: str | None
    display_title: str
    year: int | None
    surname: str | None
    chapter_version: int | None
    is_preprint: bool

    def field(self, name: str) -> str | None:
        """The value of field `name`, or `None` if the record does not have one worth reading."""
        return _field_value(self.entry, name)


def _field_value(entry: Entry, name: str) -> str | None:
    """The text of `entry`'s field `name`, or `None` when it is absent or empty."""
    try:
        value = entry[name]
    except KeyError:
        return None
    if not isinstance(value, str):  # a library read with name splitting; not what this module reads
        return None
    return value.strip() or None


def _first_surname(maybe_author: str | None) -> str | None:
    """The first author's surname, reduced the way `normalize_title` reduces a title.

    `bibtexparser`'s own name splitter does the work, called directly as a function — this module reads
    without the name-splitting middleware, but the middleware is only a wrapper around
    `parse_single_name_into_parts`, so nothing has to be parsed twice or copied to get at it.

    The particle goes with the surname (`van Beethoven`, not `Beethoven`), which is what makes the two
    orders agree: `van Beethoven, Ludwig` and `Ludwig van Beethoven` are one person and must land in one
    blocking key.

    **Two records can still block apart, and where they do the ambiguity is in the file rather than here.**
    Written without a comma, `A B C` could be two given names and a surname or one given name and a
    compound surname, and nothing in the string says which. BibTeX resolves it by rule — only the last
    token is the surname, absent a lowercase particle — so `Petra Johanna Lagerkvist` comes out right and
    `Aksel Holm Dahl` comes out as `last=[Dahl]`, which is wrong if the surname was meant to be "Holm
    Dahl". The comma form is how a file says which it meant, and `Holm Dahl, Aksel` blocks differently
    from the same name written without it.

    Following the rule is right anyway: it is correct for the common case, and the alternative is guessing
    against the format. The cost is a blocking miss, which the fuzzy pass can afford.

    The suffix has the same shape. `A. B. Fenwick, Jr.` has one comma, so it is `Last, First` and the
    surname reads as "A. B. Fenwick"; the intended name needs the two-comma form, `Fenwick, Jr., A. B.`.

    Falls back to reading up to the first comma when the splitter refuses the name outright, which it does
    for `Bloggs, PhD, MSc, Joan` — three commas where BibTeX allows two. That record is still a paper and
    still worth blocking, and giving up on it here would silently exclude it from the fuzzy pass.
    """
    if not maybe_author:
        return None
    try:
        first = names.split_multiple_persons_names(maybe_author.strip())[0]
    except Exception:  # noqa: BLE001 -- an author list this cannot split is a blocking miss, not an error
        first = re.split(r"\s+and\s+", maybe_author.strip(), maxsplit=1)[0]
    first = first.strip().strip("{}")
    if not first:
        return None
    try:
        parts = names.parse_single_name_into_parts(first)
        surname = " ".join(parts.von + parts.last)
    except Exception:  # noqa: BLE001 -- see the fallback in the docstring
        surname = first.split(",", 1)[0]
    return normalize_title(surname)


def _year_of(maybe_year: str | None) -> int | None:
    """The four-digit year in a `year` field, or `None`. Exports write `2024`, `2024-06`, and `c2024`."""
    if not maybe_year:
        return None
    match = re.search(r"(1[6-9]\d{2}|20\d{2}|21\d{2})", maybe_year)
    return int(match.group()) if match else None


def _make_record(index: int, entry: Entry) -> Record:
    """Derive a `Record` from one parsed entry."""
    doi = normalize_doi(_field_value(entry, "doi"))
    if doi is None:
        # arXiv records often carry the identifier and no DOI. The registered form is derivable and is
        # what the published copy's own `doi` field will say if it has one, so deriving it here is what
        # lets a preprint match its twin at all.
        eprint = _field_value(entry, "eprint")
        if eprint and _field_value(entry, "archiveprefix"):
            doi = normalize_doi(f"10.48550/arXiv.{eprint.strip()}")

    version_match = _CHAPTER_VERSION_PATTERN.search(doi) if doi else None
    return Record(index=index,
                  entry=entry,
                  key=entry.key,
                  doi=doi,
                  title=normalize_title(_field_value(entry, "title")),
                  display_title=_field_value(entry, "title") or "",
                  year=_year_of(_field_value(entry, "year")),
                  surname=_first_surname(_field_value(entry, "author")),
                  chapter_version=int(version_match.group(2)) if version_match else None,
                  is_preprint=doi is not None and doi.startswith(papers_config.preprint_doi_prefixes))


def read_records(source: str) -> tuple[list[Record], list[fixbib.RepairReport]]:
    """Read BibTeX text into `Record`s, repairing what the parser would otherwise refuse.

    Returns `(records, unreadable)`, the second a list of `fixbib.RepairReport` naming the records that
    could not be read even after repair, so a caller can report how much of the input it is speaking for.

    The repair is **all of what `raven-fixbib` does** — the entity decoding as much as the structural
    rescue — applied to a copy, so the caller's file is untouched. `raven-fixbib` remains the tool for
    repairing the file itself, and running it first changes nothing about the result.
    """
    source, _decoded = bibtex.decode_html_entities(source)
    repaired, _recovered, unrecovered = fixbib.repair_bibtex(source)
    # Read without name splitting, which serves two ends at once: the library can be written back out
    # (see `bibtex.write_string`), and a record whose author BibTeX cannot express — `Bloggs, PhD, MSc,
    # Joan`, three commas where the format allows two — is read rather than refused. Merging such a
    # record is unaffected by the fault, since its author string is only ever copied.
    library = bibtex.parse_string(repaired, split_names=False)
    records = [_make_record(index, entry) for index, entry in enumerate(library.entries)]
    return records, [report for report in unrecovered
                     if report.kind != fixbib.KIND_UNREADABLE or _still_missing(report, library)]


def _still_missing(report: fixbib.RepairReport, library: Library) -> bool:
    """Whether a record `fixbib` could not repair is genuinely absent from `library`.

    `fixbib` parses with name splitting and this module does not, so a record refused only for an
    unsplittable author is reported by `fixbib` and read here perfectly well. Reporting it as lost would
    be a lie about the corpus, in the direction that makes a coverage figure look worse than it is.
    """
    return not any(entry.key == report.key for entry in library.entries)


# --------------------------------------------------------------------------------
# Deterministic clustering
#
# No model involved anywhere below: union-find over exact normalized DOI and exact normalized title, with
# `_title_edge_holds` deciding when a title match is allowed to carry a merge on its own.

@dataclasses.dataclass(frozen=True)
class Cluster:
    """A set of records taken to be one paper, and how they came to be one.

    `records`: In merge preference order, so `records[0]` is the base. See `_merge_rank`.
    `rules`: Which keys joined this cluster — `"doi"`, `"title"`, `"judge"`. More than one is ordinary:
             a cluster of three can be held together by a DOI at one end and a title at the other.
    """
    records: tuple[Record, ...]
    rules: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.records)


def _merge_rank(record: Record) -> tuple:
    """Sort key putting the record that should be the base of a merge first.

    In order: the version of record beats a preprint or repository deposit; a higher Springer chapter
    version beats a lower one; more fields beats fewer; and the earlier record in the input breaks the
    remaining ties, so that a re-run picks the same base.

    Publication status leads deliberately. Ranking by field count first usually picks the same record and
    sometimes picks the deposit, which leaves a Zenodo DOI on a paper that has a journal one — wrong in a
    bibliography in a way that is invisible once written.
    """
    return (record.is_preprint,
            -(record.chapter_version or 0),
            -len(record.entry.fields),
            record.index)


def _disagree_on_author(a: Record, b: Record) -> bool:
    """Whether both records name a first author and the two are different people.

    Absence is never disagreement: a record with no `author` field agrees with everything, because the
    alternative is treating a database that omits authors as evidence against a merge.
    """
    return bool(a.surname and b.surname and a.surname != b.surname)


def _disagree_on_year(a: Record, b: Record) -> bool:
    """Whether both records give a year and the two are more than `papers_config.max_year_drift` apart."""
    return bool(a.year and b.year and abs(a.year - b.year) > papers_config.max_year_drift)


def _disagree_on_identity(a: Record, b: Record) -> bool:
    """Whether any of `config.identifying_fields` says these are different items.

    Used where the title cannot carry a merge by itself. Only a *positive* disagreement counts: a field
    one record has and the other lacks says nothing, since two databases export different subsets of one
    record, and demanding agreement would refuse nearly every genuine pair.

    Values are reduced the way titles are, so `101--103` and `101-103` are one page range rather than two.
    The DOI is taken from the record's already-normalized form, which additionally sees past a resolver
    prefix and a stray en-dash.
    """
    if a.doi is not None and b.doi is not None and a.doi != b.doi:
        return True
    for name in papers_config.identifying_fields:
        if name == "doi":
            continue  # handled above, and better, by the normalized form
        x, y = normalize_title(a.field(name)), normalize_title(b.field(name))
        if x is not None and y is not None and x != y:
            return True
    return False


def _title_edge_holds(a: Record, b: Record) -> bool:
    """Whether two records sharing a normalized title may be merged on that evidence.

    Not every equal title means one paper, so a title match is weighed against what else the two records
    say — and how much weighing it needs depends on how much else there is:

      - **A generic title** — `Editorial`, `Book Review` — proves nothing by itself, so it is accepted
        only where the records positively *agree*: the same first author, and years within
        `config.max_year_drift`. Silence is not agreement here; a record with no author is not merged
        into another `Editorial` on the strength of the word.

        **Agreement on author and year is not sufficient either**, because one person writes several book
        reviews in a year and every one of them is titled `Book Review`. So the pair must additionally not
        contradict itself on `config.identifying_fields` — a different DOI, page range or issue means
        different items. (Raised by Juha, 2026-08-29; the corpus happens to contain no such pair, which is
        exactly why counting merges could not have found it.)
      - **Where neither record names an author**, the title is the only evidence there is, so the same
        identifying-field disagreement is enough to overrule it.
      - **Otherwise** the title is distinctive enough to carry a merge on its own, and is refused only
        where the records positively contradict each other: a different first author *and* a year too far
        apart to be one paper appearing twice. Both are required. Databases disagree about author order
        often enough that a surname mismatch alone is weak evidence, and a preprint and its published
        version straddling a New Year make a year gap alone weaker still. Together they are conclusive.

    The authorless clause came from a corpus rather than from first principles, and it is worth knowing
    what it catches. A serial's recurring section headings: `II Political Science: Method and Theory` and
    `Abstracts Abstracts` head an item in every issue of their journals, carry no author, and are ordinary
    titles by every other test — so the title alone merged four issues into one record.

    And, once the check widened past the DOI, **multi-volume conference proceedings**. `23rd International
    Conference on Artificial Intelligence in Education, AIED 2022` is the title of both LNCS 13355 and
    LNCS 13356 — Parts I and II, two different books with one name, no authors and no DOIs between them.
    Five such pairs were merging in that corpus, each collapsing a whole volume out of the bibliography,
    and nothing but `volume` distinguishes them.

    That last case is also why the earlier account here was wrong. It read: of 36 authorless merges, the
    three with disagreeing DOIs were the three that were wrong and the other 33 were all right. Five of
    the 33 were not, and no DOI could have shown it.

    Applied per pair rather than per title, which is what lets four records carrying one heading come
    apart into the two pairs that are each one item.
    """
    if is_generic_title(a.title):
        return (a.surname is not None and a.surname == b.surname
                and a.year is not None and b.year is not None
                and abs(a.year - b.year) <= papers_config.max_year_drift
                and not _disagree_on_identity(a, b))
    if a.surname is None and b.surname is None:
        return not _disagree_on_identity(a, b)
    return not (_disagree_on_author(a, b) and _disagree_on_year(a, b))


def cluster_records(records: list[Record],
                    maybe_judgements: dict[tuple[int, int], bool] | None = None) -> list[Cluster]:
    """Group `records` into clusters of one paper each, by exact DOI and exact normalized title.

    `maybe_judgements`: the judge's verdicts, keyed by a pair of record indices in ascending order, or
                        `None` when the judge did not run. True adds an edge the exact keys missed; False
                        withdraws a *title* edge between that pair, which is how a rejected merge comes
                        apart. A DOI match is never withdrawn — equality there is conclusive, and the
                        judge is not asked to overrule it.

    Every record comes back in exactly one cluster, singletons included, ordered by where the cluster's
    base sat in the input — so the output of a whole run stays in reading order.

    Matching is transitive: a record sharing a DOI with one twin and a title with another puts all three
    in one cluster. That is what makes the two keys complementary rather than two passes, since neither
    key is present on every record. A title match must additionally satisfy `_title_edge_holds`.
    """
    judgements = maybe_judgements or {}
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    edges = []  # (index, index, rule), kept so the audit can say which key matched

    doi_groups = collections.defaultdict(list)
    for index, record in enumerate(records):
        if record.doi is not None:
            doi_groups[record.doi].append(index)
    for indices in doi_groups.values():
        for other in indices[1:]:
            edges.append((indices[0], other, "doi"))
            union(indices[0], other)

    title_groups = collections.defaultdict(list)
    for index, record in enumerate(records):
        if record.title is not None:
            title_groups[record.title].append(index)
    for indices in title_groups.values():
        # Every pair, not a chain from the first: an edge that `_title_edge_holds` refuses must not take
        # the rest of the group down with it, and which record happens to be first is an accident of the
        # input order. Groups are small — the largest in a 6934-record corpus holds six.
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                if judgements.get((i, j)) is False:
                    continue  # the judge looked at this one and said the two are different works
                if _title_edge_holds(records[i], records[j]):
                    edges.append((i, j, "title"))
                    union(i, j)

    for (i, j), same in judgements.items():
        if same:
            edges.append((i, j, "judge"))
            union(i, j)

    return _clusters_from(records, find, edges)


def _clusters_from(records: list[Record], find, edges: list[tuple[int, int, str]]) -> list[Cluster]:
    """Assemble `Cluster` objects once the union-find is settled."""
    members = collections.defaultdict(list)
    for index in range(len(records)):
        members[find(index)].append(index)

    rules = collections.defaultdict(set)
    for i, _j, rule in edges:
        rules[find(i)].add(rule)

    clusters = []
    for root, indices in members.items():
        ordered = tuple(sorted((records[i] for i in indices), key=_merge_rank))
        clusters.append(Cluster(records=ordered, rules=tuple(sorted(rules[root]))))
    return sorted(clusters, key=lambda cluster: cluster.records[0].index)


# --------------------------------------------------------------------------------
# The LLM judge (opt-in, `--judge`)
#
# It sees only what the deterministic pass could not settle, it proposes rather than decides — every "same
# work" verdict must still clear `_judge_admits` in Python — and it is never asked a question a rule
# already answers; see `settled_by_rule`.

def _describe_for_judge(record: Record) -> str:
    """One record as the few lines the judge is asked to read.

    Deliberately narrow. Everything here is something a bibliography record states about itself; the
    abstract is left out because two databases' copies of one abstract differ in ways that have nothing to
    do with whether the papers are the same, and it would be most of the prompt.
    """
    parts = [f"title: {record.display_title or '(none)'}"]
    for label, name in (("authors", "author"), ("year", "year"), ("venue", "journal"),
                        ("booktitle", "booktitle"), ("publisher", "publisher"), ("doi", "doi")):
        value = record.field(name)
        if value:
            parts.append(f"{label}: {_tsv_cell(value)[:200]}")
    return "\n".join(f"    {part}" for part in parts)


def _title_similarity(a: str, b: str) -> float:
    """How alike two normalized titles are, as `difflib.SequenceMatcher.ratio()`.

    That is `2M/T` — matching characters over the combined length of both strings — so 0 means nothing in
    common and 1 means identical. **Character overlap, not meaning**: no embeddings and no semantics, so
    two titles about one subject in different words score low, while two spellings of one title score
    high. The second is the question being asked here.

    Returns 0.0 rather than the true ratio for a pair that cannot reach `config.title_similarity`, since
    the caller only ever compares against that threshold. `difflib`'s two cheap upper bounds do that
    rejection in O(len) where `ratio` is O(len²), which is what keeps a quadratic pass over a blocking
    group affordable.
    """
    threshold = papers_config.title_similarity
    matcher = difflib.SequenceMatcher(None, a, b)
    if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
        return 0.0
    return matcher.ratio()


def fuzzy_candidates(records: list[Record], clusters: list[Cluster]) -> list[tuple[Record, Record]]:
    """Pairs of records that look like one paper but that no exact key joined.

    Blocked by first-author surname and by year within `papers_config.max_year_drift`, then filtered by
    `papers_config.title_similarity`. Blocking is what makes this tractable — comparing all pairs of a 7000-record
    corpus is 24 million comparisons, and comparing within a surname is a few thousand.

    The cost of blocking is what it cannot see: a record with no author or no year is in no block, and
    two databases spelling the first author differently put one paper in two blocks. Both are missed
    merges, which is the direction to fail in — the deterministic keys have already found everything that
    agrees exactly, so what is left here is a bonus rather than a floor.
    """
    cluster_of = {}
    for position, cluster in enumerate(clusters):
        for record in cluster.records:
            cluster_of[record.index] = position

    blocks = collections.defaultdict(list)
    for record in records:
        if record.surname and record.year and record.title:
            for year in range(record.year - papers_config.max_year_drift, record.year + papers_config.max_year_drift + 1):
                blocks[(record.surname, year)].append(record)

    seen, candidates = set(), []
    for members in blocks.values():
        for position, a in enumerate(members):
            for b in members[position + 1:]:
                if cluster_of[a.index] == cluster_of[b.index]:
                    continue  # already one paper; nothing for the judge to add
                pair = (min(a.index, b.index), max(a.index, b.index))
                if pair in seen:
                    continue  # a pair reachable from two adjacent year blocks is still one pair
                seen.add(pair)
                if _title_similarity(a.title, b.title) >= papers_config.title_similarity:
                    candidates.append((a, b) if a.index < b.index else (b, a))
    return sorted(candidates, key=lambda pair: (pair[0].index, pair[1].index))


def settled_by_rule(a: Record, b: Record) -> bool:
    """Whether a DOI disagreement between two records is one the project has already decided.

    True where either DOI carries a Springer living-reference chapter version — `..._12-1` against
    `..._12-2`, or a versioned chapter against the book chapter it became. Those are one work, the higher
    version preferred, which is the same thing `raven.papers.utils.deduplicate_arxiv_ids` does with arXiv
    versions; the suffix is a documented convention rather than an inference.

    Used to keep such a pair away from the judge, which is a guard and not a saving. Measured 2026-08-28:
    shown four version pairs, Qwen3.6 refused all four, reasoning that "different DOI suffixes indicate
    separate chapters" — fluent, plausible, and wrong in the way a documented convention is invisible to
    a reader seeing the string cold. Acting on that would have split four works into eight.

    The general form is worth keeping in view when adding to what the judge sees: a rule that encodes a
    convention is not a rule a model can be expected to rediscover from the data, so a question already
    answered by one should not be asked.
    """
    return a.chapter_version is not None or b.chapter_version is not None


def conflicting_clusters(clusters: list[Cluster]) -> list[Cluster]:
    """Merged clusters whose records do not agree on a DOI.

    Not an error — a preprint beside its published version, or a Zenodo deposit beside the journal's own,
    is the usual reason — but it is the one shape of merge that a title match made on its own where the
    records had a way to contradict it. Worth a second opinion when one is available.
    """
    return [cluster for cluster in clusters
            if len(cluster) > 1 and len({record.doi for record in cluster.records if record.doi}) > 1]


def _parse_json_payload(text: str):
    """The JSON in a model reply, tolerating code fences and stray prose around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in reply: {text[:200]!r}")


def _ask_judge(llm_settings, prompt: str) -> str:
    """One stateless turn: no character, no tools, no retrieval, no history."""
    from ..librarian import agent
    record = agent.turn(llm_settings,
                        prompt,
                        use_character_card=False,
                        tools_enabled=False,
                        internet_enabled=False,
                        docs_enabled=False,
                        markup=None)
    if record.generation is None:
        raise RuntimeError("the backend returned no generation")
    return record.reply or ""


def judge_batch(llm_settings, batch: list[tuple[str, str, str]]) -> dict[int, dict]:
    """Ask about one batch of `(pair id, description, description)`. Returns `{position: answer}`.

    Answers whose index does not resolve are dropped rather than trusted, and a batch that comes back
    short simply leaves those pairs unanswered — which is also what makes a re-run the recovery path for
    a failed batch. Following `investigations/agent-batch-classification/`, where a batch of forty did
    come back with thirty-nine.
    """
    items = "\n\n".join(f"{position}.\n  RECORD A:\n{a}\n  RECORD B:\n{b}"
                        for position, (_pair_id, a, b) in enumerate(batch))
    answers = _parse_json_payload(_ask_judge(llm_settings, papers_config.judge_instructions.format(items=items)))
    if not isinstance(answers, list):
        raise ValueError(f"expected a JSON array, got {type(answers).__name__}")

    resolved = {}
    for answer in answers:
        if not isinstance(answer, dict) or "i" not in answer:
            continue
        try:
            position = int(answer["i"])
        except (TypeError, ValueError):
            continue
        if 0 <= position < len(batch):
            resolved[position] = {"same": answer.get("same") is True,
                                  "why": str(answer.get("why") or "").strip()}
    return resolved


def _judge_admits(a: Record, b: Record) -> bool:
    """Whether a "same work" verdict may be acted on, decided in Python rather than by the model.

    The one durable lesson of `investigations/agent-batch-classification/`: a model's own confidence must
    never be the thing that decides whether to look harder, because the case where the self-report is
    wrong is exactly the case that then goes unexamined. Here the same idea makes the model a proposer and
    not a decider — it may suggest a merge the exact keys missed, and a suggestion contradicted by what
    the records themselves say is dropped whatever the model said about it.

    The condition is the one an ordinary title match already has to clear: a different first author *and*
    a year too far apart cannot be one paper.
    """
    return not (_disagree_on_author(a, b) and _disagree_on_year(a, b))


def _load_judge_state(path: pathlib.Path) -> dict[str, dict]:
    """Answers already recorded, keyed by pair id."""
    state = {}
    if not path.exists():
        return state
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            answer = json.loads(line)
        except json.JSONDecodeError:
            continue  # a run killed mid-write leaves a partial last line; the pair is simply re-asked
        if isinstance(answer, dict) and "pair" in answer:
            state[answer["pair"]] = answer
    return state


def _pair_id(a: Record, b: Record) -> str:
    """A stable name for a pair, so a resumed run recognizes what it already asked.

    Built from the BibTeX keys rather than from the record indices, which shift the moment the input
    files are given in a different order — and a resumable file that silently answers a different
    question after an argument is reordered would be worse than no resumability at all.
    """
    return "\t".join(sorted((a.key, b.key)))


def judge_pairs(llm_settings,
                pairs: list[tuple[Record, Record, str]],
                maybe_state_path: pathlib.Path | None = None,
                on_progress=None) -> dict[tuple[int, int], bool]:
    """Ask the judge about `pairs`, returning the verdicts clustering can act on.

    `pairs`: `(record, record, why_asked)`, the last naming which question this pair came from —
             `"fuzzy"` for a near-miss the exact keys did not join, `"conflict"` for a merge whose
             records disagree about the DOI.

    `maybe_state_path`: a JSONL appended to as answers arrive. A re-run skips what is already there, so
                        a backend that falls over two-thirds of the way through costs the batch in
                        flight rather than the run.

    A verdict of "same" is kept only where `_judge_admits` agrees, which is where this stops being a
    model deciding and becomes a model proposing. A pair the model does not answer at all is simply
    absent from the result and stays unmerged.
    """
    done = _load_judge_state(maybe_state_path) if maybe_state_path else {}
    todo = [(a, b, why) for a, b, why in pairs if _pair_id(a, b) not in done]

    for start in range(0, len(todo), papers_config.judge_batch):
        chunk = todo[start:start + papers_config.judge_batch]
        batch = [(_pair_id(a, b), _describe_for_judge(a), _describe_for_judge(b)) for a, b, _why in chunk]
        try:
            answers = judge_batch(llm_settings, batch)
        except Exception as exc:  # noqa: BLE001 -- a failed batch is a result: re-running is the retry
            logger.warning(f"judge_pairs: batch at {start} failed, {type(exc)}: {exc}")
            answers = {}
        for position, (a, b, why) in enumerate(chunk):
            if position not in answers:
                continue
            answer = dict(answers[position], pair=_pair_id(a, b), asked=why,
                          keys=[a.key, b.key], titles=[a.display_title, b.display_title])
            done[answer["pair"]] = answer
            if maybe_state_path is not None:
                with maybe_state_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
        if on_progress is not None:
            on_progress(min(start + papers_config.judge_batch, len(todo)), len(todo))

    verdicts = {}
    for a, b, _why in pairs:
        answer = done.get(_pair_id(a, b))
        if answer is None:
            continue
        same = bool(answer.get("same")) and _judge_admits(a, b)
        verdicts[(min(a.index, b.index), max(a.index, b.index))] = same
    return verdicts


def _judge_state_path(opts) -> pathlib.Path | None:
    """Where to keep the judge's answers, or `None` if there is nowhere sensible.

    `--judge-state` if given; otherwise beside whichever output the run is producing, so that a re-run
    resumes without the user having had to think about it in advance — which is the only moment at which
    thinking about it would have helped. A run writing nothing at all keeps nothing.
    """
    if opts.judge_state:
        return pathlib.Path(opts.judge_state).expanduser().resolve()
    beside = opts.output or opts.audit
    if not beside:
        return None
    path = pathlib.Path(beside).expanduser().resolve()
    return path.with_name(f"{path.stem}_judge.jsonl")


def _apply_judge(records: list[Record], clusters: list[Cluster], opts) -> list[Cluster]:
    """Run the opt-in judge pass and return the re-clustered result. CLI glue for `judge_pairs`."""
    from ..librarian import config as librarian_config, llmclient

    fuzzy = fuzzy_candidates(records, clusters)
    conflicts, settled = [], 0
    for cluster in conflicting_clusters(clusters):
        base = cluster.records[0]
        for other in cluster.records[1:]:
            if not (other.doi and base.doi and other.doi != base.doi):
                continue
            if settled_by_rule(base, other):
                settled += 1
                continue
            conflicts.append((base, other, "conflict"))
    pairs = [(a, b, "fuzzy") for a, b in fuzzy] + conflicts
    print(f"judge: {len(fuzzy)} near-miss pair(s), {len(conflicts)} DOI-disagreement pair(s)"
          + (f", {settled} settled by rule and not asked about" if settled else ""))
    if not pairs:
        return clusters

    backend_url = opts.backend_url or librarian_config.llm_backend_url
    # Stop here rather than at the first batch: this run can take a while, and a precise diagnosis now
    # beats the same failure once per batch for the rest of the corpus. Reachable and
    # reachable-with-a-model are separate questions, and the second is the one that reads as a bug when
    # it is not checked — the backend answers, so nothing looks wrong until every verdict is empty.
    if not llmclient.test_connection(backend_url):
        print(f"judge: cannot reach an LLM backend at {backend_url}.", file=sys.stderr)
        sys.exit(1)
    llm_settings = llmclient.setup(backend_url=backend_url, quiet=True)
    if (status := llmclient.backend_status(llm_settings)) is llmclient.backend_has_no_model:
        headline, advice = llmclient.describe_backend_status(status, backend_url)
        print(f"judge: {headline} {advice}", file=sys.stderr)
        sys.exit(1)
    print(f"judge: {llm_settings.model} at {backend_url}")

    state_path = _judge_state_path(opts)
    verdicts = judge_pairs(llm_settings, pairs, state_path,
                           on_progress=lambda done, total: print(f"  judged {done}/{total}", flush=True))

    was_together = {}
    for position, cluster in enumerate(clusters):
        for record in cluster.records:
            was_together[record.index] = position
    merges = sum(1 for (i, j), same in verdicts.items()
                 if same and was_together[i] != was_together[j])
    splits = sum(1 for same in verdicts.values() if not same)
    print(f"judge: {merges} pair(s) newly merged, {splits} pair(s) refused")
    if state_path is not None:
        print(f"judge: answers in {state_path}")
    return cluster_records(records, verdicts)


# --------------------------------------------------------------------------------
# Merging a cluster, and the audit trail that accounts for it

# The TSV schema. Not a knob despite being public: `AuditRow.to_row` produces these cells in this order,
# so the two change together or not at all. A caller reading the audit wants the names, which is why it is
# exported.
AUDIT_COLUMNS = ("kept", "removed", "matched_by", "size", "title", "dois", "differences")


@dataclasses.dataclass(frozen=True)
class AuditRow:
    """What one merge did, in the form the audit TSV records it.

    A scoping review reports the number of duplicates it removed and has to be able to stand behind it,
    so this is the tool's real output — enough per merge for a reader to disagree with it.

    Fields:

    `kept`: BibTeX key of the surviving record.
    `removed`: Keys of the records merged into it, in merge preference order.
    `matched_by`: Which keys joined the cluster.
    `size`: How many records the cluster held.
    `title`: The surviving record's title, as written.
    `dois`: Every distinct normalized DOI in the cluster. More than one is a disagreement worth seeing,
            not an error — see the module docstring.
    `differences`: Every field where a merged-away record held a different non-empty value than the one
                   kept, as `field: kept … / dropped …`. Values are truncated; this says what was
                   dropped, and the input file remains the place to read it in full.

                   **Abstracts are compared with their rights notices removed**, so two copies of one
                   abstract carrying two publishers' copyright lines are not reported as a difference.
                   They are the overwhelmingly common case — a database's notice is usually the *only*
                   thing separating its copy from another's — and reporting each one would bury the
                   differences that are about the paper under hundreds that are about the exporter.
                   Nothing else is compared that way, and nothing else is exempt.
    """
    kept: str
    removed: tuple[str, ...]
    matched_by: tuple[str, ...]
    size: int
    title: str
    dois: tuple[str, ...]
    differences: tuple[str, ...]

    def to_row(self) -> tuple[str, ...]:
        """The cells of this row, in `AUDIT_COLUMNS` order, safe to join with tabs."""
        return tuple(_tsv_cell(cell) for cell in (self.kept,
                                                  "; ".join(self.removed),
                                                  "+".join(self.matched_by),
                                                  str(self.size),
                                                  self.title,
                                                  "; ".join(self.dois),
                                                  " | ".join(self.differences)))


def _tsv_cell(value: str) -> str:
    """One TSV cell: no tabs, no newlines, no carriage returns, since those end a cell or a row."""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clip(value: str) -> str:
    """`value` shortened to something an audit row can carry."""
    value = _tsv_cell(value)
    return value if len(value) <= papers_config.audit_value_chars else value[:papers_config.audit_value_chars - 1] + "…"


def _best_abstract(cluster: Cluster) -> tuple[str | None, list[str]]:
    """The abstract to keep, as the source wrote it, and the ones that lost while saying something else.

    Candidates are *compared* through `text.strip_boilerplate`, and that is what makes "keep the longest"
    the right rule rather than a coin toss. Databases append their own rights notice to the abstracts they
    export, so on raw text the longest copy is usually just the one carrying the most boilerplate, and
    taking it would pick a record for the size of its copyright line. Stripped, most of these copies
    become the same string and there is nothing left to choose.

    What remains after that is either a database's truncation of the same abstract — where the longest is
    plainly right — or a handful of genuinely different texts, where it is the least bad rule and the
    audit records what it passed over.
    """
    # Stripping decides *which* abstract; it does not edit the one that wins. Writing the stripped text
    # would leave the output carrying two kinds of abstract — trimmed where a record happened to have a
    # twin, untouched where it did not — and would be a content edit in a tool whose whole promise is
    # that the values it writes are the ones it read. Removing a rights notice from a `.bib` is
    # `raven-fixbib`'s kind of job, and the Visualizer's importer strips at read time regardless.
    candidates = []
    for record in cluster.records:
        raw = record.field("abstract")
        if raw:
            stripped = textutil.strip_boilerplate(raw)
            if stripped:
                candidates.append((record, raw, stripped))
    if not candidates:
        return None, []
    best_record, best_raw, best_stripped = max(candidates,
                                               key=lambda triple: (len(triple[2]), -triple[0].index))
    losers = [raw for record, raw, stripped in candidates
              if record is not best_record and stripped != best_stripped]
    return best_raw, losers


def merge_cluster(cluster: Cluster) -> tuple[Entry, AuditRow | None]:
    """Merge one cluster into a single entry, and say what that cost.

    Returns `(entry, maybe_audit_row)`. A cluster of one is returned unchanged with no audit row, since
    nothing happened to it.

    The base is `cluster.records[0]`; every field it lacks is filled from the first twin that has one, in
    the same preference order. A field the base already has is never overwritten, with two exceptions:

      - **the abstract**, chosen across the whole cluster by `_best_abstract`, because the base being the
        most complete record does not make its abstract the least truncated one;
      - **`copyright`**, unioned across the cluster rather than chosen, since a merged record came from
        several exports and each notice names one of them.

    The entry that comes back shares no `Field` objects with the input, so a caller may write it out
    without the merge having disturbed the records it drew from.
    """
    base = cluster.records[0]
    if len(cluster.records) == 1:
        return base.entry, None

    fields: dict[str, str] = {}
    for record in cluster.records:
        for field in record.entry.fields:
            value = _field_value(record.entry, field.key)
            if value is not None and field.key not in fields:
                fields[field.key] = value

    maybe_abstract, abstract_losers = _best_abstract(cluster)
    if maybe_abstract is not None:
        fields["abstract"] = maybe_abstract

    # `copyright` is unioned rather than chosen, because a merged record genuinely came from several
    # exports and each notice names one of them — which is most of what a rights notice is worth here,
    # since nobody redistributes a bibliography pulled out of a paywalled aggregator. Picking one would
    # throw away the only thing saying where the other copy came from. Joined the way
    # `bibtex.repair_duplicate_field_keys` joins repeated fields, for the same reason and so that the two
    # read alike.
    notices = list(dict.fromkeys(value for record in cluster.records
                                 if (value := record.field(papers_config.rights_field)) is not None))
    if notices:
        fields[papers_config.rights_field] = "\n".join(notices)

    # Field order follows the base's own, so a merged record still reads like the record it came from,
    # with whatever was filled in from its twins after it.
    base_order = [field.key for field in base.entry.fields]
    ordered = base_order + sorted(key for key in fields if key not in base_order)
    merged = Entry(entry_type=base.entry.entry_type,
                   key=base.key,
                   fields=[Field(key, fields[key]) for key in ordered if key in fields])

    differences = [f"abstract: kept {_clip(maybe_abstract or '')} / dropped {_clip(text)}"
                   for text in abstract_losers]
    for record in cluster.records[1:]:
        for field in record.entry.fields:
            value = _field_value(record.entry, field.key)
            kept = fields.get(field.key)
            if (value is not None and kept is not None and value != kept
                    and field.key not in ("abstract", papers_config.rights_field)):
                differences.append(f"{field.key}: kept {_clip(kept)} / dropped {_clip(value)}")

    dois = sorted({record.doi for record in cluster.records if record.doi})
    row = AuditRow(kept=base.key,
                   removed=tuple(record.key for record in cluster.records[1:]),
                   matched_by=cluster.rules,
                   size=len(cluster.records),
                   title=_tsv_cell(base.display_title),
                   dois=tuple(dois),
                   differences=tuple(differences))
    return merged, row


def deduplicate(clusters: list[Cluster]) -> tuple[Library, list[AuditRow]]:
    """Merge every cluster, returning the deduplicated library and the audit rows for what merged.

    One entry per cluster, in cluster order, so the output is in the reading order of the input. A cluster
    of one contributes no audit row: nothing happened to it.
    """
    library = Library()
    rows = []
    for cluster in clusters:
        entry, maybe_row = merge_cluster(cluster)
        library.add(entry)
        if maybe_row is not None:
            rows.append(maybe_row)
    return library, rows


def write_audit(path: pathlib.Path, rows: list[AuditRow], sources: list[str]) -> None:
    """Write the audit TSV, preceded by comment lines naming the tool version and the inputs.

    The version stamp is what makes the file citable: a method section says which tool produced these
    numbers, and "the script said so" is not a method section.
    """
    lines = [f"# raven-deduplicate {__version__}",
             f"# input: {'; '.join(sources)}",
             f"# clusters merged: {len(rows)}",
             f"# records removed: {sum(len(row.removed) for row in rows)}",
             "\t".join(AUDIT_COLUMNS)]
    lines += ["\t".join(row.to_row()) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(records: list[Record],
            clusters: list[Cluster],
            unreadable: list[fixbib.RepairReport]) -> None:
    """Print what the run found, in the shape a method section asks for."""
    merged = [cluster for cluster in clusters if len(cluster) > 1]
    removed = sum(len(cluster) - 1 for cluster in merged)
    print(f"read {len(records)} record(s)"
          + (f", {len(unreadable)} unreadable" if unreadable else ""))
    print(f"  {len(records) - removed} unique, {removed} duplicate(s) merged away "
          f"from {len(merged)} cluster(s)")

    # A `doi` field holding something that is not a DOI is treated as no DOI at all, which is the safe
    # reading and a silent one — those records then match on title alone, and nothing would say why.
    # Cheap to count, and the answer is either zero or a data-quality problem the user wants to know
    # about. (Not hypothetical: the fixtures written for this tool's own tests contained five.)
    rejected = sum(1 for record in records if record.doi is None and record.field("doi"))
    if rejected:
        print(f"    {rejected} record(s) have a `doi` field that is not a DOI; matched on title only")

    by_rule = collections.Counter("+".join(cluster.rules) for cluster in merged)
    for rule, count in by_rule.most_common():
        print(f"    matched by {rule}: {count} cluster(s)")

    sizes = collections.Counter(len(cluster) for cluster in merged)
    if sizes:
        shape = ", ".join(f"{count}×{size}" for size, count in sorted(sizes.items()))
        print(f"    cluster sizes: {shape}")

    conflicted = [cluster for cluster in merged
                  if len({record.doi for record in cluster.records if record.doi}) > 1]
    if conflicted:
        print(f"    {len(conflicted)} cluster(s) hold more than one DOI (see the audit)")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="""Merge the duplicate records in a bibliography assembled from several databases. Matches on DOI and on title; writes an audit of every merge. Reads through the repair `raven-fixbib` applies, so records the parser would refuse are still seen; the input file is never modified.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-o", "--output", dest="output", default=None, type=str, metavar="deduped.bib",
                        help="Where to write the deduplicated bibliography. Without it, the run reports what it would do and writes nothing.")
    parser.add_argument("-a", "--audit", dest="audit", default=None, type=str, metavar="audit.tsv",
                        help="Where to write the audit of every merge: what was kept, what was merged away, which key matched, and every value that differed from the one kept.")
    parser.add_argument("--judge", dest="judge", action="store_true", default=False,
                        help="Also ask an LLM about near-miss titles that no exact key joined. Needs a backend; off by default.")
    parser.add_argument("--judge-state", dest="judge_state", default=None, type=str, metavar="PATH",
                        help="Resumable JSONL of the judge's answers. Defaults to sitting beside the output or the audit; a re-run skips what is already there, so a backend that falls over costs the batch in flight rather than the run.")
    parser.add_argument("--backend-url", dest="backend_url", default=None, type=str, metavar="URL",
                        help="LLM backend to judge with, overriding the configured one.")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="Mirror the log to this file (overwritten each run).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Root logger level.")
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="search.bib",
                        help="BibTeX file(s) to deduplicate. Several are read as one corpus, which is what a multi-database search produces.")
    opts = parser.parse_args()

    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)

    # `bibtexparser` logs a warning per record it cannot read, and this tool reports the same records
    # itself with a line number. Two accounts of one problem, interleaved, is worse than one. Set here
    # rather than in a library function, because muting someone else's logger is an application's
    # decision to make.
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    sources, pieces = [], []
    for filename in opts.filenames:
        path = pathlib.Path(filename).expanduser().resolve()
        try:
            pieces.append(path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"{path}: cannot read ({type(exc).__name__}: {exc})", file=sys.stderr)
            sys.exit(1)
        sources.append(path.name)

    records, unreadable = read_records("\n".join(pieces))
    for report in unreadable:
        print(f"unreadable: {report.describe()}", file=sys.stderr)
    if not records:
        print("no records to deduplicate.", file=sys.stderr)
        sys.exit(1)

    clusters = cluster_records(records)
    if opts.judge:
        clusters = _apply_judge(records, clusters, opts)

    _report(records, clusters, unreadable)

    library, rows = deduplicate(clusters)

    if opts.audit:
        audit_path = pathlib.Path(opts.audit).expanduser().resolve()
        write_audit(audit_path, rows, sources)
        print(f"audit written to {audit_path}")
    if not opts.output:
        print("nothing written (no -o/--output given).")
        return
    out_path = pathlib.Path(opts.output).expanduser().resolve()
    out_path.write_text(bibtex.write_string(library), encoding="utf-8")
    print(f"written to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
