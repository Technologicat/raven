"""The lorebook: definitions of terms, given to the LLM when a term comes up in the conversation.

An entry is a Markdown section — a header, and everything under it up to the next header at the same level.
The header is the entry's title, and the title is the term: when it appears in the text being scanned, the
entry fires, and its whole section is what the model gets. Entries live in Markdown files in one folder, any
number to a file; see `parse` for how a file is split.

The stateless half is parsing, matching and choosing which entries fire. `select` is the whole of the last,
and needs nothing but the entries and the text, so a batch tool can call it to compose definitions into a
prompt of its own. The stateful half is `Lorebook`, which reads a folder, follows it, and offers the model the
`lorebook_lookup` tool.
"""

__all__ = ["Entry", "slugify",
           "parse", "load",
           "Match", "find_matches",
           "Firing", "select",
           "CANONICAL_NOT_DEFINED", "CANONICAL_NEAR_MISS", "CANONICAL_CANDIDATES",
           "Lookup", "lookup", "format_entry",
           "Lorebook", "loaded"]

import contextlib
import dataclasses
import difflib
import logging
import pathlib
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import Any

from unpythonic.env import env

from ..common import filewatch
from ..common import utils as common_utils
from . import config as librarian_config
from . import llmtools

logger = logging.getLogger(__name__)

README_FILENAME = "README.md"  # the folder's own README, which says what goes there; never a source

@dataclasses.dataclass(frozen=True)
class Entry:
    """One definition.

    `id`: identifies the entry, stable for as long as its title is: `slugify(title)`.
    `title`: the term, as the header names it, aliases removed.
    `aliases`: other spellings that fire it too, from parentheses at the end of the header.
    `content`: the section under the header, whole.
    `crossrefs`: the anchors of the section's `[text](#anchor)` links, slugified as written.
    `anchors`: the anchors that lead to this entry: the slug of the title, and of the header as written.
    `source`: the file it came from.
    `order`: its position among all the entries, the files taken in name order. What decides between entries
             when not all of them fit.
    """
    id: str
    title: str
    aliases: tuple[str, ...]
    content: str
    crossrefs: tuple[str, ...]
    anchors: tuple[str, ...]
    source: pathlib.Path
    order: int


def slugify(text: str) -> str:
    """The anchor GitHub gives a header with this text: lowercase, punctuation dropped, spaces as hyphens."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


# ----------------------------------------
# Parsing

_HEADER = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")  # a horizontal rule: `---`, `***`, `___`, spaced or not
_ALIASES = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
_CROSSREF = re.compile(r"\]\(#([^)\s]+)\)")

def _headers(lines: list[str]) -> list[tuple[int, int, str]]:
    """`(line index, level, text)` of each header, skipping fenced code, where a `#` is not one."""
    headers, fenced = [], False
    for index, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced and (match := _HEADER.match(line)):
            headers.append((index, len(match.group(1)), match.group(2)))
    return headers

def _is_blank(line: str) -> bool:
    return not line.strip() or bool(_RULE.match(line))

def parse(text: str, *, name: str, source: pathlib.Path | None = None, entry_level: int | None = None,
          first_order: int = 0) -> list[Entry]:
    """Split one Markdown file into entries.

    `name`: titles the entry when the file has no headers at all; usually the filename's stem.
    `source`: recorded in each entry. Default: `name`.
    `entry_level`: the header level that delimits entries, overriding the rule below.
    `first_order`: the `order` of the first entry; the rest count up from it.

    The rule, when no `entry_level` is given. A header with nothing under it before the next header — blank lines
    and horizontal rules aside — is a *grouping*, never an entry: the letter sections of a glossary, say. Of the
    level-1 headers that are not groupings, several are themselves the entries. One is the collection's title,
    and the entries are the shallowest level below it — as they are when there is no level-1 header at all. A
    lone level-1 header with nothing deeper is the one entry of a single-entry file, and a file with no headers is
    one entry titled `name`.

    An entry's section runs to the next header at its own level or shallower, so deeper headers are part of it.
    """
    source = source if source is not None else pathlib.Path(name)
    lines = text.splitlines()
    headers = _headers(lines)
    if not headers:
        content = text.strip()
        return [_make_entry(name, content, source, first_order)] if content else []

    def has_content(header_number: int) -> bool:
        start = headers[header_number][0] + 1
        end = headers[header_number + 1][0] if header_number + 1 < len(headers) else len(lines)
        return not all(_is_blank(line) for line in lines[start:end])

    with_content = [number for number in range(len(headers)) if has_content(number)]
    if entry_level is None:
        top = [number for number in with_content if headers[number][1] == 1]
        deeper = sorted({headers[number][1] for number in with_content if headers[number][1] > 1})
        entry_level = 1 if len(top) >= 2 or not deeper else deeper[0]

    entries = []
    for number in with_content:
        index, level, header_text = headers[number]
        if level != entry_level:
            continue
        end = next((later_index for later_index, later_level, _ in headers[number + 1:] if later_level <= entry_level),
                   len(lines))
        body = lines[index + 1:end]
        while body and _is_blank(body[-1]):  # a trailing rule separates sections; it is not part of this one
            body.pop()
        entries.append(_make_entry(header_text, "\n".join(body).strip(), source, first_order + len(entries)))
    return entries

def _make_entry(header_text: str, content: str, source: pathlib.Path, order: int) -> Entry:
    title, aliases = header_text.strip(), ()
    if (match := _ALIASES.match(title)) and match.group(1):
        title = match.group(1)
        aliases = tuple(alias.strip() for alias in match.group(2).split(",") if alias.strip())
    anchors = tuple(dict.fromkeys((slugify(title), slugify(header_text))))  # the two coincide without aliases
    crossrefs = tuple(dict.fromkeys(slugify(anchor) for anchor in _CROSSREF.findall(content)))
    return Entry(id=slugify(title), title=title, aliases=aliases, content=content, crossrefs=crossrefs,
                 anchors=anchors, source=source, order=order)

def load(directory: str | pathlib.Path,
         *,
         entry_levels: Mapping[str, int] | None = None) -> tuple[dict[str, Entry], dict[str, str]]:
    """Read every lorebook file in `directory`: each `*.md` except the folder's `README.md`, in name order.

    `entry_levels`: filename -> the header level that delimits that file's entries, for a file the rule in `parse`
                    reads wrongly.

    Returns `(entries, problems)`. `entries` maps ids to entries, in order. `problems` maps a filename to why it,
    or an entry in it, could not be used. Two entries with one id keep the first. A missing directory is an empty
    lorebook with no problems.
    """
    directory = pathlib.Path(directory).expanduser()
    entry_levels = entry_levels or {}
    entries: dict[str, Entry] = {}
    problems: dict[str, str] = {}
    if not directory.is_dir():
        return entries, problems
    order = 0
    for path in sorted(directory.glob("*.md")):
        if path.name == README_FILENAME or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            problems[path.name] = f"cannot be read: {type(exc)}: {exc}"
            continue
        for entry in parse(text, name=path.stem, source=path, entry_level=entry_levels.get(path.name), first_order=order):
            order += 1
            if entry.id in entries:
                problems[path.name] = f"'{entry.title}' is defined in '{entries[entry.id].source.name}' already; the first is kept"
                continue
            entries[entry.id] = entry
    return entries, problems


# ----------------------------------------
# Matching

# A word: letters and digits, joined by internal hyphens or apostrophes (`co-pter`, `Chekhov's`), and keeping a
# trailing hyphen that nothing word-like follows — a suspended hyphen, as in "co- and re-", which is what lets
# the key `co-` match the prefix written on its own and not "co-pter" or "CO".
_WORD = re.compile(r"\w+(?:['-]\w+)*(?:-(?!\w))?")

def _normalize(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")  # curly apostrophes, as typed by many editors

def _words(text: str) -> list[re.Match]:
    return list(_WORD.finditer(_normalize(text)))

def _key(phrase: str, *, sentence_case: bool) -> tuple[str, ...]:
    """The words of a key, in the case they are matched in.

    `sentence_case`: the phrase is a title, whose first letter is capitalized as the start of a sentence would
                     be rather than to make the word exact.
    """
    words = [match.group() for match in _words(phrase)]
    # The first letter says nothing, so the rest of the word decides: `Hydra` is forgiving, while `EQ` and `AI`
    # carry an uppercase letter past the first and stay exact, as written.
    if sentence_case and words and words[0][1:] == words[0][1:].lower():
        words[0] = words[0].lower()
    return tuple(words)

def _keys(entry: Entry) -> list[tuple[str, ...]]:
    keys = [_key(entry.title, sentence_case=True)]
    keys.extend(_key(alias, sentence_case=False) for alias in entry.aliases)
    return [key for key in keys if key]

@dataclasses.dataclass(frozen=True)
class Match:
    """Where an entry's term was found. `wording` is the text as written there, which may differ from the title."""
    entry_id: str
    start: int
    end: int
    wording: str

def _window_matches(key: tuple[str, ...], window: list[str]) -> bool:
    """Whether the words of `window` are the words of `key`, in any order, each matched with smart case.

    A key word in lowercase matches any case; a key word with an uppercase letter matches only as written.
    """
    if Counter(word.lower() for word in key) != Counter(word.lower() for word in window):
        return False
    # Within each lowercase spelling, a forgiving key word takes any of the window's words, so an assignment
    # exists exactly when every exact form has at least as many words in the window as exact key words need it.
    exact_needed = Counter(word for word in key if word != word.lower())
    exact_present = Counter(window)
    return all(exact_present[form] >= count for form, count in exact_needed.items())

def find_matches(entries: Mapping[str, Entry], text: str) -> dict[str, Match]:
    """Which entries' terms appear in `text`: entry id -> the last place one does.

    A term is matched as words, never inside one ("cat" is not in "concatenate"), and a term of several words
    matches them in any order, next to each other ("different bath, same shampoo" for "Same bath, different
    shampoo"). Case is smart: a word in lowercase matches any case, and a word with an uppercase letter only as
    written, except that a title's first letter does not count, being capitalized as a sentence's would be.
    """
    normalized = _normalize(text)
    words = _words(normalized)
    lowered = [match.group().lower() for match in words]
    positions: dict[str, list[int]] = {}
    for position, word in enumerate(lowered):
        positions.setdefault(word, []).append(position)

    found: dict[str, Match] = {}
    for entry in entries.values():
        best: tuple[int, int] | None = None  # (start word, end word) of the last match
        for key in _keys(entry):
            n = len(key)
            # Every window holding the key must hold its rarest word, so only the windows around that word's
            # occurrences are worth looking at.
            rarest = min((word.lower() for word in key), key=lambda word: len(positions.get(word, ())))
            for position in positions.get(rarest, ()):
                for first in range(max(0, position - n + 1), min(position, len(words) - n) + 1):
                    window = [match.group() for match in words[first:first + n]]
                    if _window_matches(key, window) and (best is None or first > best[0]):
                        best = (first, first + n - 1)
        if best is not None:
            start, end = words[best[0]].start(), words[best[1]].end()
            found[entry.id] = Match(entry_id=entry.id, start=start, end=end, wording=normalized[start:end])
    return found


# ----------------------------------------
# Choosing what fires

@dataclasses.dataclass(frozen=True)
class Firing:
    """An entry chosen to be given to the model.

    `depth`: 0 for an entry whose term is in the text, n for one reached through n links from such an entry.
    `maybe_wording`: at depth 0, the text as written where the term was found; `None` deeper.
    `maybe_via`: deeper than 0, the id of the entry whose link brought this one in; `None` at depth 0.
    `tokens`: what the entry costs in the budget, as `format_entry` gives it to the model.
    """
    entry: Entry
    depth: int
    maybe_wording: str | None
    maybe_via: str | None
    tokens: int = 0

def select(entries: Mapping[str, Entry],
           text: str,
           *,
           count_tokens: Callable[[str], int],
           budget: int,
           max_depth: int,
           already_fired: Collection[str] = ()) -> list[Firing]:
    """Choose which entries fire for `text`, and the entries they lead to.

    `count_tokens`: the size of a text in tokens, for the budget.
    `budget`: the most tokens the chosen entries may take together, counting each entry's title and content.
    `max_depth`: how many links deep to follow; 0 takes only the entries whose terms are in `text`.
    `already_fired`: ids of entries the model has already been given, which neither fire nor lead anywhere.

    An entry whose term is in `text` fires at depth 0. Each entry chosen then leads to more: through its
    `[text](#anchor)` links when it has any, which say exactly what the author meant to point at, and otherwise
    to the entries whose terms appear in its content. Nothing fires twice, so a cycle of links ends.

    Shallower entries come first, and within a depth they come in `order`. An entry that does not fit the budget
    is skipped, and leads nowhere; later ones that fit are still taken. Deterministic: the same entries, text and
    arguments choose the same firings.
    """
    anchors = {anchor: entry.id for entry in entries.values() for anchor in entry.anchors}
    visited = set(already_fired)
    chosen: list[Firing] = []
    remaining = budget

    def take(candidates: list[Firing]) -> list[Firing]:
        nonlocal remaining
        taken = []
        for firing in sorted(candidates, key=lambda firing: firing.entry.order):
            cost = count_tokens(format_entry(firing.entry))
            if cost > remaining:
                logger.info(f"select: '{firing.entry.title}' fired but does not fit the budget ({cost} tokens, {remaining} left); skipped.")
                continue
            remaining -= cost
            taken.append(dataclasses.replace(firing, tokens=cost))
        return taken

    matches = find_matches(entries, text)
    level = take([Firing(entry=entries[entry_id], depth=0, maybe_wording=match.wording, maybe_via=None)
                  for entry_id, match in matches.items() if entry_id not in visited])
    visited.update(entry_id for entry_id in matches)  # fired or skipped, either way not again at another depth
    for depth in range(1, max_depth + 1):
        if not level:
            break
        chosen.extend(level)
        candidates: dict[str, Firing] = {}
        for firing in level:
            entry = firing.entry
            if entry.crossrefs:
                targets = [anchors[anchor] for anchor in entry.crossrefs if anchor in anchors]
            else:
                targets = list(find_matches(entries, entry.content))
            for target in targets:
                if target not in visited and target not in candidates:
                    candidates[target] = Firing(entry=entries[target], depth=depth, maybe_wording=None, maybe_via=entry.id)
        visited.update(candidates)
        level = take(list(candidates.values()))
    else:
        chosen.extend(level)
    return chosen


# ----------------------------------------
# Looking a term up

CANONICAL_NOT_DEFINED = ("The term '{term}' is not defined in the lorebook. It holds only the terms the user has "
                         "defined, so most words are not in it.")
CANONICAL_NEAR_MISS = "There is no entry for '{term}'. The closest is '{title}':"
CANONICAL_CANDIDATES = "There is no entry for '{term}'. The closest entries are: {titles}. Look one of them up by its title."

@dataclasses.dataclass(frozen=True)
class Lookup:
    """The answer to looking up a term.

    `maybe_entry`: the entry the term names, or the one near miss there is; `None` otherwise.
    `exact`: whether `maybe_entry` is the term's own entry rather than a near miss.
    `candidates`: titles of the closest entries, when there are several near misses and no exact entry.
    """
    maybe_entry: Entry | None
    exact: bool
    candidates: tuple[str, ...]

def lookup(entries: Mapping[str, Entry], term: str) -> Lookup:
    """Find the entry for `term`, as a model asking for it by name would want.

    Exact when the term's words are a key's words, in any order and any case: whoever asks by name has already
    picked the term, so smart case, which keeps scanned prose from firing on a common word, has nothing to do
    here. Otherwise the keys within `difflib`'s default similarity of the term are near misses: one of them is
    taken as meant, several are offered by title, and none means the term is not defined.
    """
    wanted = sorted(word.lower() for word in _key(term, sentence_case=False))
    spellings: dict[str, str] = {}  # a key as written, lowercased -> the entry it belongs to
    for entry in entries.values():
        for key in _keys(entry):
            if sorted(word.lower() for word in key) == wanted:
                return Lookup(maybe_entry=entry, exact=True, candidates=())
        for phrase in (entry.title, *entry.aliases):
            spellings.setdefault(_normalize(phrase).lower(), entry.id)
    close = difflib.get_close_matches(_normalize(term).lower(), list(spellings), n=5)
    ids = list(dict.fromkeys(spellings[spelling] for spelling in close))  # several spellings, one entry
    if len(ids) == 1:
        return Lookup(maybe_entry=entries[ids[0]], exact=False, candidates=())
    return Lookup(maybe_entry=None, exact=False, candidates=tuple(entries[entry_id].title for entry_id in ids))

def format_entry(entry: Entry) -> str:
    """An entry as the model reads it: its title, then its content."""
    return f"{entry.title}\n\n{entry.content}"

def _answer(entries: Mapping[str, Entry], term: str) -> str:
    """The text `lorebook_lookup` answers `term` with."""
    found = lookup(entries, term)
    if found.maybe_entry is not None:
        if found.exact:
            return format_entry(found.maybe_entry)
        return f"{CANONICAL_NEAR_MISS.format(term=term, title=found.maybe_entry.title)}\n\n{format_entry(found.maybe_entry)}"
    if found.candidates:
        return CANONICAL_CANDIDATES.format(term=term, titles=", ".join(f"'{title}'" for title in found.candidates))
    return CANONICAL_NOT_DEFINED.format(term=term)

def _tool_spec() -> dict[str, Any]:
    return {"type": "function",
            "function": {"name": llmtools.LOREBOOK_TOOL_NAME,
                         "description": ("Look up a term in the lorebook: the user's own definitions of terms, mostly their "
                                         "coinages and the special senses they give to ordinary words. Only the terms the user "
                                         "has defined are in it, so most words are not. Definitions of terms that come up in "
                                         "the conversation are given to you as they come up; look up one you need that has not been given."),
                         "parameters": {"type": "object",
                                        "properties": {"term": {"type": "string",
                                                                "description": "The term to look up, as it was written."}},
                                        "required": ["term"],
                                        "additionalProperties": False}}}


# ----------------------------------------
# The lorebook of a running app or script

_LOREBOOK_README = """# Put your lorebook entries here

Definitions of terms, which Raven-librarian gives the AI when a term comes up in the conversation: your own
coinages, a field's jargon, the special sense you give an ordinary word.

Each entry is a Markdown section: its header is the term, and everything under it is the definition. Write one
file per term, or many terms to a file — a glossary is one file. Other spellings of a term go in parentheses
in its header, comma-separated: `## Fup (fuppable)`. A file here can be a link to one kept elsewhere.

More in the *Lorebook* section of Librarian's README:
https://github.com/Technologicat/raven/blob/main/raven/librarian/README.md#lorebook

Changes are picked up while Librarian runs. This file is not an entry, and you can delete it.
"""

class Lorebook:
    """The lorebook in a folder, offered to the LLM through `settings`.

    `settings`: from `llmclient.setup` or `llmclient.connect`. `start` adds the lookup tool to it when there are
                entries, and `settings.lorebook_entries` holds them for the per-turn scan; `shutdown` removes both.

    `directory`: where the lorebook files are. Default is `librarian_config.lorebook_dir`.

    `entry_levels`: filename -> the header level delimiting that file's entries, for a file `parse`'s rule reads
                    wrongly. Default is `librarian_config.lorebook_entry_levels`.

    `on_changed`: 0-argument callable, called on a background thread when the set of entries has changed on disk.
    """
    def __init__(self,
                 settings: env,
                 *,
                 directory: str | pathlib.Path | None = None,
                 entry_levels: Mapping[str, int] | None = None,
                 on_changed: Callable[[], None] | None = None) -> None:
        self.settings = settings
        self.directory = pathlib.Path(directory if directory is not None else librarian_config.lorebook_dir).expanduser()
        self.entry_levels = dict(entry_levels if entry_levels is not None else librarian_config.lorebook_entry_levels)
        self.on_changed = on_changed
        self._entries: dict[str, Entry] = {}
        self._problems: dict[str, str] = {}
        self._registered = False
        self._maybe_watcher: filewatch.DirectoryWatcher | None = None

    def _get_entries(self) -> dict[str, Entry]:
        """Copy, so that a caller cannot edit the lorebook's record."""
        return dict(self._entries)
    entries = property(fget=_get_entries, doc="The entries, by id, in order.")

    def _get_problems(self) -> dict[str, str]:
        """Copy, so that a caller cannot edit the lorebook's record."""
        return dict(self._problems)
    problems = property(fget=_get_problems, doc="Why a file, or an entry in it, could not be used, by filename.")

    def start(self, *, watch: bool = False) -> None:
        """Read the lorebook, and add the lookup tool to the settings if there are entries.

        `watch`: follow changes to the folder, creating it if it is missing. For a long-lived frontend. A script
                 leaves this off, so that what it offers cannot change during a run.
        """
        if watch:
            if not self.directory.is_dir():  # a watcher needs something to watch; empty, it shows where entries go
                logger.info(f"Lorebook.start: '{self.directory}' does not exist; creating it.")
                common_utils.create_directory(self.directory, readme=_LOREBOOK_README)
            # Started before the first read, so that an edit right after the read is seen.
            watcher = filewatch.DirectoryWatcher(self.directory,
                                                 on_changed=lambda relative_path, kind: self._on_file_changed(),
                                                 accept=lambda path: path.suffix == ".md" and path.name != README_FILENAME)
            watcher.start()
            self._maybe_watcher = watcher
        self._refresh()

    def shutdown(self) -> None:
        """Remove the tool and the entries from the settings, and stop following the folder. Safe to call more than once."""
        if self._maybe_watcher is not None:
            self._maybe_watcher.stop()
            self._maybe_watcher = None
        self._entries = {}
        self._install()

    def _refresh(self) -> None:
        self._entries, self._problems = load(self.directory, entry_levels=self.entry_levels)
        for filename, problem in self._problems.items():
            logger.error(f"Lorebook: '{filename}': {problem}")
        self._install()
        noun = "entry" if len(self._entries) == 1 else "entries"  # plural
        logger.info(f"Lorebook: {len(self._entries)} {noun} in '{self.directory}'.")

    def _install(self) -> None:
        # Readers take the dict as it is at that moment, and it is replaced whole rather than changed in place,
        # so a scan running alongside a reload sees one lorebook or the other, never a mixture.
        entries = self._entries
        self.settings.lorebook_entries = entries
        specs = [_tool_spec()] if entries else []
        entrypoints = {llmtools.LOREBOOK_TOOL_NAME: lambda term: _answer(entries, term)} if entries else {}
        previous = {llmtools.LOREBOOK_TOOL_NAME} if self._registered else set()
        llmtools.replace_tools(self.settings, previous_names=previous, specs=specs, entrypoints=entrypoints)
        self._registered = bool(entries)

    def _on_file_changed(self) -> None:
        """Read the lorebook again after a file changed. On the watcher's thread."""
        before = list(self._entries)
        try:
            self._refresh()
        except Exception:  # noqa: BLE001 -- the next change to the folder gets another try
            logger.exception("Lorebook: re-reading the lorebook failed")
            return
        if list(self._entries) != before and self.on_changed is not None:
            try:
                self.on_changed()
            except Exception:  # noqa: BLE001 -- a frontend's handler must not stop the watcher
                logger.exception("Lorebook: ignoring exception from `on_changed`")


@contextlib.contextmanager
def loaded(settings: env,
           *,
           directory: str | pathlib.Path | None = None,
           entry_levels: Mapping[str, int] | None = None) -> Iterator[Lorebook]:
    """Have the lorebook available in `settings` for the duration of a `with` block.

    Arguments as for `Lorebook`. Yields it, whose `problems` say which files could not be used.
    """
    lorebook = Lorebook(settings, directory=directory, entry_levels=entry_levels)
    try:
        lorebook.start()
        yield lorebook
    finally:
        lorebook.shutdown()
