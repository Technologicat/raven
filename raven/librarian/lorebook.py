"""The lorebook: definitions of terms, given to the LLM when a term comes up in the conversation.

`Lorebook` reads a folder of Markdown files, follows it, and offers the model the `lorebook_lookup` tool;
`loaded` does so for the length of a `with` block. Everything else — parsing, matching, choosing what fires,
answering a lookup — is `raven.librarian.lorebookcore`, kept to the standard library so that other frontends
can take it as it is, and re-exported here.
"""

__all__ = ["README_FILENAME", "Entry", "slugify",
           "parse", "load",
           "terms", "Expansion", "clean_expansion", "Match", "find_matches",
           "Firing", "select",
           "CANONICAL_NOT_DEFINED", "CANONICAL_NEAR_MISS", "CANONICAL_NO_ENTRY", "CANONICAL_CANDIDATES",
           "CANONICAL_MENTIONED_IN", "CANONICAL_LOOK_ONE_UP",
           "Lookup", "lookup", "format_entry", "answer",
           "TOOL_NAME", "tool_spec",
           "Lorebook", "loaded"]

import contextlib
import logging
import pathlib
from collections.abc import Callable, Iterator, Mapping

from unpythonic.env import env

from ..common import filewatch
from ..common import utils as common_utils
from . import config as librarian_config
from . import llmtools
# The core, re-exported by name, so that `lorebook.select` and the rest go on resolving: where the stateless half
# lives is not the callers' business. A new public name in the core has to be added here too, as `__all__` asks.
from .lorebookcore import (README_FILENAME, Entry, slugify,  # noqa: F401 -- re-export
                           parse, load,
                           terms, Expansion, clean_expansion, Match, find_matches,
                           Firing, select,
                           CANONICAL_NOT_DEFINED, CANONICAL_NEAR_MISS, CANONICAL_NO_ENTRY, CANONICAL_CANDIDATES,
                           CANONICAL_MENTIONED_IN, CANONICAL_LOOK_ONE_UP,
                           Lookup, lookup, format_entry, answer,
                           TOOL_NAME, tool_spec)

logger = logging.getLogger(__name__)

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
                entries, and `settings.lorebook_entries` holds them for the per-turn scan, with
                `settings.lorebook_expansions` holding the key words' other forms; `shutdown` removes all three.

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
        self._expansions: dict[str, Expansion] = {}
        self._problems: dict[str, str] = {}
        self._generation = 0
        self._registered = False
        self._maybe_watcher: filewatch.DirectoryWatcher | None = None

    def _get_entries(self) -> dict[str, Entry]:
        """Copy, so that a caller cannot edit the lorebook's record."""
        return dict(self._entries)
    entries = property(fget=_get_entries, doc="The entries, by id, in order.")

    def _get_expansions(self) -> dict[str, Expansion]:
        """Copy, so that a caller cannot edit the lorebook's record."""
        return dict(self._expansions)
    def _set_expansions(self, expansions: Mapping[str, Expansion]) -> None:
        """Replace the expansions, and offer them to the scan and the lookup tool from now on."""
        self._expansions = dict(expansions)
        self._install()
    def _get_generation(self) -> int:
        """A counter, compared rather than read: only whether it moved means anything."""
        return self._generation
    generation = property(fget=_get_generation,
                          doc="How many times the folder has been read. Changes whenever a file does, whether or not "
                              "the set of entries did, so a poll can notice an edited alias.")

    expansions = property(fget=_get_expansions, fset=_set_expansions,
                          doc="The key words' other forms, as `find_matches` takes them; `lorebook_expansion` makes "
                              "them. Setting this offers them to the scan and the lookup tool from then on.")

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
        self._expansions = {}
        self._install()

    def _refresh(self) -> None:
        self._entries, self._problems = load(self.directory, entry_levels=self.entry_levels)
        self._generation += 1
        for filename, problem in self._problems.items():
            logger.error(f"Lorebook: '{filename}': {problem}")
        self._install()
        noun = "entry" if len(self._entries) == 1 else "entries"  # plural
        logger.info(f"Lorebook: {len(self._entries)} {noun} in '{self.directory}'.")

    def _install(self) -> None:
        # Readers take the dict as it is at that moment, and it is replaced whole rather than changed in place,
        # so a scan running alongside a reload sees one lorebook or the other, never a mixture.
        entries, expansions = self._entries, self._expansions
        self.settings.lorebook_entries = entries
        self.settings.lorebook_expansions = expansions
        specs = [tool_spec()] if entries else []
        entrypoints = ({llmtools.LOREBOOK_TOOL_NAME: lambda term: answer(entries, term, expansions=expansions)}
                       if entries else {})
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
