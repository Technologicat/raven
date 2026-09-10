"""Who the shipped avatar characters are: the link from a character's *name* to everything else about it.

The avatar system never needs this — it is handed an image path and animates it. What needs it is anything
holding only a name, which is Raven-librarian: a chat message records the character that wrote it as a bare
name, and drawing that message's speaker glyph means getting from "Aria" back to the files that depict it.

**A character is a JSON file.** That file is what makes a character exist under a name; everything else is
optional, and is found beside it under the same stem::

    aria1.json        the character itself: its name, and its voice.    Required.
    aria1.md          its character card, as the LLM is told it.        Optional.
    aria1.png         its avatar image, which the avatar animates.      Optional.
    aria1_icon.png    the glyph beside its messages in the chat.        Optional.

::

    {
        "character_definition_version": 1,
        "name": "Aria",
        "voice": "af_nova"
    }

Every part but the JSON is optional, including the face: `raven-minichat` is a terminal REPL that shows no
avatar at all, and a Librarian may be run without one. A character with a card and no image is an ordinary
character. That keeps the system's "plonk in some suitably formatted files and go" property.

**An image with no JSON is not a character**, and in Raven-librarian that is a real loss rather than a
lesser mode: it cannot be selected by name, so its face, voice and card have to be set separately in the
configuration — which is the arrangement this file exists to replace. The image still animates, so the
pose editor and the settings editor are unaffected; they are handed a path and never ask who it is.

**The declaration holds only what a filename cannot.** Everything that is a *file* is found by the stem
rather than named in the JSON, so there is one way to find a character's files and it is the same way the
avatar loader finds its cels. The card in particular is a file because it is paragraphs of prose, and JSON
is a poor place to edit those: every line break becomes a `\\n` and the whole card becomes one unreadable
line.

**`character_definition_version` is what says this JSON is a character**, rather than the presence of a
`name` — the assets tree holds other JSON, and anything at all might have a `name`. A file without the key
is quietly not a character; a file with it is one, and anything wrong with it is worth complaining about.
The number is there so the format can change later and say so.

**Reading happens in two phases, and the split is deliberate.** `scan` walks the tree and reads each JSON,
which is small; `Character.read_card` reads one card, on demand. A card is a page of prose per character,
and only the character actually being spoken as is ever wanted — so folding the cards into the scan would
read the whole cast every time anything asked a question about any of them.

**Why not derive the name from the filename**, which is the obvious cheaper idea: filenames are not names.
Measured against the shipped tree: "strip a trailing number and capitalize" gets ten of thirteen and fails
on the three that matter — `jj1.png` is the character *Juha*, `nerd1.png` and `nerd2.png` are two images
that collapse onto one name, and `ol1.png` is an initialism that capitalization mangles. The first is not
recoverable by any rule, being an abbreviation.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["METADATA_EXT", "CARD_EXT", "IMAGE_EXT", "ICON_SUFFIX",
           "VERSION_KEY", "FORMAT_VERSION",
           "Character",
           "scan", "characters", "rescan", "find"]

import dataclasses
import json
import logging
import pathlib

logger = logging.getLogger(__name__)

from ..common import text as common_text

from . import assets_path

METADATA_EXT = ".json"  # `aria1.json`, the character itself
CARD_EXT = ".md"        # `aria1.md`, its card; Markdown, which is what the LLM is shown elsewhere too
IMAGE_EXT = ".png"      # `aria1.png`, its avatar image
ICON_SUFFIX = "_icon"   # `aria1_icon.png`, the chat glyph; see `raven.server.modules.avatarutil`

# What marks a JSON file as a character rather than as any other object that happens to carry a `name`.
VERSION_KEY = "character_definition_version"
FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class Character:
    """One character: the name it goes by, and what a caller needs in order to present it.

    Attributes:
        name: What the character is called, as a chat message records it.
        image_path: The avatar image, which the avatar system animates, or `None` if this character has no
                    face — an ordinary state rather than an incomplete one.
        icon_path: The chat glyph, or `None` if this character has none — also ordinary, and callers fall
                   back to a generic glyph.
        voice: The TTS voice this character speaks in, or `None` to leave the caller's own default alone.
               Not validated here: which voices exist is the speech server's answer, and it may not be
               running when a character is read.
        card_path: The character card — who this character is, as the LLM is told it — or `None` if this
                   character ships none. Read it with `read_card`.
    """

    name: str
    image_path: pathlib.Path | None
    icon_path: pathlib.Path | None
    voice: str | None
    card_path: pathlib.Path | None

    def read_card(self) -> str | None:
        """Return the character card's text, or `None` if this character has none.

        Returned as written, with `{char}` and `{user}` still in it, because **`{user}` cannot be resolved
        here**: who the user is comes from the running app's configuration, and this module knows nothing
        about a chat. Whoever does — `raven.librarian.llmclient` — fills both in.

        `{char}` is a convenience rather than a necessity, a character's name being right there in its own
        JSON. It earns its place where a passage is shared between characters, or copied from one to start
        another, and it costs nothing to leave available.
        """
        if self.card_path is None:
            return None
        try:
            return self.card_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            # The file was there when the tree was scanned, so this is a disappearance rather than an
            # absence. Worth a complaint rather than a crash: the caller's fallback is whatever it would
            # have used for a character that ships no card at all.
            logger.warning(f"read_card: cannot read '{self.card_path}' for '{self.name}': "
                           f"{type(exc)}: {exc}")
            return None


def _read_declaration(metadata_path: pathlib.Path) -> Character | None:
    """Turn one `*.json` into a `Character`, or `None` if it is not a character declaration.

    Silent about JSON that does not claim to be one — the assets tree holds other kinds — and vocal about
    anything that does and is then unusable, which is a file somebody meant to work.
    """
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"_read_declaration: cannot read '{metadata_path}': {type(exc)}: {exc}")
        return None

    if not isinstance(metadata, dict) or VERSION_KEY not in metadata:
        return None  # not a character declaration; nothing to say about it

    version = metadata[VERSION_KEY]
    if version != FORMAT_VERSION:
        logger.warning(f"_read_declaration: '{metadata_path}' says {VERSION_KEY} is {version!r}, and this "
                       f"Raven reads {FORMAT_VERSION}; ignoring it.")
        return None

    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        logger.warning(f"_read_declaration: '{metadata_path}' is a character declaration with no usable "
                       f"'name' ({name!r}); ignoring it.")
        return None

    def beside(filename: str) -> pathlib.Path | None:
        """The named file next to the declaration, if it is there."""
        path = metadata_path.with_name(filename)
        return path if path.exists() else None

    stem = metadata_path.stem
    # A face is optional. A character with none is a perfectly good character wherever the face is not what
    # is wanted — a terminal frontend, or a Librarian run without an avatar — and rejecting the declaration
    # would cost it its card and its voice as well.
    image_path = beside(f"{stem}{IMAGE_EXT}")
    if image_path is None:
        logger.info(f"_read_declaration: character '{name}' has no avatar image of its own "
                    f"('{stem}{IMAGE_EXT}' is not there); it will have no face.")

    return Character(name=name,
                     image_path=image_path,
                     icon_path=beside(f"{stem}{ICON_SUFFIX}{IMAGE_EXT}"),
                     voice=metadata.get("voice"),
                     card_path=beside(f"{stem}{CARD_EXT}"))


def scan(directory: pathlib.Path | None = None) -> dict[str, Character]:
    """Return `{name: Character}` for every character declared under `directory`.

    **Exactly the characters that have a JSON declaration**, and nothing else: an image with no JSON beside
    it is not here, and cannot be found by name at all. Every other file a character may have is optional
    and does not affect whether it appears.

    `directory`: Where to look, searched recursively. Defaults to the shipped
                 `raven/avatar/assets/characters/`.

    Touches the filesystem on every call; `characters` is the cached front for it.

    Two characters claiming one name are resolved in path order, which is arbitrary but stable — and
    logged, since it means one of them can never be found.
    """
    if directory is None:
        directory = assets_path("characters")
    found: dict[str, Character] = {}
    declared_by: dict[str, pathlib.Path] = {}  # only to name both files in the duplicate-name warning
    for metadata_path in sorted(pathlib.Path(directory).rglob(f"*{METADATA_EXT}")):
        character = _read_declaration(metadata_path)
        if character is None:
            continue
        if character.name in found:
            # Both named by their declaration rather than by their image: a character need not have one,
            # so reaching for `image_path.name` here would crash on exactly the case being reported.
            logger.warning(f"scan: '{character.name}' is declared twice, by "
                           f"'{declared_by[character.name].name}' and by '{metadata_path.name}'; keeping "
                           "the first, so the second cannot be found by name")
            continue
        found[character.name] = character
        declared_by[character.name] = metadata_path
    return found


_cache: dict[str, Character] | None = None


def characters() -> dict[str, Character]:
    """Return `{name: Character}` for the shipped characters, scanning the first time and caching after.

    Cached because the callers ask per drawn message: a chat graph resolves a glyph for every box in the
    picture, on the render thread, at every rebuild. The answer changes only when somebody adds a file, so
    `rescan` is how that is picked up.
    """
    global _cache
    if _cache is None:
        _cache = scan()
        logger.info(f"characters: {len(_cache)} character{common_text.plural_s(len(_cache))} declared: {sorted(_cache)}")
    return _cache


def rescan() -> dict[str, Character]:
    """Forget the cached scan and read the tree again. Returns the new mapping."""
    global _cache
    _cache = None
    return characters()


def find(name: str | None) -> Character | None:
    """Return the character called `name`, or `None` if no declaration claims that name.

    `None` in and `None` out, so a caller holding a message's recorded persona — which is `None` for the
    roles that have no character behind them — can ask without checking first.
    """
    if name is None:
        return None
    return characters().get(name)
