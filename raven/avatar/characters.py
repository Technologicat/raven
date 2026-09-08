"""Who the shipped avatar characters are: the link from a character's *name* to everything else about it.

The avatar system never needs this — it is handed an image path and animates it. What needs it is anything
holding only a name, which is Raven-librarian: a chat message records the character that wrote it as a bare
name, and drawing that message's speaker glyph means getting from "Aria" back to `aria1.png`.

**A character is declared by a JSON file beside its image**, `aria1.json` next to `aria1.png`::

    {
        "name": "Aria",
        "voice": "af_nova"
    }

`name` is the only required key, and the file is optional: a character without one still works everywhere
it worked before, it simply cannot be found by name. That keeps the system's "plonk in some suitably
formatted files and go" property.

**The declaration holds only what a filename cannot.** Everything that is a *file* is found by the
convention the tree already uses — `aria1_icon.png` for the chat glyph, `aria1_card.md` for the character
card — rather than being named in the JSON, so there is one way to find a character's files and it is the
same way the avatar loader finds its cels.

**The settings a character brings with it are settings the caller then does not have to keep in step.**
Switching character used to mean editing four things that had to agree — the name, the image, the voice,
and which card function got called — and getting it wrong meant the new face speaking in the old voice, or
answering as the previous character.

**The card is a file rather than a string in the JSON**, because it is paragraphs of prose and JSON is a
poor place to edit those: every line break becomes a `\\n` and the whole card becomes one unreadable line.
A sidecar keeps it editable, and keeps the declaration small enough to read at a glance.

**Why not derive the name from the filename**, which is the obvious cheaper idea: filenames are not names.
Measured against the shipped tree: "strip a trailing number and capitalize" gets ten of thirteen and fails
on the three that matter — `jj1.png` is the character *Juha*, `nerd1.png` and `nerd2.png` are two images
that collapse onto one name, and `ol1.png` is an initialism that capitalization mangles. The first is not
recoverable by any rule, being an abbreviation.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["METADATA_EXT", "IMAGE_EXT", "ICON_SUFFIX", "CARD_SUFFIX", "CARD_EXT",
           "Character",
           "scan", "characters", "find", "rescan"]

import dataclasses
import json
import logging
import pathlib
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from . import assets_path

METADATA_EXT = ".json"  # `aria1.json`, beside `aria1.png`
IMAGE_EXT = ".png"      # every shipped character image
ICON_SUFFIX = "_icon"   # `aria1_icon.png`, the chat glyph; see `raven.server.modules.avatarutil`
CARD_SUFFIX = "_card"   # `aria1_card.md`, the character card
CARD_EXT = ".md"        # cards are Markdown, which is what the LLM is shown elsewhere too


@dataclasses.dataclass(frozen=True)
class Character:
    """One character: the name it goes by, and what a caller needs in order to present it.

    Attributes:
        name: What the character is called, as a chat message records it.
        image_path: The main character image, which the avatar system animates, or `None` if this character
                    has no face. Optional because a character is a name, a personality and a voice, and a
                    face is one thing it may additionally have: `raven-minichat` is a terminal REPL that
                    shows no avatar at all, and a Librarian may be run without one. Requiring an image
                    would mean a faceless character could not be declared, and an undeclared character
                    loses its card and its voice too — so the coupling would cost far more than the face.
        icon_path: The chat glyph, or `None` if this character has none — an ordinary state, and callers
                   fall back to a generic glyph.
        voice: The TTS voice this character speaks in, or `None` to leave the caller's own default alone.
               Not validated here: which voices exist is the speech server's answer, and it may not be
               running when a character is read.
        card_path: The character card — who this character is, as the LLM is told it — or `None` if this
                   character ships none. Read it with `read_card`.
    """

    name: str
    image_path: Optional[pathlib.Path]
    icon_path: Optional[pathlib.Path]
    voice: Optional[str]
    card_path: Optional[pathlib.Path]

    def read_card(self) -> Optional[str]:
        """Return the character card's text, or `None` if this character has none.

        The text is a *template*, not the finished card: it is written with `{char}` and `{user}` in it,
        and whoever knows what those are — `raven.librarian.config` — fills them in. Returned raw for that
        reason, this module knowing nothing about a chat.

        Read on demand rather than at scan time. A card is a page of prose per character and only the one
        being spoken as is ever wanted, so reading every card to answer a question about icons would be
        the whole cast's worth of file I/O for nothing.
        """
        if self.card_path is None:
            return None
        try:
            return self.card_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            # Between the scan and now, so it existed a moment ago. Worth a complaint rather than a
            # crash: the caller's fallback is the card it would have used before this file appeared.
            logger.warning(f"read_card: cannot read '{self.card_path}' for '{self.name}': "
                           f"{type(exc)}: {exc}")
            return None


def _read_declaration(metadata_path: pathlib.Path) -> Optional[Character]:
    """Turn one `*.json` into a `Character`, or `None` if it is not one (or names files that are absent)."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        name = metadata["name"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Not fatal: the assets tree holds JSON that is nobody's character declaration — emotion presets,
        # for one — and a malformed file should cost that character rather than every character after it.
        logger.warning(f"_read_declaration: ignoring '{metadata_path}': {type(exc)}: {exc}")
        return None

    def sidecar(suffix: str, ext: str) -> Optional[pathlib.Path]:
        """`aria1.json` -> `aria1_icon.png`, if it is there.

        Keyed off the declaration's own stem rather than the image's, which are the same string — and has
        to be, since a faceless character has no image to key off.
        """
        path = metadata_path.with_name(f"{metadata_path.stem}{suffix}{ext}")
        return path if path.exists() else None

    # A face is optional. A character with none is a perfectly good character everywhere the face is not
    # what is wanted — a terminal frontend, or a Librarian run without an avatar — and rejecting the
    # declaration would cost it its card and its voice as well, which is far more than the face.
    image_path = metadata_path.with_suffix(IMAGE_EXT)
    if not image_path.exists():
        logger.info(f"_read_declaration: character '{name}' has no avatar image of its own "
                    f"('{image_path.name}' is not there); it will have no face.")
        image_path = None

    return Character(name=name,
                     image_path=image_path,
                     icon_path=sidecar(ICON_SUFFIX, IMAGE_EXT),
                     voice=metadata.get("voice"),
                     card_path=sidecar(CARD_SUFFIX, CARD_EXT))


def scan(directory: Optional[pathlib.Path] = None) -> Dict[str, Character]:
    """Read every character declared under `directory`, returning `{name: Character}`.

    `directory`: Where to look, searched recursively. Defaults to the shipped
                 `raven/avatar/assets/characters/`.

    Touches the filesystem on every call; `characters` is the cached front for it.

    Two characters claiming one name are resolved in path order, which is arbitrary but stable — and
    logged, since it means one of them can never be found.
    """
    if directory is None:
        directory = assets_path("characters")
    found: Dict[str, Character] = {}
    for metadata_path in sorted(pathlib.Path(directory).rglob(f"*{METADATA_EXT}")):
        character = _read_declaration(metadata_path)
        if character is None:
            continue
        if character.name in found:
            logger.warning(f"scan: '{character.name}' is declared twice, by "
                           f"'{found[character.name].image_path.name}' and by "
                           f"'{character.image_path.name}'; keeping the first, so the second cannot be "
                           "found by name")
            continue
        found[character.name] = character
    return found


_cache: Optional[Dict[str, Character]] = None


def characters() -> Dict[str, Character]:
    """Return `{name: Character}` for the shipped characters, scanning the first time and caching after.

    Cached because the callers ask per drawn message: a chat graph resolves a glyph for every box in the
    picture, on the render thread, at every rebuild. The answer changes only when somebody adds a file, so
    `rescan` is how that is picked up.
    """
    global _cache
    if _cache is None:
        _cache = scan()
        logger.info(f"characters: {len(_cache)} character(s) declared: {sorted(_cache)}")
    return _cache


def rescan() -> Dict[str, Character]:
    """Forget the cached scan and read the tree again. Returns the new mapping."""
    global _cache
    _cache = None
    return characters()


def find(name: Optional[str]) -> Optional[Character]:
    """Return the character called `name`, or `None` if no declaration claims that name.

    `None` in and `None` out, so a caller holding a message's recorded persona — which is `None` for the
    roles that have no character behind them — can ask without checking first.
    """
    if name is None:
        return None
    return characters().get(name)
