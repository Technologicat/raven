"""Who the user is: the link from a *name* to the card describing the person the AI is talking to.

The mirror of `raven.avatar.characters`, one step simpler. A character is declared beside its avatar image
in Raven's own assets; a user is declared in the user's own configuration directory, because a user is not
one of Raven's assets and Raven has no business shipping a default one.

**A profile is a JSON file** in `~/.config/raven/librarian/users/`, with its card beside it under the same
stem::

    juha.json        the profile: what the user is called.        Required.
    juha.md          their user card, as the LLM is told it.      Optional.
    juha_icon.png    the glyph beside their messages in the chat.  Optional.

::

    {
        "user_profile_version": 1,
        "name": "Juha"
    }

`llm_user_name` then selects one by that name, exactly as `llm_char_name` selects a character — so the
same setting that names you in the chat log also brings your card, and the two cannot disagree.

**Nothing ships, and nothing is required.** A Raven with no `users/` directory behaves as it always has:
the user has a name and no card, which is what the shipped `prompts/user.md` amounted to before 0.2.9.
Writing a profile is worth doing — current models respond well to knowing who they are talking to — but
it is the user's to write, and there is no sensible default for somebody we have never met.

**Why a directory rather than one `user.json`.** A name selects from a set, so a machine shared between
people, or one person keeping a work profile and a personal one, is a matter of changing `llm_user_name`
rather than editing prose. That is the same property the character side has, and it costs nothing extra:
the format is a directory scan either way.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["METADATA_EXT", "CARD_EXT", "IMAGE_EXT", "ICON_SUFFIX",
           "VERSION_KEY", "FORMAT_VERSION",
           "UserProfile",
           "scan", "profiles", "rescan", "find"]

import dataclasses
import json
import logging
import pathlib

logger = logging.getLogger(__name__)

from ..common import text as common_text

from . import config as librarian_config

METADATA_EXT = ".json"  # `juha.json`, the profile itself
CARD_EXT = ".md"        # `juha.md`, their user card
IMAGE_EXT = ".png"      # what an icon is
ICON_SUFFIX = "_icon"   # `juha_icon.png`, the chat glyph -- the same convention a character's icon uses

# What marks a JSON file as a user profile rather than as any other object that happens to carry a `name`.
# Same guard, and for the same reason, as `raven.avatar.characters.VERSION_KEY`.
VERSION_KEY = "user_profile_version"
FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class UserProfile:
    """One user: the name they go by, and the card describing them.

    Attributes:
        name: What the user is called, as `llm_user_name` names them and as the chat log records them.
        card_path: Their user card, or `None` if this profile ships none — an ordinary state, and the
                   caller then has a name and no description. Read it with `read_card`.
        icon_path: The glyph shown beside their messages in the chat, or `None` for the generic one. Same
                   `_icon.png` convention a character's icon follows, so the two sides of a conversation
                   are decorated the same way.
    """

    name: str
    card_path: pathlib.Path | None
    icon_path: pathlib.Path | None

    def read_card(self) -> str | None:
        """Return the user card's text, or `None` if this profile has none.

        Returned as written, with `{user}` and `{char}` still in it: who the AI character is comes from
        elsewhere entirely, and filling either in is the caller's job. See `raven.avatar.characters` for
        the same split on the other side.
        """
        if self.card_path is None:
            return None
        try:
            return self.card_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(f"read_card: cannot read '{self.card_path}' for '{self.name}': "
                           f"{type(exc)}: {exc}")
            return None


def _read_declaration(metadata_path: pathlib.Path) -> UserProfile | None:
    """Turn one `*.json` into a `UserProfile`, or `None` if it is not one."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"_read_declaration: cannot read '{metadata_path}': {type(exc)}: {exc}")
        return None

    if not isinstance(metadata, dict) or VERSION_KEY not in metadata:
        return None  # not a user profile; nothing to say about it

    version = metadata[VERSION_KEY]
    if version != FORMAT_VERSION:
        logger.warning(f"_read_declaration: '{metadata_path}' says {VERSION_KEY} is {version!r}, and this "
                       f"Raven reads {FORMAT_VERSION}; ignoring it.")
        return None

    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        logger.warning(f"_read_declaration: '{metadata_path}' is a user profile with no usable 'name' "
                       f"({name!r}); ignoring it.")
        return None

    def beside(filename: str) -> pathlib.Path | None:
        """The named file next to the declaration, if it is there."""
        path = metadata_path.with_name(filename)
        return path if path.exists() else None

    stem = metadata_path.stem
    return UserProfile(name=name,
                       card_path=beside(f"{stem}{CARD_EXT}"),
                       icon_path=beside(f"{stem}{ICON_SUFFIX}{IMAGE_EXT}"))


def scan(directory: pathlib.Path | None = None) -> dict[str, UserProfile]:
    """Return `{name: UserProfile}` for every profile declared under `directory`.

    `directory`: Where to look, searched recursively. Defaults to `~/.config/raven/librarian/users/`.

    **A missing directory is the ordinary case**, not an error: nothing ships one, and a user who has not
    written a profile has none. It reads as an empty result, and the caller then has a name and no card.

    Touches the filesystem on every call; `profiles` is the cached front for it.
    """
    if directory is None:
        directory = librarian_config.user_profiles_dir
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        logger.info(f"scan: no user profile directory at '{directory}'; the user will have a name and no "
                    "card. Writing one is optional -- see `raven.librarian.userprofile`.")
        return {}

    found: dict[str, UserProfile] = {}
    declared_by: dict[str, pathlib.Path] = {}  # only to name both files in the duplicate-name warning
    for metadata_path in sorted(directory.rglob(f"*{METADATA_EXT}")):
        profile = _read_declaration(metadata_path)
        if profile is None:
            continue
        if profile.name in found:
            logger.warning(f"scan: '{profile.name}' is declared twice, by "
                           f"'{declared_by[profile.name].name}' and by '{metadata_path.name}'; keeping "
                           "the first, so the second cannot be found by name")
            continue
        found[profile.name] = profile
        declared_by[profile.name] = metadata_path
    return found


_cache: dict[str, UserProfile] | None = None


def profiles() -> dict[str, UserProfile]:
    """Return `{name: UserProfile}`, scanning the first time and caching after.

    The answer changes only when somebody writes a file, so `rescan` is how that is picked up.
    """
    global _cache
    if _cache is None:
        _cache = scan()
        if _cache:
            logger.info(f"profiles: {len(_cache)} user profile{common_text.plural_s(len(_cache))} declared: {sorted(_cache)}")
    return _cache


def rescan() -> dict[str, UserProfile]:
    """Forget the cached scan and read the directory again. Returns the new mapping."""
    global _cache
    _cache = None
    return profiles()


def find(name: str | None) -> UserProfile | None:
    """Return the profile called `name`, or `None` if no declaration claims that name.

    `None` in and `None` out, so a caller can ask without checking first.
    """
    if name is None:
        return None
    return profiles().get(name)
