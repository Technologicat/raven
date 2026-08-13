"""Raven's AI-animated avatar: its shipped assets, and the two editors that work on them.

Licensed under the 2-clause BSD license, like the project default — deliberately, and load-bearing: the
avatar *service* (`raven.server`) and the pose editor (`raven.avatar.pose_editor`) are AGPL-3.0, and both
need `assets_path` below. AGPL code may use BSD code; the reverse is what is not allowed. Keeping this
module BSD is therefore what lets the whole constellation share one answer to "where are the assets?".
"""

__all__ = ["assets_path"]

import pathlib


def assets_path(*parts: str) -> pathlib.Path:
    """Return an absolute path into the shipped avatar assets, `raven/avatar/assets/`.

    `assets_path()` is the assets directory itself; `assets_path("characters")` its character images;
    `assets_path("emotions", "_defaults.json")` one file.

    Resolves from *this* module's location, so a caller gets the same answer wherever it lives. That is the
    point of having it: every call site used to spell out `os.path.dirname(__file__)` followed by however
    many `".."` its own depth required, which made an asset path something a module could break by moving —
    silently, since a wrong path is only discovered when something tries to read it.
    """
    return pathlib.Path(__file__).parent.joinpath("assets", *parts).resolve()
