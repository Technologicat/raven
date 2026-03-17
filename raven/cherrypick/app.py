"""raven-cherrypick — fast image triage tool.

Main application module: startup, GUI layout, render loop, hotkey dispatch.

This module is licensed under the 2-clause BSD license.
"""

# WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
# https://github.com/hoffstadt/DearPyGui/issues/554
# See raven/librarian/app.py:27-32 for the canonical example.
import platform
import os
if platform.system().upper() == "LINUX":
    os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

import logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Entry point for raven-cherrypick."""
    # TODO: implement
    print("raven-cherrypick: not yet implemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
