"""An MCP server over a music library on another machine, reached over SSH.

Usage: `python rhythmbox_mcp_server.py --host HOST --music-dir DIR --queue-file PATH`.

`queue_track` appends to `--queue-file` instead of playing anything.
"""

import argparse
import pathlib
import subprocess

from mcp.server.mcpserver import MCPServer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", required=True, help="the machine holding the music, for `ssh`")
parser.add_argument("--music-dir", default="Music", help="the music folder, relative to the home directory there")
parser.add_argument("--queue-file", required=True, help="where to record the queued tracks")
opts = parser.parse_args()

_EXTENSIONS = {".mp3", ".flac", ".ogg", ".m4a", ".opus"}

listing = subprocess.run(["ssh", "-o", "BatchMode=yes", opts.host, f'find "$HOME/{opts.music_dir}" -type f'],
                         capture_output=True, text=True, check=True).stdout
library = [path for path in listing.splitlines() if pathlib.PurePosixPath(path).suffix.lower() in _EXTENSIONS]

server = MCPServer("rhythmbox")

def _title(path: str) -> str:
    """A track's title, from the convention `NN - Artist - Title.ext`."""
    return pathlib.PurePosixPath(path).stem.rsplit(" - ", 1)[-1]

@server.tool()
def search_tracks(query: str, limit: int = 20) -> str:
    """Find tracks in the music library whose title contains the query, ignoring case. Returns one track per line, as `title | path`."""
    needle = query.lower()
    matches = [path for path in library if needle in _title(path).lower()]
    if not matches:
        return f"No track title contains '{query}'."
    return "\n".join(f"{_title(path)} | {path}" for path in matches[:limit])

@server.tool()
def queue_track(path: str) -> str:
    """Add a track to the end of the play queue, by its path from `search_tracks`."""
    if path not in library:
        return f"There is no track at '{path}'."
    with open(opts.queue_file, "a", encoding="utf-8") as file:
        file.write(path + "\n")
    return f"Queued: {_title(path)}"

if __name__ == "__main__":
    server.run("stdio")
