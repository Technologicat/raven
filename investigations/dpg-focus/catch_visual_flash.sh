#!/usr/bin/env bash
# Catch a sub-100 ms visual artifact in a running GUI app, and say which frames carry it.
#
# Written 2026-08-18 for a blue flash in `FileDialog`'s path field that lasted 25-100 ms per Tab. Three
# headless probes failed to reproduce it, because the artifact needs a real focus change in flight; what
# settled it was recording the window and scanning frames.
#
# Screenshots cannot do this. `import` costs 50-200 ms per capture, so at 60 Hz it samples roughly one
# frame in five and will usually miss the thing entirely. `ffmpeg -f x11grab` records every presented frame.
#
# Usage:  catch_visual_flash.sh <window-name-pattern> <seconds> <crop WxH+X+Y in window coords>
# Then drive the app by hand, or with xdotool, while it records.
#
# Example, the case it was written for:
#   catch_visual_flash.sh Raven-cherrypick 6 1100x26+340+95

set -euo pipefail
PATTERN=${1:?window name pattern}
SECONDS_TO_RECORD=${2:-6}
CROP=${3:?crop as WxH+X+Y, in window coordinates}
OUT=${TMPDIR:-/tmp}/flash-$$

WID=$(xdotool search --name "$PATTERN" | tail -1)
eval "$(xwininfo -id "$WID" | awk '/Absolute upper-left X/{printf "AX=%s ",$NF}
                                   /Absolute upper-left Y/{printf "AY=%s ",$NF}
                                   /Width:/{printf "W=%s ",$NF} /Height:/{printf "H=%s ",$NF}')"

mkdir -p "$OUT/frames"
echo "recording ${SECONDS_TO_RECORD}s of ${W}x${H} at +${AX}+${AY} — drive the app now"
ffmpeg -loglevel error -f x11grab -framerate 60 -video_size "${W}x${H}" \
       -i ":0.0+${AX},${AY}" -t "$SECONDS_TO_RECORD" -y "$OUT/capture.mp4"

# Crop to the region of interest and rank the frames by how blue they are relative to the median. Blue
# because ImGui paints selected text that way; adapt the metric to whatever the artifact looks like.
ffmpeg -loglevel error -i "$OUT/capture.mp4" -vf "crop=${CROP/+/:}" "$OUT/frames/f%04d.png" 2>/dev/null || \
ffmpeg -loglevel error -i "$OUT/capture.mp4" \
       -vf "crop=$(echo "$CROP" | sed 's/x/:/; s/+/:/g')" "$OUT/frames/f%04d.png"

python3 - "$OUT/frames" <<'PY'
import pathlib, sys
import numpy as np
from PIL import Image

frames = sorted(pathlib.Path(sys.argv[1]).glob("*.png"))
scores = [(float((np.asarray(Image.open(p).convert("RGB")).astype(int)[:, :, 2] -
                  np.asarray(Image.open(p).convert("RGB")).astype(int)[:, :, 0]).mean()), p.name)
          for p in frames]
median = float(np.median([s for s, _ in scores]))
odd = [(s, n) for s, n in scores if s > median + 3]
print(f"  {len(frames)} frames, median blueness {median:.2f}")
print(f"  {len(odd)} frames stand out:")
for s, n in odd[:12]:
    print(f"    {n}  {s:.2f}")
PY
echo "frames in $OUT/frames — open the ones named above"
