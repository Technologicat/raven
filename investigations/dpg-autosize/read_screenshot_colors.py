import collections
import sys
import numpy as np
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
flat = img.reshape(-1, 3)
counts = collections.Counter(map(tuple, flat))
print(f"{flat.shape[0]} px, {len(counts)} distinct colours")
print("brightest, with pixel counts (a plateau means saturated glyph interiors):")
for c, n in sorted(counts.items(), key=lambda kv: -sum(kv[0]))[:8]:
    print(f"    {c}   n={n}")
print("channel-wise max:", tuple(flat.max(axis=0)))
