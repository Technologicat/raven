# How DearPyGui filters a texture drawn at another size

**Question.** When DPG draws a texture larger or smaller than it is, how does it sample it? Asked for scaling
Raven-librarian's avatar up on the client, where a server-side upscale would cost network bandwidth and
postprocessing time, and to settle a recollection from 2024 that DPG scaling was nearest-neighbour only.

**Answer.** Bilinear, without mipmaps. Enlarging is smooth at any factor, integer or not. Shrinking by much
more than 2× point-samples: each screen pixel reads only the few texels under its centre, so fine detail
aliases into a moiré. The same on both paths measured — a raw texture under `add_image` (the avatar's) and a
static texture under `draw_image` in a drawlist (the chat log's role icons).

Measured 2026-09-30, DearPyGui 2.3.1, Linux.

## The probe

`probe_filtering.py` draws black-and-white checkerboards of 1-texel squares, saves the framebuffer from inside
the process (`dpg.output_frame_buffer`), and reads each drawn region. Maps a small window for a few seconds.

| case | result | reading |
|---|---|---|
| 4×4 board at 256×256 (64×), `add_image` | 214 grey levels, 0.02..0.98 | bilinear |
| the same, `draw_image` of a static texture | 214 grey levels, 0.02..0.98 | bilinear |
| 64×64 board at 160×160 (2.5×), `add_image` | 7 grey levels, 0.18..0.82 | bilinear at a fractional factor |
| 256×256 board at 40×40 (6.4× smaller), `add_image` | 0.18..0.82, std 0.153 | point-sampled: aliases |
| the same, `draw_image` | 0.18..0.82, std 0.153 | point-sampled: aliases |

How each reading follows:

- **Enlarging.** Nearest-neighbour puts only the board's two values on screen; bilinear puts ramps between
  them at every square's edge, hence the hundreds of levels.
- **Shrinking.** Averaging the texels a pixel covers (mipmaps, or area sampling) turns a 1-texel board into a
  flat mid-grey. Reading only the texels under the pixel's centre leaves whatever phase of the board that
  centre happens to land on, which varies across the image — a spread of greys. The factor is deliberately
  not a power of two: at exactly 8× every sample lands on a texel corner, where bilinear averages four texels
  and reads a flat 0.5 whatever it does about minification.
- **The control.** A region that missed its image holds one grey level, the window background, and says so
  rather than reading as "nearest". The first run of the probe placed its drawlist with `pos`, which a
  drawlist ignores, and the control caught it.

## What follows

- **The avatar can be enlarged on the client** at any factor. Rounding the factor down to an integer is a
  matter of taste rather than of correctness: to the maintainer's eye, bilinear at 2.0× looks noticeably
  sharper than at 2.2×.
- **Anything drawn much smaller than its texture should be prescaled**, since nothing on DPG's side will
  average it. The chat log's role icons are prescaled, and were at first drawn shrunk from larger images;
  aliasing from exactly this is the likeliest reading of the 2024 impression of nearest-neighbour.
