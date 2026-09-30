"""Does DearPyGui sample a texture linearly or by nearest neighbour when it draws it larger than it is?

The question behind scaling the avatar's video up on the client: a raw texture, shown with `add_image` at a
larger size, which is exactly how `raven.client.avatar_renderer` shows it. A static texture under
`draw_image` is measured beside it, since that is the other common path.

A 4×4 black-and-white checkerboard is drawn at 256×256 (64×), and at 2.5× a 64×64 one. The framebuffer is
saved from inside the process (`dpg.output_frame_buffer`), and the grey levels in each drawn region are
counted: nearest-neighbour leaves only black and white, linear filtering leaves grey ramps at the edges.

Maps a small window for a few seconds. Run: python probe_filtering.py
"""

import time

import numpy as np
import dearpygui.dearpygui as dpg

from raven.common.gui import utils as guiutils

N = 4          # checkerboard size, texels
SHOW = 256     # size it is drawn at, pixels
M = 64         # the second board
SHOW_M = 160   # 2.5x
BIG = 256      # the shrinking case: a board of 1-texel squares...
SHOW_BIG = 40  # ...drawn 6.4x smaller. Not a power of two: at exactly 8x every sample lands on a texel corner,
               # where bilinear averages four texels and reads a flat 0.5 however it treats minification.


def checkerboard(n: int) -> np.ndarray:
    """RGBA float32, flat, as DPG wants: black and white squares one texel each."""
    y, x = np.mgrid[0:n, 0:n]
    lum = ((x + y) % 2).astype(np.float32)
    rgba = np.stack([lum, lum, lum, np.ones_like(lum)], axis=-1)
    return rgba.ravel()


def grey_levels(region: np.ndarray) -> int:
    """How many distinct luminance values a region holds, ignoring alpha."""
    return len(np.unique(region[..., 0]))


def main():
    dpg.create_context()
    with dpg.texture_registry():
        raw = dpg.add_raw_texture(width=N, height=N, default_value=checkerboard(N), format=dpg.mvFormat_Float_rgba)
        static = dpg.add_static_texture(width=N, height=N, default_value=checkerboard(N))
        raw_m = dpg.add_raw_texture(width=M, height=M, default_value=checkerboard(M), format=dpg.mvFormat_Float_rgba)
        raw_big = dpg.add_raw_texture(width=BIG, height=BIG, default_value=checkerboard(BIG), format=dpg.mvFormat_Float_rgba)
        static_big = dpg.add_static_texture(width=BIG, height=BIG, default_value=checkerboard(BIG))

    # Laid out by groups rather than `pos`, which a drawlist ignores; each region is then found by asking
    # where its widget ended up.
    with dpg.window(tag="probe") as window:
        with dpg.group(horizontal=True):
            avatar_like = dpg.add_image(raw, width=SHOW, height=SHOW)            # the avatar's path
            with dpg.drawlist(width=SHOW, height=SHOW) as icon_like:
                dpg.draw_image(static, (0, 0), (SHOW, SHOW))                     # the chat log's role icons
        with dpg.group(horizontal=True):
            fractional = dpg.add_image(raw_m, width=SHOW_M, height=SHOW_M)       # a non-integer factor
            shrunk_image = dpg.add_image(raw_big, width=SHOW_BIG, height=SHOW_BIG)  # shrinking, the avatar's path
            with dpg.drawlist(width=SHOW_BIG, height=SHOW_BIG) as shrunk_icon:
                dpg.draw_image(static_big, (0, 0), (SHOW_BIG, SHOW_BIG))         # shrinking, the icons' path
    dpg.set_primary_window(window, True)

    dpg.create_viewport(title="dpg texture filtering probe", width=2 * SHOW + 32, height=SHOW + SHOW_M + 48)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    captured = {}
    def on_frame_buffer(sender, buffer):
        w, h = buffer.get_width(), buffer.get_height()
        captured["pixels"] = np.array(buffer, dtype=np.float32).reshape(h, w, 4)

    frame = 0
    while dpg.is_dearpygui_running() and "pixels" not in captured and frame < 300:
        dpg.render_dearpygui_frame()
        frame += 1
        if frame == 30:
            origins = {item: guiutils.get_widget_pos(item)
                       for item in (avatar_like, icon_like, fractional, shrunk_image, shrunk_icon)}
            dpg.output_frame_buffer(callback=on_frame_buffer)
    time.sleep(0.1)
    dpg.destroy_context()

    px = captured["pixels"]
    inset = 4  # stay clear of the regions' outer edges
    def region(item, size):
        x0, y0 = origins[item]
        return px[y0 + inset:y0 + size - inset, x0 + inset:x0 + size - inset]
    regions = {"add_image, raw texture, 64x (the avatar)": region(avatar_like, SHOW),
               "draw_image, static texture, 64x (the role icons)": region(icon_like, SHOW),
               "add_image, raw texture, 2.5x": region(fractional, SHOW_M)}
    for name, pixels in regions.items():
        levels = grey_levels(pixels)
        lo, hi = float(pixels[..., 0].min()), float(pixels[..., 0].max())
        # One level is the control firing: the region holds only background, having missed the image.
        # Nearest-neighbour puts only the board's two values on screen; anything more is smoothing.
        if levels == 1:
            verdict = "the region missed the image; no verdict"
        elif levels == 2:
            verdict = "nearest"
        else:
            verdict = "linear (or other smoothing)"
        print(f"{name}: {levels} grey level(s), range {lo:.2f}..{hi:.2f} -> {verdict}")

    # Shrinking asks a different question: whether minification averages the texels a pixel covers
    # (mipmaps, or area sampling), which turns a 1-texel board into flat mid-grey, or samples only the few
    # under the pixel's centre, which leaves a moire of greys from near-black to near-white.
    for name, item in (("add_image, raw texture, shrunk 6.4x (the avatar)", shrunk_image),
                       ("draw_image, static texture, shrunk 6.4x (the role icons)", shrunk_icon)):
        pixels = region(item, SHOW_BIG)[..., 0]
        lo, hi, spread = float(pixels.min()), float(pixels.max()), float(pixels.std())
        if grey_levels(region(item, SHOW_BIG)) == 1 and abs(lo - 0.5) > 0.1:
            verdict = "the region missed the image; no verdict"
        elif spread < 0.05:
            verdict = "averaged (mipmaps or area sampling)"
        elif grey_levels(region(item, SHOW_BIG)) == 2:
            verdict = "nearest"
        else:
            verdict = "point-sampled with bilinear, no mipmaps: aliases"
        print(f"{name}: range {lo:.2f}..{hi:.2f}, std {spread:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
