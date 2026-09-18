"""How far up do font sizes still work, now that DPG picks the character ranges itself?

If every codepoint of the TTF were rasterized eagerly, ~1150 glyphs would need more than a 16384-px square
texture somewhere past 480 px, so the ladder would break there. If glyphs are added on demand, size buys
little and the ladder stays linear far past that. Timing each load separates the two the same way: eager
rasterization of a whole face at 1024 px is not free.
"""
import sys
import time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

SIZES = (64, 128, 256, 512, 1024)
LATIN = "Hamburgefonstiv"
GREEK = "Ψυχολογία"        # never drawn anywhere in this process
CYRILLIC = "Привет мир"    # ...nor this

dpg.create_context()
dpg.create_viewport(title="atlas size probe", width=200, height=100)
dpg.setup_dearpygui()
themes_and_fonts = guiutils.bootup(font_size=20)

fonts = {}
for size in SIZES:
    t0 = time.perf_counter()
    fonts[size] = guiutils.load_extra_font(themes_and_fonts, size, "OpenSans", "Regular")[1]
    print(f"load {size:>5} px: {1000 * (time.perf_counter() - t0):7.1f} ms")

dpg.show_viewport()
t0 = time.perf_counter()
for _ in range(3):
    dpg.render_dearpygui_frame()
print(f"\nthree frames after loading them all: {1000 * (time.perf_counter() - t0):.1f} ms")

print(f"\n{'size':>6} {'latin':>10} {'per px':>8} {'greek':>10} {'cyrillic':>10}")
for size in SIZES:
    latin = dpg.get_text_size(LATIN, font=fonts[size])
    greek = dpg.get_text_size(GREEK, font=fonts[size])
    cyr = dpg.get_text_size(CYRILLIC, font=fonts[size])
    lw = latin[0] if latin else 0.0
    print(f"{size:>6} {lw:10.1f} {lw / size:8.3f} "
          f"{(greek[0] if greek else 0.0):10.1f} {(cyr[0] if cyr else 0.0):10.1f}")

dpg.destroy_context()
sys.exit(0)
