"""Does the graph's four-face font ladder survive the atlas? Load it, map a window, and measure every rung.

A lost glyph is silent, so the check is that every (face, size) rung measures a known string plausibly:
nonzero, growing with the atlas size, and bold wider than regular. A rung whose atlas failed reports zero,
or a width that does not track the others because it fell back to another font.
"""
import sys
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

SIZES = (4, 8, 16, 32, 64)
VARIANTS = {(False, False): "Regular", (True, False): "Bold",
            (False, True): "Italic", (True, True): "BoldItalic"}
SAMPLE = "Hamburgefonstiv"

dpg.create_context()
dpg.create_viewport(title="atlas probe", width=200, height=100)
dpg.setup_dearpygui()
themes_and_fonts = guiutils.bootup(font_size=20)

fonts = {face: [(size, guiutils.load_extra_font(themes_and_fonts, size, "OpenSans", variant)[1])
                for size in SIZES]
         for face, variant in VARIANTS.items()}

dpg.show_viewport()
for _ in range(3):
    dpg.render_dearpygui_frame()

print(f"{'size':>5} " + " ".join(f"{VARIANTS[f]:>12}" for f in VARIANTS))
widths = {}
for i, size in enumerate(SIZES):
    row = []
    for face in VARIANTS:
        w = dpg.get_text_size(SAMPLE, font=fonts[face][i][1])
        widths[(face, size)] = w[0] if w else 0.0
        row.append(f"{widths[(face, size)]:12.1f}")
    print(f"{size:>5} " + " ".join(row))

bad = []
for face, name in VARIANTS.items():
    seq = [widths[(face, size)] for size in SIZES]
    if any(w <= 0 for w in seq):
        bad.append(f"{name}: a rung measured nothing: {seq}")
    elif sorted(seq) != seq:
        bad.append(f"{name}: widths do not grow with size: {seq}")
for size in SIZES:
    if widths[((True, False), size)] <= widths[((False, False), size)]:
        bad.append(f"bold is not wider than regular at {size}px: "
                   f"{widths[((True, False), size)]} vs {widths[((False, False), size)]}")

print()
print("PROBLEMS:" if bad else "every rung measured, widths grow with size, and bold is wider than regular")
for line in bad:
    print("  " + line)
dpg.destroy_context()
sys.exit(1 if bad else 0)
