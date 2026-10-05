"""Compare each visible Markdown decoration with the text it decorates, in a running app.

Run inside the app through its REPL (every Raven GUI app takes `--repl`), from the repository root. The path
is absolute because the REPL opens files relative to the app's working directory, not this shell's:

    printf 'exec(open("%s/investigations/dpg-markdown-decorations/probe_live_offsets.py").read())\n' "$PWD" \
        | python -m unpythonic.net.client localhost

A decoration is a group created at `pos=dpg.get_item_pos(text_group)`, holding a drawlist sized to
`dpg.get_item_rect_size(text_group)` (`text_attributes.Code.render`). For each visible drawlist whose parent
is a group, this prints the position and size recorded at decoration time beside the text group's position
and size now. `BAD` means they disagree. It also matches drawlists that are not code spans (list bullets, at
20x8, and a few unrelated ones); read the sizes.
"""

import dearpygui.dearpygui as dpg

rows = []
for item in dpg.get_all_items():
    try:
        if dpg.get_item_info(item)["type"] != "mvAppItemType::mvDrawlist":
            continue
        group = dpg.get_item_parent(item)
        if dpg.get_item_info(group)["type"] != "mvAppItemType::mvGroup":
            continue
        if not dpg.is_item_visible(item):
            continue
        text_group = dpg.get_item_parent(group)
        config = dpg.get_item_configuration(item)
        rows.append((item,
                     tuple(dpg.get_item_pos(group)), (config["width"], config["height"]),
                     tuple(dpg.get_item_pos(text_group)), tuple(dpg.get_item_rect_size(text_group)),
                     tuple(dpg.get_item_rect_min(item))))
    except Exception:  # an item deleted while we walk, by the app's own threads; skip it
        pass

print("NROWS", len(rows))
for item, recorded_pos, recorded_size, text_pos, text_size, screen_pos in rows:
    agrees = all(abs(a - b) < 2 for a, b in zip(recorded_pos + recorded_size, text_pos + text_size))
    print("ROW", "OK " if agrees else "BAD", "drawlist", item,
          "recorded pos", recorded_pos, "size", recorded_size,
          "| text group now pos", text_pos, "size", text_size,
          "| on screen", screen_pos)
