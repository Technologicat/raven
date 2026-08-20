"""What an `InputText(on_enter=True)` tells you, and when.

Three questions, one run:

  1. Does an item-edited handler fire per keystroke on a field whose `callback` is gated on Enter?
     If it does, a per-keystroke recolour has somewhere to hook.
  2. When the *global* key-press handler sees Enter, is that field still active? still focused?
     A dialog whose Enter also acts on a listing has to be able to tell that Enter was the field's.
  3. In what order do the two fire — the field's own `on_enter` callback, and the global handler?

Every line is stamped with the frame number and the monotonic clock, so the order is readable rather
than inferred.
"""

import time

import dearpygui.dearpygui as dpg

LOG = []
T0 = time.monotonic()


def note(what, **extra):
    LOG.append(f"[frame {dpg.get_frame_count():5d}  t={1000 * (time.monotonic() - T0):8.1f}ms] {what}"
               + ("  " + "  ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""))


dpg.create_context()
dpg.create_viewport(title="probe: InputText(on_enter=True)", width=600, height=200)
dpg.setup_dearpygui()

with dpg.window(tag="win"):
    field = dpg.add_input_text(hint="type here", on_enter=True, default_value="", width=-1)
    other = dpg.add_button(label="somewhere else to click")
dpg.set_primary_window("win", True)  # no title bar, so a click lands where the arithmetic says

dpg.set_item_callback(field, lambda: note("field callback (on_enter)", value=repr(dpg.get_value(field))))

with dpg.item_handler_registry() as reg:
    dpg.add_item_edited_handler(callback=lambda: note("edited handler", value=repr(dpg.get_value(field))))
    dpg.add_item_activated_handler(callback=lambda: note("activated handler"))
    dpg.add_item_deactivated_handler(callback=lambda: note("deactivated handler"))
    dpg.add_item_deactivated_after_edit_handler(callback=lambda: note("deactivated_after_edit handler"))
dpg.bind_item_handler_registry(field, reg)


def on_key(sender, app_data):
    note(f"GLOBAL key press {app_data}",
         active=dpg.is_item_active(field),
         focused=dpg.is_item_focused(field),
         value=repr(dpg.get_value(field)))


with dpg.handler_registry():
    dpg.add_key_press_handler(callback=on_key)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

print("\n".join(LOG))
