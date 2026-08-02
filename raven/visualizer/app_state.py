"""Shared app-level state for the Visualizer.

A single namespace for state that needs to be read or written by multiple
submodules. Beats `from .app import name` / circular-import gymnastics, and
matches the Zen of Python's *explicit is better than implicit*: every
cross-module access is `app_state.foo`, not a bare name whose origin is
ambiguous.

Entries get added when a new cross-module dependency surfaces, and can leave
when their state finds a more natural home elsewhere.

Module-local state (state only one submodule needs to read or write) stays in
that submodule as module-level variables — not here.

## Expected fields

The fields below are the currently-known shared state. Most are assigned by
`app.py` at the point where the corresponding resource becomes available;
the rest are published or maintained by the submodule that owns them, named
per row. Either way, readers need to be aware of initialization ordering —
reading a field before its owner has assigned it raises `AttributeError`.

The `Populated by` column is therefore where to look when a field turns up
empty or missing: it names the code that has to have run first.

| Field                            | Type                        | Populated by                          | Purpose                                                        |
|----------------------------------|-----------------------------|---------------------------------------|----------------------------------------------------------------|
| `dataset`                        | `unpythonic.env.env` / None | `app.py` top-level + `open_file`      | Currently-loaded dataset. `None` when no file is open.         |
| `bg`                             | `ThreadPoolExecutor`        | `app.py` lifecycle section            | Shared thread-pool executor for background tasks.              |
| `themes_and_fonts`               | `unpythonic.env.env`        | `app.py` DPG bootup                   | DPG theme + font handles produced by `guiutils.bootup`.        |
| `selection_data_idxs_box`        | `box(np.ndarray)`           | `app.py` selection-management section | Boxed current selection (indices into `dataset.sorted_*`).     |
| `filedialog_save`                | `FileDialog` / None         | `app.py` `initialize_filedialogs`     | The save-word-cloud `FileDialog` instance.                     |
| `enter_modal_mode`               | callable                    | `app.py` modal-window utilities       | Prepare GUI for showing a modal (hide annotation, etc.).       |
| `exit_modal_mode`                | callable                    | `app.py` modal-window utilities       | Restore GUI after closing a modal.                             |
| `is_any_modal_window_visible`    | callable → bool             | `app.py` modal-window utilities       | Whether some modal window is currently open.                   |
| `mouse_inside_plot_widget`       | callable → bool             | `app.py` event-handlers section       | Whether the mouse cursor is over the plotter.                  |
| `search_string_box`              | `box(str)`                  | `app.py` search section               | Boxed current search string (empty when no search active).     |
| `search_result_data_idxs_box`    | `box(np.ndarray)`           | `app.py` search section               | Boxed indices (into `sorted_*`) of items matching the search.  |
| `selection_changed`              | `bool`                      | `raven.visualizer.selection`          | Set when the selection changes; cleared by the info panel once it has finalized an update. Used for scroll anchoring. |
| `selection_anchor_data_idxs_set` | `set[int]`                  | `raven.visualizer.selection`          | Items common to the previous and current selection, so they can serve as scroll anchors. Indices into `sorted_*`. |
| `update_mouse_hover`             | callable                    | `raven.visualizer.annotation.update`  | Submit a plotter-tooltip refresh. Published by `annotation`.   |
| `update_info_panel`              | callable                    | `app.py` info-panel section           | `info_panel.update`, published so cross-module callers reach it without importing `app`. |
| `update_search`                  | callable                    | `app.py` search section               | Re-run the current search. Published for the same reason.      |

Subsystems that own their own task managers or per-subsystem state keep those
private (e.g. `word_cloud._task_manager`, `word_cloud._image_box`,
`annotation._task_manager`, `annotation._build_number`); only the genuinely
cross-cutting pieces appear above.
"""

__all__ = ["app_state"]

from unpythonic.env import env

app_state = env()
