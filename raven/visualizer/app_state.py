"""Shared app-level state for the Visualizer.

During the ongoing refactor (splitting `app.py` into focused submodules), this
module holds what used to be module-level globals in `app.py` that need to be
read or written by multiple extracted submodules. A central namespace beats
`from .app import name` / circular-import gymnastics, and matches the Zen of
Python's *explicit is better than implicit*: every cross-module access is
`app_state.foo`, not a bare name whose origin is ambiguous.

Entries get added as each submodule extraction surfaces a new cross-module
dependency. Entries leave once a later refactor pass wraps related state into
a class whose instance can live here under a single name.

Module-local state (state that only one submodule needs to read or write)
stays in that submodule as module-level variables — not here.

## Expected fields (populated by `app.py` during startup)

The fields below are the currently-known shared state. They are assigned in
`app.py` at the point where the corresponding resource becomes available, so
readers need to be aware of initialization ordering — reading a field before
the owning section of `app.py` has run raises `AttributeError`. (This mirrors
the behaviour of the original module-level globals, which were also `None`
or undefined until initialized.)

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
| `info_panel_content_lock`        | `threading.RLock`           | `app.py` info-panel section           | Guards the info-panel content swap (will move into info_panel).|
| `info_panel_entry_title_widgets` | `dict[int, int]`            | `app.py` info-panel section           | `data_idx` → DPG entry-title group (will move into info_panel).|
| `get_entries_for_selection`      | callable                    | `app.py` shared-helpers section       | Gather items by cluster; shared by annotation and info panel.  |
| `update_mouse_hover`             | callable                    | `raven.visualizer.annotation.update`  | Submit a plotter-tooltip refresh. Published by `annotation`.   |

Subsystems that own their own task managers or per-subsystem state keep those
private (e.g. `word_cloud._task_manager`, `word_cloud._image_box`,
`annotation._task_manager`, `annotation._build_number`); only the genuinely
cross-cutting pieces appear above.
"""

__all__ = ["app_state"]

from unpythonic.env import env

app_state = env()
