# raven-cherrypick — Specification v0.3

## 1. Name & Concept

**raven-cherrypick** — fast image triage tool. The user opens a folder, reviews images, and sorts them into three piles: **cherries** (keepers), **lemons** (rejects), and **neutral** (undecided, the default). Optimized for comparing variants of the same shot.

Module: `raven/cherrypick/`. Entry point: `raven-cherrypick`.

Icons: `ICON_STAR` (cherry/keeper) and `ICON_LEMON` (reject) from FontAwesome.

## 2. Module Structure

```
raven/cherrypick/
    __init__.py          # re-export main
    app.py               # startup, GUI layout, render loop, hotkey dispatch (~600 lines)
    config.py            # all constants (~80 lines)
    imageview.py         # main image pane: drawlist, pan, zoom, display scaling (~350 lines)
    grid.py              # thumbnail grid: layout, tile rendering, selection (~500 lines)
    loader.py            # image I/O, thumbnail generation (GPU batch), preload cache (~400 lines)
    triage.py            # triage state, file move operations, virtual directory merge (~250 lines)

raven/common/
    lanczos.py           # GPU Lanczos-3 resize, reusable (~200 lines)
```

No `__main__.py` — follows existing Raven convention. Entry point via `pyproject.toml`
`[project.scripts]`, with an `if __name__ == "__main__"` guard at the end of `app.py`.

### Testing

```
raven/common/tests/
    test_lanczos.py      # Lanczos kernel: correctness, edge cases, multi-stage vs single-pass

raven/cherrypick/tests/
    test_triage.py        # virtual directory merge, file move logic, collision handling
    test_loader.py        # thumbnail pipeline, mipchain generation, format loading (PNG/JPG/QOI)
    test_grid.py          # grid layout math (column count, pagination, filter navigation)
```

**Test-as-we-go**: unit tests written alongside each module, not deferred. Every algorithm (Lanczos kernel, triage state machine, virtual directory merge, grid layout calculation, filter navigation, VRAM budget logic) gets tests. The Lanczos kernel in particular needs:
- Correctness: downscale a known test pattern (e.g. checkerboard, frequency sweep), verify against PIL Lanczos as reference.
- Identity: resize to same size ≈ identity (within floating-point tolerance).
- Separability: horizontal-then-vertical matches a 2D reference.
- Multi-stage: verify multi-stage output matches single-pass output for large ratios.
- Device: run on both CPU and CUDA (if available), results should match.

## 3. GUI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Toolbar]  Open  +  -  Fit  1:1  │ Grid: 32 64 [128] 256 512      │
│            * Mark  L Mark  N Clear │ View: All * L N │ Compare     │
│            Dark  FS  Help                                           │
├──────────────────────────────────────┬──────────────────────────────┤
│                                      │  ┌──┐ ┌──┐ ┌──┐ ┌──┐       │
│                                      │  │  │ │▓▓│ │  │ │  │       │
│         Main Image View              │  └──┘ └──┘ └──┘ └──┘       │
│         (drawlist with pan/zoom)     │  a.jpg b.jpg c.jpg d.jpg    │
│                                      │  ┌──┐ ┌──┐ ┌──┐ ┌──┐       │
│                                      │  │  │ │  │ │  │ │  │       │
│                                      │  └──┘ └──┘ └──┘ └──┘       │
│                                      │  e.jpg f.jpg g.jpg h.jpg    │
│                                      │         ...                  │
├──────────────────────────────────────┴──────────────────────────────┤
│ [Status] IMG_1234.jpg | 4032x3024 | Zoom: 42% | 847 imgs | 12* 3L │
└─────────────────────────────────────────────────────────────────────┘
```

- **Left ~70%**: Main image pane (drawlist inside child_window).
- **Right ~30%**: Thumbnail grid (scrollable child_window).
- **Top**: Toolbar row(s). All buttons follow the xdot_viewer pattern (30px icon buttons, FontAwesome, tooltips with `"Action [Hotkey]"` format).
- **Bottom**: Status bar — current filename, pixel dimensions, zoom %, total image count, triage counts (star/lemon counts with icons). Also shows compare mode indicator and image-pane-focus indicator when active.

**Triage toolbar buttons**: Star, Lemon, and Neutral buttons with tooltips — makes the single-letter hotkeys discoverable. E.g. `"Mark cherry [C]"`, `"Mark lemon [L]"`, `"Clear mark (Neutral) [N]"`.

**Split ratio**: Configurable in `config.py` (default 70/30). Fixed in v1. Architect the layout so that widths are computed from a single ratio variable, making a future draggable splitter straightforward. Draggable splitter will need debounced grid layout recalculation (via `make_managed_task` from `raven.common.bgtask`). Note: splitter resize does NOT require thumbnail regeneration — the thumbnails are resolution-independent of the grid panel width. Only the grid layout (column count, row arrangement) needs recalculating.

## 4. Image Pipeline

### 4.1 Supported Formats

- **PNG, JPG**: via PIL (`Image.open`)
- **QOI**: via `qoi.decode()` (the `qoi` package, already a dependency — used in avatar streaming)
- Anything else PIL handles (BMP, TIFF, WebP) comes free; no explicit support needed.

Detection by extension (`.png`, `.jpg`, `.jpeg`, `.qoi`). The directory scanner collects files matching these.

### 4.2 Thumbnail Generation

**Engine**: PyTorch GPU batch processing, with automatic CPU fallback via `raven.common.deviceinfo`.

**Resize algorithm**: Custom Lanczos-3 kernel in `raven/common/lanczos.py` (see §4.6). Config toggle for `torch.nn.functional.interpolate(mode='area')` as alternative — may auto-switch to `area` at the smallest grid sizes (32, 64) where the extreme reduction ratio makes `area` a reasonable choice.

For efficiency at extreme reduction ratios (e.g. 4000→128, ~30×), use multi-stage downsampling: repeatedly halve with Lanczos until within 2× of the target, then one final Lanczos resize. This keeps the kernel compact (~6 taps per stage) while maintaining quality.

**Pipeline**:
1. Load batch of images on CPU (PIL/QOI → numpy RGBA uint8)
2. Stack as `torch.Tensor`, transfer to configured device
3. Resize to target thumbnail size (square; letterbox non-square originals)
4. Transfer result to CPU, convert to float32 [0,1] RGBA
5. Create DPG dynamic textures

**DPG texture management**:
- Use counter-based unique tags (e.g. `f"cherrypick_thumb_{counter}"`) — DPG tag collision = crash.
- Use descriptive tag prefixes throughout (e.g. `"cherrypick_main_tex"`, `"cherrypick_grid_child"`) for easier debugging.
- Hidden debug hotkeys ("Mr. T Lite"): Ctrl+Shift+M (metrics), R (registry), T (fonts), L (style editor).

**NVIDIA workaround**: At module top, before importing DPG (see `raven/librarian/app.py:27–32` for the canonical example):
```python
if platform.system().upper() == "LINUX":
    os.environ["__GLVND_DISALLOW_PATCHING"] = "1"
```
Required to avoid segfault when deleting textures on Linux/NVIDIA (DPG issue #554).

**Batching**: Configurable batch size (default 32). Process in background thread via `TaskManager`. Grid shows placeholder tiles (neutral gray) until thumbnails are ready; tiles pop in progressively.

**No disk cache** — regenerate on every app launch.

### 4.3 VRAM Budget

DPG textures are float32 RGBA → 16 bytes/pixel. Per thumbnail:

| Tile size | Bytes/thumb | 1000 images | 5000 images |
|-----------|-------------|-------------|-------------|
| 32×32     | 4 KB        | 4 MB        | 20 MB       |
| 64×64     | 16 KB       | 16 MB       | 80 MB       |
| 128×128   | 64 KB       | 64 MB       | 320 MB      |
| 256×256   | 256 KB      | 256 MB      | 1.28 GB     |
| 512×512   | 1 MB        | 1 GB        | 5 GB        |

These are OpenGL texture allocations (shared physical VRAM with CUDA). The torch processing tensors are transient (freed after each batch).

**Dynamic VRAM detection**: On startup, if running on a CUDA device, query available VRAM via `torch.cuda.mem_get_info()`. Set the thumbnail VRAM budget as a configurable fraction of available memory (default: `THUMBNAIL_VRAM_FRACTION = 0.5`, capped at `THUMBNAIL_VRAM_BUDGET_MAX_MB`). On CPU, no VRAM budget applies (system RAM is the limit — much larger, no auto-downgrade).

**Auto-downgrade strategy**: On folder open:
```
max_tiles = vram_budget / bytes_per_tile(current_size)
if num_images > max_tiles:
    switch to largest tile size that fits
    notify user via status bar
```

At a 2 GB effective budget, the breakpoints are: 512→2K images, 256→8K, 128→32K, 64→128K. In practice, auto-downgrade kicks in mostly at the 512 setting.

**Auto-upgrade**: Remember the user's last manually chosen tile size. When opening a new folder with fewer images, auto-upgrade back to that preferred size if VRAM permits. The user's intent is preserved — auto-downgrade is a constraint, not a preference change.

### 4.4 Main Image Display

Full-resolution image loaded as a single DPG dynamic texture. Displayed on a drawlist via `dpg.draw_image()` with custom pan/zoom handlers (our own affine math, as in xdot_viewer).

**Zoom modes**:
- Zoom to fit (whole image visible)
- Zoom to 1:1 (actual pixels)
- Zoom in/out (step factor from config)
- Mouse wheel zoom (centered on cursor)

**Pan**: Click-and-drag on main view. Arrow keys when image pane is focused (see §8).

**Display scaling via Lanczos mipmaps**: GPU bilinear texture sampling alone produces visible artifacts at intermediate zoom-out levels (confirmed by xdot_viewer's text rendering experience). Solution: pre-compute Lanczos mip levels at image load time.

1. On image load, generate mip chain (1/2, 1/4, 1/8, … of original) using `raven.common.lanczos` on GPU.
2. Store each mip as a separate DPG dynamic texture.
3. At display time, pick the mip level closest to (but not smaller than) the current zoom level.
4. GPU bilinear handles the remaining interpolation (at most 2× range — well within bilinear's comfort zone).

Mip generation is a one-time cost per image load, amortized by the preload cache. No per-frame Lanczos computation during zoom or pan. No debouncing needed — mip selection is instant (a table lookup based on zoom level).

### 4.5 Preloading

Maintain a cache window of ±N images (configurable, default N=3) around the current position. Preload full-resolution images (including their mip chains) in background threads. When the user navigates to next/previous, the image and its mips are already in memory.

**GIL consideration**: Image loading (PIL decode, QOI decode) is mostly C code, so background threads release the GIL effectively. Numpy array conversion is also C-level. Torch GPU ops release the GIL during kernel execution. Threads should be fine.

**Cache structure**: `OrderedDict` or similar, keyed by filename. Evict furthest-from-current entries when cache exceeds 2×N+1. Each entry holds the numpy array, DPG texture ID for the full-res image, and DPG texture IDs for mip levels.

**Selection-aware preloading**: When tiles are multi-selected (e.g. for comparison), also preload those images. They get priority over positional neighbors. Cache eviction accounts for this — don't evict a selected image just because it's far from the current position.

### 4.6 Lanczos-3 GPU Kernel (`raven/common/lanczos.py`)

Custom implementation, ~150–200 lines of pure PyTorch. Same pattern as the video postprocessor filters — all standard tensor operations, no raw CUDA code. GPU acceleration comes from PyTorch's automatic dispatch.

**Key operations**: `torch.sinc()` (available since PyTorch 1.7; Raven requires ≥2.4), separable 1D horizontal + vertical passes, gather/scatter for neighbor sampling.

**Kernel definition**: `L(x) = sinc(x) · sinc(x/3)` for |x| < 3, else 0.

**Multi-stage downsampling** for large reduction ratios: repeatedly halve (each step uses a compact 6-tap Lanczos kernel) until within 2× of the target, then one final Lanczos pass. Equivalent quality to single-pass Lanczos, much faster.

**API sketch**:
```python
def lanczos_resize(tensor: torch.Tensor,
                   target_h: int,
                   target_w: int) -> torch.Tensor:
    """Resize a (B, C, H, W) tensor using Lanczos-3 interpolation.

    Uses multi-stage downsampling for ratios > 2×.
    Input and output are float32 tensors in [0, 1] range.
    """

def lanczos_mipchain(tensor: torch.Tensor,
                     min_size: int = 64) -> list[torch.Tensor]:
    """Generate a chain of Lanczos-downsampled mip levels.

    `tensor` is (B, C, H, W). Returns [original, 1/2, 1/4, ...] down to
    `min_size` on the short edge.
    """
```

Lives in `raven/common/` for reuse by other Raven apps (e.g. improving the Anime4K final downscale step).

## 5. Grid View

### 5.1 Layout

Scrollable child_window. Tiles laid out in a wrapping grid — number of columns derived from `grid_width / tile_size`. Each tile: thumbnail image + filename (truncated to tile width, full name in tooltip).

**Implementation (v1)**: DPG table with dynamic column count, recalculated on tile size change or panel resize. Each cell is a group containing `dpg.add_image()` + `dpg.add_text()`. For borders/highlights, use per-cell DPG themes or small drawlists. If performance is a problem at 1000+ images, upgrade to virtualized rendering (only create DPG items for visible rows, recycle on scroll).

### 5.2 Tile Visual States

Each tile has two orthogonal visual properties:

**Triage state** (persistent, affects border + icon):
- Neutral — default (subtle gray border, no icon)
- Cherry — **golden/amber** border + star icon at tile corner
- Lemon — **muted gray** border (darker than neutral) + lemon icon at tile corner

Color choice is **colorblind-accessible**: the distinction is luminance-based (warm gold vs cool gray), not red/green. The star/lemon icons provide redundant shape coding.

**Selection state** (transient):
- **Current** — the image shown in the main view. Bright blue thick border. Always exactly one.
- **Multi-selected** — part of a selection group. Highlighted background tint. Zero or more.
- **Neither** — default appearance.

These combine: a tile can be current + cherry + multi-selected simultaneously. Rendering priority (outermost to innermost): current border > triage border > selection tint > triage icon.

### 5.3 Interaction

- **Single click**: Set as current image (shown in main view). Clears multi-selection.
- **Ctrl+click**: Toggle tile in/out of multi-selection. Does NOT change current image.
- **Shift+click**: Range-select from last-clicked to this tile. Does NOT change current image.
- **Double-click**: Set as current + zoom to fit in main view.
- **Right-click**: Context menu (Mark cherry / Mark lemon / Clear mark / Select all / Invert selection).
- **Scroll**: Standard vertical scroll of the grid.

### 5.4 Filter Views

Toolbar toggle buttons (radio group): **All** | **Cherries** | **Lemons** | **Neutral**.

When a filter is active, only matching tiles are shown in the grid. The current image persists even if filtered out (the main view doesn't change, but the grid may not show it).

**Navigation with active filter**: Next/prev respects the filter — skips non-matching images. When the current image is hidden by the filter, navigation starts from the hidden image's position in the full (unfiltered) list and finds the nearest matching image in the requested direction. Never jumps to beginning/end.

## 6. Triage System

### 6.1 States & Operations

Three states per image: `neutral`, `cherry`, `lemon`. Default: `neutral`.

**Mark operations** (apply to current image, or to multi-selection if any):
- Mark as cherry: `C` key (also toolbar button with star icon + tooltip `"Mark cherry [C]"`)
- Mark as lemon: `L` key (also toolbar button with lemon icon + tooltip `"Mark lemon [L]"`)
- Clear mark (back to neutral): `N` key (also toolbar button + tooltip `"Clear mark (Neutral) [N]"`)

### 6.2 File Operations

Marking an image physically moves the file:
- `neutral → cherry`: move `base/foo.jpg` → `base/cherries/foo.jpg`
- `neutral → lemon`: move `base/foo.jpg` → `base/lemons/foo.jpg`
- `cherry → neutral`: move `base/cherries/foo.jpg` → `base/foo.jpg`
- etc.

Subdirectories `cherries/` and `lemons/` are created on first use. The app never deletes them.

### 6.3 Virtual Directory

The grid is built from the union of three directories: `base/`, `base/cherries/`, `base/lemons/`. All image files are collected, tagged with their triage state (derived from which directory they're in), and sorted by filename. The sort order is stable across triage operations — moving a file doesn't change its grid position.

On app launch, any pre-existing files in `cherries/`/`lemons/` are recognized as previously triaged.

**Filename collision handling**: If marking would overwrite an existing file in the target directory, refuse and show an error in the status bar. (Shouldn't happen in normal use, but be defensive.)

## 7. Animation / Compare Mode (Stretch Goal)

Activated via toolbar button or `Space` hotkey when ≥2 tiles are multi-selected.

**Behavior**:
1. Selected tiles are numbered 1–9 (max 9 in v1; if more selected, use first 9 by grid order).
2. The main view cycles through them at configurable speed (default: 3 FPS).
3. A large, prominent number overlay appears on the main view (top-left corner, semi-transparent).
4. Corresponding number badges appear on the grid tiles.
5. The grid tile for the currently displayed image pulses or highlights during its frame.
6. Status bar shows "Compare mode" indicator with current speed.

**Controls during compare mode**:
- `1`–`9`: Stop animation, select that image as current, exit compare mode.
- `Escape`: Stop animation, return to previous current image, exit compare mode.
- `,` / `.`: Slower / faster.
- `Space`: Pause/resume cycling.

Compare mode is an overlay on normal operation — the underlying state (triage, etc.) is unchanged.

## 8. Hotkeys

| Key | Action |
|-----|--------|
| **Navigation** | |
| `←` / `→` | Previous / next image (or pan when image pane focused) |
| `↑` / `↓` | Previous / next row in grid (or pan when image pane focused) |
| `Home` / `End` | First / last image |
| `Page Up` / `Page Down` | Scroll grid by visible rows (last row → first, etc.) |
| **Zoom** | |
| `+` / Numpad `+` | Zoom in |
| `-` / Numpad `-` | Zoom out |
| `F` | Zoom to fit |
| `1` | Zoom to 1:1 (100%) — only outside compare mode |
| Mouse wheel | Zoom (centered on cursor) |
| Click+drag (main view) | Pan |
| **Image pane focus** | |
| `Tab` | Toggle image pane focus |
| Arrow keys | Pan (when image pane focused) |
| `Escape` | Unfocus image pane (return to grid navigation) |
| **Triage** | |
| `C` | Mark current/selected as cherry |
| `L` | Mark current/selected as lemon |
| `N` | Clear mark (neutral) |
| **Selection** | |
| `Ctrl+A` | Select all (visible, per filter) |
| `Ctrl+D` | Deselect all |
| `Ctrl+I` | Invert selection |
| **Compare mode** | |
| `Space` | Enter compare mode (if multi-selected) / pause-resume (if in compare) |
| `1`–`9` | Select numbered image and exit compare |
| `Escape` | Exit compare mode |
| `,` / `.` | Slower / faster |
| **Grid tile size** | |
| `Ctrl+1`–`Ctrl+5` | Switch tile size (32, 64, 128, 256, 512) |
| **App** | |
| `Ctrl+O` | Open folder |
| `F1` | Help card |
| `F11` | Fullscreen toggle |
| `F12` | Dark mode toggle |
| **Debug (hidden)** | |
| `Ctrl+Shift+M` | DPG metrics |
| `Ctrl+Shift+R` | DPG item registry |
| `Ctrl+Shift+T` | DPG font manager |
| `Ctrl+Shift+L` | DPG style editor |

**Arrow key semantics**: Arrows navigate by default. `Tab` toggles image pane focus, where arrows pan instead. `Escape` returns to navigation mode. A visual indicator signals the current mode — e.g. bright border glow on the image pane, and/or status bar text `"Image pane focused — arrows pan, Esc to return"`.

**Note on `1` key**: Outside compare mode, `1` = zoom to 1:1. Inside compare mode, `1` = select image #1. Context-dependent but non-conflicting — compare mode is a clear modal state.

**DPG 2.0 key constant gotcha**: `dpg.mvKey_Prior` (Page Up) and `dpg.mvKey_Next` (Page Down) no longer work in DPG 2.0.0. The key codes are now mysterious magic numbers 517 and 518. Must check both: `key == dpg.mvKey_Prior or key == 517` etc. See `raven/visualizer/app.py:4316–4318` for the workaround pattern.

## 9. Config (`config.py`)

```python
import torch

# GPU configuration.
# Uses raven.common.deviceinfo pattern: {"component": {"device_string": ..., "dtype": ...}}.
# Validated at startup by deviceinfo.validate() — auto-falls back to CPU if CUDA unavailable,
# auto-promotes float16 → float32 on CPU.
gpu_config = {
    "thumbnails": {"device_string": "cuda:0",
                   "dtype": torch.float32},
}

# Font & Layout (derived from DPG style, as in xdot_viewer)
FONT_SIZE = 20
DPG_WINDOW_PADDING_Y = 8
DPG_FRAME_PADDING_Y = 3
DPG_ITEM_SPACING_Y = 4
DPG_SCROLLBAR_SIZE = 14
TOOLBAR_H = FONT_SIZE + 2 * DPG_FRAME_PADDING_Y
STATUS_H = FONT_SIZE

# Window
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1040

# Split
IMAGE_PANE_RATIO = 0.70               # main view gets 70% of width

# Grid
DEFAULT_TILE_SIZE = 128                # initial thumbnail size
TILE_SIZES = [32, 64, 128, 256, 512]
THUMBNAIL_BATCH_SIZE = 32              # images per GPU batch
THUMBNAIL_VRAM_FRACTION = 0.5          # fraction of free VRAM for thumbnails
THUMBNAIL_VRAM_BUDGET_MAX_MB = 4096    # hard cap
THUMBNAIL_INTERPOLATION = "lanczos"    # "lanczos" or "area"
THUMBNAIL_AREA_THRESHOLD = 64          # auto-switch to "area" at tile sizes <= this

# Colors (colorblind-accessible)
CHERRY_COLOR = (220, 180, 50, 255)     # golden/amber
LEMON_COLOR = (100, 100, 105, 255)     # muted gray (darker than neutral)
NEUTRAL_BORDER_COLOR = (60, 60, 65, 255)  # subtle gray
CURRENT_COLOR = (80, 160, 255, 255)    # bright blue
SELECTION_TINT = (255, 255, 255, 40)   # subtle overlay

# Zoom
ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 1.25
MOUSE_WHEEL_ZOOM_FACTOR = 1.1
PAN_AMOUNT = 30                        # pixels per arrow keypress (at 1:1 zoom)

# Preload
PRELOAD_WINDOW = 3                     # ±3 images around current

# Mipmaps
MIP_MIN_SIZE = 64                      # smallest mip level (short edge, pixels)

# Compare mode
COMPARE_DEFAULT_FPS = 3.0
COMPARE_MIN_FPS = 0.5
COMPARE_MAX_FPS = 15.0

# Appearance
DARK_MODE = True
DARK_MODE_BACKGROUND = (45, 45, 48, 255)
LIGHT_MODE_BACKGROUND = (255, 255, 255, 255)

# Help
HELP_WINDOW_W = 1400
HELP_WINDOW_H = 760
```

## 10. Startup Sequence

Following the xdot_viewer / Librarian pattern:

1. NVIDIA workaround (`__GLVND_DISALLOW_PATCHING`) — before any DPG import. See `raven/librarian/app.py:27–32` for the canonical example.
2. Parse CLI args (`argparse`: optional folder path, `--width`, `--height`, `--tile-size`, `--device`)
3. Validate GPU config via `raven.common.deviceinfo.validate(config.gpu_config)` — auto-fallback to CPU, dtype fixup, device info logging.
4. If running on a CUDA device, query available VRAM via `torch.cuda.mem_get_info()`, compute thumbnail VRAM budget.
5. `dpg.create_context()`
6. `guiutils.bootup(font_size=20)` → `themes_and_fonts`
7. `dpg.create_viewport(...)`, `dpg.setup_dearpygui()`
8. Initialize FileDialog (`dirs_only=True` for folder selection)
9. Build GUI layout (toolbar, main view drawlist, grid child_window, status bar)
10. Initialize ImageView (pan/zoom state)
11. Initialize Grid (tile layout state)
12. Initialize Triage manager
13. Register global key handler
14. Build help card
15. Set primary window, resize callback, exit callback
16. Show viewport
17. If folder arg given, scan and load
18. Frame callback for initial resize settling
19. Render loop: `animator.render_frame()` + `dpg.render_dearpygui_frame()`

## 11. Resolved Design Decisions

1. **Resize algorithm**: Custom Lanczos-3 kernel on GPU (v1). Config toggle for `area` mode; auto-switch at smallest tile sizes.
2. **Grid**: DPG table for v1. Virtualize later if needed.
3. **Arrow keys**: Navigate by default. `Tab` toggles image pane focus — arrows pan when focused. `Escape` returns.
4. **Compare mode**: Max 9 images (digit keys) for v1.
5. **Split**: Fixed ratio from config in v1. Future draggable splitter with debounced grid layout regen.
6. **Undo**: Deferred to v2.
7. **Icons**: `ICON_STAR` / `ICON_LEMON` from FontAwesome.
8. **Colors**: Colorblind-accessible (golden/amber vs muted gray), with icon redundancy.
9. **GPU config**: Server-style `{component: {device_string, dtype}}` dict, validated by `deviceinfo.validate()`. Supports CPU fallback.

## 12. Future Enhancements (v2+)

- Draggable split pane (with `make_managed_task` debouncing for grid layout regen)
- Undo/redo for triage operations (`Ctrl+Z` / `Ctrl+Shift+Z`)
- Compare mode with >9 images (paged groups?)
- Custom graphical icons via Qwen-Image
- Grid virtualization for 5000+ image folders
- Metadata overlay (EXIF date, file size) on tiles
- Improve Anime4K final downscale step using `raven.common.lanczos`
