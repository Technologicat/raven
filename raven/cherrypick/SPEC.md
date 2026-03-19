# raven-cherrypick — Specification v0.4

## 1. Name & Concept

**raven-cherrypick** — fast image triage tool. The user opens a folder, reviews images, and sorts them into three piles: **cherries** (keepers), **lemons** (rejects), and **neutral** (undecided, the default). Optimized for comparing variants of the same shot.

Module: `raven/cherrypick/`. Entry point: `raven-cherrypick`.

Icons: `ICON_STAR` (cherry/keeper) and `ICON_LEMON` (reject) from FontAwesome.

## 2. Module Structure

```
raven/cherrypick/
    __init__.py          # re-export main
    app.py               # startup, GUI layout, render loop, hotkey dispatch (~1100 lines)
    config.py            # all constants (~130 lines)
    imageview.py         # main image pane: drawlist, pan, zoom, mip display, texture pool (~960 lines)
    grid.py              # thumbnail grid: layout, tile rendering, selection, filter (~730 lines)
    loader.py            # image I/O, thumbnail generation (GPU batch) (~200 lines)
    preload.py           # preload cache: flat numpy arrays, cross-neighborhood (~350 lines)
    triage.py            # triage state, file move operations, virtual directory merge (~200 lines)

raven/common/image/
    lanczos.py           # GPU Lanczos resize (configurable order 3–5), reusable
```

No `__main__.py` — follows existing Raven convention. Entry point via `pyproject.toml`
`[project.scripts]`, with an `if __name__ == "__main__"` guard at the end of `app.py`.

### Testing

```
raven/common/image/tests/
    test_lanczos.py      # Lanczos kernel: correctness, edge cases, multi-stage vs single-pass

raven/cherrypick/tests/
    test_triage.py        # virtual directory merge, file move logic, collision handling
    test_loader.py        # thumbnail pipeline, mipchain generation, format loading (PNG/JPG/QOI)
```

**Testing philosophy**: Test the algorithm layer; keep the GUI layer thin enough for manual integration testing. Every pure-logic algorithm (Lanczos kernel, triage state machine, virtual directory merge, VRAM budget logic) gets tests. Grid layout math is tightly coupled to DPG rendering, making unit tests fragile — manual testing of the running app covers this adequately. The Lanczos kernel in particular needs:
- Correctness: downscale a known test pattern (e.g. checkerboard, frequency sweep), verify against PIL Lanczos as reference.
- Identity: resize to same size ≈ identity (within floating-point tolerance).
- Separability: horizontal-then-vertical matches a 2D reference.
- Multi-stage: verify multi-stage output matches single-pass output for large ratios.
- Device: run on both CPU and CUDA (if available), results should match.

## 3. GUI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Toolbar]  Open  +  -  Fit  1:1  ★  🍋  ☐  │ Filter ▼  128 ▼     │
│                                               │ FS  Help            │
├──────────────────────────────────────┬──────────────────────────────┤
│                                      │  ┌──┐ ┌──┐ ┌──┐ ┌──┐       │
│                                      │  │  │ │▓▓│ │  │ │  │       │
│         Main Image View              │  └──┘ └──┘ └──┘ └──┘       │
│         (drawlist with pan/zoom)     │  a.jpg b.jpg c.jpg d.jpg    │
│                  [spinner]           │  ┌──┐ ┌──┐ ┌──┐ ┌──┐       │
│                                      │  │  │ │  │ │  │ │  │       │
│                                      │  └──┘ └──┘ └──┘ └──┘       │
│                                      │  e.jpg f.jpg g.jpg h.jpg    │
│                                      │         ...                  │
├──────────────────────────────────────┴──────────────────────────────┤
│ IMG_1234.jpg  [42 / 128] | 4032×3024 | Zoom: 42% | 128 images     │
└─────────────────────────────────────────────────────────────────────┘
```

- **Left ~70%**: Main image pane (drawlist inside child_window). Blue spinning indicator (DPG `loading_indicator`, Visualizer style) overlaid at bottom-left during mip generation and augmentation.
- **Right ~30%**: Thumbnail grid (scrollable child_window).
- **Top**: Toolbar row. All buttons follow the xdot_viewer pattern (30px icon buttons, FontAwesome, tooltips with `"Action [Hotkey]"` format). Filter and tile size are comboboxes.
- **Bottom**: Status bar — current filename with position `[pos / count]`, pixel dimensions, zoom %, total image count with triage counts. Also shows selection count and image-pane-focus indicator when active. Position tracks the visible list, so it stays meaningful under any filter.

**Triage toolbar buttons**: Star, Lemon, and Neutral buttons with tooltips — makes the single-letter hotkeys discoverable. E.g. `"Mark cherry [C]"`, `"Mark lemon [X]"`, `"Clear mark [V]"`.

**Split ratio**: Configurable in `config.py` (default 70/30). Fixed in v1. Architect the layout so that widths are computed from a single ratio variable, making a future draggable splitter straightforward. Draggable splitter will need debounced grid layout recalculation (via `make_managed_task` from `raven.common.bgtask`). Note: splitter resize does NOT require thumbnail regeneration — the thumbnails are resolution-independent of the grid panel width. Only the grid layout (column count, row arrangement) needs recalculating.

## 4. Image Pipeline

### 4.1 Supported Formats

- **PNG, JPG**: via PIL (`Image.open`)
- **QOI**: via `qoi.decode()` (the `qoi` package, already a dependency — used in avatar streaming)
- Anything else PIL handles (BMP, TIFF, WebP) comes free; no explicit support needed.

Detection by extension (`.png`, `.jpg`, `.jpeg`, `.qoi`). The directory scanner collects files matching these.

### 4.2 Thumbnail Generation

**Engine**: PyTorch GPU batch processing, with automatic CPU fallback via `raven.common.deviceinfo`.

**Resize algorithm**: Custom Lanczos kernel in `raven/common/image/lanczos.py` (see §4.6). Configurable kernel order (3, 4, or 5; default 4). Extremely fast on GPU and produces excellent quality at all reduction ratios.

For efficiency at extreme reduction ratios (e.g. 4000→128, ~30×), use multi-stage downsampling: repeatedly halve with Lanczos until within 2× of the target, then one final Lanczos resize. This keeps the kernel compact per stage while maintaining quality.

**Pipeline**:
1. Load batch of images on CPU (PIL/QOI → numpy RGBA uint8)
2. Stack as `torch.Tensor`, transfer to configured device
3. Resize to target thumbnail size (square; letterbox non-square originals)
4. Transfer result to CPU, convert to float32 [0,1] RGBA
5. Create DPG dynamic textures

**DPG texture management**:
- Use counter-based unique tags (e.g. `f"grid_thumb_tex_{counter}"`) — DPG tag collision = crash.
- Use descriptive tag prefixes throughout for easier debugging.
- Hidden debug hotkeys ("Mr. T Lite"): Ctrl+Shift+M (metrics), R (registry), T (fonts), L (style editor).

**NVIDIA workaround**: At module top, before importing DPG (see `raven/librarian/app.py:27–32` for the canonical example):
```python
if platform.system().upper() == "LINUX":
    os.environ["__GLVND_DISALLOW_PATCHING"] = "1"
```
Required to avoid segfault when deleting textures on Linux/NVIDIA (DPG issue #554).

**Batching**: Configurable batch size (default 32). Process in background thread via `TaskManager`. Grid shows VHS noise placeholder tiles (unique per tile, generated from `raven.common.video.postprocessor.vhs_noise_pool`) until thumbnails are ready; tiles pop in progressively.

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

**Auto-downgrade strategy** [not yet implemented]: On folder open:
```
max_tiles = vram_budget / bytes_per_tile(current_size)
if num_images > max_tiles:
    switch to largest tile size that fits
    notify user via status bar
```

At a 2 GB effective budget, the breakpoints are: 512→2K images, 256→8K, 128→32K, 64→128K. In practice, auto-downgrade kicks in mostly at the 512 setting.

**Auto-upgrade** [not yet implemented]: Remember the user's last manually chosen tile size. When opening a new folder with fewer images, auto-upgrade back to that preferred size if VRAM permits. The user's intent is preserved — auto-downgrade is a constraint, not a preference change.

### 4.4 Main Image Display

Displayed on a drawlist via `dpg.draw_image()` with custom pan/zoom handlers (our own affine math, as in xdot_viewer). Mip textures are managed via a texture pool (`set_value` on pooled textures for fast recycling without OpenGL allocation churn).

**Zoom modes**:
- Zoom to fit (whole image visible)
- Zoom to 1:1 (raw pixel blit, no scaling — important for screenshots with text). Draw position snapped to integer coordinates to prevent GPU bilinear interpolation artifacts.
- Zoom in/out (step factor from config)
- Mouse wheel zoom (centered on cursor)

**Pan**: Click-and-drag on main view. Arrow keys when image pane is focused (see §8).

**Display scaling via Lanczos mipmaps**: GPU bilinear texture sampling alone produces visible artifacts at intermediate zoom-out levels (confirmed by xdot_viewer's text rendering experience). Solution: pre-compute Lanczos mip levels at image load time.

1. On image load, generate mip chain (1/2, 1/4, 1/8, … of original) using `raven.common.image.lanczos` on GPU. Background thread, smallest-first for progressive sharpening.
2. Store each mip as a separate DPG dynamic texture (via texture pool).
3. At display time, pick the mip level closest to (but not smaller than) the current zoom level.
4. GPU bilinear handles the remaining interpolation (at most 2× range — well within bilinear's comfort zone).

Mip generation is a one-time cost per image load, amortized by the preload cache. No per-frame Lanczos computation during zoom or pan. No debouncing needed — mip selection is instant (a table lookup based on zoom level).

**Loading indicator**: Blue DPG `loading_indicator` (style 0, spinning dots) overlaid at the bottom-left of the image pane. Shows during both full mip generation (new image) and augmentation of preloaded images. Positioned via `pos` inside a `child_window` wrapper to avoid affecting layout of the status bar.

### 4.5 Preloading

Maintain a cache of nearby images in a cross-neighborhood pattern: ±N tiles horizontally and vertically in the grid (configurable, default N=2). Preload images (including capped mip chains) in background threads. When the user navigates to a cached image, the switch is instant.

**GIL consideration**: Image loading (PIL decode, QOI decode) is mostly C code, so background threads release the GIL effectively. Numpy array conversion is also C-level. Torch GPU ops release the GIL during kernel execution. Threads should be fine.

**Cache structure**: `PreloadCache` class in `preload.py`. Stores **flat numpy arrays** (not DPG textures) to avoid per-frame DPG registry overhead — DPG dynamic textures have O(n) per-frame cost that becomes significant at ~80+ textures. GPU texture readback (`dpg.get_value`) is also catastrophically slow on gaming-grade GPUs (observed ~540ms over Thunderbolt vs ~24ms for upload), so textures are never read back — the cache stores CPU-side arrays and uploads on demand. Each cache entry holds a list of `(scale, w, h, flat_array)` tuples for the mip levels. Eviction is furthest-from-current-first, with a configurable RAM budget (default: 25% of system RAM, clamped to [512 MB, 16 GB]).

**Preload mip cap**: Speculative preloads are capped at `PRELOAD_MAX_SCALE` (default 0.25×) — the full-res and 0.5× levels are skipped. When the user actually navigates to a preloaded image, `augment_mips` generates the missing larger levels in the background. This keeps the cache compact while still providing instant perceived navigation.

**Selection-aware preloading** [not yet implemented]: When tiles are multi-selected (e.g. for comparison), also preload those images. They get priority over positional neighbors. Cache eviction accounts for this — don't evict a selected image just because it's far from the current position.

### 4.6 Lanczos GPU Kernel (`raven/common/image/lanczos.py`)

Custom implementation, pure PyTorch. Same pattern as the video postprocessor filters — all standard tensor operations, no raw CUDA code. GPU acceleration comes from PyTorch's automatic dispatch.

**Configurable kernel order**: Lanczos-*a* for *a* ∈ {3, 4, 5}. Default order 4 (good balance of stopband rejection and computation). Higher order = sharper but slightly more compute.

**Key operations**: `torch.sinc()` (available since PyTorch 1.7; Raven requires ≥2.4), separable 1D horizontal + vertical passes, gather/scatter for neighbor sampling.

**Kernel definition**: `L(x) = sinc(x) · sinc(x/a)` for |x| < a, else 0.

**Multi-stage downsampling** for large reduction ratios: repeatedly halve (each step uses a compact kernel) until within 2× of the target, then one final Lanczos pass. Equivalent quality to single-pass Lanczos, much faster.

**API**:
```python
def resize(tensor: torch.Tensor,
           target_h: int,
           target_w: int,
           order: int = 4) -> torch.Tensor:
    """Resize a (B, C, H, W) tensor using Lanczos interpolation."""

def mipchain(tensor: torch.Tensor,
             min_size: int = 64,
             order: int = 4) -> list[torch.Tensor]:
    """Generate a chain of Lanczos-downsampled mip levels.

    Returns [original, 1/2, 1/4, ...] down to `min_size` on the short edge.
    """
```

Lives in `raven/common/image/` for reuse by other Raven apps (e.g. improving the Anime4K final downscale step).

## 5. Grid View

### 5.1 Layout

Scrollable child_window. Tiles laid out in a wrapping grid — number of columns derived from `grid_width / (tile_size + item_spacing)`. Each tile: per-tile drawlist (for full control over borders, image, and icon overlays) + filename label (truncated to tile width, full name in tooltip).

**Implementation**: Horizontal groups of per-tile drawlists, torn down and rebuilt on filter/size changes. O(1) hit detection from mouse position via layout math. If performance is a problem at 1000+ images, upgrade to virtualized rendering (only create DPG items for visible rows, recycle on scroll).

### 5.2 Tile Visual States

Each tile has two orthogonal visual properties:

**Triage state** (persistent, affects border + icon + dimming):
- Neutral — default (subtle gray border, no icon)
- Cherry — **golden/amber** border + star icon at tile corner
- Lemon — **muted dark gray** border (darker than neutral) + lemon icon + dimming overlay (rejects fade into background)

Color choice is **colorblind-accessible**: the distinction is luminance-based (warm gold vs cool gray), not red/green. The star/lemon icons provide redundant shape coding.

**Selection state** (transient):
- **Current** — the image shown in the main view. Bright blue inner border. Always exactly one.
- **Multi-selected** — part of a selection group. Highlighted background tint. Zero or more.
- **Neither** — default appearance.

These combine: a tile can be current + cherry + multi-selected simultaneously. Rendering priority (outermost to innermost): lemon dimming > selection tint > triage border > current border > triage icon.

### 5.3 Interaction

- **Single click**: Set as current image (shown in main view) and select it (replaces multi-selection).
- **Ctrl+click**: Toggle tile in/out of multi-selection. Does NOT change current image.
- **Shift+click**: Range-select from last-clicked to this tile in visible order.
- **Double-click**: Set as current + zoom to fit in main view.
- **Right-click**: Currently unused. A context menu may not be needed — the app is simple enough that hotkeys and toolbar buttons cover all operations.
- **Scroll**: Standard vertical scroll of the grid.

### 5.4 Filter Views

Toolbar combobox: **All** | **Cherries** | **Lemons** | **Neutral**. Keyboard: `G` cycles forward, `Shift+G` cycles backward.

When a filter is active, only matching tiles are shown in the grid. The current image persists even if filtered out (the main view doesn't change, but the grid may not show it).

**Navigation with active filter**: Next/prev respects the filter — skips non-matching images. When the current image is hidden by the filter, navigation starts from the hidden image's position in the full (unfiltered) list and finds the nearest matching image in the requested direction. Never jumps to beginning/end.

## 6. Triage System

### 6.1 States & Operations

Three states per image: `neutral`, `cherry`, `lemon`. Default: `neutral`.

**Mark operations** (apply to current image; Ctrl variants apply to multi-selection):
- Mark as cherry: `C` key, `Ctrl+C` for selection (also toolbar button with star icon)
- Mark as lemon: `X` key, `Ctrl+X` for selection (also toolbar button with lemon icon)
- Clear mark (back to neutral): `V` key, `Ctrl+V` for selection (also toolbar button)

The triage keys (`X`/`C`/`V`) are on the left hand, freeing the right for the mouse. The jump-to-state keys (`B`/`N`/`M`) are the next three-column set to the right on QWERTY — same order, easy to remember.

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

Not yet implemented. Config values (`COMPARE_DEFAULT_FPS`, etc.) are in place.

Activated via toolbar button or hotkey when ≥2 tiles are multi-selected.

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
| `Page Up` / `Page Down` | Scroll grid by visible rows |
| **Jump to triage state** (All view only; wraps around) | |
| `B` / `Shift+B` | Next / prev lemon |
| `N` / `Shift+N` | Next / prev cherry |
| `M` / `Shift+M` | Next / prev neutral |
| **Zoom** | |
| `+` / Numpad `+` | Zoom in |
| `-` / Numpad `-` | Zoom out |
| `F` | Zoom to fit |
| `1` | Zoom to 1:1 (100%) |
| Mouse wheel | Zoom (centered on cursor) |
| Click+drag (main view) | Pan |
| **Image pane focus** | |
| `Tab` | Toggle image pane focus |
| Arrow keys | Pan (when image pane focused) |
| `Escape` | Unfocus image pane (return to grid navigation) |
| **Triage** | |
| `C` / `Ctrl+C` | Mark cherry (current / all selected) |
| `X` / `Ctrl+X` | Mark lemon (current / all selected) |
| `V` / `Ctrl+V` | Clear mark (current / all selected) |
| **Filter** | |
| `G` | Cycle filter forward (All → Cherries → Lemons → Neutral) |
| `Shift+G` | Cycle filter backward |
| **Selection** | |
| `Space` | Toggle select current image |
| `Ctrl+A` | Select all (visible, per filter) |
| `Ctrl+D` | Deselect all |
| `Ctrl+I` | Invert selection |
| **Grid tile size** | |
| `Ctrl+1`–`Ctrl+5` | Switch tile size (32, 64, 128, 256, 512) |
| **App** | |
| `Ctrl+O` | Open folder |
| `F1` | Help card |
| `F11` | Fullscreen toggle |
| **Debug (hidden)** | |
| `Ctrl+Shift+M` | DPG metrics + toggle debug overlay |
| `Ctrl+Shift+R` | DPG item registry |
| `Ctrl+Shift+T` | DPG font manager |
| `Ctrl+Shift+L` | DPG style editor |

**Arrow key semantics**: Arrows navigate by default. `Tab` toggles image pane focus, where arrows pan instead. `Escape` returns to navigation mode. A bright blue border on the image pane and status bar text `"IMAGE PANE FOCUSED"` signal the current mode.

**Triage key layout**: `X`/`C`/`V` (mark lemon/cherry/neutral) on the left hand, `B`/`N`/`M` (jump to lemon/cherry/neutral) are the next three-column set to the right on QWERTY. Same order, easy to remember.

**DPG 2.0 key constant gotcha**: `dpg.mvKey_Prior` (Page Up) and `dpg.mvKey_Next` (Page Down) no longer work in DPG 2.0.0. The key codes are now mysterious magic numbers 517 and 518. Must check both: `key == dpg.mvKey_Prior or key == 517` etc. See `raven/visualizer/app.py:4316–4318` for the workaround pattern.

## 9. Startup Sequence

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
10. Initialize ImageView (pan/zoom, texture pool, spinner overlay)
11. Initialize Grid (tile layout, VHS noise placeholder pool)
12. Register global key handler
13. Build help card
14. Set primary window, resize callback, exit callback, vsync
15. Show viewport
16. If folder arg given: scan with TriageManager, start thumbnail pipeline, initialize PreloadCache
17. Frame callback for initial resize settling
18. Render loop: poll thumbnail pipeline, poll deferred noise pool, update components, trigger preloading, `animator.render_frame()` + `dpg.render_dearpygui_frame()`
19. Two-phase shutdown: cancel tasks in exit callback (without waiting), wait for threads after render loop exits, destroy DPG context.

## 10. Resolved Design Decisions

1. **Resize algorithm**: Custom Lanczos kernel on GPU, configurable order (3–5, default 4). Extremely fast and high quality; no need for `area` interpolation fallback.
2. **Grid**: Per-tile drawlists in horizontal groups (not DPG table). Full rebuild on filter/size change. Virtualize later if needed.
3. **Arrow keys**: Navigate by default. `Tab` toggles image pane focus — arrows pan when focused. `Escape` returns.
4. **Compare mode**: Max 9 images (digit keys) for v1. Not yet implemented.
5. **Split**: Fixed ratio from config in v1. Future draggable splitter with debounced grid layout regen.
6. **Undo**: Deferred to v2.
7. **Icons**: `ICON_STAR` / `ICON_LEMON` from FontAwesome.
8. **Colors**: Colorblind-accessible (golden/amber vs muted dark gray), with icon redundancy. Lemons get dimming overlay.
9. **GPU config**: Server-style `{component: {device_string, dtype}}` dict, validated by `deviceinfo.validate()`. Supports CPU fallback.
10. **Triage keys**: `X`/`C`/`V` (left hand, near mouse). Jump keys `B`/`N`/`M` mirror them one column right.
11. **Preload cache**: Flat numpy arrays (not DPG textures). DPG dynamic textures have O(n) per-frame overhead — at 80+ cached textures, frame time degrades significantly. Numpy arrays have zero per-frame cost; DPG textures are created on demand via the texture pool when the image is actually displayed.
12. **Texture pool**: Reuse DPG textures via `set_value` instead of create/delete. Avoids OpenGL allocation churn and the associated glitches on NVIDIA/Linux.
13. **Dark mode**: Not needed. The default color scheme is already dark, and images must be shown as-is for triage accuracy.

## 11. Future Enhancements (v2+)

- Draggable split pane (with `make_managed_task` debouncing for grid layout regen)
- Undo/redo for triage operations (`Ctrl+Z` / `Ctrl+Shift+Z`)
- Compare mode (§7)
- Compare mode with >9 images (paged groups?)
- Custom graphical icons via Qwen-Image
- Grid virtualization for 5000+ image folders
- Metadata overlay (EXIF date, file size) on tiles
- Improve Anime4K final downscale step using `raven.common.image.lanczos`
