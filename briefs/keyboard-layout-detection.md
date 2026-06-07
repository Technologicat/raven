# Keyboard layout detection / positional hotkeys — research notes

Prep material for the deferred item *"Keyboard-layout-aware positional hotkeys across the fleet"*
(`TODO_DEFERRED.md`). The motivating case is `raven-cherrypick`'s WASD navigation (a positional
alias for the arrow keys): on AZERTY (French/Belgian) the WASD physical cluster is **ZQSD**, and on
QWERTZ (German/Swiss) `Z`/`Y` are swapped — so a character-based WASD binding lands under the wrong
fingers for those users.

This brief records what's true about layout detection on Linux / macOS / Windows so the eventual
implementation starts from facts, not guesses. Confidence is flagged per claim; the one item marked
**CONFIRM** is a 2-minute local test worth doing before writing any code.

## The core constraint: DPG gives us *translated* keys, and hides scancodes

There are two ways a toolkit can report a key press:

- **Physical / untranslated** (a *scancode*): "the key in the top-left letter position," stable
  regardless of layout. This is what `KeyboardEvent.code` on the web and `QKeyEvent::nativeScanCode`
  on Qt expose. Bind to these and WASD *automatically* becomes ZQSD on AZERTY — no detection needed,
  because they are the same physical keys.
- **Translated / virtual** (a *character*): "the key that produces 'W' on the current layout."

Dear ImGui's GLFW and SDL backends submit **translated** keys: `ImGuiKey_W` is delivered for the key
the user must press to type a 'W' (ImGui issues [#2959](https://github.com/ocornut/imgui/issues/2959),
[#3141](https://github.com/ocornut/imgui/pull/3141)). DearPyGui sits on ImGui, and its own
`briefs/dpg-keycodes.md` describes incoming codes as "Windows virtual-key-style" — and Windows VK
codes for letter keys are themselves layout-translated. So **`dpg.mvKey_W` is layout-dependent**, and
a grep of the installed `dearpygui` Python package finds **no scancode and no `glfwGetKeyName`** —
DPG's Python API does not expose the physical layer at all. (High confidence from the above; see
**CONFIRM** below for the cheap local check.)

Consequence: within DPG as it ships, we **cannot** bind by physical position. The physical route would
require extending our DPG usage to reach ImGui's untranslated/scancode path or GLFW's
`glfwGetKeyName` (GLFW scancodes are documented as "unique for every key … consistent over time, …
safe to save to disk"). That's a vendored-DPG extension, not a config tweak — noted as Strategy C
below.

### CONFIRM (do this first, ~2 min)

On a dev box: `setxkbmap fr` (X11), launch `raven-cherrypick`, and in `_on_key` log the raw `app_data`
while pressing the **physical** WASD-up key (top-left letter area). If it reports `mvKey_Z` (not
`mvKey_W`), DPG is confirmed layout-dependent and Strategy C is the only way to get true physical
binding. Restore with `setxkbmap us`. (Under Wayland, `setxkbmap` may not take — switch layout via the
desktop's input-source UI instead.)

## Strategies, ranked

**A. Config override (ship this first — trivial, reliable everywhere).**
A `config.py` setting picking the cluster: `WASD` (default), `ZQSD`, or an explicit list. Zero
detection, works on every OS and on Wayland. Immediately unblocks AZERTY/QWERTZ users who set it. This
is the backbone the other strategies *fall back to* — detection should only ever change the *default*.

**B. OS position→char query (best detection within DPG's by-character model).**
DPG fires `mvKey_<char>`, so the bridge is: ask the OS *"what characters do the physical WASD/QE
position keys produce on the current layout?"* and bind **those** character-mvKeys — and reuse the same
characters as the labels in the help card / tooltips. The physical scancodes of the WASD positions are
fixed US-QWERTY constants (e.g. Windows scancodes W=0x11, A=0x1E, S=0x1F, D=0x20; Linux evdev = those
+8). This handles **every** layout (AZERTY, QWERTZ, Dvorak, Colemak, …) with no hardcoded remap table,
because it reads the actual layout. Per-OS APIs in the appendix. Wayland is the gap (see below).

**C. True physical binding (most robust; biggest lift).**
Extend the vendored DPG / our key handling to expose scancodes (ImGui untranslated key path) or
`glfwGetKeyName`, then bind by physical position and read labels from `glfwGetKeyName`. Needs zero
runtime detection and auto-adapts to all layouts — but it's a real change to how Raven receives key
events fleet-wide, so it's a separate project, not part of the first pass.

**Layout-name → lookup-table** (detect "fr" → swap to ZQSD) is the obvious-but-inferior middle path:
it only covers layouts we enumerate and needs maintenance. Prefer **B** (position→char) over it — same
per-OS plumbing, strictly more coverage.

Recommendation: ship **A** now (one config knob), layer **B** on top as a best-effort auto-default,
keep **C** on the someday list. Whatever the binding strategy, the **help card and tooltips must show
the labels actually on the user's keycaps** — so the label-resolution half of B is needed even if
binding stays character-based.

## Per-OS appendix (detection APIs)

### Windows
- **Layout name:** `user32.GetKeyboardLayout(threadId)` → HKL (low word = LANGID); or
  `GetKeyboardLayoutNameW` → 8-hex **KLID** string (`00000409` US, `0000040C` French, `00000407`
  German). Pure `ctypes`, no third-party dep.
- **Position→char (Strategy B):** `MapVirtualKeyExW` (scancode↔VK) + `ToUnicodeEx` / `VkKeyScanExW`
  to ask what a scancode currently produces. Layout-agnostic.
- **Live changes:** `WM_INPUTLANGCHANGE`. A DPG app doesn't get the raw window message easily, so
  poll on focus-gain or periodically instead.
- Win8+ also has WinRT `Windows.Globalization.Language.CurrentInputMethodLanguageTag`, but `ctypes` +
  user32 is lighter and dependency-free.

### macOS
- **Layout name:** Text Input Sources (TIS) API in the **HIToolbox** framework via `pyobjc`. The C
  functions aren't auto-bridged — load them from the bundle:
  `NSBundle.bundleWithIdentifier_("com.apple.HIToolbox")` then `objc.loadBundleFunctions(...)` for
  `TISCopyCurrentKeyboardInputSource` and `TISGetInputSourceProperty`. Read
  `kTISPropertyInputSourceID` → `"com.apple.keylayout.US"` / `".French"` / `".German"` etc.
- **Position→char (Strategy B):** `UCKeyTranslate` with the layout's
  `kTISPropertyUnicodeKeyLayoutData` (uchr data).
- **Live changes:** `kTISNotifySelectedKeyboardInputSourceChanged` distributed notification.
- **Dep:** `pyobjc` (Cocoa/Quartz). Note Apple removed the public TIS docs; the
  [pudquick gist](https://gist.github.com/pudquick/cff1ecdc02b4cabe5aa0dc6919d97c6d) is the working
  reference. A 2022 note there flags `kTISPropertyScriptCode` as gone on newer macOS — verify property
  constants against the running OS at implementation time.

### Linux — X11 / Xorg
- **Layout name:** `setxkbmap -query` (parse `layout: fr`), or the XKB extension via `python-xlib`,
  or the [`xkbgroup`](https://pypi.org/project/xkbgroup/) PyPI package (active **group** + layout
  symbols). The active *group* matters: multi-layout users switch groups, so read the group, not just
  the configured list.
- **Position→char (Strategy B):** XKB `xkb_state_key_get_utf8` (libxkbcommon) for the WASD scancodes.
- **Dep:** `python-xlib` or `libxkbcommon` bindings (or just shell out to `setxkbmap`).

### Linux — Wayland (the gap)
By design there is **no global "what is my layout" query** — it's per-compositor and per-client. The
compositor delivers the active XKB keymap to the *focused* client over `wl_keyboard.keymap` (an fd of
XKB text); the toolkit (GLFW) parses it, but DPG doesn't surface it. Practical options, all
best-effort:
- Read the desktop's input-source config: GNOME
  `gsettings get org.gnome.desktop.input-sources sources`; KDE `~/.config/kxkbrc`. Fragile, per-DE.
- `localectl status` gives the *system* default, not necessarily the active per-session layout.
- Under **XWayland**, X11 queries (`setxkbmap -query`) often still report correctly — worth testing on
  the target systems.
- Otherwise: **lean on Strategy A** (config override) on Wayland. This is the strongest reason the
  config knob must exist regardless of how good detection gets.

## Sources / further reading

- Web games — physical vs label keys: <https://www.bram.us/2022/03/31/wasd-controls-on-the-web/>
- Qt cross-platform layout handling (nativeScanCode): <https://forum.qt.io/topic/95527>
- Dear ImGui translated-vs-untranslated keys: <https://github.com/ocornut/imgui/issues/2959>,
  <https://github.com/ocornut/imgui/pull/3141>
- GLFW input guide (scancodes, `glfwGetKeyName`): <https://www.glfw.org/docs/3.3/input_guide.html>
- Windows `GetKeyboardLayoutName` / KLID:
  <https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getkeyboardlayoutnamew>
- macOS TIS via pyobjc: <https://gist.github.com/pudquick/cff1ecdc02b4cabe5aa0dc6919d97c6d>
- Linux `xkbgroup`: <https://pypi.org/project/xkbgroup/>
