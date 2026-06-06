# DPG key constants: the `mvKey_*` ↔ runtime-code mismatch

*2026-06-06, Claude Opus 4.8 (1M context)*

Reference note (not an architectural brief). Measured against **dearpygui 2.1.1**.

## TL;DR

DPG hands a key-press handler the **live ImGuiKey code** in `app_data`. For most
keys, the `dpg.mvKey_*` named constant equals that code, so `app_data ==
dpg.mvKey_C` works. But for a handful of keys, the `mvKey_*` constant is a
**stale legacy Windows-VK value** that does *not* match the code actually
delivered at runtime. Comparing `app_data` against those constants silently
never matches.

The "mysterious 517/518" in the Visualizer were the first instances we hit:

- **Page Up** arrives as **517**, but `dpg.mvKey_Prior` == **266**.
- **Page Down** arrives as **518**, but `dpg.mvKey_Next` == **267**.

There is no correct named constant for these — the only reliable comparison is
against the literal ImGuiKey code.

## The trapped keys

Legacy `mvKey_*` value (< 512) vs. the ImGuiKey code DPG actually delivers
(≥ 512, left as a gap in the constant table):

| Key | DPG constant (value) | Runtime code | Status |
|---|---|---|---|
| Page Up | `mvKey_Prior` (266) | **517** | field-confirmed |
| Page Down | `mvKey_Next` (267) | **518** | field-confirmed |
| Left Super / Win | `mvKey_LWin` (343) | 530 | inferred (gap) |
| Right Super / Win | `mvKey_RWin` (347) | 534 | inferred (gap) |
| `'` apostrophe | `mvKey_Quote` (39) | 596 | inferred (gap) |
| `;` semicolon | `mvKey_Colon` (59) | 601 | inferred (gap) |
| `=` equal | `mvKey_Plus` (61) | 602 | inferred (gap) |
| `` ` `` grave | `mvKey_Tilde` (96) | 606 | inferred (gap) |

"Field-confirmed" = observed in `app_data` from a real keypress. "Inferred
(gap)" = the constant carries a legacy value and the corresponding ImGuiKey slot
is empty in the constant table, so the same mechanism applies; confirm with a
keypress logger before relying on the exact number if you wire one of these.

Two further phantoms have **no** ImGuiKey equivalent at all and are never
delivered: `mvKey_Clear` (259) and `mvKey_F25` (314) — ImGui's function-key
range stops at F24.

Everything else is safe: letters, digits, F1–F24, Tab, the four arrows,
Home/End/Insert/Delete, Backspace/Enter/Space/Escape, the L/R Control-Shift-Alt
pairs, Menu, the numpad, and the remaining punctuation all have constants that
equal their runtime codes.

## Why this happens

History, because it's load-bearing here. In **DPG 1.x**, key presses were
reported with Windows virtual-key-style codes, and the `mvKey_*` constants
matched them — Page Up genuinely arrived as 266 (`mvKey_Prior`), Page Down as
267 (`mvKey_Next`). **DPG 2.0** rebased the *delivered* codes onto the ImGuiKey
enum (named keys start at `ImGuiKey_Tab` = 512), so Page Up now arrives as 517
and Page Down as 518. Most `mvKey_*` constants were regenerated to match the new
scheme — hence `mvKey_Tab` == 512, `mvKey_A` == 546, `mvKey_C` == 548 — but a
scattering of them (Prior/Next, LWin/RWin, Quote/Colon/Plus/Tilde) were **left
at their 1.x values**. The migration simply forgot them.

So each stale constant now points at a code that is *never delivered*, and the
key's true ImGuiKey code shows up as a **hole** in the constant table. The
517/518 here were captured empirically exactly that way: the old Prior/Next
codes stopped arriving after the 2.0 upgrade, and a keypress logger reported the
new ones.

The reason `app_data` carries the ImGuiKey code (not the constant) is that
`IsKeyPressed`-style edge detection in ImGui keys off the enum directly; DPG
passes that through verbatim.

## Practical guidance

- For the trapped keys, compare `app_data` against the **literal** runtime code
  (517, 518, …), not the `mvKey_*` constant. Comment the literal so it isn't
  mistaken for a magic number.
- For anything not in the trapped-keys table, `dpg.mvKey_*` is fine.
- `mvKey_Mod*` (`ModCtrl` 4096, `ModShift` 8192, `ModAlt` 16384, `ModSuper`
  32768) are **bit flags**, not key codes — used for chord/modifier state, not
  matched against `app_data`.
- The group of constants reported as **-1** (`mvKey_Apps`, the `Browser_*`,
  `Media_*`, `Launch_*`, `Volume_*`, `Sleep`, `Help`, …) are unmapped in this
  build — they all collapse to -1, so they can't be distinguished and shouldn't
  be relied on.

## Related: same-frame dispatch order

A practical consequence of the table above: a keyless key-press handler
(`dpg.add_key_press_handler(callback=...)`) is dispatched **once per key pressed
that frame, in ascending keycode order** — *not* in the temporal order the keys
were physically struck. ImGui's per-frame edge detection collapses everything
since the last `glfwPollEvents` into "pressed this frame" and discards the
sub-frame ordering.

Every triage letter sorts *after* every arrow (`C`=548, `V`=567, `X`=569 vs.
`Left`=513 … `Down`=516), so two keys struck within one ~16 ms frame are always
processed arrow-first. In raven-cherrypick this surfaced as a fast `C`+`Right`
(cherry, then next) being applied as `Right`+`C` (move, then cherry) — landing
the tag on the wrong image. If correctness depends on the order of two
near-simultaneous keys, don't rely on dispatch order; defer the
lower-keycode action (e.g. navigation) by a frame so the higher-keycode action
(triage) still sees pre-navigation state.

## How to reproduce the table

No display/context needed — the constants are plain module attributes:

```python
"""Dump DPG's mvKey_* constants as a code -> name(s) table, and flag the
constants whose value does not match the ImGuiKey code delivered at runtime.

Run:  python briefs/dpg-keycodes.py    (or paste into a REPL)
"""
import importlib.metadata as _md
from collections import defaultdict

import dearpygui.dearpygui as dpg

print("dearpygui", _md.version("dearpygui"))

by_val = defaultdict(list)
for name in dir(dpg):
    if name.startswith("mvKey_"):
        v = getattr(dpg, name)
        if isinstance(v, int):
            by_val[v].append(name)

print(f"{sum(len(n) for n in by_val.values())} names, {len(by_val)} distinct codes\n")
for v in sorted(by_val):
    print(f"{v:>6}  {'  '.join(sorted(by_val[v]))}")

# Gaps in the ImGuiKey named range = codes delivered at runtime with no constant.
# A legacy constant (value < 512) almost always corresponds to one of these gaps.
print("\nGaps in 512..666 (runtime codes with no mvKey_* constant):")
for v in range(512, 667):
    if v not in by_val:
        print(f"  {v}")
print("\nLegacy constants (value < 512, excluding None/-1):")
for v in sorted(k for k in by_val if 0 < k < 512):
    print(f"  {v:>4}  {'  '.join(by_val[v])}")
```

## Appendix: full table (dearpygui 2.1.1)

Codes below 512 (except 0) and the 517/518/530/534/596/601/602/606 gaps are the
trap; see above. `-1` is the catch-all for unmapped keys.

```
  -1  mvKey_Apps  mvKey_Browser_Favorites  mvKey_Browser_Home  mvKey_Browser_Refresh
      mvKey_Browser_Search  mvKey_Browser_Stop  mvKey_Execute  mvKey_Help
      mvKey_Launch_App1  mvKey_Launch_App2  mvKey_Launch_Mail  mvKey_Launch_Media_Select
      mvKey_Media_Next_Track  mvKey_Media_Play_Pause  mvKey_Media_Prev_Track
      mvKey_Media_Stop  mvKey_ModDisabled  mvKey_Select  mvKey_Sleep
      mvKey_Volume_Down  mvKey_Volume_Mute  mvKey_Volume_Up
   0  mvKey_None
  39  mvKey_Quote          (apostrophe; runtime 596)
  59  mvKey_Colon          (semicolon;  runtime 601)
  61  mvKey_Plus           (equal;      runtime 602)
  96  mvKey_Tilde          (grave;      runtime 606)
 259  mvKey_Clear          (phantom; no ImGuiKey)
 266  mvKey_Prior          (Page Up;    runtime 517)
 267  mvKey_Next           (Page Down;  runtime 518)
 314  mvKey_F25            (phantom; ImGui stops at F24)
 343  mvKey_LWin           (LeftSuper;  runtime 530)
 347  mvKey_RWin           (RightSuper; runtime 534)
 512  mvKey_Tab
 513  mvKey_Left
 514  mvKey_Right
 515  mvKey_Up
 516  mvKey_Down
 519  mvKey_Home
 520  mvKey_End
 521  mvKey_Insert
 522  mvKey_Delete
 523  mvKey_Back           (Backspace)
 524  mvKey_Spacebar
 525  mvKey_Return         (Enter)
 526  mvKey_Escape
 527  mvKey_LControl
 528  mvKey_LShift
 529  mvKey_LAlt
 531  mvKey_RControl
 532  mvKey_RShift
 533  mvKey_RAlt
 535  mvKey_Menu
 536–545  mvKey_0 … mvKey_9
 546–571  mvKey_A … mvKey_Z   (C=548, V=567, X=569)
 572–595  mvKey_F1 … mvKey_F24
 597  mvKey_Comma
 598  mvKey_Minus
 599  mvKey_Period
 600  mvKey_Slash
 603  mvKey_Open_Brace
 604  mvKey_Backslash
 605  mvKey_Close_Brace
 607  mvKey_CapsLock
 608  mvKey_ScrollLock
 609  mvKey_NumLock
 610  mvKey_Print
 611  mvKey_Pause
 612–621  mvKey_NumPad0 … mvKey_NumPad9
 622  mvKey_Decimal
 623  mvKey_Divide
 624  mvKey_Multiply
 625  mvKey_Subtract
 626  mvKey_Add
 627  mvKey_NumPadEnter
 628  mvKey_NumPadEqual
 629  mvKey_Browser_Back
 630  mvKey_Browser_Forward
4096   mvKey_ModCtrl    (bit flag)
8192   mvKey_ModShift   (bit flag)
16384  mvKey_ModAlt     (bit flag)
32768  mvKey_ModSuper   (bit flag)
```
