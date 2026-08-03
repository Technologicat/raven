# Fonts bundled with Raven

Every font here is licensed under the **SIL Open Font License, Version 1.1**. The licence text is in
[`OFL.txt`](OFL.txt); the copyright notices it requires are below, one per family.

None of these fonts is Raven's work, and none of them is covered by Raven's own licence. They are
redistributed unmodified.

| Files | Family | Copyright | Source |
|---|---|---|---|
| `fa-regular-400.ttf`, `fa-solid-900.ttf` | Font Awesome 6 Free | Copyright (c) 2024 Fonticons, Inc. (https://fontawesome.com) | https://github.com/FortAwesome/Font-Awesome |
| `InterTight-Regular.ttf`, `InterTight-Bold.ttf`, `InterTight-Italic.ttf`, `InterTight-BoldItalic.ttf` | Inter Tight | Copyright 2022 The Inter Project Authors (https://github.com/rsms/inter-tight) | https://github.com/rsms/inter-tight |
| `OpenSans-Regular.ttf`, `OpenSans-Bold.ttf`, `OpenSans-Italic.ttf`, `OpenSans-BoldItalic.ttf` | Open Sans | Copyright 2020 The Open Sans Project Authors (https://github.com/googlefonts/opensans) | https://github.com/googlefonts/opensans |

## Notes

**Font Awesome ships under three licences and only one of them applies here.** Its `LICENSE.txt` puts the
icons (as SVG and JS) under CC BY 4.0, "all non-font and non-icon files" under MIT, and the web and desktop
*font files* — which is what these two `.ttf`s are — under OFL 1.1. The codepoint constants Raven uses to
address the glyphs are a separate matter and live with the code, in
[`raven/vendor/IconsFontAwesome6.py`](../vendor/IconsFontAwesome6.py).

**Open Sans was relicensed.** Older releases were under Apache License 2.0; the copyright line above is from
the current OFL-1.1 release, which is what these files are. If a font here is ever replaced, re-check the
licence of the build actually downloaded rather than assuming this file still applies.

**Attribution is not optional.** OFL 1.1 requires that the copyright notice and the licence text accompany
the font whenever it is redistributed, which is what this directory now does. Keep them together: a build
process that copies the `.ttf` files without these two files reintroduces the problem.
