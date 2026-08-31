# Video pipeline: tests, and two instruments that are not tests

The `test_*.py` modules here are collected by pytest in the usual way and need no introduction.

The other two do not start with `test_`, so pytest ignores them, and they are not console scripts either —
they are **bench instruments for the video pipeline**, run by hand from the project root with the venv
active. They are indexed here because nothing else points at them: both were invisible in every `.md` in
the repo until 2026-08-31, which is how a tool that exists gets rebuilt from scratch by the next person
who needs it.

| Instrument | The question it answers |
|---|---|
| `bench_postprocessor.py` | What does each filter cost, and the default chain as a whole, at 512²/768²/1024² on the GPU? |
| `preview_postprocessor.py` | What does a filter *look like*, at several settings side by side? |

```bash
python -m raven.common.video.tests.bench_postprocessor            # default chain + every filter
python -m raven.common.video.tests.bench_postprocessor chain      # the default chain only
python -m raven.common.video.tests.preview_postprocessor crt      # a labelled contact sheet
python -m raven.common.video.tests.preview_postprocessor crt --crop   # 1:1 crops of the head
```

## Why a look instrument earns its place beside a test suite

The postprocessor's tests are contract tests: right shape, right dtype, output in range, the claimed
channel modulated. That is the correct thing for them to be, and it leaves a gap that this module's real
defects have lived in.

The worked example is `crt`, 2026-08-31. It passed every contract test while rendering a washed-out,
half-transparent character, because it applied its scanline term to the colour *and* to alpha, which
squares the modulation in a straight-alpha frame. No assertion available at the time could have caught
it; a rendered still showed it immediately. Two of the three defects that filter shipped with were found
by looking, and the third by someone noticing stripes in the running app.

So: before believing a filter is right, render a still through it.

**Judge fine structure at 1:1, or not at all.** A contact sheet is tiled and then downscaled again by
whatever views it, and a raster at the pixel pitch does not survive either step — one bright row and one
dark row average into a uniform haze. The filter then gets blamed for what the resampling did. `--crop`
exists for exactly this, and the habit is worth more than the flag: the first version of the `crt`
investigation reached a wrong conclusion twice from a downscaled sheet before the crops settled it.

## Adding a filter to either

`bench_postprocessor.py` takes an entry in `ALL_FILTERS`, optionally with a third element to label it
when the same filter appears twice at different settings.

`preview_postprocessor.py` takes a list under `VARIANTS`, keyed by filter name. Make the first entry one
that switches the filter off, so every sheet carries an untouched reference to compare against — a
change is much easier to see beside the thing it changed than remembered from the previous run.
