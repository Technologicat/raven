# XDot Widget Test Data

## contrast_test

Visual regression test for dark-mode text contrast on colored node fills.
Generated from the xdot widget and viewer source with Pyan3, laid out with fdp.

```bash
# Generate DOT (from project root)
pyan3 raven/xdot_viewer/*.py raven/common/gui/xdotwidget/*.py \
    --dot --colored --no-defines --concentrate --depth 1 \
    --file raven/common/gui/xdotwidget/testdata/contrast_test.dot

# Layout with fdp
fdp -Txdot raven/common/gui/xdotwidget/testdata/contrast_test.dot \
    -o raven/common/gui/xdotwidget/testdata/contrast_test.xdot
```

Open in `raven-xdot-viewer` and toggle dark mode to verify text readability
on green, yellow, and other colored nodes.

**Note:** `--concentrate` causes GraphViz to produce near-miss edge endpoints
(~0.02–0.09 graph units) at split/merge points, visible as small gaps at high
zoom. This is a GraphViz precision issue in the xdot data, not a rendering bug.
