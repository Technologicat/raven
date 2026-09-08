"""DPG renderer for xdot graph elements.

This module provides rendering functions that draw graph shapes using
DearPyGUI's drawlist primitives.
"""

__all__ = ["set_dark_mode", "get_dark_mode", "color_to_dpg", "render_graph"]

import colorsys
import math
from collections.abc import Callable, Sequence

import dearpygui.dearpygui as dpg

from .graph import (
    Graph, Element, Shape, Pen,
    TextShape, EllipseShape, PolygonShape, LineShape, BezierShape, ImageShape,
    CompoundShape, tessellate_bezier
)
from .constants import Color, DPGColor, Point
from .viewport import Viewport


# Dark mode state — module-level because we have a single XDotWidget instance
# and dark mode is an app-wide display concern. If multiple widget instances
# are ever needed, this should move into the widget or renderer instance.
_dark_mode: bool = False


def set_dark_mode(enabled: bool) -> None:
    """Enable or disable dark mode (HSL lightness inversion)."""
    global _dark_mode
    _dark_mode = enabled


def get_dark_mode() -> bool:
    """Return whether dark mode is enabled."""
    return _dark_mode


# Dark mode lightness remap endpoints (in [0,1] lightness space).
# Original L=0 (black) maps to L_MAX; original L=1 (white) maps to L_MIN.
# Tuned so black text becomes light gray (not blinding white) and white
# backgrounds become DPG's dark gray (not pitch black).
_DARK_MODE_L_MAX = 220 / 255  # brightest output — light gray, not white
_DARK_MODE_L_MIN = 45 / 255   # darkest output — DPG dark gray, not black


def _invert_lightness(color: Color) -> Color:
    """Remap the lightness of an RGBA color for dark mode (all channels [0,1]).

    RGB → HLS, linearly remap L from [0,1] to [L_MAX, L_MIN], HLS → RGB.
    Alpha passthrough. Preserves hue and saturation.
    """
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741 -- `l` is the standard name for lightness in HLS
    new_l = _DARK_MODE_L_MAX - l * (_DARK_MODE_L_MAX - _DARK_MODE_L_MIN)
    r2, g2, b2 = colorsys.hls_to_rgb(h, new_l, s)
    return (r2, g2, b2, a)


def _perceived_luminance(r: float, g: float, b: float) -> float:
    """Perceived luminance (ITU-R BT.709). Input channels in [0,1]."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def color_to_dpg(color: Color) -> DPGColor:  # TODO: move to a utility module, maybe `raven.common.video.colorspace`? OTOH, not video, but GUI, and we don't have a colorspace module in that namespace.
    """Convert RGBA color from [0,1] to DPG format [0,255].

    If dark mode is enabled, applies lightness inversion first.
    """
    if _dark_mode:
        color = _invert_lightness(color)
    r, g, b, a = color
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))


def _get_effective_pen(shape: Shape,
                       element: Element | None,
                       highlight_intensities: dict[Element, float]) -> Pen:
    """Get the effective pen for rendering, accounting for highlighting.

    `shape`: The shape being rendered.
    `element`: The element (Node/Edge) containing this shape, if any.
    `highlight_intensities`: Maps highlighted elements to their intensity [0,1].

    Returns the Pen to use for rendering.
    """
    if shape.pen is None:
        pen = Pen()
    else:
        pen = shape.pen

    # Check if the containing element is highlighted
    if element is not None and element in highlight_intensities:
        intensity = highlight_intensities[element]
        highlighted_pen = pen.highlighted_final()
        result = pen.copy()
        Pen.mix(result, pen, highlighted_pen, intensity)
        return result

    return pen


def _transform_point(point: Point, viewport: Viewport) -> Point:
    """Transform a point from graph to screen coordinates."""
    return viewport.graph_to_screen(point[0], point[1])


def _transform_points(points: list[Point],
                      viewport: Viewport) -> list[Point]:
    """Transform a list of points from graph to screen coordinates."""
    return [viewport.graph_to_screen(p[0], p[1]) for p in points]


def _dashify_polyline(points: list[Point],
                      dash: tuple[float, ...]) -> list[list[Point]]:
    """Split a polyline into dashed segments.

    Walks along `points`, toggling between "on" (visible) and "off" (gap)
    phases according to the dash pattern. The pattern repeats cyclically —
    e.g. ``(6,)`` means 6 on, 6 off; ``(2, 4)`` means 2 on, 4 off.

    Returns a list of sub-polylines (the "on" segments).
    """
    if len(points) < 2:
        return [list(points)] if points else []

    # Expand a single-value dash pattern to on/off pair
    cycle = dash if len(dash) >= 2 else (dash[0], dash[0])
    segments: list[list[Point]] = []
    current: list[Point] = []   # points in the current "on" segment
    phase_idx = 0               # index into `cycle`
    phase_remaining = cycle[0]  # distance remaining in current on/off phase

    # phase_idx even → drawing ("on"), odd → gap ("off")
    current.append(points[0])

    for k in range(1, len(points)):
        ax, ay = points[k - 1]
        bx, by = points[k]
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue
        ux, uy = dx / seg_len, dy / seg_len  # unit direction

        consumed = 0.0
        while consumed < seg_len:
            step = min(phase_remaining, seg_len - consumed)
            consumed += step
            phase_remaining -= step

            # Interpolated point at `consumed` along this segment
            px = ax + ux * consumed
            py = ay + uy * consumed

            if phase_idx % 2 == 0:
                # "on" phase — accumulate points
                current.append((px, py))

            if phase_remaining <= 1e-9:
                # Phase exhausted — finalize "on" segment or start new one
                if phase_idx % 2 == 0 and len(current) >= 2:
                    segments.append(current)
                    current = []
                elif phase_idx % 2 == 1:
                    # Transition off → on: start a new segment at this point
                    current = [(px, py)]
                phase_idx = (phase_idx + 1) % len(cycle)
                phase_remaining = cycle[phase_idx]

    # Flush the last "on" segment
    if phase_idx % 2 == 0 and len(current) >= 2:
        segments.append(current)

    return segments


def _render_text_shape(drawlist: int | str,
                       shape: TextShape,
                       viewport: Viewport,
                       pen: Pen,
                       text_compaction_cb: Callable | None = None,
                       graph_text_fonts: Sequence[tuple[float, int | str]] | None = None,
                       element_fillcolor: Color | None = None) -> None:
    """Render a text shape."""
    # Transform position
    sx, sy = viewport.graph_to_screen(shape.x, shape.y)
    zoom = viewport.zoom.current

    # Calculate font size in screen pixels
    font_size_px = pen.fontsize * zoom

    # Skip text that's too small to read
    min_readable_size = 4
    if font_size_px < min_readable_size:
        return

    # Get text content, possibly compacted
    text = shape.t
    screen_width = shape.w * zoom
    if text_compaction_cb is not None and font_size_px < 8:
        text = text_compaction_cb(text, screen_width)
        if not text:
            return

    # Calculate position based on justification.
    # `shape.w` is the text width in graph coordinates (from GraphViz).
    # DPG's draw_text is always left-aligned, so we offset manually.
    if shape.j == TextShape.LEFT:
        x = sx
    elif shape.j == TextShape.CENTER:
        x = sx - screen_width / 2
    else:  # RIGHT
        x = sx - screen_width

    # Adjust y position (DPG draws from top-left, xdot uses baseline)
    # Approximate adjustment based on font size
    y = sy - font_size_px * 0.8

    # In dark mode, text on colored fills needs contrast-aware color selection.
    # The standard lightness inversion can produce near-white text on medium-
    # lightness fills (e.g., green, yellow), which is unreadable.
    # `element_fillcolor` comes from the element's filled shape (parsed from
    # `_draw_`), not from this text shape's pen (parsed from `_ldraw_`).
    if _dark_mode and element_fillcolor is not None:
        inv_fill = _invert_lightness(element_fillcolor)
        lum = _perceived_luminance(inv_fill[0], inv_fill[1], inv_fill[2])
        if lum > 0.5:
            v = int(_DARK_MODE_L_MIN * 255)
        else:
            v = int(_DARK_MODE_L_MAX * 255)
        color = (v, v, v, int(pen.color[3] * 255))
    else:
        color = color_to_dpg(pen.color)

    # DPG's draw_text size parameter is in pixels
    item = dpg.draw_text((x, y), text, size=font_size_px, color=color, parent=drawlist)

    # Bind the font whose atlas size is closest to the rendered size.
    # DPG uses bilinear filtering on font atlas textures, so minimal
    # scaling ratio → sharpest text.
    if graph_text_fonts:
        best_font = min(graph_text_fonts, key=lambda sf: abs(sf[0] - font_size_px))[1]
        dpg.bind_item_font(item, best_font)


def _render_ellipse_shape(drawlist: int | str,
                          shape: EllipseShape,
                          viewport: Viewport,
                          pen: Pen) -> None:
    """Render an ellipse shape."""
    # Transform center
    cx, cy = viewport.graph_to_screen(shape.x0, shape.y0)
    zoom = viewport.zoom.current

    # Scale radii
    rx = shape.w * zoom
    ry = shape.h * zoom

    # DPG's draw_ellipse takes a bounding box (pmin, pmax), not center+radius.
    pmin = (cx - rx, cy - ry)
    pmax = (cx + rx, cy + ry)

    if shape.filled:
        fill_color = color_to_dpg(pen.fillcolor)
        dpg.draw_ellipse(pmin, pmax,
                         color=(0, 0, 0, 0), fill=fill_color,
                         parent=drawlist)
    else:
        stroke_color = color_to_dpg(pen.color)
        thickness = max(1, pen.linewidth * zoom)
        dpg.draw_ellipse(pmin, pmax,
                         color=stroke_color, thickness=thickness,
                         parent=drawlist)


def _render_polygon_shape(drawlist: int | str,
                          shape: PolygonShape,
                          viewport: Viewport,
                          pen: Pen) -> None:
    """Render a polygon shape."""
    if not shape.points:
        return

    points = _transform_points(shape.points, viewport)
    zoom = viewport.zoom.current

    if shape.filled:
        fill_color = color_to_dpg(pen.fillcolor)
        dpg.draw_polygon(points, color=(0, 0, 0, 0), fill=fill_color,
                         parent=drawlist)
    else:
        stroke_color = color_to_dpg(pen.color)
        thickness = max(1, pen.linewidth * zoom)
        # `draw_polygon` strokes an *open* path -- measured, and it leaves the edge back to the first
        # vertex undrawn -- so the outline is a polyline either way.
        if pen.dash:
            # Same treatment a `LineShape` gets. Polygons went without it for as long as this renderer has
            # existed, so a dashed outline asked for by a caller -- or by GraphViz's `style=dashed` -- came
            # out solid, silently and with nothing to notice but the picture.
            #
            # The walk needs the closing edge spelled out, and `closed` would not help it: what comes back
            # is a list of open dashes, each of which is meant to have caps.
            scaled_dash = tuple(d * zoom for d in pen.dash)
            for seg in _dashify_polyline(points + [points[0]], scaled_dash):
                dpg.draw_polyline(seg, color=stroke_color, thickness=thickness, parent=drawlist)
        else:
            # `closed`, rather than appending the first point again. The two close the same outline and
            # differ at the seam: repeating the vertex ends one stroke and starts another, so two butt
            # caps meet there instead of a join, and the outer corner is left unfilled. The width of that
            # notch is the line width, which scales with the zoom -- invisible at 1:1 and a bite out of
            # the corner once a reader has zoomed in. It lands on the outline's *first vertex*, wherever
            # whoever built the outline put that: GraphViz writes a box starting at its top right, while
            # a rectangle built corner-by-corner in Python usually starts at the top left.
            dpg.draw_polyline(points, closed=True, color=stroke_color,
                              thickness=thickness, parent=drawlist)


def _render_line_shape(drawlist: int | str,
                       shape: LineShape,
                       viewport: Viewport,
                       pen: Pen) -> None:
    """Render a line/polyline shape."""
    if len(shape.points) < 2:
        return

    points = _transform_points(shape.points, viewport)
    zoom = viewport.zoom.current

    stroke_color = color_to_dpg(pen.color)
    thickness = max(1, pen.linewidth * zoom)

    if pen.dash:
        # Dash pattern is in graph-coordinate points; scale by zoom
        scaled_dash = tuple(d * zoom for d in pen.dash)
        for seg in _dashify_polyline(points, scaled_dash):
            dpg.draw_polyline(seg, color=stroke_color, thickness=thickness,
                              parent=drawlist)
    else:
        dpg.draw_polyline(points, color=stroke_color, thickness=thickness,
                          parent=drawlist)


def _render_bezier_shape(drawlist: int | str,
                         shape: BezierShape,
                         viewport: Viewport,
                         pen: Pen) -> None:
    """Render a bezier curve shape.

    xdot bezier format: [start, ctrl1, ctrl2, end, ctrl1, ctrl2, end, ...]
    Each segment is: start_point, control_point_1, control_point_2, end_point
    After first segment, the end becomes the start of the next.
    """
    if len(shape.points) < 4:
        return

    if shape.filled:
        # DPG has no filled bezier support, so tessellate into a polygon.
        tess_graph = tessellate_bezier(shape.points, n=32)
        tess_screen = _transform_points(tess_graph, viewport)
        fill_color = color_to_dpg(pen.fillcolor)
        dpg.draw_polygon(tess_screen, color=(0, 0, 0, 0), fill=fill_color,
                         parent=drawlist)
    else:
        zoom = viewport.zoom.current
        stroke_color = color_to_dpg(pen.color)
        thickness = max(1, pen.linewidth * zoom)

        if pen.dash:
            # Tessellate to polyline for dash pattern rendering.
            # Use n=32 for visual smoothness (the default n=10 is fine
            # for hit detection but looks angular at high zoom).
            tess_graph = tessellate_bezier(shape.points, n=32)
            tess_screen = _transform_points(tess_graph, viewport)
            scaled_dash = tuple(d * zoom for d in pen.dash)
            for seg in _dashify_polyline(tess_screen, scaled_dash):
                dpg.draw_polyline(seg, color=stroke_color, thickness=thickness,
                                  parent=drawlist)
        else:
            # Solid — use DPG's native bezier for smooth curves.
            points = _transform_points(shape.points, viewport)
            p0 = points[0]
            for i in range(1, len(points), 3):
                if i + 2 >= len(points):
                    break
                c1 = points[i]
                c2 = points[i + 1]
                p1 = points[i + 2]
                dpg.draw_bezier_cubic(p0, c1, c2, p1,
                                      color=stroke_color, thickness=thickness,
                                      parent=drawlist)
                p0 = p1


def _render_image_shape(drawlist: int | str,
                        shape: ImageShape,
                        viewport: Viewport) -> None:
    """Render an image shape.

    No pen is involved, so nothing here goes through `color_to_dpg` and dark mode does not touch the
    picture — which is the wanted behaviour: the lightness inversion is for ink drawn on the background,
    and inverting a photograph or an icon would make it wrong rather than dark-friendly. A caller that
    wants the image to sit in the drawing outlines it, and *that* line is a pen and does invert.
    """
    if not shape.levels:  # still being prepared; the caller draws whatever stands in for it
        return

    x1, y1 = viewport.graph_to_screen(min(shape.x1, shape.x2), min(shape.y1, shape.y2))
    x2, y2 = viewport.graph_to_screen(max(shape.x1, shape.x2), max(shape.y1, shape.y2))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    if shape.max_screen_size is not None and max(w, h) > shape.max_screen_size:
        # Shrink about the centre, uniformly, so the image keeps its proportions and its place.
        t = shape.max_screen_size / max(w, h)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        x1, x2 = cx - 0.5 * t * w, cx + 0.5 * t * w
        y1, y2 = cy - 0.5 * t * h, cy + 0.5 * t * h
        w, h = x2 - x1, y2 - y1

    dpg.draw_image(_texture_for_screen_size(shape, w, h), (x1, y1), (x2, y2),
                   uv_min=(0.0, 0.0), uv_max=(1.0, 1.0), parent=drawlist)


def _texture_for_screen_size(shape: ImageShape, w: float, h: float) -> int | str:
    """Return the level of `shape`'s mip chain to draw at `w` x `h` screen pixels.

    The coarsest level that still covers the drawn size in both axes, since a level that has to be
    downsampled to fit stays sharp where the next one down would have to be stretched. Nothing coarse
    enough — the picture is drawn larger than any level was prepared at — leaves the finest, which DPG
    then upsamples.

    Both axes, because the rectangle need not carry the texture's proportions: a picture stretched to fit
    wants whichever axis is the more demanding of the two, and asking for both is the answer that never
    upsamples.

    **What decides the level is the size in screen pixels, which is the size in graph units times the
    zoom.** A graph shown at 1:1 says nothing on its own — a card 55 graph units across is 55 pixels
    there, and a level prepared at 1024 would be an eighteen-fold downsample.
    """
    chosen = shape.levels[0]
    for level in shape.levels[1:]:  # finest first
        if level.width < w or level.height < h:
            break
        chosen = level
    return chosen.texture


def _get_element_fillcolor(element: Element | None) -> Color | None:
    """Extract the fill color from an element's filled shapes, if any.

    Returns the fillcolor of the first filled EllipseShape or PolygonShape
    found in the element's shape list, or ``None`` if the element has no
    filled shapes (e.g. edges, background shapes).
    """
    if element is None:
        return None
    for shape in element.shapes:
        if isinstance(shape, (EllipseShape, PolygonShape)) and shape.filled:
            return shape.pen.fillcolor if shape.pen is not None else None
    return None


def _render_shape(drawlist: int | str,
                  shape: Shape,
                  viewport: Viewport,
                  element: Element | None,
                  highlight_intensities: dict[Element, float],
                  text_compaction_cb: Callable | None,
                  graph_text_fonts: Sequence[tuple[float, int | str]] | None = None,
                  element_fillcolor: Color | None = None) -> None:
    """Render a single shape."""
    pen = _get_effective_pen(shape, element, highlight_intensities)

    if isinstance(shape, TextShape):
        _render_text_shape(drawlist, shape, viewport, pen,
                           text_compaction_cb, graph_text_fonts,
                           element_fillcolor=element_fillcolor)
    elif isinstance(shape, EllipseShape):
        _render_ellipse_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, PolygonShape):
        _render_polygon_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, LineShape):
        _render_line_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, BezierShape):
        _render_bezier_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, ImageShape):
        _render_image_shape(drawlist, shape, viewport)
    elif isinstance(shape, CompoundShape):
        for child in shape.shapes:
            _render_shape(drawlist, child, viewport, element,
                          highlight_intensities, text_compaction_cb,
                          graph_text_fonts,
                          element_fillcolor=element_fillcolor)


def _is_element_visible(element: Element, viewport: Viewport) -> bool:
    """Check if an element is visible in the current viewport.

    Asked of everything the element *draws* rather than of the layout cell it occupies, the two differing
    for a node that carries decorations in its margins — see `Element.get_drawn_bounding_box`. Culling on
    the cell drops the decorations along with it, and the further in the view is zoomed the more of the
    screen they are.
    """
    bbox = element.get_drawn_bounding_box()
    if bbox is None:
        return True  # If no bbox, assume visible
    return viewport.is_visible(*bbox)


def render_graph(drawlist: int | str,
                 graph: Graph,
                 viewport: Viewport,
                 highlight_intensities: dict[Element, float] | None = None,
                 text_compaction_cb: Callable | None = None,
                 graph_text_fonts: Sequence[tuple[float, int | str]] | None = None,
                 background_color: DPGColor | None = None) -> None:
    """Render a graph to a DPG drawlist.

    `drawlist`: DPG drawlist ID or tag.
    `graph`: The Graph to render.
    `viewport`: Viewport for coordinate transforms.
    `highlight_intensities`: Per-element highlight intensity {element: [0,1]}.
    `text_compaction_cb`: Optional callback for text compaction.
                          Signature: (text: str, available_width: float) -> str
    `graph_text_fonts`: Optional list of (atlas_size_px, dpg_font_id) tuples.
                         The renderer picks the font whose atlas size is closest
                         to the rendered text size, for sharpest results.
    `background_color`: Optional DPG color for the graph background rectangle.
    """
    if highlight_intensities is None:
        highlight_intensities = {}

    # Clear the drawlist
    dpg.delete_item(drawlist, children_only=True)

    # Draw background rectangle
    if background_color is not None:
        w = viewport.width
        h = viewport.height
        dpg.draw_rectangle((0, 0), (w, h), color=(0, 0, 0, 0),
                           fill=background_color, parent=drawlist)

    # Render background shapes
    for shape in graph.shapes:
        _render_shape(drawlist, shape, viewport, None,
                      highlight_intensities, text_compaction_cb, graph_text_fonts)

    # Render edges (before nodes so nodes appear on top)
    for edge in graph.edges:
        if not _is_element_visible(edge, viewport):
            continue
        for shape in edge.shapes:
            _render_shape(drawlist, shape, viewport, edge,
                          highlight_intensities, text_compaction_cb, graph_text_fonts)

    # Render nodes
    for node in graph.nodes:
        if not _is_element_visible(node, viewport):
            continue
        fillcolor = _get_element_fillcolor(node)
        for shape in node.shapes:
            _render_shape(drawlist, shape, viewport, node,
                          highlight_intensities, text_compaction_cb,
                          graph_text_fonts, element_fillcolor=fillcolor)
