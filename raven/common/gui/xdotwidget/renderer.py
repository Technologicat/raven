"""DPG renderer for xdot graph elements.

This module provides rendering functions that draw graph shapes using
DearPyGUI's drawlist primitives.
"""

__all__ = ["render_graph", "color_to_dpg"]

from typing import Callable, Dict, List, Optional, Union

import dearpygui.dearpygui as dpg

from .graph import (
    Graph, Element, Shape, Pen,
    TextShape, EllipseShape, PolygonShape, LineShape, BezierShape,
    CompoundShape
)
from .constants import Color, DPGColor, Point
from .viewport import Viewport


def color_to_dpg(color: Color) -> DPGColor:  # TODO: move to a utility module, maybe `raven.common.video.colorspace`? OTOH, not video, but GUI, and we don't have a colorspace module in that namespace.
    """Convert RGBA color from [0,1] to DPG format [0,255]."""
    r, g, b, a = color
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))


def _get_effective_pen(shape: Shape,
                       element: Optional[Element],
                       highlight_intensities: Dict[Element, float]) -> Pen:
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


def _transform_points(points: List[Point],
                      viewport: Viewport) -> List[Point]:
    """Transform a list of points from graph to screen coordinates."""
    return [viewport.graph_to_screen(p[0], p[1]) for p in points]


def _render_text_shape(drawlist: Union[int, str],
                       shape: TextShape,
                       viewport: Viewport,
                       pen: Pen,
                       text_compaction_cb: Optional[Callable] = None) -> None:
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

    color = color_to_dpg(pen.color)

    # DPG's draw_text size parameter is in pixels
    dpg.draw_text((x, y), text, size=font_size_px, color=color, parent=drawlist)


def _render_ellipse_shape(drawlist: Union[int, str],
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


def _render_polygon_shape(drawlist: Union[int, str],
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
        # DPG's draw_polygon doesn't close automatically for stroke,
        # so we need to use draw_polyline with the first point appended
        closed_points = points + [points[0]]
        dpg.draw_polyline(closed_points, color=stroke_color,
                          thickness=thickness, parent=drawlist)


def _render_line_shape(drawlist: Union[int, str],
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

    dpg.draw_polyline(points, color=stroke_color, thickness=thickness,
                      parent=drawlist)


def _render_bezier_shape(drawlist: Union[int, str],
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

    points = _transform_points(shape.points, viewport)
    zoom = viewport.zoom.current

    if shape.filled:
        # TODO: DPG has no filled Bezier support. Tessellate the bezier segments into polyline points and then fill that polygon. Possibly rare case in xdot files.
        # For filled beziers, approximate with polygon
        # (DPG doesn't have filled bezier support)
        fill_color = color_to_dpg(pen.fillcolor)
        dpg.draw_polygon(points, color=(0, 0, 0, 0), fill=fill_color,
                         parent=drawlist)
    else:
        stroke_color = color_to_dpg(pen.color)
        thickness = max(1, pen.linewidth * zoom)

        # Draw each cubic bezier segment
        # Points format: [p0, c1, c2, p1, c1, c2, p1, ...]
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


def _render_shape(drawlist: Union[int, str],
                  shape: Shape,
                  viewport: Viewport,
                  element: Optional[Element],
                  highlight_intensities: Dict[Element, float],
                  text_compaction_cb: Optional[Callable]) -> None:
    """Render a single shape."""
    pen = _get_effective_pen(shape, element, highlight_intensities)

    if isinstance(shape, TextShape):
        _render_text_shape(drawlist, shape, viewport, pen, text_compaction_cb)
    elif isinstance(shape, EllipseShape):
        _render_ellipse_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, PolygonShape):
        _render_polygon_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, LineShape):
        _render_line_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, BezierShape):
        _render_bezier_shape(drawlist, shape, viewport, pen)
    elif isinstance(shape, CompoundShape):
        for child in shape.shapes:
            _render_shape(drawlist, child, viewport, element,
                          highlight_intensities, text_compaction_cb)


def _is_element_visible(element: Element, viewport: Viewport) -> bool:
    """Check if an element is visible in the current viewport."""
    bbox = element.get_bounding_box()
    if bbox is None:
        return True  # If no bbox, assume visible
    return viewport.is_visible(*bbox)


def render_graph(drawlist: Union[int, str],
                 graph: Graph,
                 viewport: Viewport,
                 highlight_intensities: Optional[Dict[Element, float]] = None,
                 text_compaction_cb: Optional[Callable] = None) -> None:
    """Render a graph to a DPG drawlist.

    `drawlist`: DPG drawlist ID or tag.
    `graph`: The Graph to render.
    `viewport`: Viewport for coordinate transforms.
    `highlight_intensities`: Per-element highlight intensity {element: [0,1]}.
    `text_compaction_cb`: Optional callback for text compaction.
                          Signature: (text: str, available_width: float) -> str
    """
    if highlight_intensities is None:
        highlight_intensities = {}

    # Clear the drawlist
    dpg.delete_item(drawlist, children_only=True)

    # Render background shapes
    for shape in graph.shapes:
        _render_shape(drawlist, shape, viewport, None,
                      highlight_intensities, text_compaction_cb)

    # Render edges (before nodes so nodes appear on top)
    for edge in graph.edges:
        if not _is_element_visible(edge, viewport):
            continue
        for shape in edge.shapes:
            _render_shape(drawlist, shape, viewport, edge,
                          highlight_intensities, text_compaction_cb)

    # Render nodes
    for node in graph.nodes:
        if not _is_element_visible(node, viewport):
            continue
        for shape in node.shapes:
            _render_shape(drawlist, shape, viewport, node,
                          highlight_intensities, text_compaction_cb)
