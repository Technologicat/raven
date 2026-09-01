"""Turn the chat forest into a picture: the `Graph` that Librarian's chat graph view renders.

Pure by design — a datastore and a description of what the user is looking at go in, a `Graph` plus a table
saying what each graph node *means* comes out. No DearPyGui, no widget, no mutation of the forest. That is
what makes the layout checkable at all: a position cannot be asserted through a rendered widget, and this
view's entire difficulty is positions.

The picture is a *focus-plus-context* view rather than the whole forest. Real chat trees are one very wide
level — every chat ever started under the current character card is a sibling there — plus narrow chains
hanging off it, so a layout that tries to show everything is unreadable long before it is slow. Instead the
branch leading to HEAD is drawn as a vertical spine, a few siblings are shown either side of it at each
level, and everything omitted is drawn as a clickable gap. The gaps are not decoration: a node with no
visible links has to mean the graph genuinely ends there, or the picture lies.

Coordinates are the widget's: x to the right, y downward, origin at the top left of the graph's bounding
box, which is what `Viewport.zoom_to_fit` expects.
"""

__all__ = ["SPINE_FILL_COLOR",
           "OFF_SPINE_FILL_COLOR",
           "LINE_COLOR",
           "GAP_LINE_COLOR",
           "PREVIEW_COLOR",

           "MeasureText",

           "Ref",
           "ChatNodeRef",
           "SiblingGapRef",
           "DepthGapRef",
           "SubtreeGapRef",
           "RootGapRef",

           "ViewState",
           "LayoutConfig",
           "ChatGraph",

           "build"]

import dataclasses
import logging
import math
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

from ..common.gui.xdotwidget import constants as xdotconstants
from ..common.gui.xdotwidget import graph as xdotgraph

from . import chattree
from . import chatutil
from . import config as librarian_config

# Authored for a light background and inverted by the renderer, which is the path a parsed graph takes too
# -- so the two cannot drift apart, and there is one place to look when a colour is wrong.
SPINE_FILL_COLOR: xdotconstants.Color = (0.78, 0.93, 0.78, 1.0)  # the linearized branch, green as in the README's tree diagram
OFF_SPINE_FILL_COLOR: xdotconstants.Color = (0.94, 0.94, 0.94, 1.0)  # everything the current branch did not take
LINE_COLOR: xdotconstants.Color = (0.15, 0.15, 0.15, 1.0)
GAP_LINE_COLOR: xdotconstants.Color = (0.45, 0.45, 0.45, 1.0)
# The ring around the box a click has selected. A colour of its own, and used for nothing else, because
# the two things it must not be mistaken for are both already on screen: the hover highlight, and HEAD.
PREVIEW_COLOR: xdotconstants.Color = (0.10, 0.35, 0.80, 1.0)

# Dash pattern for the outline of a gap, in graph units: on, off. A gap stands for content that is not
# here, and a broken outline says that before any label is read.
_GAP_DASH: Tuple[float, float] = (6.0, 4.0)

# And for the ring around a tentatively selected box. Shorter marks than the gap's pattern, so the two
# broken lines do not read as the same thing -- they say related but different things, "this is not here"
# against "this is not settled".
#
# Kept thin and comfortably longer than the stroke is thick, for reasons in `dpg-notes.md` under the
# ragged-dashes entry: a mark's sub-pixel phase varies, the error is roughly constant while the mark is
# not, and a thin stroke additionally takes ImGui's texture-based antialiasing path, whose coverage does
# not depend on that phase at all. Making the cycle divide evenly would fix it too, and only at the zoom
# it was computed for -- the cycle in pixels is the cycle times the zoom, and the zoom is continuous.
_PREVIEW_DOTS: Tuple[float, float] = (3.0, 3.0)

_ROUNDED_CORNER_SEGMENTS = 4  # per corner; four is already indistinguishable from a curve at these radii

# Average glyph advance as a fraction of font size. Two figures, because the two kinds of text here are
# not the same width: the pills are short uppercase words, and capitals are appreciably wider than the
# mixed-case prose of a chat message. Using the uppercase figure for a message cut its label about a
# quarter early -- "How can I help you to..." for a line that ends "today?". Anything wanting a real
# measurement has to ask DPG, which this module deliberately cannot do.
_PILL_ADVANCE_PER_CHAR = 0.62
_LABEL_ADVANCE_PER_CHAR = 0.5

# How a caller lets this module ask what text actually measures: `(text, font size) -> width or None`, in
# graph units. Optional, because the module is pure and DPG is where the answer lives -- with no measurer
# it falls back to an average advance, which is good enough to size a box and not good enough to centre
# text inside one.
#
# `None` is the answer for "cannot say right now", which is an ordinary state rather than a fault: a font
# atlas does not exist until a frame has been rendered, so anything drawing before the first one -- or in
# a test suite, which renders none -- gets the estimate and no complaint. An exception means something
# actually went wrong, and is logged.
MeasureText = Callable[[str, float], Optional[float]]


def _text_width(text: str, font_size: float,
                measure_text: Optional[MeasureText], advance_per_char: float) -> float:
    """Return how wide `text` is at `font_size`, measured where that is possible and estimated where not."""
    if measure_text is not None:
        try:
            measured = measure_text(text, font_size)
        except Exception:  # noqa: BLE001 -- a measurer that fails must not cost the whole picture
            logger.warning(f"_text_width: measuring '{text}' failed; falling back to an estimate",
                           exc_info=True)
        else:
            if measured is not None:
                return measured
    return font_size * advance_per_char * len(text)


# Two outline vertices closer together than this are the same point. In graph units, which are pixels at
# zoom 1, so this is far below anything a display can tell apart and far above float rounding.
_COINCIDENT_POINT_TOLERANCE = 1e-9

# Horizontal breathing room between a box's edge and its text, in graph units.
_LABEL_INSET = 8.0

# Vertical space between the speaker line and the label below it, in graph units.
_LINE_GAP = 2.0


# --------------------------------------------------------------------------------
# What a graph node means

class Ref:
    """Base class for the descriptors saying what a graph node stands for.

    The widget hands a click back as the node's `internal_name` and nothing else, so every name a `Graph`
    built here carries has an entry in `ChatGraph.refs`. A caller dispatches on the subclass rather than
    parsing the name, which is why the naming scheme is private to this module.
    """

    def __init__(self, name: str):
        self.name = name


class ChatNodeRef(Ref):
    """A real node of the chat forest.

    `node_id`: Its ID in the datastore. This is also the graph node's name, so `XDotWidget.pan_to_node`
               takes a chat node ID directly.
    `role`: "system", "user", "assistant" or "tool".
    `on_current_branch`: Whether it lies on the branch HEAD is on — which is what the fill colour encodes.
                         Not the same as being on the drawn spine: previewing another branch draws that
                         one, and the shared prefix stays coloured while the divergence does not, which is
                         the picture of *where you would be going against where you are*.
    `tool_call_count`: How many tool calls this message made, for the badge. Zero for everything that made
                       none, which is nearly every node.
    `pills`: The pointer labels resting on it, e.g. `("SYS", "NEW")`. A tuple rather than one value because
             more than one pointer can land on the same node: with the AI greeting turned off, a new chat
             starts at the system prompt node, so SYS and NEW coincide there.
    """

    def __init__(self, name: str, node_id: str, role: str, on_current_branch: bool,
                 tool_call_count: int, pills: Tuple[str, ...]):
        super().__init__(name)
        self.node_id = node_id
        self.role = role
        self.on_current_branch = on_current_branch
        self.tool_call_count = tool_call_count
        self.pills = pills


class SiblingGapRef(Ref):
    """Siblings omitted from one level's window.

    `parent_node_id`: Whose children these are.
    `hidden_node_ids`: What is not shown, in sibling order.
    `recenter_on`: The sibling to move the window to when this gap is clicked — the middle of the run, so
                   that repeated clicks bisect a wide fan rather than walking it at a fixed stride.
    """

    def __init__(self, name: str, parent_node_id: str,
                 hidden_node_ids: Tuple[str, ...], recenter_on: str):
        super().__init__(name)
        self.parent_node_id = parent_node_id
        self.hidden_node_ids = hidden_node_ids
        self.recenter_on = recenter_on

    def _get_hidden_count(self) -> int:
        """Return how many siblings this gap stands for."""
        return len(self.hidden_node_ids)

    hidden_count = property(fget=_get_hidden_count,
                            doc="How many siblings this gap stands for.")


class DepthGapRef(Ref):
    """Ancestors omitted between the root and the shown part of the spine.

    `hidden_node_ids`: What is not shown, oldest first.
    """

    def __init__(self, name: str, hidden_node_ids: Tuple[str, ...]):
        super().__init__(name)
        self.hidden_node_ids = hidden_node_ids

    def _get_hidden_count(self) -> int:
        """Return how many ancestors this gap stands for."""
        return len(self.hidden_node_ids)

    hidden_count = property(fget=_get_hidden_count,
                            doc="How many ancestors this gap stands for.")


class SubtreeGapRef(Ref):
    """The conversation continuing below an off-spine sibling, which this view does not descend into.

    `node_id`: The sibling it hangs under.
    `child_count`: How many children that sibling has.
    """

    def __init__(self, name: str, node_id: str, child_count: int):
        super().__init__(name)
        self.node_id = node_id
        self.child_count = child_count


class RootGapRef(Ref):
    """Other roots — that is, chats written under other versions of the character card.

    Inert in v1, and shown anyway: the alternative is a root that looks like the only one there has ever
    been. Clicking through to another card would leave the configured avatar and voice running against a
    different system prompt, which wants a decision of its own.

    `hidden_node_ids`: The other roots, in datastore order.
    """

    def __init__(self, name: str, hidden_node_ids: Tuple[str, ...]):
        super().__init__(name)
        self.hidden_node_ids = hidden_node_ids

    def _get_hidden_count(self) -> int:
        """Return how many other roots this gap stands for."""
        return len(self.hidden_node_ids)

    hidden_count = property(fget=_get_hidden_count,
                            doc="How many other roots this gap stands for.")


# --------------------------------------------------------------------------------
# Inputs

@dataclasses.dataclass
class ViewState:
    """What the user is currently looking at. Owned by the panel; read here.

    `head_node_id`: The tip of the current branch — `app_state["HEAD"]`. Where the user actually is, which
                    is what the fill colour and the HEAD pill report.
    `previewed_node_id`: The node a click has selected, drawn with a ring of its own. Part of the picture
                         rather than of the widget's highlight state, so that it cannot be confused with a
                         hover — they would otherwise share one pair of colours — and so that it survives
                         a rebuild without anyone re-applying it.
    `focus_node_id`: The node the picture is drawn around, defaulting to `head_node_id`. These come apart
                     while previewing: clicking a node on another branch re-lays the graph out around it
                     and refreshes the siblings near it, without moving HEAD. Browsing the multiverse
                     changes nothing; only a deliberate second act does.
    `new_chat_node_id`: Where a new chat starts — `app_state["new_chat_HEAD"]`. Taken as a parameter rather
                        than derived, because *what* it points at is changing: it is the AI's greeting
                        today, and the root itself for a chat started with the greeting turned off. One
                        datastore will hold both shapes.
    `expanded_tool_turns`: Assistant node IDs whose tool-result nodes are shown individually instead of
                           being counted onto a badge.
    `sibling_focus`: Parent node ID -> which of its children the sibling window is centred on. An override:
                     a level not listed here centres on whichever child the spine goes through, which is
                     what the user sees before touching anything.
    """

    head_node_id: str
    focus_node_id: Optional[str] = None
    previewed_node_id: Optional[str] = None
    new_chat_node_id: Optional[str] = None
    expanded_tool_turns: Set[str] = dataclasses.field(default_factory=set)
    sibling_focus: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LayoutConfig:
    """Sizes and counts, in graph units. One graph unit is one pixel at zoom 1.

    `siblings_each_side`: How many siblings to show either side of the focused one. The first and last are
                          shown regardless, as anchors, so a level can hold up to
                          `2 * siblings_each_side + 3` items counting the two gaps.
    `max_visible_depth`: How many spine nodes to draw, the root included. Anything between the root and the
                         last `max_visible_depth - 1` of them becomes one depth gap.
    `label_chars`: Coarse cut for a node's label, or `None` to derive it from the node width and the font
                   size — the default, and what keeps the three in step when any one of them moves. The
                   widget compacts further when the text will not fit at the current zoom; this is the cut
                   that stops a whole chat message from becoming the graph's width at zoom 1.
    """

    node_w: float = 300.0
    node_h: float = 84.0
    gap_node_w: float = 120.0
    horizontal_spacing: float = 24.0
    vertical_spacing: float = 44.0
    corner_radius: float = 10.0
    line_width: float = 1.5
    # HEAD's box is drawn heavier than the rest. Being where the reader actually is is the most important
    # thing on screen, and until now the only thing saying so was a small pill in the margin.
    head_line_width: float = 3.5
    preview_ring_offset: float = 5.0  # how far outside the box the selection ring sits
    preview_line_width: float = 1.0  # thin on purpose; see `_PREVIEW_DOTS`
    # The same size the rest of Raven's interface uses, so a node reads like the app it belongs to once the
    # reader has zoomed to 1:1. Sourced rather than repeated: two numbers both meaning "the UI font" drift.
    font_size: float = librarian_config.gui_config.font_size
    role_font_size: float = 0.7 * librarian_config.gui_config.font_size
    # Where the text starts, measured down from the box's top edge. Anchored to the top rather than
    # centred, because a message wraps to one line or two and centring would leave the short ones floating
    # at a different height from their neighbours -- in a row of boxes that reads as raggedness rather
    # than as a shorter message. Lower this to move the text up.
    text_top_inset: float = 8.0
    pill_font_size: float = 10.0
    pill_h: float = 16.0  # a pill's width follows its own label; see `_box_shapes`
    arrowhead_length: float = 10.0
    arrowhead_halfwidth: float = 4.5
    margin: float = 20.0
    label_chars: Optional[int] = None
    label_lines: int = 2

    def _get_effective_label_chars(self) -> int:
        """Return `label_chars`, or how many characters fit across a node when it was left unset."""
        if self.label_chars is not None:
            return self.label_chars
        return max(1, int((self.node_w - 2 * _LABEL_INSET) / (self.font_size * _LABEL_ADVANCE_PER_CHAR)))

    # These two are the ones a user has a reason to change, so they live in `config` and are picked up from
    # there; the rest of this class is drawing detail. Neither is speed-bound in any range worth using --
    # see `investigations/chatgraph-rebuild-cost/`, and the comment beside them in the config.
    siblings_each_side: int = librarian_config.gui_config.chat_graph_siblings_each_side
    max_visible_depth: int = librarian_config.gui_config.chat_graph_max_visible_depth


# --------------------------------------------------------------------------------
# Output

class ChatGraph:
    """A `Graph` ready for `XDotWidget.set_graph`, and the table that makes its clicks meaningful.

    `graph`: The graph.
    `refs`: Graph node name -> `Ref`. Every node in `graph` appears here.
    `spine`: The chat node IDs on the branch to HEAD, root first — including the ones the depth window
             elided and the tool nodes the collapse hid, so a caller can ask "is this node on the current
             branch?" without rebuilding the picture.
    `spine_bbox`: `(x1, y1, x2, y2)` around the boxes of the drawn spine — the branch, and nothing beside
                  it. This is what a view should frame on opening: the whole picture is far too wide to
                  fit and still be legible (a windowed level runs to thousands of units against a panel
                  of hundreds), while the branch is a narrow column, so fitting *this* is fitting the
                  answer to "where am I" at a zoom that leaves the words readable.
    """

    def __init__(self, graph: xdotgraph.Graph, refs: Dict[str, Ref], spine: Tuple[str, ...],
                 spine_bbox: Tuple[float, float, float, float]):
        self.graph = graph
        self.refs = refs
        self.spine = spine
        self.spine_bbox = spine_bbox

    def ref_for(self, name: str) -> Optional[Ref]:
        """Return what the graph node called `name` stands for, or `None` if there is no such node."""
        return self.refs.get(name)


# --------------------------------------------------------------------------------
# Geometry helpers

def _rounded_rect_points(x1: float, y1: float, x2: float, y2: float,
                         radius: float) -> List[xdotconstants.Point]:
    """Return the vertices of a rectangle with rounded corners, clockwise from the top left.

    `radius` is clamped to half the shorter side, so a small box degenerates to a stadium rather than
    turning itself inside out.
    """
    radius = min(radius, 0.5 * (x2 - x1), 0.5 * (y2 - y1))
    if radius <= 0.0:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    points: List[xdotconstants.Point] = []
    # (centre of the corner's arc, angle at which that arc starts), in drawing order. Angles are measured
    # with y downward, so increasing angle runs clockwise on screen.
    corners = [((x1 + radius, y1 + radius), math.pi),
               ((x2 - radius, y1 + radius), 1.5 * math.pi),
               ((x2 - radius, y2 - radius), 0.0),
               ((x1 + radius, y2 - radius), 0.5 * math.pi)]
    for (cx, cy), start_angle in corners:
        for k in range(_ROUNDED_CORNER_SEGMENTS + 1):
            angle = start_angle + 0.5 * math.pi * (k / _ROUNDED_CORNER_SEGMENTS)
            point = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            # Drop a vertex that repeats the one before it. Ordinarily there is none -- one corner's arc
            # ends where a straight edge begins -- but when the radius is clamped to half the shorter side
            # the box is a stadium, the two arcs on that side share a centre, and each meets the next at
            # the same point. A repeated vertex is a zero-length segment, which a stroked polyline renders
            # as a spur at the join: visible as a horizontal flick at the left and right extremes of a
            # pill, which is exactly where those joins are.
            #
            # Compared with a tolerance, not for equality. The two spellings of that shared point are
            # `cos(2*pi)` and `cos(0)`, which agree to about one part in 10^16 and not to the bit -- so an
            # exact test drops nothing and the spur survives, looking for all the world like the fix is in.
            if not points or math.dist(point, points[-1]) > _COINCIDENT_POINT_TOLERANCE:
                points.append(point)
    # And the seam. The outline closes from the last vertex back to the first, so a last vertex sitting on
    # the first is a zero-length *closing* segment -- the same spur as an interior one, at the point where
    # the walk began. For a stadium that is the leftmost point, which is where it shows.
    if len(points) > 1 and math.dist(points[0], points[-1]) <= _COINCIDENT_POINT_TOLERANCE:
        points.pop()
    return points


def _arrowhead_points(tip: xdotconstants.Point, tail: xdotconstants.Point,
                      length: float, halfwidth: float) -> List[xdotconstants.Point]:
    """Return the three vertices of an arrowhead at `tip`, pointing away from `tail`."""
    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return [tip, tip, tip]
    ux, uy = dx / norm, dy / norm
    base = (tip[0] - length * ux, tip[1] - length * uy)
    return [tip,
            (base[0] - halfwidth * uy, base[1] + halfwidth * ux),
            (base[0] + halfwidth * uy, base[1] - halfwidth * ux)]


def _wrap(text: str, width: int, max_lines: int) -> List[str]:
    """Fold `text` into at most `max_lines` lines of about `width` characters, marking anything cut.

    The message's own line breaks go first — `split` collapses every run of whitespace, so a message that
    opens "Hi!" and continues after a blank line becomes one flowing string. That is the point: a blank
    line copied faithfully into a two-line label spends half of it on nothing.
    """
    text = " ".join(text.split())
    if not text:
        return []
    return textwrap.wrap(text, width=width, max_lines=max_lines, placeholder="…") or [text[:width]]


# --------------------------------------------------------------------------------
# Reading the forest
#
# Tolerant on purpose, all of it. This runs against a forest another thread is writing to, and a node that
# vanishes between the lineage walk and the label lookup should cost the picture one blank box rather than
# the frame.

def _payload_of(datastore: chattree.Forest, node_id: str) -> Dict[str, Any]:
    """Return the active payload of `node_id`, or an empty dict if the node or its payload is missing."""
    try:
        payload = datastore.get_payload(node_id)
    except KeyError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _role_of(datastore: chattree.Forest, node_id: str) -> str:
    """Return the role of the message in `node_id`: "system", "user", "assistant", "tool", or "" if unknown."""
    message = _payload_of(datastore, node_id).get("message") or {}
    return message.get("role") or ""


def _tool_call_count(datastore: chattree.Forest, node_id: str) -> int:
    """Return how many tool calls the message in `node_id` requested.

    Read from the assistant message's own `tool_calls` rather than by counting the result nodes below it: a
    turn that was interrupted has fewer results than calls, and the badge should say what was asked for.
    """
    message = _payload_of(datastore, node_id).get("message") or {}
    return len(message.get("tool_calls") or ())


# What a node's role line says, when the message carries no persona name of its own. "system" and "tool"
# never have one; "user" and "assistant" normally do, and theirs is the character's name, which says more
# than the role does.
_ROLE_CAPTIONS = {"system": "SYSTEM", "tool": "TOOL", "user": "USER", "assistant": "AI"}


def _speaker_and_label_of(datastore: chattree.Forest, node_id: str,
                          width: int, max_lines: int) -> Tuple[str, List[str]]:
    """Return `(who said it, the lines of what they said)` for `node_id`, ready to draw.

    The speaker is the message's stored persona where it has one, and the role otherwise — the same
    preference the chat log shows, so the two views name the same participants the same way.
    """
    try:
        role, persona, text = chatutil.get_node_message_text_without_persona(datastore, node_id)
    except (KeyError, TypeError):
        return "?", ["(missing)"]
    speaker = persona or _ROLE_CAPTIONS.get(role, (role or "?").upper())
    return speaker, (_wrap(text, width, max_lines) or ["(empty)"])


def _collapse_tool_rounds(datastore: chattree.Forest,
                          lineage: Sequence[str],
                          expanded: Set[str]) -> List[str]:
    """Drop the tool-result nodes of collapsed rounds from a lineage.

    The agent loop chains one `role="tool"` node per call under the assistant message that requested them,
    so with documents and tools switched on a single conversational turn is three to six nodes, and a
    visitor looking for "the things it could have said instead" would mostly be shown plumbing. A round is
    what gets a node here; the calls within it are a count on that node.
    """
    kept: List[str] = []
    owner: Optional[str] = None  # the assistant node whose round we are inside, if any
    for node_id in lineage:
        role = _role_of(datastore, node_id)
        if role == "tool":
            if owner is not None and owner not in expanded:
                continue
        elif role == "assistant":
            owner = node_id
        else:
            owner = None
        kept.append(node_id)
    return kept


def _pills_for(node_id: str, state: ViewState, is_root: bool) -> Tuple[str, ...]:
    """Return the pointer labels resting on `node_id`.

    More than one can land on the same node, which is why this is a tuple: with the AI greeting turned off,
    a new chat starts at the system prompt itself, and SYS and NEW coincide there.
    """
    pills = []
    if is_root:
        pills.append("SYS")
    if state.new_chat_node_id is not None and node_id == state.new_chat_node_id:
        pills.append("NEW")
    if node_id == state.head_node_id:
        pills.append("HEAD")
    return tuple(pills)


# --------------------------------------------------------------------------------
# Choosing what to show

class _Slot:
    """One position in a row: either a chat node, or a gap standing for several of them.

    Internal to the layout pass — what leaves this module is a `Ref` per graph node.
    """

    def __init__(self, node_id: Optional[str], hidden: Tuple[str, ...] = ()):
        self.node_id = node_id
        self.hidden = hidden

    def _get_is_gap(self) -> bool:
        """Return whether this slot stands for omitted nodes rather than for one node."""
        return self.node_id is None

    is_gap = property(fget=_get_is_gap,
                      doc="Whether this slot stands for omitted nodes rather than for one node.")


def _window(items: Sequence[str],
            focus_index: int,
            must_include: Set[int],
            each_side: int) -> List[_Slot]:
    """Reduce a level to the slots that will be drawn: some items, and a gap per run of omitted ones.

    `focus_index`: Index in `items` the window is centred on.
    `must_include`: Indices shown whatever the window says — in practice the child the spine goes through,
                    which cannot be hidden without disconnecting the picture.
    `each_side`: How many neighbours of the focus to show either side.

    The first and last items are always shown, so a wide level keeps its ends visible and the reader can
    tell how far the fan reaches.
    """
    n = len(items)
    if n == 0:
        return []

    shown = {0, n - 1}
    shown |= {i for i in range(focus_index - each_side, focus_index + each_side + 1) if 0 <= i < n}
    shown |= {i for i in must_include if 0 <= i < n}

    slots: List[_Slot] = []
    previous = -1
    for index in sorted(shown):
        if index > previous + 1:
            slots.append(_Slot(node_id=None, hidden=tuple(items[previous + 1:index])))
        slots.append(_Slot(node_id=items[index]))
        previous = index
    if previous < n - 1:
        slots.append(_Slot(node_id=None, hidden=tuple(items[previous + 1:])))
    return slots


class _Row:
    """One level of the picture, with the slots to draw and what to line them up on.

    `slots`: Left to right.
    `anchor_index`: Which slot sits on the spine's vertical line. Aligning every row on its anchor is what
                    makes the current branch come out as a straight line however wide the fans around it
                    get — which in turn is what keeps positions steady as the tree grows, since a new
                    sibling then moves only its own row.
    `parent_node_id`: Whose children these are, or `None` at the root level.
    """

    def __init__(self, slots: List[_Slot], anchor_index: int, parent_node_id: Optional[str]):
        self.slots = slots
        self.anchor_index = anchor_index
        self.parent_node_id = parent_node_id


# --------------------------------------------------------------------------------
# Emitting shapes

def _box_shapes(x: float, y: float, width: float, config: LayoutConfig,
                label_lines: Sequence[str], fill: Optional[xdotconstants.Color],
                dashed: bool, pills: Tuple[str, ...],
                speaker: Optional[str] = None,
                measure_text: Optional[MeasureText] = None,
                emphasized: bool = False, previewed: bool = False) -> List[xdotgraph.Shape]:
    """Return the shapes for one box: its outline, its text, and any pointer pills above it.

    `width`: The box's width. A gap is narrower than a node, and the row layout allocates it that much
             room, so drawing it at any other width would put it under its neighbour.
    `fill`: `None` for an unfilled box, which is what a gap is.
    `dashed`: Draw the outline broken.
    `speaker`: Who said it, drawn small above the label. `None` for a gap, which nobody said.
    `emphasized`: Draw the outline heavy. This is HEAD, and where the reader actually is deserves to be
                  the loudest thing in the picture.
    `previewed`: Draw a dotted ring outside the box. This is what a click has selected, and what a second
                 click would commit to — dotted because that selection is tentative until the second one.
    """
    x1, y1 = x - 0.5 * width, y - 0.5 * config.node_h
    x2, y2 = x + 0.5 * width, y + 0.5 * config.node_h

    outline_pen = xdotgraph.Pen()
    outline_pen.color = GAP_LINE_COLOR if dashed else LINE_COLOR
    outline_pen.linewidth = config.head_line_width if emphasized else config.line_width
    if dashed:
        outline_pen.dash = _GAP_DASH

    shapes: List[xdotgraph.Shape] = []
    corners = _rounded_rect_points(x1, y1, x2, y2, config.corner_radius)
    if fill is not None:
        fill_pen = outline_pen.copy()
        fill_pen.fillcolor = fill
        fill_pen.dash = ()  # a fill has no outline to break
        shapes.append(xdotgraph.PolygonShape(fill_pen, corners, filled=True))
    shapes.append(xdotgraph.PolygonShape(outline_pen, corners, filled=False))

    if previewed:
        # A ring outside the box rather than a change to the box itself. The box's own outline is already
        # saying something -- solid or dashed, heavy for HEAD -- and a selection has to be legible on top
        # of every combination of those without overwriting any of them.
        ring_pen = xdotgraph.Pen()
        ring_pen.color = PREVIEW_COLOR
        ring_pen.linewidth = config.preview_line_width
        ring_pen.dash = _PREVIEW_DOTS  # dotted, because the selection is tentative until a second click
        offset = config.preview_ring_offset
        shapes.append(xdotgraph.PolygonShape(
            ring_pen,
            _rounded_rect_points(x1 - offset, y1 - offset, x2 + offset, y2 + offset,
                                 config.corner_radius + offset),
            filled=False))

    text_pen = xdotgraph.Pen()
    text_pen.color = LINE_COLOR
    text_pen.fontsize = config.font_size

    # A text shape's y is its baseline, so a line sits on the y given plus about a third of its cap height.
    # With a speaker the two lines straddle the centre; without one the label takes the centre itself.
    if speaker is not None:
        # Anchored to the top edge, not centred. A message wraps to one line or two, and centring would
        # hang the short ones at a different height from their neighbours -- which along a row reads as
        # raggedness rather than as a shorter message.
        speaker_pen = xdotgraph.Pen()
        speaker_pen.color = LINE_COLOR
        speaker_pen.fontsize = config.role_font_size
        cursor = y1 + config.text_top_inset + config.role_font_size
        # Left-aligned, unlike the label. The speaker is the same handful of short words on every node, so
        # a common left edge lets the eye read the column of them without tracking a centre that moves.
        shapes.append(xdotgraph.TextShape(speaker_pen, x1 + _LABEL_INSET, cursor,
                                          xdotgraph.TextShape.LEFT,
                                          width - 2 * _LABEL_INSET, speaker))
        cursor += _LINE_GAP
    else:
        # A gap box has one line and nobody to attribute it to, so it takes the middle.
        cursor = y - 0.5 * config.font_size

    for line in label_lines:
        cursor += config.font_size
        shapes.append(xdotgraph.TextShape(text_pen, x, cursor,
                                          xdotgraph.TextShape.CENTER, width - 2 * _LABEL_INSET, line))
        cursor += _LINE_GAP

    # Pills are a separate visual class from nodes -- outlined rather than filled -- so that SYS, NEW and
    # HEAD read as labels attached to a node rather than as part of what the node says. Two reasons for
    # each half of that, neither of them taste:
    #
    # They sit in the vertical space above the box, right-aligned to it, rather than out to its side. A
    # column is only as wide as its box, so anything drawn beside one lands on the neighbour; the gap
    # between rows is empty by construction and is the only free space a node has.
    #
    # Outlined, because the renderer picks a contrasting text colour in dark mode from *the element's*
    # fill, and a node carrying a second filled shape would have some of its text coloured for the wrong
    # background.
    pill_pen = xdotgraph.Pen()
    pill_pen.color = LINE_COLOR
    pill_pen.linewidth = 1.0
    pill_text_pen = xdotgraph.Pen()
    pill_text_pen.color = LINE_COLOR
    pill_text_pen.fontsize = config.pill_font_size

    # A pill is sized to its own label rather than to a fixed width, and its `TextShape` is given the width
    # of the *text* rather than of the box.
    #
    # Both halves matter, and the second is the subtle one. The renderer centres a text shape by starting
    # it at `centre - w/2` and drawing left-aligned from there, so `w` has to be what the text actually
    # measures: pass the box width instead and a label narrower than the box begins half the difference too
    # far left, which puts "SYS" inside its own rounded cap.
    #
    # The width comes from `measure_text` when the caller supplied one, and from an average advance
    # otherwise. Measuring matters here more than the size of the box suggests: the renderer centres text
    # by starting it at `centre - w/2` and drawing left-aligned, so an error in `w` displaces the glyphs by
    # half of it. Estimating "HEAD" at 10 px gave 24.8 against a true 19.5, and the label sat visibly left
    # inside its own pill -- an error invisible in the box's geometry and obvious on screen.
    text_widths = [_text_width(pill, config.pill_font_size, measure_text, _PILL_ADVANCE_PER_CHAR)
                   for pill in pills]
    box_widths = [text_w + config.pill_h for text_w in text_widths]  # a cap's worth of room at each end
    pill_span = sum(box_widths) + max(0, len(pills) - 1) * 3.0

    cursor = x2 - pill_span
    for pill, text_w, box_w in zip(pills, text_widths, box_widths):
        px1, px2 = cursor, cursor + box_w
        py2 = y1 - 4.0
        py1 = py2 - config.pill_h
        shapes.append(xdotgraph.PolygonShape(pill_pen,
                                             _rounded_rect_points(px1, py1, px2, py2, 0.5 * config.pill_h),
                                             filled=False))
        shapes.append(xdotgraph.TextShape(pill_text_pen,
                                          0.5 * (px1 + px2), py2 - 0.3 * config.pill_h,
                                          xdotgraph.TextShape.CENTER, text_w, pill))
        cursor = px2 + 3.0
    return shapes


def _edge_between(src: xdotgraph.Node, dst: xdotgraph.Node, config: LayoutConfig) -> xdotgraph.Edge:
    """Return an edge from the bottom of `src` to the top of `dst`, with an arrowhead at the destination."""
    start = (src.x, src.y2)
    end = (dst.x, dst.y1)

    line_pen = xdotgraph.Pen()
    line_pen.color = LINE_COLOR
    head_pen = line_pen.copy()
    head_pen.fillcolor = LINE_COLOR

    head = _arrowhead_points(end, start, config.arrowhead_length, config.arrowhead_halfwidth)
    shapes: List[xdotgraph.Shape] = [xdotgraph.LineShape(line_pen, [start, head[0]]),
                                     xdotgraph.PolygonShape(head_pen, head, filled=True)]
    return xdotgraph.Edge(src, dst, [start, end], shapes)


# --------------------------------------------------------------------------------
# The build

def build(datastore: chattree.Forest,
          state: ViewState,
          config: Optional[LayoutConfig] = None,
          measure_text: Optional[MeasureText] = None) -> ChatGraph:
    """Build the picture of the chat forest around `state.focus_node_id`, or HEAD if none is given.

    `datastore`: The chat forest. Read under its own lock, and not modified.
    `state`: What the user is looking at. See `ViewState`.
    `config`: Sizes and counts. See `LayoutConfig`.
    `measure_text`: How to ask what a string actually measures — `(text, font size) -> width`. Optional;
                    without it, widths are estimated from an average glyph advance, which is enough to
                    size a box and not enough to centre text inside one. See `MeasureText`.

    Returns a `ChatGraph`: the `Graph` to hand to `XDotWidget.set_graph`, plus the table saying what each
    of its nodes stands for.

    Raises `KeyError` if the node the picture is drawn around is not in the datastore.
    """
    config = config or LayoutConfig()

    with datastore.lock:
        focus_node_id = state.focus_node_id or state.head_node_id
        full_spine = datastore.linearize_up(focus_node_id)
        visible_spine = _collapse_tool_rounds(datastore, full_spine, state.expanded_tool_turns)

        # Two different questions, and conflating them is what would make a preview look like a move.
        # `drawn_spine` is the branch on screen and decides the layout; `current_branch` is where HEAD
        # actually is and decides the colour. They agree until somebody previews another branch, and then
        # the shared prefix stays green while the divergence does not.
        try:
            current_branch = set(datastore.linearize_up(state.head_node_id))
        except KeyError:  # HEAD is gone, mid-cleanup or mid-delete; the picture is still worth drawing
            current_branch = set()

        visible_spine, elided_ancestors, depth_gap_row = _depth_window(
            visible_spine, state.new_chat_node_id, config.max_visible_depth)

        drawn_spine = set(visible_spine)
        rows = _rows_for(datastore, state, config, visible_spine, focus_node_id)
        subtree_counts = _subtree_counts_for(datastore, rows, drawn_spine)

        # ------------------------------------------------------------------
        # Vertical placement. A row that has subtree gaps hanging off it, and the root row when ancestors
        # were elided, get a whole empty row's worth of space below them to hold those gaps -- otherwise a
        # gap drawn one row down lands on top of the row that is already there.

        needs_band = [bool(subtree_counts[index]) or index == depth_gap_row
                      for index in range(len(rows))]
        row_step = config.node_h + config.vertical_spacing
        row_y: List[float] = []
        band_y: List[Optional[float]] = []
        cursor = 0.5 * config.node_h
        for index in range(len(rows)):
            row_y.append(cursor)
            if needs_band[index]:
                band_y.append(cursor + row_step)
                cursor += 2 * row_step
            else:
                band_y.append(None)
                cursor += row_step

        # ------------------------------------------------------------------
        # Horizontal placement and shapes.

        refs: Dict[str, Ref] = {}
        nodes_by_name: Dict[str, xdotgraph.Node] = {}
        graph_nodes: List[xdotgraph.Node] = []
        drawn: List[List[Tuple[_Slot, xdotgraph.Node]]] = []  # parallel to `rows`, for wiring edges after

        gap_serial = 0
        for row_index, row in enumerate(rows):
            y = row_y[row_index]
            widths = [config.gap_node_w if slot.is_gap else config.node_w for slot in row.slots]
            centers: List[float] = []
            cursor = 0.0
            for width in widths:
                centers.append(cursor + 0.5 * width)
                cursor += width + config.horizontal_spacing
            shift = -centers[row.anchor_index]  # put the anchor on x = 0

            drawn_row: List[Tuple[_Slot, xdotgraph.Node]] = []
            for slot_index, slot in enumerate(row.slots):
                x = centers[slot_index] + shift
                width = widths[slot_index]
                if slot.is_gap:
                    gap_serial += 1
                    if row_index == 0:
                        name = "gap:roots"
                        ref: Ref = RootGapRef(name, hidden_node_ids=slot.hidden)
                        label = f"…{len(slot.hidden)} more cards"
                    else:
                        name = f"gap:siblings:{gap_serial}"
                        ref = SiblingGapRef(name, parent_node_id=row.parent_node_id,
                                            hidden_node_ids=slot.hidden,
                                            recenter_on=slot.hidden[len(slot.hidden) // 2])
                        label = f"…{len(slot.hidden)} more"
                    shapes = _box_shapes(x, y, width, config, [label], fill=None, dashed=True, pills=(),
                                         measure_text=measure_text)
                else:
                    name = slot.node_id
                    ref = ChatNodeRef(name, node_id=slot.node_id,
                                      role=_role_of(datastore, slot.node_id),
                                      on_current_branch=(slot.node_id in current_branch),
                                      tool_call_count=_tool_call_count(datastore, slot.node_id),
                                      pills=_pills_for(slot.node_id, state, is_root=(row_index == 0)))
                    speaker, label_lines = _speaker_and_label_of(
                        datastore, slot.node_id,
                        config._get_effective_label_chars(), config.label_lines)
                    shapes = _box_shapes(x, y, width, config, label_lines,
                                         fill=(SPINE_FILL_COLOR if slot.node_id in current_branch
                                               else OFF_SPINE_FILL_COLOR),
                                         dashed=False, pills=ref.pills, speaker=speaker,
                                         measure_text=measure_text,
                                         emphasized=(slot.node_id == state.head_node_id),
                                         previewed=(slot.node_id == state.previewed_node_id))

                graph_node = xdotgraph.Node(x=x, y=y, w=width, h=config.node_h,
                                            shapes=shapes, internal_name=name)
                refs[name] = ref
                nodes_by_name[name] = graph_node
                graph_nodes.append(graph_node)
                drawn_row.append((slot, graph_node))
            drawn.append(drawn_row)

        # ------------------------------------------------------------------
        # Gap nodes that live in the bands, and then the edges.

        graph_edges: List[xdotgraph.Edge] = []

        def add_box(name: str, ref: Ref, x: float, y: float, label: str) -> xdotgraph.Node:
            """Draw one gap box in a band, register it, and return it."""
            node = xdotgraph.Node(x=x, y=y, w=config.gap_node_w, h=config.node_h,
                                  shapes=_box_shapes(x, y, config.gap_node_w, config, [label],
                                                     fill=None, dashed=True, pills=(),
                                                     measure_text=measure_text),
                                  internal_name=name)
            refs[name] = ref
            nodes_by_name[name] = node
            graph_nodes.append(node)
            return node

        depth_gap_node: Optional[xdotgraph.Node] = None
        if elided_ancestors:
            depth_gap_node = add_box("gap:depth",
                                     DepthGapRef("gap:depth", hidden_node_ids=elided_ancestors),
                                     x=0.0, y=band_y[depth_gap_row],
                                     label=f"…{len(elided_ancestors)} more")

        for row_index, drawn_row in enumerate(drawn):
            for slot, graph_node in drawn_row:
                if slot.is_gap:
                    continue
                child_count = subtree_counts[row_index].get(slot.node_id)
                if child_count is None:
                    continue
                # An off-spine sibling with children of its own would otherwise look like a chat that
                # stopped after one message. The same primitive as a sibling gap, pointing down.
                gap_node = add_box(f"gap:subtree:{slot.node_id}",
                                   SubtreeGapRef(f"gap:subtree:{slot.node_id}",
                                                 node_id=slot.node_id, child_count=child_count),
                                   x=graph_node.x, y=band_y[row_index],
                                   label=f"…{child_count} more")
                graph_edges.append(_edge_between(graph_node, gap_node, config))

        for row_index in range(1, len(drawn)):
            parent_id = rows[row_index].parent_node_id
            parent_node = nodes_by_name.get(parent_id) if parent_id is not None else None
            if parent_node is None:
                # The parent fell outside the depth window. The depth gap stands in for the whole elided
                # chain, so it is what this row hangs from -- every slot of it, since they are all
                # descendants of what the gap is hiding.
                parent_node = depth_gap_node
                if parent_node is None:
                    continue
            for _slot, graph_node in drawn[row_index]:
                graph_edges.append(_edge_between(parent_node, graph_node, config))

        if depth_gap_node is not None:
            graph_edges.append(_edge_between(nodes_by_name[visible_spine[depth_gap_row]],
                                             depth_gap_node, config))

    # ------------------------------------------------------------------
    # Normalize into the widget's coordinate box, which `zoom_to_fit` reads as (0, 0)-(width, height).

    x1, y1, _x2, _y2 = _content_bbox(graph_nodes)
    _translate(graph_nodes, graph_edges, -(x1 - config.margin), -(y1 - config.margin))

    _x1, _y1, x2, y2 = _content_bbox(graph_nodes)
    graph = xdotgraph.Graph(width=x2 + config.margin, height=y2 + config.margin,
                            nodes=graph_nodes, edges=graph_edges)

    # Measured after the translation above, so it is in the same coordinates as everything else -- and
    # measured over what is *drawn*, not over the node boxes. A pointer pill hangs above its node, so the
    # topmost box of the branch has its pill outside its own rectangle, and a frame computed from
    # rectangles clips the SYS pill off the top of the view.
    spine_nodes = [nodes_by_name[node_id] for node_id in visible_spine if node_id in nodes_by_name]
    if spine_nodes:
        spine_bbox = _content_bbox(spine_nodes)
    else:  # nothing of the branch survived; framing the whole picture is the only answer left
        spine_bbox = (0.0, 0.0, graph.width, graph.height)

    return ChatGraph(graph=graph, refs=refs, spine=tuple(full_spine), spine_bbox=spine_bbox)


def _content_bbox(nodes: Sequence[xdotgraph.Node]) -> Tuple[float, float, float, float]:
    """Return the box enclosing every node *and everything drawn on it*.

    A node's own bounding box is the layout cell it occupies, which is what the row placement reasons
    about. It is not what has to fit on screen: a pointer pill is drawn in the space above its node, so
    the topmost row's pills sit outside every node box there is, and a fit computed from those alone
    clips the one label that says which node HEAD is on.
    """
    boxes = []
    for node in nodes:
        boxes.append(node.get_bounding_box())
        boxes.extend(box for box in (shape.get_bounding_box() for shape in node.shapes)
                     if box is not None)
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _rows_for(datastore: chattree.Forest,
              state: ViewState,
              config: LayoutConfig,
              visible_spine: Sequence[str],
              focus_node_id: str) -> List[_Row]:
    """Choose what each level of the picture holds: one row per spine node, plus HEAD's children.

    The caller holds the datastore lock.
    """
    rows: List[_Row] = []

    for depth, node_id in enumerate(visible_spine):
        if depth == 0:
            # The root level. v1 shows HEAD's own root and a count of the others: a root is a version of
            # the character card, and clicking through to another one would leave the configured avatar
            # and voice running against a different system prompt.
            other_roots = tuple(root_id for root_id in datastore.get_all_root_nodes() if root_id != node_id)
            slots = [_Slot(node_id=node_id)]
            anchor_index = 0
            if other_roots:
                slots.insert(0, _Slot(node_id=None, hidden=other_roots))
                anchor_index = 1
            rows.append(_Row(slots, anchor_index, parent_node_id=None))
            continue

        parent_id = datastore.get_parent(node_id)
        siblings, own_index = datastore.get_siblings(node_id)
        if siblings is None or own_index is None:  # a broken link; draw the node alone rather than nothing
            rows.append(_Row([_Slot(node_id=node_id)], 0, parent_node_id=parent_id))
            continue

        focus_id = state.sibling_focus.get(parent_id, node_id)
        focus_index = siblings.index(focus_id) if focus_id in siblings else own_index
        slots = _window(siblings, focus_index, must_include={own_index},
                        each_side=config.siblings_each_side)
        rows.append(_Row(slots, _index_of_slot(slots, node_id), parent_node_id=parent_id))

    # The children of the node at the bottom of the spine, so that the branch does not appear to end
    # there when it does not.
    children = datastore.get_children(focus_node_id)
    if children:
        chosen = state.sibling_focus.get(focus_node_id, children[0])
        chosen_index = children.index(chosen) if chosen in children else 0
        slots = _window(children, chosen_index, must_include=set(),
                        each_side=config.siblings_each_side)
        rows.append(_Row(slots, _index_of_slot(slots, children[chosen_index]),
                         parent_node_id=focus_node_id))
    return rows


def _depth_window(visible_spine: Sequence[str],
                  new_chat_node_id: Optional[str],
                  max_visible_depth: int) -> Tuple[List[str], Tuple[str, ...], int]:
    """Choose which of the branch's nodes to draw. Returns (kept, elided, row the gap sits below).

    The budget goes to a prefix at the top of the tree and a run at the bottom, with one gap between them.
    The bottom is where the reader is. The top is what says *where* they are, and it is two things rather
    than one:

    - **The root**, which carries SYS and names the version of the character card this was written under.
    - **The session level** — the child of `new_chat_node_id` this branch began at, and its siblings, which
      are every other chat started under the same card. That level doubles as the list of recent chats, so
      losing it costs the only way out of the current conversation.

    The second is the reason this is not simply "keep the root". A chat twenty messages deep has a spine
    longer than the budget, and the elided middle swallows the session level — which is to say the way out
    disappears exactly when the conversation is long enough to want one.

    The prefix is kept whole rather than as the two pinned nodes alone. `new_chat_node_id` normally sits
    directly under the root, so the whole prefix is three nodes; pinning only the ends of it would split
    the elision into two runs and spend a gap box on hiding a single node.

    Falls back to the root alone when the prefix would leave no room for the tail, or when
    `new_chat_node_id` is not on this branch — which is what a chat under an older card looks like.

    `row the gap sits below`: index into `kept`. Meaningless when nothing was elided.
    """
    if len(visible_spine) <= max_visible_depth:
        return list(visible_spine), (), 0

    prefix_length = 1  # the root
    if new_chat_node_id is not None and new_chat_node_id in visible_spine:
        prefix_length = visible_spine.index(new_chat_node_id) + 2  # the anchor, plus the session node
    # Leave at least half the budget for the nodes nearest HEAD; a prefix that crowds those out has
    # answered "where am I" at the cost of "what is happening".
    if prefix_length > max_visible_depth // 2:
        prefix_length = 1
    prefix_length = min(prefix_length, len(visible_spine))

    tail_length = max_visible_depth - prefix_length
    elided = tuple(visible_spine[prefix_length:len(visible_spine) - tail_length])
    kept = list(visible_spine[:prefix_length]) + list(visible_spine[-tail_length:])
    return kept, elided, prefix_length - 1


def _index_of_slot(slots: Sequence[_Slot], node_id: str) -> int:
    """Return which slot holds `node_id`, or 0 if none does."""
    for index, slot in enumerate(slots):
        if slot.node_id == node_id:
            return index
    return 0


def _subtree_counts_for(datastore: chattree.Forest,
                        rows: Sequence[_Row],
                        drawn_spine: Set[str]) -> List[Dict[str, int]]:
    """Return, per row, how many children each of its off-spine nodes has — omitting the childless ones.

    Off the *drawn* spine, not off HEAD's branch: a node whose children are already the row below needs no
    gap announcing that it has some.

    The caller holds the datastore lock.
    """
    counts: List[Dict[str, int]] = []
    for row in rows:
        row_counts: Dict[str, int] = {}
        for slot in row.slots:
            if slot.is_gap or slot.node_id in drawn_spine:
                continue
            child_count = len(datastore.get_children(slot.node_id))
            if child_count:
                row_counts[slot.node_id] = child_count
        counts.append(row_counts)
    return counts


def _translate(nodes: Sequence[xdotgraph.Node], edges: Sequence[xdotgraph.Edge],
               dx: float, dy: float) -> None:
    """Move every node, edge and shape by (`dx`, `dy`), in place."""
    for node in nodes:
        node.x += dx
        node.y += dy
        node.x1 += dx
        node.x2 += dx
        node.y1 += dy
        node.y2 += dy
        _translate_shapes(node.shapes, dx, dy)
    for edge in edges:
        edge.points = [(x + dx, y + dy) for x, y in edge.points]
        _translate_shapes(edge.shapes, dx, dy)


def _translate_shapes(shapes: Sequence[xdotgraph.Shape], dx: float, dy: float) -> None:
    """Move a list of shapes by (`dx`, `dy`), in place."""
    for shape in shapes:
        if isinstance(shape, xdotgraph.TextShape):
            shape.x += dx
            shape.y += dy
        elif isinstance(shape, xdotgraph.EllipseShape):
            shape.x0 += dx
            shape.y0 += dy
        elif isinstance(shape, (xdotgraph.PolygonShape, xdotgraph.LineShape, xdotgraph.BezierShape)):
            shape.points = [(x + dx, y + dy) for x, y in shape.points]
        elif isinstance(shape, xdotgraph.CompoundShape):
            _translate_shapes(shape.shapes, dx, dy)
