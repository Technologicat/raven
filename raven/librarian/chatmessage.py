"""Chat messages as drawn in the chat log: `DPGChatMessage`, and its stored and streaming kinds.

Each message renders one chat node, and reaches its view and the chat controller through
`parent_view`. `raven.librarian.chat_controller` holds the view that lays messages out, and the controller.
"""

__all__ = ["DPGChatMessage",
           "DPGCompleteChatMessage",
           "DPGStreamingChatMessage"]

import logging
logger = logging.getLogger(__name__)

import io
import threading
import time
from typing import Any, Callable, TYPE_CHECKING
import urllib.parse
import uuid
import webbrowser

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown

if TYPE_CHECKING:  # a type only: `chat_controller` imports this module, to build messages
    from .chat_controller import DPGLinearizedChatView

from ..common import utils as common_utils

from ..common.gui import animation as gui_animation
from ..common.gui import keyboardmark
from ..common.gui import tooltip as gui_tooltip
from ..common.gui import utils as guiutils

from . import chatutil
from . import config as librarian_config
from . import llmclient
from . import messagetext
from . import scaffold
from . import sidecarstore

gui_config = librarian_config.gui_config  # shorthand, this is used a lot


# The field a message opens into for editing: as tall as the text's line count, within these bounds, plus
# room for the frame padding and a horizontal scrollbar — the field does not wrap, so a paragraph is one
# line and usually wider than the field.
_EDITOR_MIN_LINES = 3
_EDITOR_MAX_LINES = 20
_EDITOR_EXTRA_H = 24  # pixels

# The grey line above a message is drawn as separate widgets, so the revision number can be a link. The
# spacing between them stands in for the single space the line had as one string.
_METADATA_SPACING = 5  # pixels
_LINK_COLOR = (85, 135, 205)  # the Markdown renderer's link colour, in `DearPyGui_Markdown`'s `text_attributes`


role_to_colors = {"assistant": {"front": gui_config.chat_color_ai_front, "back": gui_config.chat_color_ai_back},
                  "system": {"front": gui_config.chat_color_system_front, "back": gui_config.chat_color_system_back},
                  "tool": {"front": gui_config.chat_color_tool_front, "back": gui_config.chat_color_tool_back},
                  "user": {"front": gui_config.chat_color_user_front, "back": gui_config.chat_color_user_back},
                  }


def _open_source_url(url: str) -> None:
    """Open an image's recorded provenance source: a `file://` local original in its default application,
    anything else (an `https://` page) in the web browser. Raises like the underlying opener when a local
    original has moved or been deleted, so the caller can flash a non-intrusive failure acknowledgment."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        common_utils.open_file(urllib.parse.unquote(parsed.path))
    else:
        webbrowser.open(url)


# What the phase breakdown says under its table. Held here rather than inline so the two halves of the
# tooltip are written in one place. Rendered as Markdown, so no hand-wrapping — `wrap` sets the width, and
# a single newline would come out as a space anyway.
#
# Markdown is safe here despite the "no wrapped Markdown before the first frame" rule: a message's tooltip
# is built with the message, long after the render loop is up, and the message body itself already renders
# this way. Prose only, though — a code span or a list inside a *hidden* container
# loses its decoration outright: those are sized from a laid-out read, a hidden widget has no metrics, and
# the quad comes out zero-sized. Measured; see `investigations/dpg-markdown-decorations/`.
_PHASE_TOOLTIP_WRAP_W = 430  # pixels; about the width the table above it comes out at

_PHASE_BREAKDOWN_FOOTNOTE = ("*Prompt processing* is the wait before the model generates anything: how much of "
                             "the prompt the backend's cache did not already hold. Its speed is not shown, "
                             "because a warm KV cache still reports the whole prompt as its size.")

# On the thinking trace's own figures, beside the cloud. Phrased to hold while a reply is still streaming,
# where the line is a live count and the message's figures do not exist yet.
_THINKING_STATS_TOOLTIP = ("The *thinking* alone: tokens, wall time, and the speed between them, for this "
                           "reply's reasoning.",
                           "The figures under the finished message cover the whole turn, and break it down "
                           "when hovered.")

# Added only when the turn ended in a tool call, since it explains a row that is otherwise not there.
_PHASE_BREAKDOWN_TOOL_CALL_NOTE = ("The tool call's *time* is counted under *Thinking*. A call does not arrive "
                                   "as generated text, so there is no way to see where the reasoning stopped "
                                   "and the call began; only its tokens can be told apart, and those are what "
                                   "its row shows.")

def _highlights_anything(text: str, maybe_highlight: tuple | None) -> bool:
    """Whether search highlighting `maybe_highlight` (a `chatsearch.SearchQuery.highlight`, or `None`) marks anything in `text`."""
    return maybe_highlight is not None and common_utils.has_search_highlight(text, *maybe_highlight)


class DPGChatMessage:
    # Whether this class renders a reply that is still arriving. The one thing the shared `build` cannot
    # decide for itself: an unfinished node means "cut short" to a stored message and "working" to a live
    # one, and only the class knows which it is.
    renders_live_reply = False

    def __init__(self,
                 gui_parent: str | int,
                 parent_view: "DPGLinearizedChatView"):
        """Base class for a chat message displayed in the linearized chat view.

        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).
        """
        super().__init__()
        self.gui_parent = gui_parent  # GUI container to render in (DPG ID or tag)
        self.gui_uuid = None  # populated by `_create_container_group`; used in GUI widget tags
        self.gui_container_group = None  # populated by `_create_container_group`
        self._create_container_group()
        self.parent_view = parent_view
        self.role = None  # populated by `build`
        self.persona = None  # populated by `build`
        self.paragraphs = []  # [{"text": ..., "rendered": True}, ...]
        self.paragraphs_lock = threading.RLock()
        # Counts paragraph widgets ever built by this instance, for their tags: a paragraph re-rendered for a
        # search highlight is built while the old widget still exists, and DPG frees deleted tags lazily.
        self.paragraph_build_count = 0
        # System message only: the two kinds of per-turn system inject, as last drawn.
        self.rendered_system_preamble = None
        self.rendered_system_postamble = None
        self.node_id = None  # populated by `build`
        self.gui_text_group = None  # populated by `build`
        # The thought bubble, built on demand by `_thought_bubble` when a thinking paragraph first arrives:
        # the cloud button, and the column of trace paragraphs it shows and hides. Both stay `None` on a
        # message from a model that did not think, which is what "is there a trace here" is read from.
        self.gui_thought_button = None
        self.gui_thought_group = None
        self.gui_thought_stats = None
        # Whether this message's thinking trace opens as it is built. View state belonging to this
        # *rendering*, like `show_full_text` below — see `_thought_bubble` for why the `show_thinking`
        # preference is not read there directly.
        self.start_thinking_open = False
        self.gui_keyboard_mark_widget = None  # populated by `build`; the dot the keyboard mark lights when this message is the current one
        self.gui_buttons_group = None  # populated by `build`; whether this is on screen decides which message the hotkeys act on
        self.gui_button_callbacks = {}  # {name0: callable0, ...} - to trigger button features programmatically
        self.text_indent_w = 0  # how far the text currently being rendered is inset from the message's left edge
        # Item handler registries created by `_make_clickable`. They live in DPG's handler-registry tree, not
        # under `gui_container_group`, so `demolish`'s children-only delete does not reach them - this is what
        # it deletes them by. A rebuilt message would otherwise leak one per attachment, per rebuild.
        self.owned_handler_registries = []
        # Self-sizing tooltips created by `_add_tooltip`. Same story as the registries above: a `Tooltip`
        # is a window at the root, so the children-only delete does not reach it either.
        self.owned_tooltips = []

        # for "delete subtree" confirmation (cannot be undone)
        self.last_delete_click_time = None

    def _get_text(self) -> str:
        with self.paragraphs_lock:
            return "\n".join(paragraph["text"] for paragraph in self.paragraphs)
    text = property(fget=_get_text,
                    doc="Full text of this GUI chat message as `str`. Read-only.")

    def _get_next_or_prev_sibling_in_datastore(self,
                                               node_id: str,
                                               direction: str = "next",
                                               step: int | None = 1) -> str | None:
        """Get the next or previous sibling of `node_id` in the chat datastore.

        `direction`: One of "next", "prev".

        `step`: How many siblings to jump. Will jump up to as many as available in `direction`.
                Special value `None` means "jump to end" in the given `direction`.

        Returns the node ID of the sibling, or `None` if no such sibling.

        May return `node_id` itself.

        Works at the top of the tree as well: a root's siblings are the forest's other roots, so this walks
        between system prompts — which is how a chat held under an earlier card is reached.
        """
        siblings, this_node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
        if direction == "next":
            if step is None:  # jump to end
                return siblings[-1]
            elif this_node_index + step < len(siblings):
                return siblings[this_node_index + step]
            return siblings[-1]
        else:  # direction == "prev":
            if step is None:
                return siblings[0]
            elif this_node_index - step >= 0:
                return siblings[this_node_index - step]
            return siblings[0]

    def get_chat_text_width(self) -> int:
        """Get the current text wrap width of the chat.

        Narrowed by `text_indent_w` while a block is being rendered indented (the document-body column sits
        to the right of its toggle button). Wrapping is measured from the text's own left edge, so an
        indented block given the full width would run past the right margin by exactly the indent — visible
        only once the window is narrow enough for the margin to stop absorbing it.
        """
        w, h = guiutils.get_widget_size(self.parent_view.gui_parent)  # The view's GUI parent is the actual panel (DPG child window), whose width changes in a window resize.
        chat_text_w = w - gui_config.chat_text_right_margin_w - self.text_indent_w
        return chat_text_w

    def build(self,
              role: str,
              persona: str | None,
              node_id: str | None) -> None:
        """Build the GUI widgets for this chat message instance, thus rendering the chat message (and its buttons and such) in the GUI.

        Runs into a fresh container: the constructor calls this, and `rebuild_in_place` is how to redraw an
        existing message. Raises `RuntimeError` on a demolished instance.

        `role`: One of the roles supported by `raven.librarian.llmclient`.
                Typically, one of "assistant", "system", "tool", or "user".

        `persona`: The persona name speaking `text`, or `None` if the role has no persona name ("system" and "tool" are like this).

                   To get the **current session's** persona, use::

                       persona=llm_settings.personas.get(role, None)

                   where `role` is one of "assistant", "system", "tool", "user".

                   To get the **stored** persona from a chat node::

                       persona=node_payload["general_metadata"]["persona"]

                   This may differ from the current session's persona, e.g. if the chat node was generated with a different AI character.

        `node_id`: The chat node ID of this message in the datastore, if applicable.

                   NOTE: Particularly, an incoming streaming message from the LLM does not have a node in the datastore.

        NOTE: You still need to `add_paragraph` the text you want to show in the chat message widget.

              We require explicit adding in order to be able to handle messages that *contain* thought blocks
              (i.e. any complete message from a thinking model), because the `is_thought` state (which is
              required when adding a paragraph) needs to be different for the think-block and final-message segments.

              The derived class `DPGCompleteChatMessage` automates this; it parses the content from a chat node,
              and adds the text to the widget.

              The derived class `DPGStreamingChatMessage`, on the other hand, requires full manual control, by design,
              so that the GUI driver handling the incoming message (`DPGChatController.ai_turn`) gets full control
              of what is displayed in the widget.
        """
        global role_to_colors  # intent only - we only read the color settings from this.

        # Loud rather than a quiet re-creation: a new container could only go at the end of the view, which
        # for a message already on screen is the wrong place.
        if self.gui_container_group is None:
            raise RuntimeError(f"{type(self).__name__}.build: this message was demolished and cannot be built again; use `rebuild_in_place` to redraw a message.")

        self.role = role
        self.persona = persona
        self.node_id = node_id

        # Always a fresh container here (the constructor's, or the one `rebuild_in_place` just made), so there
        # is nothing to clear; forget the thought bubble's widgets all the same, since the next thinking
        # paragraph would otherwise be rendered into a container that is not this one.
        self.gui_thought_button = None
        self.gui_thought_group = None
        self.gui_thought_stats = None

        # --------------------------------------------------------------------------------
        # lay out the role icon and the text content areas horizontally

        icon_and_text_container_group = dpg.add_group(horizontal=True,
                                                      tag=f"chat_icon_and_text_container_group_{self.gui_uuid}",
                                                      parent=self.gui_container_group)

        # ----------------------------------------
        # role icon

        # The drawlist sits inside a group so that it can have a tooltip: DPG accepts only draw items as a
        # drawlist's children, and refuses a tooltip there. The group holds nothing else, so it occupies
        # exactly the space the drawlist did.
        icon_group = dpg.add_group(tag=f"chat_icon_group_{self.gui_uuid}",
                                   parent=icon_and_text_container_group)
        icon_drawlist = dpg.add_drawlist(width=(2 * gui_config.margin + gui_config.chat_icon_size),
                                         height=(2 * gui_config.margin + gui_config.chat_icon_size),
                                         tag=f"chat_icon_drawlist_{self.gui_uuid}",
                                         parent=icon_group)  # empty drawlist acts as placeholder if no icon
        icon_texture = self.parent_view.chat_controller.speaker_glyphs.icon_texture_for(role, persona)
        if icon_texture is not None:
            dpg.draw_image(icon_texture,
                           (gui_config.margin, gui_config.margin),
                           (gui_config.margin + gui_config.chat_icon_size, gui_config.margin + gui_config.chat_icon_size),
                           uv_min=(0, 0),
                           uv_max=(1, 1),
                           parent=icon_drawlist)

        # Who wrote this, named. The log shows a timestamp and the text, so the glyph is its only speaker
        # indicator — and a message by anyone but the configured pair draws the *generic* glyph, which is
        # where the stored persona would otherwise be unrecoverable from the screen. Plain `dpg.add_tooltip`
        # rather than `_add_tooltip`: the name is fixed for the life of the widget, and a caption written
        # once cannot show the autosize glitch the `Tooltip` class exists to hide.
        #
        # A system prompt and a tool result have no speaker, so their caption says what the message is.
        maybe_caption = persona if persona is not None else {"system": "System prompt",
                                                             "tool": "Tool result"}.get(role)
        if maybe_caption is not None:
            dpg.add_text(maybe_caption, parent=dpg.add_tooltip(icon_group))

        # ----------------------------------------
        # text content

        # # colored border
        # dpg.add_drawlist(width=4,
        #                  height=4,  # to be updated after the text is rendered
        #                  tag=f"chat_colored_border_drawlist_{self.gui_uuid}",
        #                  parent=icon_and_text_container_group)

        # adjust text vertical positioning
        text_vertical_layout_group = dpg.add_group(tag=f"chat_message_vertical_layout_group_{self.gui_uuid}",
                                                   parent=icon_and_text_container_group)
        dpg.add_spacer(height=gui_config.margin,
                       parent=text_vertical_layout_group)

        # The grey line: when the revision on screen was written, and which revision it is
        if node_id is not None:
            node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)  # auto-selects active revision
            node_active_revision = self.parent_view.chat_controller.datastore.get_revision(node_id)
            # The cogs icon on a tool result says one ran and not which, so a turn that called three tools
            # is a column of identical badges; naming them is the whole of what tells them apart. Composed
            # in a function of its own so that what the line says can be tested without building a widget.
            #
            # Tagged so a navigation jump can flash it: it is the one widget every stored message has, at a
            # fixed place at its top, which makes it the natural "here is the message you asked for" marker.
            # A reply still arriving has not recorded its model yet, so it shows the one writing it, which is the
            # string the stored reply will record: both come from `llm_settings.model`.
            maybe_live_model = None
            if self.renders_live_reply:
                maybe_live_model = self.parent_view.chat_controller.llm_settings.model
                if maybe_live_model == llmclient.NO_MODEL_INFO:
                    maybe_live_model = None
            when, revision_label, producer_label = messagetext.format_message_metadata_parts(node_payload, role, node_active_revision,
                                                                                              maybe_live_model=maybe_live_model)
            metadata_row = dpg.add_group(horizontal=True, horizontal_spacing=_METADATA_SPACING,
                                         parent=text_vertical_layout_group)
            dpg.add_text(when,
                         color=(120, 120, 120),
                         tag=f"chat_message_timestamp_{self.gui_uuid}",  # tag
                         parent=metadata_row)
            # The revision number is the way into this message's revision history, so where there is a history
            # to see, it is drawn as a link and the line says how many revisions there are.
            n_revisions = len(self.parent_view.chat_controller.datastore.get_revisions(node_id))
            if n_revisions > 1:
                revision_link = self._add_clickable_text(revision_label, parent=metadata_row, color=_LINK_COLOR,
                                                         action=lambda: self.parent_view.chat_controller.open_revision_history(node_id))
                dpg.add_text(f"Revision {node_active_revision} of this message. Click to see all its revisions [Ctrl+Shift+E]",
                             parent=dpg.add_tooltip(revision_link))
                # A count rather than "R2/3": revision numbers stay unique after a deletion, so a message can
                # hold R1 and R3 only, and the number shown can exceed the count.
                dpg.add_text(f"({n_revisions} revisions)", color=(120, 120, 120), parent=metadata_row)
            else:
                dpg.add_text(revision_label, color=(120, 120, 120), parent=metadata_row)
            if producer_label:
                dpg.add_text(producer_label, color=(120, 120, 120), parent=metadata_row)

        # render the actual text
        self.gui_text_group = dpg.add_group(tag=f"chat_message_text_container_group_{self.gui_uuid}",
                                            parent=text_vertical_layout_group)  # create another group to act as container so that we can update/replace just the text easily
        # NOTE: We now have an empty group, for `add_paragraph`/`replace_last_paragraph`.

        # Show LLM performance statistics for AI chat node, if linked to a chat node, and the chat node has them stored
        if role == "assistant" and node_id is not None:
            ai_message_node_payload = self.parent_view.chat_controller.datastore.get_payload(node_id)
            # Tests for the figures rather than for the dict that would hold them: a reply still being
            # written carries `generation_metadata` saying so and no stats yet, and a message Raven authored
            # carries no dict at all. Both mean the same thing here — nothing to report — and the absence of
            # the line is how each has always appeared.
            generation_metadata = ai_message_node_payload.get("generation_metadata") or {}

            # Why a reply stops early, said under the text rather than stored in it. Suppressed while the
            # reply is still being written: an unfinished node is what a live reply *is*, and reporting it
            # there would put "Raven exited" under a message arriving in front of the reader.
            if not self.renders_live_reply:
                maybe_note = messagetext.incompleteness_note(generation_metadata)
                if maybe_note is not None:
                    dpg.add_text(maybe_note,
                                 color=(120, 120, 120),  # same gray as the stats line: a caption, not an alarm
                                 parent=text_vertical_layout_group)

            if "n_tokens" in generation_metadata:
                n_tokens = generation_metadata["n_tokens"]
                dt = generation_metadata["dt"]
                # Unchanged in meaning: the whole reply, thinking included. An old node cannot be
                # recomputed, so this line must not come to mean two things depending on the node's age.
                # The breakdown goes in a tooltip, where it costs no space in the log.
                stats_widget = dpg.add_text(messagetext.format_generation_stats(n_tokens=n_tokens, dt=dt),
                                            color=(120, 120, 120),
                                            parent=text_vertical_layout_group)
                ended_in_tool_call = bool(ai_message_node_payload["message"].get("tool_calls"))
                breakdown_rows = messagetext.phase_breakdown_rows(generation_metadata,
                                                                  ended_in_tool_call=ended_in_tool_call)
                autosearch_rows = messagetext.autosearch_rows(generation_metadata)
                # Absent on a node written before the model was recorded, which is why this is asked for
                # rather than indexed.
                maybe_model = generation_metadata.get("model")
                if maybe_model is not None or breakdown_rows is not None or autosearch_rows is not None:
                    stats_tooltip = dpg.add_tooltip(stats_widget)
                    # Which model produced *this* message, said per message rather than once for the
                    # app. In a branching chat the siblings of one node can come from different models,
                    # and a chat reloaded from disk predates whatever happens to be loaded now.
                    if maybe_model is not None:
                        dpg.add_text(maybe_model, parent=stats_tooltip)
                        if breakdown_rows is not None:
                            dpg.add_spacer(height=gui_config.margin, parent=stats_tooltip)
                    if breakdown_rows is not None:
                        dpg.add_text("Where this reply's time went.", parent=stats_tooltip)
                        dpg.add_spacer(height=gui_config.margin, parent=stats_tooltip)
                        self._add_figures_table(breakdown_rows, parent=stats_tooltip)
                        dpg.add_spacer(height=gui_config.margin, parent=stats_tooltip)
                        dpg_markdown.add_text(_PHASE_BREAKDOWN_FOOTNOTE, wrap=_PHASE_TOOLTIP_WRAP_W, parent=stats_tooltip)
                        # Only after thinking: without it, the call's time is its own row's.
                        if ended_in_tool_call and generation_metadata["phases"].get("thinking") is not None:
                            dpg.add_spacer(height=gui_config.margin, parent=stats_tooltip)
                            dpg_markdown.add_text(_PHASE_BREAKDOWN_TOOL_CALL_NOTE, wrap=_PHASE_TOOLTIP_WRAP_W, parent=stats_tooltip)
                    # A table of its own rather than more rows above: the auto-search ran before this message's
                    # model call, so its time is not in the figures above, and the first table's total must
                    # stay the line it explains.
                    if autosearch_rows is not None:
                        if maybe_model is not None or breakdown_rows is not None:
                            dpg.add_spacer(height=gui_config.margin, parent=stats_tooltip)
                        dpg.add_text("Before it, the auto-search:", parent=stats_tooltip)
                        dpg.add_spacer(height=gui_config.margin, parent=stats_tooltip)
                        self._add_figures_table(autosearch_rows, parent=stats_tooltip)

                # Say when nothing was retrieved for this reply. Recorded only when the documents were in
                # play, or an attachment was present; absent means there is nothing to say, which is why
                # this tests `is False` rather than falsiness.
                #
                # The wording states what was *retrieved*, not what the model did with it, because that is
                # all we can observe: retrieval reporting matches does not mean the reply used them, and
                # against a real corpus a search nearly always returns something. Claiming "answered from
                # general knowledge" would assert the unobservable half. (What would make the stronger claim
                # sayable: relevance-aware retrieval scores, or the model citing its own sources.)
                #
                # A marker, not a warning: on a general question this state is correct and expected, since
                # no document database answers "what is 2+2?". Hence the muted colour rather than a red one.
                # Said under the message for whoever reads the chat later; DOCUMENTS said it at the time.
                if generation_metadata.get("docs_query_failed"):
                    query_note = dpg.add_text("[no document search]",
                                              color=(170, 145, 90),
                                              parent=text_vertical_layout_group)
                    query_tooltip = dpg.add_tooltip(query_note)
                    dpg.add_text("The automatic search of your documents did not run before this reply:\n"
                                 "the AI was asked for a search query, and gave no usable one.",
                                 parent=query_tooltip)
                if generation_metadata.get("grounded") is False:
                    grounding_marker = dpg.add_text("[no sources retrieved]",
                                                    color=(170, 145, 90),
                                                    parent=text_vertical_layout_group)
                    grounding_tooltip = dpg.add_tooltip(grounding_marker)
                    dpg.add_text("Nothing was retrieved for this reply: no document matches,\n"
                                 "no attachments, no tool results.\n\n"
                                 "Without this marker, something was retrieved for the reply.\n"
                                 "That does not mean the reply relied on it.",
                                 parent=grounding_tooltip)

        # If there is no linked chat node, this is a live streaming chat message, so the GUI widget should end here - it doesn't need the datastore control buttons or end spacers.
        # This makes the GUI look calmer while rendering a streaming message.
        if node_id is None:
            return

        # text area end spacer
        dpg.add_spacer(height=2,
                       parent=text_vertical_layout_group)

        # ----------------------------------------
        # buttons (below text)

        # Held, because "is this message's button row on screen?" is what decides which message the
        # per-message hotkeys act on. See `DPGChatController.get_current_message`.
        buttons_horizontal_layout_group = dpg.add_group(horizontal=True,
                                                        tag=f"chat_buttons_container_group_{self.gui_uuid}",
                                                        parent=text_vertical_layout_group)
        self.gui_buttons_group = buttons_horizontal_layout_group
        number_of_message_buttons = 14
        chat_text_w = self.get_chat_text_width()
        dpg.add_spacer(width=chat_text_w - number_of_message_buttons * (gui_config.toolbutton_w + 8) - 64 - keyboardmark.DOT_SLOT_W,  # 8 = DPG outer margin; 64 = some space for sibling counter
                       parent=buttons_horizontal_layout_group)

        # Where the keyboard mark goes when this is the message the per-message hotkeys would act on. A dot
        # rather than a border around the row, because a pulsating outline's claim on the eye scales with
        # its perimeter: fourteen bordered buttons is far more motion than a combo elsewhere in the
        # constellation gets for a mark that means the same thing.
        #
        self.gui_keyboard_mark_widget = keyboardmark.add_dot(parent=buttons_horizontal_layout_group,
                                                             tag=f"chat_keyboard_mark_{self.gui_uuid}")  # tag

        self.build_buttons(gui_parent=buttons_horizontal_layout_group)

        # ----------------------------------------
        # chat turn end spacers and line

        dpg.add_spacer(height=4,
                       tag=f"chat_turn_end_spacer1_{self.gui_uuid}",
                       parent=self.gui_container_group)

        if role in role_to_colors:
            dpg.add_drawlist(height=1,
                             width=(chat_text_w + 64),
                             tag=f"chat_turn_end_drawlist_{self.gui_uuid}",
                             parent=self.gui_container_group)
            dpg.draw_rectangle((64, 0), (chat_text_w + 64, 1),
                               color=(80, 80, 80),
                               fill=(80, 80, 80),
                               parent=f"chat_turn_end_drawlist_{self.gui_uuid}")  # tag

        dpg.add_spacer(height=4,
                       tag=f"chat_turn_end_spacer2_{self.gui_uuid}",
                       parent=self.gui_container_group)

    def add_paragraph(self, text: str, is_thought: bool) -> None:
        """Add a new paragraph of text to this widget.

        `is_thought`: Whether this paragraph is (part of) a `<think>...</think>` block.
                      The renderer selects the text color appropriately.
        """
        paragraph = {"text": text,
                     "is_thought": is_thought,
                     "rendered": False}
        with self.paragraphs_lock:
            self.paragraphs.append(paragraph)
            self._render_text()
        # Outside the lock: this reaches the controller, which takes `current_chat_history_lock` to find a
        # message — and `build` takes that one first and `paragraphs_lock` second, so asking from in here
        # would be the same two locks in the opposite order.
        self._recheck_awaited_thinking_trace()

    def replace_last_paragraph(self, text: str, is_thought: bool) -> None:  # TODO: Only last paragraph is replaceable for now, because it's easier for coding the GUI. :)
        """Replace the last paragraph of text in this widget. If there are no paragraphs yet, create one automatically.

       `is_thought`: Whether this paragraph is (part of) a `<think>...</think>` block.
                     The renderer selects the text color appropriately.

                     If needed, can be different from the old state of the same paragraph.
         """
        with self.paragraphs_lock:
            if not self.paragraphs:
                self.add_paragraph(text, is_thought)
                return
            paragraph = self.paragraphs[-1]
            maybe_old = paragraph.get("widget")
            paragraph["text"] = text
            paragraph["is_thought"] = is_thought
            paragraph["rendered"] = False
            # The replacement is built hidden, just before the old widget, and `WidgetSwap` shows the one and
            # deletes the other in the same frame. Deleting first and building after left frames with the
            # paragraph missing, so the log shortened and sprang back as each chunk arrived.
            swapped = False
            with guiutils.nonexistent_ok(parent_gone_ok=True):
                maybe_parent = self._prepare_paragraph(paragraph) if maybe_old is not None else None
                if maybe_parent is not None and maybe_parent == dpg.get_item_parent(maybe_old):
                    old_rows, old_height = paragraph["rows"], dpg.get_item_rect_size(maybe_old)[1]
                    new = self._build_paragraph_widget(len(self.paragraphs) - 1, paragraph,
                                                       parent=maybe_parent, before=maybe_old, show=False)
                    gui_animation.WidgetSwap.swap(self.parent_view.gui_parent, maybe_old, new,
                                                  height_change=dpg_markdown.predict_height_change(old_rows, old_height, paragraph["rows"]),
                                                  commanded_y_scroll=self.parent_view._commanded_y_scroll)
                    paragraph["widget"] = new
                    paragraph["rendered"] = True
                    swapped = True
            # No widget to swap, blank text, a paragraph that changed between thought and reply (a different
            # container, so not a swap), or a parent gone mid-build: delete what there is and render afresh.
            if not swapped:
                self._drop_paragraph_widget(paragraph)
                self._render_text()

        # As in `add_paragraph`, and outside the lock for the same reason. Here rather than only at a
        # paragraph break because this is where a reply's words actually arrive: the caller rate-limits it
        # to a newline, or half a second, or a quarter second and ten chunks, so a request outstanding on
        # this message is answered within that rather than whenever the model happens to end a paragraph —
        # which for a turn that ends in a tool call, or one the reader stops, may be never.
        self._recheck_awaited_thinking_trace()

        dpg.split_frame()  # ...and anything after this point runs in another frame.

    def _create_container_group(self) -> None:
        """Create this message's own container group inside `gui_parent`, under a fresh `gui_uuid`.

        A *new* uuid each time, so a message re-created after its old widgets were deleted cannot collide
        with them: DPG frees deleted items lazily, so the old tags may still be in its registry, and a tag
        collision crashes the process rather than raising.
        """
        self.gui_uuid = str(uuid.uuid4())
        self.gui_container_group = dpg.add_group(tag=f"chat_item_container_group_{self.gui_uuid}",
                                                 parent=self.gui_parent)

    def _drop_paragraph_widget(self, paragraph: dict) -> None:
        """Delete a paragraph's rendered widget and mark the paragraph unrendered, so it will be drawn again.

        Deletes rather than merely forgetting the widget id, and does so under `nonexistent_ok` because the
        widget may already be gone — a view rebuild clears the message container without the paragraph
        records hearing about it. A caller cannot generally tell which case it is in, and if the choice
        were the caller's, the one that guesses "already gone" leaves a live widget orphaned on screen.
        Deleting-if-present is right in both cases, so nobody has to know.
        """
        with self.paragraphs_lock:
            if "widget" in paragraph:
                # `pop` first, so the record is consistent even if the delete finds nothing:
                # `_render_text` asserts that an unrendered paragraph has no widget.
                with guiutils.nonexistent_ok():
                    dpg.delete_item(paragraph.pop("widget"))
            paragraph["rendered"] = False

    def _render_text_paragraphs(self, text: str) -> None:
        """Render one text content-part: split into paragraphs and add them.

        Also consolidates any inline `<think>...</think>` block into a single collapsible thought paragraph, but
        that handling is dead code: since the June 2026 `reasoning_content` migration, thinking is separated out
        before render (at load by `upgrade_datastore`, live by the stream parser), so `content` no longer
        carries inline `<think>`. Leftover from the pre-June-2026 inline handling; slated for removal."""
        paragraph_accumulator = io.StringIO()
        inside_think_block = False
        def commit_paragraph():
            nonlocal paragraph_accumulator
            text_to_commit = paragraph_accumulator.getvalue()
            if not text_to_commit:
                return
            self.add_paragraph(text_to_commit,
                               is_thought=inside_think_block)
            paragraph_accumulator = io.StringIO()

        paragraphs = text.split("\n")
        for idx, paragraph in enumerate(paragraphs):
            p = paragraph.strip()

            # Detect think block state (rudimentary; should detect from the token stream, not re-split a string).
            entering_think_block = (p == "<think>")
            exiting_think_block = (p == "</think>")

            if entering_think_block:
                commit_paragraph()  # commit previous text (if any) before start of think block
                inside_think_block = True

            paragraph_accumulator.write(f"{paragraph}\n")  # regardless of if it's just a newline

            # Consolidate "<think>...</think>" into one paragraph, so that we can hide/show it easily.
            # When at last paragraph, always commit (even if incomplete think block).
            if (inside_think_block and not exiting_think_block) and (idx < len(paragraphs) - 1):
                continue

            commit_paragraph()

            if exiting_think_block:
                inside_think_block = False

    def reclassify_all_paragraphs_as_thought(self) -> None:
        """Move everything shown so far into the thought bubble, as if it had arrived as reasoning.

        For the case where the model was inside its thinking block from the first token, because its chat
        template put it there, so nothing marked the beginning and only the close arrived. Until it does,
        the reasoning is indistinguishable from an answer, and this is the correction.

        Does nothing when there is nothing shown yet.
        """
        with self.paragraphs_lock:
            if not self.paragraphs:
                return
            for paragraph in self.paragraphs:
                if paragraph["is_thought"]:  # already where it belongs
                    continue
                paragraph["is_thought"] = True
                self._drop_paragraph_widget(paragraph)
            # The bubble is built on first use and reused after, so a whole reply's worth of paragraphs
            # re-renders into one of it. It is appended to `gui_text_group`, which the deletions above have
            # just emptied — so it lands ahead of the answer that is about to start, which is where the
            # reader expects a thought that preceded it.
            self._render_text()

        dpg.split_frame()

    def _thinking_stats_text(self) -> str:
        """The `[900t, 22.0s, 40.9t/s]` line for this message's thinking, or `""` when there is none.

        Empty for a message being streamed, which has no stored numbers yet and shows a live count instead,
        and for one stored before the phase breakdown was recorded — an old node simply says nothing rather
        than guessing.
        """
        if self.node_id is None:  # a reply still streaming; `set_thinking_progress` writes this line instead
            return ""
        payload = self.parent_view.chat_controller.datastore.get_payload(self.node_id)
        thinking = ((payload.get("generation_metadata") or {}).get("phases") or {}).get("thinking")
        if thinking is None:
            return ""
        return messagetext.format_generation_stats(n_tokens=thinking["n_tokens"],
                                                   dt=thinking["dt"],
                                                   exact=thinking.get("tokens_exact", False),
                                                   label="Thought for")

    def show_thinking_trace(self) -> None:
        """Open this message's thinking trace, if it has one and it is collapsed."""
        with guiutils.nonexistent_ok():
            if self.gui_thought_group is not None:
                dpg.show_item(self.gui_thought_group)

    def _recheck_awaited_thinking_trace(self) -> None:
        """Tell the chat log's search this message's text has moved, in case its trace is one somebody asked to see.

        Called as words arrive rather than once, because for a reply being generated the answer to "does the
        trace match" changes: the request was made against a count saying it did, and the words that made it
        so may not have been written yet. Opening before they arrive would show a bubble that does not yet
        hold what was promised, which is the one thing the request was for.

        Costs a search of this one message per update, and only while a request is in fact outstanding on
        this message — one node at a time, ending at the first match.

        A live reply only. A stored message adds its paragraphs too, while it is being built — before the
        view has it — and a recheck then would spend the request on a message not yet there to open. Its
        answer is final, and arrives through `DPGChatLogSearch.add_matches_for` once it is in the view.
        """
        if self.renders_live_reply and self.node_id is not None:
            self.parent_view.chat_controller.search.recheck_awaited_thinking_trace(self.node_id)

    def _thought_bubble(self) -> str | int:
        """The container the thinking trace renders into, built on first use. Returns its DPG ID.

        The trace starts collapsed and the cloud button beside it toggles it, so a reply whose reasoning is
        a wall of text does not stand between the reader and the answer. The button is a gutter to the left
        of a column, the same shape the document-body toggle uses, so the trace wraps beside it rather than
        under it.

        The same bubble serves a message being streamed and a message read back from the datastore, which is
        what keeps the two looking alike: a live trace grows inside the bubble it will still be in once the
        message is stored.

        **Up to and including 0.2.8 only stored messages had one**, and a live trace was drawn inline in the
        chat flow, tinted, then snapped into a bubble the moment the message finalized. Worth knowing here
        because it is what a user of an earlier release remembers seeing, and because the two shapes are why
        `is_thought` has to survive from the stream all the way to the renderer rather than being decided
        once at the end.
        """
        # A caller reaching this is inside `paragraphs_lock` (only `_render_text` calls it), which is what
        # makes "built on first use" safe against two threads rendering paragraphs at once.
        if self.gui_thought_group is not None:
            return self.gui_thought_group

        row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
        def toggle_message_think_callback():
            with guiutils.nonexistent_ok() as nok:
                # Shown, not visible: DPG's "visible" means drawn last frame, which a long trace scrolled out of
                # the view is not — so Ctrl+T from the button row below it would show it again, not hide it.
                if dpg.is_item_shown(self.gui_thought_group):
                    logger.info(f"DPGChatMessage._thought_bubble.toggle_message_think_callback: hiding thinking trace for chat node '{self.node_id}'")
                    dpg.hide_item(self.gui_thought_group)
                else:
                    logger.info(f"DPGChatMessage._thought_bubble.toggle_message_think_callback: showing thinking trace for chat node '{self.node_id}'")
                    dpg.show_item(self.gui_thought_group)
            if nok.errored:
                logger.info(f"DPGChatMessage._thought_bubble.toggle_message_think_callback: GUI widget for chat node '{self.node_id}' does not exist, ignoring.")
        self.gui_button_callbacks["toggle_thinking_trace"] = toggle_message_think_callback  # stash it so we can call it from the hotkey handler

        # No string tag on any of these. They are held in Python attributes instead, which sidesteps the
        # tag-reuse hazard entirely for a widget that a rebuild recreates: `gui_uuid` identifies the message
        # instance, not the build, so a tag built from it would collide with the copy DPG has not collected
        # yet.
        self.gui_thought_button = dpg.add_button(label=fa.ICON_CLOUD,
                                                 callback=toggle_message_think_callback,
                                                 width=gui_config.toolbutton_w,
                                                 parent=row)
        dpg.bind_item_font(self.gui_thought_button, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(self.gui_thought_button, "my_steady_think_theme")  # tag
        think_toggle_tooltip = dpg.add_tooltip(self.gui_thought_button)
        dpg.add_text("Show/hide thinking trace [Ctrl+T]", parent=think_toggle_tooltip)

        # A column beside the cloud, holding the numbers above the trace. The numbers stay put when the
        # trace is collapsed — only `gui_thought_group` below them is hidden — so they do not move when it
        # opens, and a collapsed bubble still says what the thinking cost.
        column = dpg.add_group(parent=row)
        self.gui_thought_stats = dpg.add_text(self._thinking_stats_text(), color=(120, 120, 120), parent=column)
        # The message's own figures explain themselves when hovered, so these must too — otherwise the two
        # readouts look alike, sit a few lines apart, and only one of them answers being asked about. No
        # breakdown here: there is only one phase to describe, and it is the one the reader is pointing at.
        thought_stats_tooltip = dpg.add_tooltip(self.gui_thought_stats)
        # A spacer between paragraphs rather than a blank line in the source: the renderer turns a
        # CommonMark paragraph break into a plain line break, so the two would run together.
        for paragraph_index, paragraph in enumerate(_THINKING_STATS_TOOLTIP):
            if paragraph_index > 0:
                dpg.add_spacer(height=gui_config.margin, parent=thought_stats_tooltip)
            dpg_markdown.add_text(paragraph, wrap=_PHASE_TOOLTIP_WRAP_W, parent=thought_stats_tooltip)

        self.gui_thought_group = dpg.add_group(parent=column)
        # Whether this opens is decided per message, by whoever built it — *not* by reading the
        # `show_thinking` preference here. The preference says how a reply being generated should arrive,
        # and only the streaming message and the complete message that replaces it at the end of that turn
        # count as that. Everything else — the history restored at startup, a branch switch, any rebuild —
        # starts collapsed however the preference is set.
        #
        # Reading the preference here instead would make it retroactive by the back door: every rebuild
        # would re-apply it to the whole conversation, which is exactly what the toggle is designed not to
        # do, and what opened every stored trace on startup before this was a per-message decision.
        if not self.start_thinking_open:
            dpg.hide_item(self.gui_thought_group)
        return self.gui_thought_group

    def _render_text(self) -> None:
        """Internal method. Render any pending new paragraphs. We assume new paragraphs are added only to the end.

        A paragraph whose render is abandoned keeps its `rendered` flag clear, so the next call draws it —
        or, if this message is gone for good, the rebuild that replaced it draws it from the chat node.
        """
        # EAFP, and `parent_gone_ok` is the point of it. This renders into widgets that a *view rebuild* can
        # delete from another thread at any moment — that is the ordinary way a branch switch or a resize
        # ends a streaming message — and the parent dying mid-render is therefore expected rather than
        # exceptional. Checking first would be a TOCTTOU with a real window: the render is a long sequence
        # of DPG calls, and it is the calls in the middle that find the widget gone.
        with guiutils.nonexistent_ok(parent_gone_ok=True), self.paragraphs_lock:
            if self.gui_text_group is None:
                # Either this instance was demolished while a render was on its way here (the race above,
                # having lost by a hair rather than mid-render), or it never finished building. Nothing to
                # draw into in both cases, and the log line is what tells them apart if it is ever the
                # second one, since that would repeat for a message that never appears.
                logger.debug(f"DPGChatMessage._render_text: no text group for chat node '{self.node_id}'; nothing to render into.")
                return
            # dpg.delete_item(self.gui_text_group, children_only=True)  # how to clear all old text if we ever need to
            for idx, paragraph in enumerate(self.paragraphs):
                if paragraph["rendered"]:
                    continue
                assert "widget" not in paragraph  # a paragraph that hasn't been rendered has no GUI text widget associated with it
                maybe_parent = self._prepare_paragraph(paragraph)
                if maybe_parent is not None:  # don't bother if text is blank
                    paragraph["widget"] = self._build_paragraph_widget(idx, paragraph, parent=maybe_parent)
                paragraph["rendered"] = True

    def _prepare_paragraph(self, paragraph: dict) -> str | int | None:
        """Set `paragraph`'s `display_text` and `wrap` from its text; return the container it goes in, or `None` if it is blank.

        Call holding `paragraphs_lock`.
        """
        text = paragraph["text"].strip()
        if not text:
            return None
        # Replace known XML tokens with something that doesn't look like HTML to avoid confusing the Markdown renderer (which silently drops unknown tags).
        #
        # Both pairs are fallbacks for output that arrived broken, which is why neither is dead
        # code despite normal traffic never reaching them. A well-formed tool call is parsed out
        # by the backend and never lands in the text; what lands here is a confabulated or
        # malformed one its parser did not recognize. Likewise reasoning is separated into
        # `reasoning_content` before render, so inline `<think>` means a backend that did not
        # separate it. In both cases this is the only thing standing between the reader and a
        # silently dropped tag.
        text = text.replace("<tool_call>", "**>>>Tool call>>>**")
        text = text.replace("</tool_call>", "**<<<Tool call<<<**")
        text = text.replace("<think>", "**>>>Thinking>>>**")
        text = text.replace("</think>", "**<<<Thinking<<<**")

        chat_text_w = self.get_chat_text_width()
        paragraph["display_text"] = text
        if paragraph["is_thought"]:
            paragraph["wrap"] = chat_text_w - gui_config.toolbutton_w
            return self._thought_bubble()
        paragraph["wrap"] = chat_text_w
        return self.gui_text_group

    def _build_paragraph_widget(self, idx: int, paragraph: dict, *,
                                parent: str | int, before: str | int = 0, show: bool = True) -> int:
        """Render `paragraph` (index `idx`) as Markdown into `parent`, with the current search highlighting. Returns the widget.

        The paragraph's `display_text` and `wrap` must be set. Records in it the `rows` laid out and the
        `highlight` drawn, which is what a later re-highlight compares against. Call holding `paragraphs_lock`.
        """
        role_color = role_to_colors[self.role]["front"] if self.role in role_to_colors else "#ffffff"
        # Passed to the renderer rather than wrapped around the text as a `<font>` tag. An open tag on the same
        # line as the content makes the whole paragraph inline raw HTML as far as CommonMark is concerned, and a
        # heading is a block construct that cannot occur inside a paragraph - so `### Heading` came through with
        # its markers intact.
        color = gui_config.chat_color_think_front if paragraph["is_thought"] else role_color
        maybe_highlight = self._paragraph_highlight(paragraph)
        markdown = dpg_markdown.MarkdownText(paragraph["display_text"],
                                             color=color,
                                             highlight=maybe_highlight,
                                             highlight_color=guiutils.SEARCH_HIGHLIGHT_COLOR)
        self.paragraph_build_count += 1
        widget = markdown.add(wrap=paragraph["wrap"],
                              parent=parent,
                              before=before,
                              show=show,
                              tag=f"chat_message_text_{self.role}_paragraph_{idx}_{self.gui_uuid}_build{self.paragraph_build_count}")
        paragraph["rows"] = markdown.rows
        paragraph["highlight"] = maybe_highlight
        return widget

    def _paragraph_highlight(self, paragraph: dict) -> tuple | None:
        """The search highlighting `paragraph` should be drawn with now: the query's regex pair, or `None`."""
        maybe_query = self.parent_view.chat_controller.search.query
        if maybe_query is None:
            return None
        if paragraph["is_thought"] and not maybe_query.include_thinking:
            return None
        if self.role == "tool" and not maybe_query.include_tools:
            return None
        return maybe_query.highlight

    def rehighlight(self, task_env: env) -> None:
        """Re-render the paragraphs whose search highlighting has changed, each swapped in without moving the view.

        A paragraph that matches neither the search it was drawn with nor the current one is left as it is, and
        only has its record updated. Takes `paragraphs_lock` per paragraph rather than for the whole message, so a
        reply still streaming into this message is held up for one paragraph at a time. Stops when `task_env` is
        cancelled.
        """
        idx = 0
        while not task_env.cancelled:
            with guiutils.nonexistent_ok(parent_gone_ok=True), self.paragraphs_lock:
                if idx >= len(self.paragraphs):
                    return
                paragraph = self.paragraphs[idx]
                idx += 1
                old = paragraph.get("widget")
                if old is None or not paragraph["rendered"]:
                    continue
                wanted = self._paragraph_highlight(paragraph)
                drawn = paragraph["highlight"]
                if wanted is drawn:
                    continue
                if not (_highlights_anything(paragraph["display_text"], drawn) or
                        _highlights_anything(paragraph["display_text"], wanted)):
                    paragraph["highlight"] = wanted
                    continue
                old_rows, old_height = paragraph["rows"], dpg.get_item_rect_size(old)[1]
                new = self._build_paragraph_widget(idx - 1, paragraph, parent=dpg.get_item_parent(old), before=old, show=False)
                paragraph["widget"] = new
                gui_animation.WidgetSwap.swap(self.parent_view.gui_parent, old, new,
                                              height_change=dpg_markdown.predict_height_change(old_rows, old_height, paragraph["rows"]),
                                              commanded_y_scroll=self.parent_view._commanded_y_scroll)

    def add_tool_call_invocation(self, index: int, name: str, arguments: str,
                                 tool_call_id: str | None = None) -> None:
        """Render one tool-call invocation as a visible sub-element: a meshing-cogs icon + the call signature.

        Raven's what-you-see-is-what-you-get design surfaces what the model did, so a tool-calling turn is not
        silently swallowed between an (often empty) assistant message and the subsequent tool result. The
        invocation may have arrived as a native `tool_calls` entry or as an inline `<tool_call>` tag — by the
        time it reaches here it's the same structured form (the `invoke` parser unified them).

        The icon is `ICON_GEARS` (meshing cogs), matching the tool-role result message's three-cogs badge
        (`icons/tool.png`) — invocation and result read as the same family. Deliberately *not* the single-gear
        `ICON_GEAR`, which is the universal "settings" glyph (reserved for the future settings dialog).

        `index`: position among this message's tool calls (for unique widget tags).
        `name`: the function name.
        `arguments`: the call arguments as a JSON string (OAI convention).
        `tool_call_id`: the call's canonical id, which the answering tool-role message carries as its
                        `tool_call_id`. When given, the row gains a button that jumps to that response.
                        `None` while streaming (the id is not known until the call is complete), and for
                        pre-migration data.
        """
        tool_color = role_to_colors["tool"]["front"]
        # `chatutil`'s, because the chat graph labels a tool-calling box with the same string, and a turn
        # that asked for a tool usually carries no text for either view to show instead.
        signature = chatutil.format_tool_call(name, arguments)

        with self.paragraphs_lock:
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
            # The jump button leads the row, ahead of the icon: a call signature can be any length, so a
            # trailing button would sit at a different x on every row, and several calls in one turn would
            # scatter their controls across the message instead of forming a column the eye can run down.
            if tool_call_id is not None:
                self._add_action_button(parent=row,
                                        icon=fa.ICON_ARROW_DOWN,  # plain directional arrow = "go to the related item", as in Visualizer's info panel
                                        tooltip_text="Go to this call's result",
                                        ok_message="Jumped to the result!",
                                        fail_message="No result recorded for this call",
                                        action=self._make_jump_to_tool_response(tool_call_id))
            icon_tag = f"chat_message_toolcall_icon_{index}_{self.gui_uuid}"  # tag
            dpg.add_text(fa.ICON_GEARS, color=tool_color, tag=icon_tag, parent=row)  # tag
            dpg.bind_item_font(icon_tag, self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.add_text(signature,
                         color=tool_color,
                         # Leave room for the leading icon, and for the jump button when there is one.
                         wrap=max(0, self.get_chat_text_width() - 40 - (gui_config.toolbutton_w if tool_call_id is not None else 0)),
                         parent=row)

    def _make_jump_to_tool_call(self, tool_call_id: str) -> Callable[[], None]:
        """Build the callback that scrolls to, and flashes, the tool-call sub-element with id `tool_call_id`."""
        def jump_to_tool_call() -> None:
            found = self.parent_view.chat_controller.find_tool_call_origin(tool_call_id)
            if found is None:  # the assistant message is on another branch, or predates the id migration
                raise LookupError(f"no originating call for id '{tool_call_id}' in the current branch")
            origin, index = found
            self.parent_view.scroll_view(scroll_target_node_id=origin.node_id, user_initiated=True)
            # Flash the specific call, not the whole message: an assistant turn may have made several, and
            # "which one produced this result" is the entire question the jump was asked to answer.
            gui_animation.highlight_widget(widget=f"chat_message_toolcall_icon_{index}_{origin.gui_uuid}",  # tag
                                           duration=gui_config.acknowledgment_duration)
        return jump_to_tool_call

    def _make_jump_to_tool_response(self, tool_call_id: str) -> Callable[[], None]:
        """Build the callback that scrolls to, and flashes, the tool result answering `tool_call_id`."""
        def jump_to_tool_response() -> None:
            target = self.parent_view.chat_controller.find_tool_response(tool_call_id)
            if target is None:  # in flight, on another branch, or never recorded
                raise LookupError(f"no tool response for call id '{tool_call_id}' in the current branch")
            self.parent_view.scroll_view(scroll_target_node_id=target.node_id, user_initiated=True)
            gui_animation.highlight_widget(widget=f"chat_message_timestamp_{target.gui_uuid}",  # tag
                                           duration=gui_config.acknowledgment_duration)
        return jump_to_tool_response

    def _add_action_button(self, *, parent: str | int, icon: str, tooltip_text: str, ok_message: str,
                           action: Callable[[], None], enabled: bool = True,
                           fail_message: str = "Couldn't open — it may have moved or been deleted") -> None:
        """Add one small icon-plus-tooltip action button, wired to run `action`.

        The shared shape for the secondary actions that hang off a message's content rather than off its main
        button row — the provenance cluster under an inline attachment, and the tool-call navigation links.

        On click `action` runs; success flashes the button green with `ok_message`, any failure flashes it red
        with `fail_message` (and logs) — a non-intrusive acknowledgment in place of a modal dialog, matching
        the global toolbar buttons. A disabled button (`enabled=False`) still shows its explanatory
        `tooltip_text` but does nothing, so a predictably-unavailable action (no recorded source, an inline
        `data:` image) is discoverable before the click rather than failing after it.

        Raising from `action` is therefore a supported way to report "this cannot be done right now" — which
        is what the navigation links use for a call whose response is not in the current branch, since whether
        one exists can change after the button is built."""
        button_id = dpg.add_button(label=icon, width=gui_config.toolbutton_w, parent=parent, enabled=enabled)
        dpg.bind_item_font(button_id, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(button_id, "disablable_widget_theme")  # tag
        if not enabled:  # nothing will ever rewrite this caption, so it needs nothing that can resize
            dpg.add_text(tooltip_text, parent=dpg.add_tooltip(button_id))
            return
        tooltip = self._add_tooltip(button_id, tooltip_text)
        def callback() -> None:
            try:
                action()
                ok, message = True, ok_message
            except Exception as exc:  # noqa: BLE001 -- a secondary action must never crash the chat view
                logger.error(f"DPGChatMessage._add_action_button: action failed: {type(exc)}: {exc}")
                ok, message = False, fail_message
            gui_animation.flash_button(button=button_id, tooltip=tooltip,
                                       ok=ok, message=message, duration=gui_config.acknowledgment_duration)
        dpg.set_item_callback(button_id, callback)

    def _add_tooltip(self, target: str | int, text: str) -> gui_tooltip.Tooltip:
        """Give `target` a self-sizing tooltip, owned by this message.

        For a caption a flash will rewrite. A `dpg.tooltip` renders one frame at its previous size when its
        text changes, which under the cursor reads as a glitch; this one never does.

        Returns the tooltip, to be handed to `gui_animation.flash_button` as its `tooltip`.
        """
        tooltip = gui_tooltip.Tooltip(target, text)
        self.owned_tooltips.append(tooltip)
        return tooltip

    def _make_clickable(self, items: list[str | int], *, action: Callable[[], None]) -> None:
        """Make `items` respond to a left click by running `action`, as a shortcut for a button below them.

        Redundant with the action button it duplicates, and deliberately so: an inline thumbnail *looks*
        clickable, so clicking it and getting nothing is a small papercut every time. The button row stays,
        since it is what distinguishes "open the saved copy" from "open the original source" — this is the
        shortcut for the one obvious action, not a replacement for the row.

        Failure is swallowed with a log line rather than flashed. The button is the affordance that reports;
        a click on the content itself has no natural place to put a red flash, and the same action one row
        down does say so.

        For a thumbnail; text that is clickable wants `_add_clickable_text`, which also shows that it is. The registry is owned by this message and deleted
        in `demolish` (DPG will not collect it with the widgets: it lives in the handler-registry tree).
        """
        def callback() -> None:
            try:
                action()
            except Exception as exc:  # noqa: BLE001 -- a secondary action must never crash the chat view
                logger.error(f"DPGChatMessage._make_clickable: action failed: {type(exc)}: {exc}")
        registry = dpg.add_item_handler_registry()
        self.owned_handler_registries.append(registry)
        dpg.add_item_clicked_handler(parent=registry, button=dpg.mvMouseButton_Left, callback=callback)
        for item in items:
            dpg.bind_item_handler_registry(item, registry)

    def _add_figures_table(self, rows: list[tuple[str, str, str, str]], *, parent: str | int) -> None:
        """Add a table of `(label, time, tokens, speed)` rows to `parent`, as a message's figures tooltip shows them."""
        # A table, because the labels differ in length and the font is proportional: padded spaces put the
        # figures at different x positions, which reads as unrelated lines rather than as a column to compare.
        table = dpg.add_table(header_row=True, policy=dpg.mvTable_SizingFixedFit,
                              borders_innerH=False, borders_outerH=False,
                              borders_innerV=False, borders_outerV=False,
                              parent=parent)
        for column_label in ("", "time [s]", "tokens", "speed [t/s]"):
            dpg.add_table_column(label=column_label, parent=table)
        for cells in rows:
            row = dpg.add_table_row(parent=table)
            for cell in cells:
                dpg.add_text(cell, parent=row)

    def _add_clickable_text(self, text: str, *, parent: str | int, action: Callable[[], None],
                            color: tuple[int, int, int] | None = None) -> int | str:
        """Add `text` to `parent` as a line that runs `action` when clicked, and highlights under the mouse.

        For text that is a shortcut for a button, as `_make_clickable` is for a thumbnail. Returns the item,
        for a tooltip to attach to.
        """
        # A selectable rather than text, for ImGui's own hover highlight: plain text has no hovered state to
        # theme, and a line of text does not otherwise say it can be clicked. Sized to the text, since a
        # selectable spans the rest of the row by default and would light up whatever follows it.
        def callback(sender, app_data, user_data) -> None:
            dpg.set_value(sender, False)  # a selectable remembers being clicked; this one is a link, not a choice
            try:
                action()
            except Exception as exc:  # noqa: BLE001 -- a secondary action must never crash the chat view
                logger.error(f"DPGChatMessage._add_clickable_text: action failed: {type(exc)}: {exc}")
        item = dpg.add_selectable(label=text, width=dpg.get_text_size(text)[0], callback=callback, parent=parent)
        if color is not None:
            dpg.bind_item_theme(item, self.parent_view.text_color_theme(color))
        return item

    def rebuild_in_place(self) -> None:
        """Rebuild this message's widgets without the panel ever getting shorter.

        `demolish` + `build` is the obvious spelling and it flickers, because `build` empties the container
        and then repopulates it: for the several frames the markdown takes to lay out, the panel is missing
        this message entirely. DPG clamps the scroll to that shorter content on the *first* of those frames,
        so the reader watches the conversation jump and then be put back — and correcting afterwards cannot
        help, because the wrong position was already displayed.

        So the replacement is built *first*, into a fresh container inserted where the old one is, and the
        old container is deleted only once the new one is standing.

        That container is built **hidden**, and shown in the same frame the old one is deleted. Built
        visible, it contributes its height while it fills, so the panel carries both copies for the several
        frames the markdown takes — and everything below the insertion point slides down and back. That is
        invisible for a short message low in the log and pronounced for a long one near the top, which is
        exactly where the system prompt sits. Hidden, the two height changes land in one frame and cancel.

        Nothing in `build` reads back a size or waits for a frame, and the text wrap width comes from the
        *panel* rather than from this container, so laying out unseen produces the same result.

        This is the same technique as the Visualizer's double-buffered info panel, applied per *message*
        rather than per panel — and the difference in scope is the point rather than an inconsistency. There
        the buffered thing is one panel whose whole content is replaced; here the chat log is arbitrarily
        long and only one message is changing, so buffering the panel would mean laying out the entire
        conversation twice to redraw a paragraph.

        The instance takes a **new `gui_uuid`** as part of this. Every tag `build` creates embeds it, and for
        a moment both copies exist — the old widgets keep the old namespace and the new ones get a fresh one,
        so nothing collides. This is the version-counted-tag pattern Raven uses wherever widgets are
        recreated dynamically, and it is not optional: a duplicate DPG tag terminates the process rather than
        raising, and `delete_item` does not free the name synchronously.
        """
        with self.paragraphs_lock:
            old_container = self.gui_container_group
            old_registries, self.owned_handler_registries = self.owned_handler_registries, []
            old_tooltips, self.owned_tooltips = self.owned_tooltips, []
            self.paragraphs = []
            self.gui_button_callbacks = {}
            self.gui_uuid = str(uuid.uuid4())
            self.gui_container_group = dpg.add_group(tag=f"chat_item_container_group_{self.gui_uuid}",
                                                     parent=self.gui_parent,
                                                     before=old_container,  # exactly where the old one sits
                                                     show=False)  # ...but contributing no height until it is finished
            self.build()
            dpg.show_item(self.gui_container_group)
            for registry in old_registries:  # not under the container group; see `_make_clickable`
                with guiutils.nonexistent_ok():
                    dpg.delete_item(registry)
            for tooltip in old_tooltips:  # nor are these; see `_add_tooltip`
                tooltip.destroy()
            with guiutils.nonexistent_ok():
                dpg.delete_item(old_container)

    def demolish(self) -> None:
        """Tear this message down: delete every GUI widget belonging to this instance, its container included.

        The instance cannot be built again afterwards. To redraw a message, use `rebuild_in_place`.

        Call this before dropping a message from the linearized chat view without rebuilding the whole view —
        taking a message off screen, or retiring a finished streaming message, whose stored rendering is a
        new `DPGCompleteChatMessage` appended in its place (it was the last message). A full
        `DPGLinearizedChatView.build` clears the view wholesale, and needs no call to this.
        """
        with self.paragraphs_lock:
            self.role = None
            self.persona = None
            self.paragraphs = []
            self.gui_text_group = None
            # Every other widget reference this instance holds is dangling once the delete below runs, so
            # none of them may survive it: another thread may still hold this instance, and must find
            # nothing to draw into. The renderer checks `gui_text_group`; `_thought_bubble` reads a
            # non-`None` `gui_thought_group` as "already built" and would hand the stale id straight back
            # as the parent for new paragraphs.
            self.gui_thought_button = None
            self.gui_thought_group = None
            self.gui_thought_stats = None
            self.gui_keyboard_mark_widget = None
            self.gui_buttons_group = None
            self.gui_button_callbacks = {}  # deleting all GUI widgets, so clear the stashed callbacks too.
            for registry in self.owned_handler_registries:  # not under the container group; see `_make_clickable`
                with guiutils.nonexistent_ok():
                    dpg.delete_item(registry)
            self.owned_handler_registries = []
            for tooltip in self.owned_tooltips:  # nor are these; see `_add_tooltip`
                tooltip.destroy()
            self.owned_tooltips = []
            # The container too, not only its children: an empty group still takes a line's item spacing in
            # the view's vertical layout, so a container left standing is a 4 px gap for every message ever
            # taken off screen this way.
            with guiutils.nonexistent_ok():
                dpg.delete_item(self.gui_container_group)
            self.gui_container_group = None

    def build_buttons(self,
                      gui_parent: str | int) -> None:
        """Build the set of control buttons for a single chat message in the GUI.

        `gui_parent`: DPG tag or ID of the GUI widget (typically a group) to add the buttons to.

                      This is not simply `self.gui_parent` due to other layout performed by `build`;
                      the buttons go into a group.
        """
        # NOTE: If you add or remove buttons here, update also `number_of_message_buttons` (search for it in this module).
        #
        # The builders below are phases of this one build, not an API: each runs exactly once, from here.
        # Being methods does not enforce that the way a nested `def` would — it is a contract, stated because
        # nothing else states it.
        #
        # They add their buttons to `g` in the order they are called, and DPG lays a horizontal group out in
        # creation order — so this call order *is* the left-to-right order on screen. Reordering them
        # rearranges the button row, which is why each builder holds one group of *adjacent* buttons and none
        # of them is independent of its neighbours' position.
        role = self.role
        g = dpg.add_group(horizontal=True, tag=f"{role}_message_buttons_group_{self.gui_uuid}", parent=gui_parent)

        self._build_copy_button(g)

        # These are needed for enabling/disabling some buttons.
        system_prompt_node_ids = chatutil.get_all_system_prompt_node_ids(datastore=self.parent_view.chat_controller.datastore)
        greeting_node_ids = chatutil.get_all_greeting_node_ids(datastore=self.parent_view.chat_controller.datastore)

        self._build_regeneration_buttons(g, greeting_node_ids)
        self._build_edit_button(g, greeting_node_ids)
        self._build_branching_buttons(g, system_prompt_node_ids, greeting_node_ids)
        self._build_navigation_buttons(g)

    def _build_copy_button(self, g) -> None:
        """Build the button that copies this message to the clipboard.

        `g`: the horizontal group the buttons go into.
        """
        # dpg.add_spacer(tag=f"ai_message_buttons_spacer_{self.gui_uuid}",
        #                parent=g)

        def copy_message_to_clipboard_callback(clicked: bool = False) -> None:
            """Copy this message, as-is or with its node ID; or, on a Ctrl+Shift+click, the node ID alone.

            `clicked`: whether this is the button's own click. Ctrl is read only then: the Ctrl+C and
                       Ctrl+Shift+C hotkeys call this too, with Ctrl down as part of their chord.
            """
            shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
            ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
            if clicked and ctrl_pressed and shift_pressed:  # for reporting a message: the ID and nothing else
                dpg.set_clipboard_text(self.node_id)
                mode = "node ID only"
            else:
                dpg.set_clipboard_text(messagetext.format_message_for_clipboard(self.parent_view.chat_controller.datastore,
                                                                                self.node_id,
                                                                                role=self.role,
                                                                                persona=self.persona,
                                                                                include_node_id=shift_pressed))
                mode = "with node ID" if shift_pressed else "as-is"
            # Acknowledge the action in the GUI.
            gui_animation.flash_button(button=copy_message_button,
                                       message=f"Copied to clipboard! ({mode})",
                                       duration=gui_config.acknowledgment_duration,
                                       tooltip=copy_message_tooltip)
        self.gui_button_callbacks["copy"] = copy_message_to_clipboard_callback
        copy_message_button = dpg.add_button(label=fa.ICON_COPY,
                                             callback=lambda: copy_message_to_clipboard_callback(clicked=True),
                                             width=gui_config.toolbutton_w,
                                             tag=f"message_copy_to_clipboard_button_{self.gui_uuid}",
                                             parent=g)
        dpg.bind_item_font(copy_message_button, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(copy_message_button, "disablable_widget_theme")  # tag
        copy_message_tooltip = self._add_tooltip(copy_message_button,
                                                 "Copy message to clipboard [Ctrl+C]\n    without Shift: as-is\n    with Shift: include message node ID\n    with Ctrl+Shift: the node ID only")

    def _build_regeneration_buttons(self, g, greeting_node_ids) -> None:
        """Build the three buttons that act on the AI's own output: run it again, continue it, speak it.

        `g`: the horizontal group the buttons go into.
        `greeting_node_ids`: from `chatutil.get_all_greeting_node_ids`; a greeting is not rerolled or continued.
        """
        role = self.role
        node_id = self.node_id

        # Rerolling for AI messages
        if role == "assistant":
            def reroll_message_callback():
                # A reroll starts a new turn, and a local backend serves one at a time: the KV cache holds
                # one conversation, and there is no throughput to spare for a second request. Refusing is
                # the whole handling: the reply in flight is a moment away, and the alternative —
                # cancelling it for a reroll the user may not want once they have read it — decides that
                # for them.
                if self.parent_view.chat_controller.is_generating():
                    logger.info("DPGCompleteChatMessage.reroll_message_callback: a turn is already in flight; refusing.")
                    return

                # A reroll replaces the reply on screen with a different one - the same swap a sibling
                # switch performs, except that the alternative is generated rather than already there.
                # Started before the rewind, so the effect is up while the old message comes down. It
                # therefore also fires on the rare path where the rewind finds nothing, which is a rebuild
                # having taken the message first — a discontinuity either way, so nothing to apologize for.
                self.parent_view.chat_controller.mark_discontinuity()

                # Rewind the linearized chat history in the GUI: this message and everything below it.
                #
                # TODO ("Make the canned AI greeting optional"): re-check this then. There used to be an
                # TODO: assertion here that at least three messages survive the rewind — the system prompt,
                # TODO: the greeting, and the user's first message — which held because reroll is offered
                # TODO: only on assistant messages and never on the greeting, putting the earliest
                # TODO: rerollable message at index 3. Without a greeting that becomes index 2, so the
                # TODO: number changes even though the thing it protects does not: *a reroll's* rewind must
                # TODO: never reach past the first user message, there being nothing to reroll an answer to
                # TODO: below that. Not a rule about rewinding generally — the approve-and-retry rewind
                # TODO: aims at a tool node deeper in the branch, and `rewind_to` imposes no constraint of
                # TODO: its own. Restoring the check needs a shape that does not derive the index
                # TODO: separately from the removal, which is what made the old one racy.
                if not self.parent_view.chat_controller.view.rewind_to(node_id):
                    return

                # Handle the RAG query: find the latest user message (above this AI message)
                user_message_text = None
                with self.parent_view.chat_controller.current_chat_history_lock:  # `build` refills this from another thread
                    for dpg_chat_message in reversed(self.parent_view.chat_controller.current_chat_history):  # ...what's remaining of the history
                        if dpg_chat_message.role == "user":
                            user_message_text = dpg_chat_message.text
                            break

                self.parent_view.chat_controller.app_state["HEAD"] = self.parent_view.chat_controller.datastore.get_parent(node_id)

                # Generate new AI message
                self.parent_view.chat_controller.ai_turn(docs_query=user_message_text,
                                                         continue_=False)
            reroll_enabled = ((node_id is not None) and (node_id not in greeting_node_ids))  # The AI's initial greeting can't be rerolled
            if reroll_enabled:
                self.gui_button_callbacks["reroll"] = reroll_message_callback  # stash it so we can call it from the hotkey handler
            dpg.add_button(label=fa.ICON_DICE_D20,  # fa.ICON_RECYCLE,
                           callback=reroll_message_callback,
                           enabled=reroll_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_reroll_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_reroll_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_reroll_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            reroll_tooltip = dpg.add_tooltip(f"message_reroll_button_{self.gui_uuid}")  # tag
            dpg.add_text("Reroll on a new branch [Ctrl+R]", parent=reroll_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        if role == "assistant":
            def continue_message_callback():
                # Both questions asked of one snapshot, and neither by bare indexing: `build` empties this
                # list before refilling it, from a background task among others, so an emptiness check and
                # the `[-1]` that followed it were two views of a list that need not have agreed.
                with self.parent_view.chat_controller.current_chat_history_lock:
                    history = list(self.parent_view.chat_controller.current_chat_history)
                if not history or history[-1].node_id != node_id:  # not the latest message --> can't continue
                    return

                # Handle the RAG query: find the latest user message (above this AI message)
                user_message_text = None
                for dpg_chat_message in reversed(history):
                    if dpg_chat_message.role == "user":
                        user_message_text = dpg_chat_message.text
                        break

                # Continue the AI message
                self.parent_view.chat_controller.ai_turn(docs_query=user_message_text,
                                                         continue_=True)
                # No button flash, because the button will be deleted immediately, when the chat message widget is replaced.
            # We should enable continue only for the last message, but when we get here, this message isn't in the view yet.
            # We currently solve this by disabling continue buttons for old messages, from the outside, once we're done rendering the view.
            continue_enabled = ((node_id is not None) and (node_id not in greeting_node_ids))  # The AI's initial greeting can't be continued
            if continue_enabled:
                self.gui_button_callbacks["continue"] = continue_message_callback  # stash it so we can call it from the hotkey handler
            dpg.add_button(label=fa.ICON_PARAGRAPH,  # fa.ICON_RIGHT_LONG,  # fa.ICON_ARROW_RIGHT,
                           callback=continue_message_callback,
                           enabled=continue_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_continue_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_continue_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_continue_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            continue_message_tooltip = dpg.add_tooltip(f"message_continue_button_{self.gui_uuid}")  # tag
            dpg.add_text("Ask the AI to continue this response (create new revision) [Ctrl+U]", parent=continue_message_tooltip)
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

        # TTS for AI messages
        if role == "assistant":
            def speak_message_callback():
                if self.parent_view.chat_controller.app_state["avatar_speech_enabled"]:
                    self.parent_view.chat_controller.avatar_controller.ping(config=self.parent_view.chat_controller.avatar_record)  # wake up the AI avatar before starting to speak
                    unused_message_role, message_persona, message_text = chatutil.get_node_message_text_without_persona(self.parent_view.chat_controller.datastore, node_id)
                    # Send only non-thought message content to TTS
                    message_text = chatutil.scrub(persona=message_persona,
                                                  text=message_text,
                                                  thoughts_mode="discard",
                                                  markup=None,
                                                  add_persona=False)
                    self.parent_view.chat_controller.avatar_controller.send_text_to_tts(config=self.parent_view.chat_controller.avatar_record,
                                                                                        text=message_text,
                                                                                        video_offset=librarian_config.avatar_config.video_offset,
                                                                                        update_emotion=True)

                    # Acknowledge the action in the GUI.
                    gui_animation.flash_button(button=speak_message_button,
                                               message="Sent to avatar!",
                                               duration=gui_config.acknowledgment_duration,
                                               tooltip=speak_message_tooltip)
            speak_enabled = (role == "assistant")
            if speak_enabled:
                self.gui_button_callbacks["speak"] = speak_message_callback
            speak_message_button = dpg.add_button(label=fa.ICON_COMMENT,
                                                  callback=speak_message_callback,
                                                  enabled=speak_enabled,
                                                  width=gui_config.toolbutton_w,
                                                  tag=f"chat_speak_button_{self.gui_uuid}",
                                                  parent=g)
            dpg.bind_item_font(speak_message_button, self.parent_view.themes_and_fonts.icon_font_solid)
            dpg.bind_item_theme(speak_message_button, "disablable_widget_theme")  # tag
            speak_message_tooltip = self._add_tooltip(speak_message_button, "Have the avatar speak this message [Ctrl+S]")
        else:
            dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)

    def _build_edit_button(self, g, greeting_node_ids) -> None:
        """Build the edit button, which opens the message's text for editing in place.

        `g`: the horizontal group the buttons go into.
        """
        view = self.parent_view
        edit_enabled = chatutil.is_editable(datastore=view.chat_controller.datastore,
                                            node_id=self.node_id,
                                            greeting_node_ids=greeting_node_ids)
        def edit_callback() -> None:
            maybe_refusal = view.start_editing(self.node_id)
            if maybe_refusal is not None:
                gui_animation.flash_button(button=edit_button, tooltip=edit_tooltip,
                                           ok=False, message=maybe_refusal,
                                           duration=gui_config.acknowledgment_duration)
        if edit_enabled:
            self.gui_button_callbacks["edit"] = edit_callback
        edit_button = dpg.add_button(label=fa.ICON_PENCIL,
                                     callback=edit_callback,
                                     enabled=edit_enabled,
                                     width=gui_config.toolbutton_w,
                                     tag=f"chat_edit_button_{self.gui_uuid}",
                                     parent=g)
        dpg.bind_item_font(edit_button, self.parent_view.themes_and_fonts.icon_font_solid)
        dpg.bind_item_theme(edit_button, "disablable_widget_theme")  # tag
        edit_tooltip = self._add_tooltip(edit_button, "Edit this message, as a new revision [Ctrl+E]")

    def _build_branching_buttons(self, g, system_prompt_node_ids, greeting_node_ids) -> None:
        """Build the two buttons that change the tree: branch the chat here, and delete this node with all below it.

        `g`: the horizontal group the buttons go into.
        """
        node_id = self.node_id

        # Branch chat at this node
        #
        # NOTE: Branching *is* setting HEAD here and nothing else, which decides both of the cases below.
        #
        #       Disallowed from a system prompt node, and from any message not linked to a chat node in the
        #       datastore. Leaving HEAD on a card is the state the view cannot show anything useful from —
        #       the chat under it, greeting included, builds downward and so falls out of sight.
        #
        #       Allowed on the AI's greeting, which amounts to starting a new chat under that card. That is
        #       what the action honestly does, and it is worth saying plainly rather than refusing a button
        #       whose effect the user can reach anyway through "new chat" (Juha).
        branch_enabled = ((node_id is not None) and
                          (node_id not in system_prompt_node_ids))
        def branch_chat_callback():
            self.parent_view.chat_controller.app_state["HEAD"] = node_id
            self.parent_view.build()
            self.parent_view.chat_controller.navigated()
        if branch_enabled:
            self.gui_button_callbacks["branch"] = branch_chat_callback  # stash it so we can call it from the hotkey handler
        dpg.add_button(label=fa.ICON_CODE_BRANCH,
                       callback=branch_chat_callback,
                       enabled=branch_enabled,
                       width=gui_config.toolbutton_w,
                       tag=f"message_new_branch_button_{self.gui_uuid}",
                       parent=g)
        dpg.bind_item_font(f"message_new_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_new_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
        new_branch_tooltip = dpg.add_tooltip(f"message_new_branch_button_{self.gui_uuid}")  # tag
        dpg.add_text("Branch the chat here [Ctrl+B]", parent=new_branch_tooltip)

        # Delete subtree starting from this node (requires a confirmation click)
        app_state = self.parent_view.chat_controller.app_state
        delete_enabled = chatutil.is_deletable(datastore=self.parent_view.chat_controller.datastore,
                                               node_id=node_id,
                                               configured_system_prompt_node_id=app_state["system_prompt_node_id"],
                                               configured_greeting_node_id=app_state["new_chat_HEAD"],
                                               greeting_node_ids=greeting_node_ids)
        def delete_subtree_callback():
            def flash_refusal(message: str) -> None:
                self.last_delete_click_time = None  # nothing is armed after a refusal
                gui_animation.flash_button(button=delete_subtree_button, tooltip=delete_subtree_tooltip,
                                           ok=False, message=message,
                                           duration=gui_config.acknowledgment_duration)

            # Before arming, so a refusal comes on the first press rather than after the reader confirmed.
            maybe_refusal = self.parent_view.chat_controller.delete_refusal()
            if maybe_refusal is not None:
                flash_refusal(maybe_refusal)
                return

            current_time = time.monotonic_ns()
            if self.last_delete_click_time is not None:
                double_okd = (current_time - self.last_delete_click_time < gui_config.delete_confirm_duration * 10**9)
            else:
                double_okd = False
            self.last_delete_click_time = current_time

            if double_okd:
                # On success the view is rebuilt and this button goes with it, so only a refusal has a
                # button left to flash. Asked again inside, since a reply may have started since the first press.
                maybe_refusal = self.parent_view.chat_controller.delete_subtree(node_id)
                if maybe_refusal is not None:
                    flash_refusal(maybe_refusal)
            else:
                gui_animation.flash_delete_confirmation(button=delete_subtree_button,
                                                        tooltip=delete_subtree_tooltip,
                                                        duration=gui_config.delete_confirm_duration)
        # The key goes through this same callable, so it inherits the two-press confirmation rather than
        # having one of its own — and the flash that asks for the second press is on the button the key
        # acts on, which is also the message the blue dot is beside.
        if delete_enabled:
            self.gui_button_callbacks["delete"] = delete_subtree_callback  # stash it so we can call it from the hotkey handler
        delete_subtree_button = dpg.add_button(label=fa.ICON_TRASH_CAN,
                                               callback=delete_subtree_callback,
                                               enabled=delete_enabled,
                                               width=gui_config.toolbutton_w,
                                               tag=f"message_delete_branch_button_{self.gui_uuid}",
                                               parent=g)
        dpg.bind_item_font(f"message_delete_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
        dpg.bind_item_theme(f"message_delete_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
        delete_subtree_tooltip = self._add_tooltip(f"message_delete_branch_button_{self.gui_uuid}",  # tag
                                                   "Delete branch (subtree starting from this node, ALL descendants!) [Ctrl+Shift+Delete]")

        # # TODO: Meh, `raven.common.gui.animation.WidgetFlash` doesn't play together with `dpg_markdown`.
        # c_red = '<font color="(255, 96, 96)">'
        # c_end = '</font>'
        # delete_subtree_tooltip_text = dpg_markdown.add_text(f"Delete branch (this node and {c_red}**all**{c_end} descendants!)", parent=delete_subtree_tooltip)

    def _build_navigation_buttons(self, g) -> None:
        """Build the buttons that step between this message's siblings, and jump to where its branch continues.

        `g`: the horizontal group the buttons go into.
        """
        node_id = self.node_id

        datastore = self.parent_view.chat_controller.datastore
        def descend(start_node_id: str) -> str:
            return chatutil.descend_to_latest(datastore, start_node_id)
        def make_navigate_to_sibling(message_node_id: str, direction: str, step: int | None) -> Callable:
            # Pick the most recent subtree, greedily
            def navigate_to_sibling_callback():
                node_id = self._get_next_or_prev_sibling_in_datastore(message_node_id,
                                                                      direction=direction,
                                                                      step=step)
                if node_id is not None:
                    head_node_id = descend(node_id)
                    self.parent_view.chat_controller.app_state["HEAD"] = head_node_id
                    # Switching branch means the conversation you are looking at was replaced by a different
                    # one, and the avatar reports that the way this app reports everything else - visually.
                    self.parent_view.chat_controller.mark_discontinuity()
                    self.parent_view.build(scroll_target_node_id=node_id)
                    self.parent_view.chat_controller.navigated()
            return navigate_to_sibling_callback
        def make_show_chat_continuation(message_node_id: str) -> Callable:
            def show_chat_continuation_callback():
                head_node_id = descend(message_node_id)
                if head_node_id is not None:
                    self.parent_view.chat_controller.app_state["HEAD"] = head_node_id
                    # Same rationale as a branch switch and a new chat: the conversation on screen is
                    # replaced by a different one, and the avatar reports the discontinuity.
                    self.parent_view.chat_controller.mark_discontinuity()
                    self.parent_view.build()  # let it scroll to end
                    self.parent_view.chat_controller.navigated()
            return show_chat_continuation_callback

        # Only messages attached to a datastore chat node can have siblings or a chat continuation in the datastore
        if node_id is not None:
            siblings, this_node_index = self.parent_view.chat_controller.datastore.get_siblings(node_id)
            prev_enabled = (this_node_index is not None and this_node_index - 1 >= 0)
            next_enabled = (this_node_index is not None and this_node_index + 1 <= len(siblings) - 1)
            navigate_to_prev1_callback = make_navigate_to_sibling(node_id, direction="prev", step=1)
            navigate_to_next1_callback = make_navigate_to_sibling(node_id, direction="next", step=1)
            navigate_to_prev10_callback = make_navigate_to_sibling(node_id, direction="prev", step=10)
            navigate_to_next10_callback = make_navigate_to_sibling(node_id, direction="next", step=10)
            navigate_to_prevend_callback = make_navigate_to_sibling(node_id, direction="prev", step=None)
            navigate_to_nextend_callback = make_navigate_to_sibling(node_id, direction="next", step=None)
            if prev_enabled:
                self.gui_button_callbacks["prev1"] = navigate_to_prev1_callback
                self.gui_button_callbacks["prev10"] = navigate_to_prev10_callback
                self.gui_button_callbacks["prevend"] = navigate_to_prevend_callback
            if next_enabled:
                self.gui_button_callbacks["next1"] = navigate_to_next1_callback
                self.gui_button_callbacks["next10"] = navigate_to_next10_callback
                self.gui_button_callbacks["nextend"] = navigate_to_nextend_callback

            children = self.parent_view.chat_controller.datastore.get_children(node_id)
            show_chat_continuation_enabled = (len(children) > 0)
            show_chat_continuation_callback = make_show_chat_continuation(node_id)
            if show_chat_continuation_enabled:
                self.gui_button_callbacks["show_chat_continuation"] = show_chat_continuation_callback

            dpg.add_button(label=fa.ICON_BACKWARD_FAST,
                           callback=navigate_to_prevend_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prevend_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prevend_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prevend_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            prevend_branch_tooltip = dpg.add_tooltip(f"message_prevend_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to first sibling [Ctrl+Home]", parent=prevend_branch_tooltip)

            dpg.add_button(label=fa.ICON_BACKWARD,
                           callback=navigate_to_prev10_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prev10_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prev10_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prev10_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            prev10_branch_tooltip = dpg.add_tooltip(f"message_prev10_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch 10 siblings left [Ctrl+Shift+Left]", parent=prev10_branch_tooltip)

            dpg.add_button(label=fa.ICON_CARET_LEFT,
                           callback=navigate_to_prev1_callback,
                           enabled=prev_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_prev1_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_prev1_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_prev1_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            prev1_branch_tooltip = dpg.add_tooltip(f"message_prev1_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to previous sibling [Ctrl+Left]", parent=prev1_branch_tooltip)

            dpg.add_button(label=fa.ICON_CARET_DOWN,
                           callback=show_chat_continuation_callback,
                           enabled=show_chat_continuation_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_show_chat_continuation_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_show_chat_continuation_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_show_chat_continuation_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            show_chat_continuation_tooltip = dpg.add_tooltip(f"message_show_chat_continuation_button_{self.gui_uuid}")  # tag
            dpg.add_text("Show chat continuation (if any) [Ctrl+Down]", parent=show_chat_continuation_tooltip)

            dpg.add_button(label=fa.ICON_CARET_RIGHT,
                           callback=navigate_to_next1_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_next1_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_next1_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_next1_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            next1_branch_tooltip = dpg.add_tooltip(f"message_next1_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to next sibling [Ctrl+Right]", parent=next1_branch_tooltip)

            dpg.add_button(label=fa.ICON_FORWARD,
                           callback=navigate_to_next10_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_next10_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_next10_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_next10_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            next10_branch_tooltip = dpg.add_tooltip(f"message_next10_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch 10 siblings right [Ctrl+Shift+Right]", parent=next10_branch_tooltip)

            dpg.add_button(label=fa.ICON_FORWARD_FAST,
                           callback=navigate_to_nextend_callback,
                           enabled=next_enabled,
                           width=gui_config.toolbutton_w,
                           tag=f"message_nextend_branch_button_{self.gui_uuid}",
                           parent=g)
            dpg.bind_item_font(f"message_nextend_branch_button_{self.gui_uuid}", self.parent_view.themes_and_fonts.icon_font_solid)  # tag
            dpg.bind_item_theme(f"message_nextend_branch_button_{self.gui_uuid}", "disablable_widget_theme")  # tag
            nextend_branch_tooltip = dpg.add_tooltip(f"message_nextend_branch_button_{self.gui_uuid}")  # tag
            dpg.add_text("Switch to last sibling [Ctrl+End]", parent=nextend_branch_tooltip)

            if siblings is not None:
                dpg.add_text(f"{this_node_index + 1} / {len(siblings)}", parent=g)
        else:
            # Add the spacers separately so we get the same margins as with separate buttons
            for _ in range(6):
                dpg.add_spacer(width=gui_config.toolbutton_w, height=1, parent=g)


class DPGCompleteChatMessage(DPGChatMessage):
    def __init__(self,
                 node_id: str,
                 gui_parent: str | int,
                 parent_view: "DPGLinearizedChatView",
                 start_thinking_open: bool = False):
        """A complete chat message displayed in the linearized chat view, linked to a node ID in the datastore.

        `node_id`: The ID of the chat node, in the datastore, from which to extract the data to show.
        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).
        `start_thinking_open`: Whether to show this message's thinking trace, if it has one, rather than
                               collapsing it behind its cloud. `True` only for the reply that has just
                               finished generating, and only when the user asked for open traces — every
                               other complete message, restored or rebuilt, starts collapsed.
        """
        super().__init__(gui_parent=gui_parent,
                         parent_view=parent_view)
        self.start_thinking_open = start_thinking_open
        self.node_id = node_id  # reference to the chat node (to ORIGINAL node data, not a copy)
        # Whether a long document result is showing in full. View state, not chat data: it belongs to this
        # rendering of the node, not to the node, so it resets whenever the view is rebuilt. That is the
        # right lifetime — an expansion is a thing you did to look at something, not a preference.
        self.show_full_text = False
        # Which parts of a document search's result are open, by part index: the same kind of view state as
        # `show_full_text`, which serves the results that have no parts to open one by one.
        self.expanded_parts = set()
        self.build()

    def build(self) -> None:
        """Build (or rebuild) the GUI widgets for this chat message.

        Automatically parse the content from the chat node, and add the text to the GUI.
        """
        if self.parent_view.edit_node_id == self.node_id:
            self.parent_view.capture_edit_draft()  # before this rebuild replaces the field holding it

        node_payload = self.parent_view.chat_controller.datastore.get_payload(self.node_id)  # auto-selects active revision
        message = node_payload["message"]
        role = message["role"]
        persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
        sidecars_meta = node_payload["general_metadata"].get("sidecars", {})  # provenance per attached-file sidecar (see imagestore / textfilestore)
        super().build(role=role,
                      persona=persona,
                      node_id=self.node_id)

        # Before the stored text, because that is where the wire puts it — see `_render_system_preamble`.
        if role == "system":
            self._render_system_preamble()

        # Reasoning (thinking) trace lives in the message's `reasoning_content` sibling field, not in `content`.
        # Render it first, as a single collapsible thought paragraph. Migration (`upgrade_datastore`, at load)
        # and the live stream parser both move thinking into `reasoning_content` before it ever reaches here, so
        # `content` no longer carries inline `<think>`. The per-part splitter below still recognizes inline
        # `<think>`, but that path is dead — leftover from the pre-June-2026 inline handling, not yet removed.
        reasoning_content = message.get("reasoning_content") or ""
        if reasoning_content.strip():
            self.add_paragraph(reasoning_content, is_thought=True)

        # Open for editing, the text goes into a field in place of the paragraphs, and the rest of the message
        # — attachments, tool calls — renders as it always does.
        editing = (self.parent_view.edit_node_id == self.node_id)
        if editing:
            self._render_editor()

        # Render the content parts in order, stacked vertically. A text part renders as markdown
        # paragraphs; multiple text parts (e.g. one per websearch result) stack into the message's vertical
        # layout, giving per-result visual separation. The persona prefix on the first line of assistant content
        # ("Aria: ...") is stripped per part — a no-op for tool/system messages, which carry no persona.
        # A *document* result — a fetched page, or a document from the knowledge base — renders collapsed to
        # an opening excerpt with a toggle, so that one fetch cannot bury the conversation it was meant to
        # inform. `websearch` is excluded by construction rather than by a name check: its result is a list
        # of links, which `messagetext.document_body` does not recognize as a document. See there.
        #
        # A *document search* collapses too, by the same threshold, and for the same reason: it returns up to
        # fifty matches of up to two thousand characters, which buries the conversation as surely as a fetched
        # page. It is not a document — copying it copies what is stored — so it is decided here, for display,
        # and not by `messagetext.document_body`. Collapsed, each match shows a handle on its document — which
        # opens it, as a fetched document's does — and a snippet of what it found, as a websearch result does.
        # A result stored before the tool recorded which document each match is from shows the model's line
        # naming it instead; one stored before matches came one part apiece collapses to an excerpt.
        document_body = messagetext.document_body(self.parent_view.chat_controller.datastore, node_payload)
        maybe_function_name = (node_payload.get("generation_metadata") or {}).get("function_name")
        stored_texts = [part["text"] for part in (message.get("content") or []) if part.get("type") == "text"]
        threshold = librarian_config.tool_result_attachment_threshold
        maybe_spans = None
        if document_body is not None and len(document_body) > threshold:
            maybe_collapse, maybe_body = "document", document_body
        elif role == "tool" and maybe_function_name == "search_documents" and sum(len(text) for text in stored_texts) > threshold:
            if len(stored_texts) > 1:
                maybe_collapse, maybe_body = "per_part", None
                # Which document each match is from, recorded by the tool. Used only when it lines up with
                # the parts — a heading, then one per match — which a result stored before it was recorded,
                # or one whose parts were cut some other way, would not.
                spans = (node_payload.get("generation_metadata") or {}).get("docs_match_spans") or []
                if len(stored_texts) == len(spans) + 1:
                    maybe_spans = spans
            else:
                maybe_collapse, maybe_body = "excerpt", "".join(stored_texts)
        else:
            maybe_collapse, maybe_body = None, None
        collapsible = maybe_collapse is not None
        # The left gutter of a tool result: the buttons that act on the *whole* message, stacked beside its
        # first line. Expand/collapse goes on top, because aligning a disclosure control with the top line of
        # the content it discloses is a convention older than this app; the jump-back link sits under it.
        #
        # These live here rather than in the message's button row (`build_buttons`) on purpose. That row's
        # placement philosophy is that a given button is always at the same x, with the ones that do not
        # apply hidden — so an *extra* button on tool results alone shifts everything after it and makes the
        # row read as misaligned without it being obvious why. The jump-back link also has a natural home
        # here: it is where the view scrolls to when its counterpart ("go to result") is clicked.
        answered_call_id = message.get("tool_call_id") if role == "tool" else None
        gutter_wanted = collapsible or answered_call_id is not None
        # *All* the text parts, because a message can have several and they all belong in the column beside
        # the gutter — `websearch` emits one per result, which is what gives its results their separation.
        # Rendering only the first would silently drop the other nineteen.
        gutter_texts = [chatutil.remove_persona_from_start_of_line(persona=persona, text=part["text"])
                        for part in (message.get("content") or [])
                        if part.get("type") == "text"] if gutter_wanted else []
        body_rendered = False

        for part in message.get("content") or []:
            part_type = part.get("type")
            if part_type == "text":
                if editing:
                    pass
                elif not gutter_wanted:
                    self._render_text_paragraphs(chatutil.remove_persona_from_start_of_line(persona=persona, text=part["text"]))
                elif not body_rendered:
                    # Rendered at the position of the *first* text part, so the body still precedes any chip
                    # below it. The remaining text parts were folded in above, so later ones are skipped.
                    self._render_gutter_and_body(texts=gutter_texts,
                                                 maybe_collapse=maybe_collapse,
                                                 maybe_body=maybe_body,
                                                 maybe_spans=maybe_spans,
                                                 answered_call_id=answered_call_id)
                    body_rendered = True
            elif part_type == "image_url":
                self._render_image_part(part, sidecars_meta)
            elif part_type == "text_file":
                self._render_text_file_part(part, sidecars_meta)
            # else: unknown part type — skip (forward-compat)

        if gutter_wanted and not body_rendered:
            # No text part to hang the gutter beside — an empty tool result, which the backend can produce.
            # The jump-back link still has to exist, or the navigation pair is one-way from this message.
            self._render_gutter_and_body(texts=[], maybe_collapse=None, maybe_body=None, maybe_spans=None,
                                         answered_call_id=answered_call_id)

        if role == "system":
            self._render_system_postamble()

        # A document the AI fetched from the local knowledge base gets the same handles as an attached one.
        # It is *not* an attachment — the file is already the user's, sitting in the documents folder, and
        # copying it into the sidecar store would archive a second copy of something that cannot go away.
        # So the affordance matches while the backing store does not: the reader gets a named handle on the
        # document and a way to open it, pointing at the original rather than at a copy.
        #
        # Scoped to `fetch_document` rather than to anything naming documents: a *search* result puts a handle
        # on each of its matches instead, beside the match (see `_render_gutter_and_body`). A reply's own
        # citations have none yet — see the deferred item on exposing the source files behind them.
        generation_metadata = node_payload.get("generation_metadata") or {}
        if generation_metadata.get("function_name") == "fetch_document":
            for document_id in generation_metadata.get("document_ids") or []:
                self._render_document_reference(document_id)

        # A fetch the client-side allowlist refused carries the host it refused, and offers to approve it
        # here, beside the result that names it.
        if role == "tool" and (maybe_denied_host := generation_metadata.get("webfetch_denied_host")) is not None:
            self._render_denied_host_override(maybe_denied_host)

        # Render any tool-call invocations this assistant message made, as visible sub-elements after the text.
        # Without this, a tool-calling turn — often with empty `content` — would show nothing
        # between the assistant message and the subsequent tool-result node.
        for index, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            self.add_tool_call_invocation(index=index,
                                          name=function.get("name", "?"),
                                          arguments=function.get("arguments", ""),
                                          tool_call_id=tool_call.get("id"))

    def _render_denied_host_override(self, host: str) -> None:
        """Render a row offering to approve `host` for this session and fetch from it again.

        For a webfetch result the allowlist refused. The retry runs on a new branch, from this tool result;
        see `scaffold.retry_tool_calls`.
        """
        node_id = self.node_id

        def approve_and_retry() -> None:
            chat_controller = self.parent_view.chat_controller
            llmclient.approve_host_for_session(host)
            # Rewind the GUI to the branch point: this tool result and every message after it.
            # `retry_tool_calls` re-adds the new branch via the ai_turn callbacks.
            if not chat_controller.view.rewind_to(node_id):  # not found: a rebuild got there first
                return
            # Re-run the refused fetch on a new branch and continue. HEAD is updated by the callbacks.
            chat_controller.ai_turn(docs_query=None,
                                    continue_=False,
                                    _retry_tool_node_id=node_id)

        with self.paragraphs_lock:
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
            # A plain button, not `_add_action_button`, whose flash would land on a widget the rewind has
            # just destroyed.
            button = dpg.add_button(label=fa.ICON_UNLOCK,
                                    callback=approve_and_retry,
                                    width=gui_config.toolbutton_w,
                                    tag=f"message_approve_retry_button_{self.gui_uuid}",  # tag
                                    parent=row)
            dpg.bind_item_font(button, self.parent_view.themes_and_fonts.icon_font_solid)
            dpg.bind_item_theme(button, "disablable_widget_theme")  # tag
            self._add_tooltip(button, f"Approve '{host}' for this session, and fetch again\n(on a new branch)")
            self._add_clickable_text(f"Approve {host} for this session, and fetch again", parent=row,
                                     action=approve_and_retry)

    def _render_editor(self) -> None:
        """Render this message's text as an editable field, with Save and Cancel below it.

        The field holds the draft the view has kept, if any, and the stored text otherwise.
        """
        view = self.parent_view
        text = view.edit_draft
        if text is None:
            unused_role, unused_persona, text = chatutil.get_node_message_text_without_persona(view.chat_controller.datastore, self.node_id)
        n_lines = min(max(text.count("\n") + 1, _EDITOR_MIN_LINES), _EDITOR_MAX_LINES)
        with self.paragraphs_lock:
            view.gui_edit_field = dpg.add_input_text(multiline=True,
                                                     default_value=text,
                                                     # The composer's chords, so the fingers need not change:
                                                     # `True` <=> Enter saves (see where "chat_field" is built).
                                                     ctrl_enter_for_new_line=(librarian_config.send_message_key == "enter"),
                                                     width=self.get_chat_text_width(),
                                                     height=n_lines * gui_config.font_size + _EDITOR_EXTRA_H,
                                                     tag=f"chat_edit_field_{self.gui_uuid}",
                                                     parent=self.gui_text_group)
            save_key = "Ctrl+Enter" if librarian_config.send_message_key == "ctrl+enter" else "Enter"
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
            view.gui_edit_save_button = dpg.add_button(label="Save",
                                                       callback=lambda: view.finish_editing(save=True),
                                                       tag=f"chat_edit_save_button_{self.gui_uuid}",
                                                       parent=row)
            view.gui_edit_save_tooltip = self._add_tooltip(view.gui_edit_save_button,
                                                           f"Save the text as a new revision of this message [{save_key}]")
            cancel_button = dpg.add_button(label="Cancel",
                                           callback=lambda: view.finish_editing(save=False),
                                           tag=f"chat_edit_cancel_button_{self.gui_uuid}",
                                           parent=row)
            dpg.add_text("Close the editor, keeping the message as it was [Esc]", parent=dpg.add_tooltip(cancel_button))

    def _render_gutter_and_body(self, *,
                                texts: list[str],
                                maybe_collapse: str | None,
                                maybe_body: str | None,
                                maybe_spans: list[dict] | None,
                                answered_call_id: str | None) -> None:
        """Render a tool result's text with its whole-message buttons stacked in a gutter to the left.

        `texts`: the message's own text parts, in order. Several is normal — `websearch` emits one per
                 result and `search_documents` one per match, and each renders as its own paragraph, which
                 is what visually separates them.
        `maybe_collapse`: how the result is shown until the reader expands it, with a toggle to do so:
                          `"document"` — an excerpt of `maybe_body`, the document the result reports;
                          `"excerpt"` — an excerpt of `maybe_body`, a long result that is not a document;
                          `"per_part"` — each of `texts` shortened to a snippet;
                          `None` — `texts` in full, and no toggle.
        `maybe_body`: the text to excerpt, for the two excerpt modes; `None` otherwise.
        `maybe_spans`: for `"per_part"`, which document each part after the first is from, as the tool
                       recorded it (`docs_match_spans`). Each such part then gets a handle on its document, and
                       its snippet leaves out the line naming it. `None` names them in the text instead.
        `answered_call_id`: the tool call this result answers, if any — adds the jump-back link.

        The expand/collapse toggle names the size it would expand to, because that is what decides between
        the two ways to read a long document. In-place is convenient and keeps you in the conversation, but a
        large one pushes the surrounding turns off the screen; opening the file gives you a separate window
        where the document and the conversation are visible at once. Fifty thousand characters and five
        thousand want different answers, and only the reader can pick — so the number goes where the choice
        is made.
        """
        # Per match, the message's own toggle commands the matches' toggles rather than keeping a state of its
        # own: it opens them all while any is closed, and closes them all once every one is open.
        all_parts = set(range(len(texts)))
        if maybe_collapse == "per_part":
            expanded = all_parts <= self.expanded_parts
        else:
            expanded = self.show_full_text
        body = maybe_body if maybe_body is not None else "".join(texts)

        def toggle() -> None:
            def open_or_close_all() -> None:
                if maybe_collapse == "per_part":
                    self.expanded_parts = set() if expanded else set(all_parts)
                else:
                    self.show_full_text = not self.show_full_text
            self._change_and_rebuild(open_or_close_all)

        def make_part_toggle(index: int) -> Callable[[], None]:
            def toggle_part() -> None:
                self._change_and_rebuild(lambda: self.expanded_parts.symmetric_difference_update({index}))
            return toggle_part

        def add_part_toggle(index: int, row: int | str) -> None:
            """Put this match's own chevron first on its handle row: the disclosure control at the start of the
            line it discloses, as the whole message's sits at the start of the message."""
            in_full = self._part_shown_in_full(index, maybe_collapse)
            button_id = dpg.add_button(label=fa.ICON_CHEVRON_UP if in_full else fa.ICON_CHEVRON_DOWN,
                                       width=gui_config.toolbutton_w, parent=row, callback=make_part_toggle(index))
            dpg.bind_item_font(button_id, self.parent_view.themes_and_fonts.icon_font_solid)
            dpg.add_text("Show less of this match" if in_full else "Show this match in full",
                         parent=dpg.add_tooltip(button_id))

        with self.paragraphs_lock:
            # Gutter to the *left* of the text, the same shape the thinking-trace toggle uses. The toggle
            # has to be somewhere that does not move when the text does: below the body, expanding a long
            # document pushes the collapse button off the bottom of the screen, so the gesture that undoes
            # the expansion is the one thing the expansion hides. Here it stays under the cursor, and a
            # second click puts the message back.
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
            gutter = dpg.add_group(parent=row)

            if maybe_collapse is not None:
                # Deliberately *not* an `_add_action_button`: that one flashes the button green or red once
                # the action returns, and this action deletes the button it is flashing. It is also not the
                # kind of action that wants an acknowledgment — the message visibly changing is the feedback.
                button_id = dpg.add_button(label=fa.ICON_CHEVRON_UP if expanded else fa.ICON_CHEVRON_DOWN,
                                           width=gui_config.toolbutton_w, parent=gutter, callback=toggle)
                dpg.bind_item_font(button_id, self.parent_view.themes_and_fonts.icon_font_solid)
                expand_tooltip = dpg.add_tooltip(button_id)
                if expanded:
                    back_to = "a snippet per match" if maybe_collapse == "per_part" else "the opening"
                    dpg.add_text(f"Show less\n(collapse back to {back_to})", parent=expand_tooltip)
                elif maybe_collapse == "document":
                    dpg.add_text(f"Show all {len(body):,} characters here\n"
                                 "(a large document will fill the view — the button below opens it\n"
                                 "in a separate window instead, so you keep the conversation in sight)",
                                 parent=expand_tooltip)
                elif maybe_collapse == "per_part":
                    # No count: the result's heading says how many, and a caption promising all of them
                    # has no need to.
                    dpg.add_text(f"Show all matches in full, {len(body):,} characters", parent=expand_tooltip)
                else:
                    dpg.add_text(f"Show all {len(body):,} characters here", parent=expand_tooltip)

            if answered_call_id is not None:
                self._add_action_button(parent=gutter,
                                        icon=fa.ICON_ARROW_UP,
                                        tooltip_text="Go to the call this result answers",
                                        ok_message="Jumped to the call!",
                                        fail_message="The originating call isn't in this branch",
                                        action=self._make_jump_to_tool_call(answered_call_id))

            # Render the body into a column beside the gutter. `add_paragraph` parents to `gui_text_group`,
            # so retarget it for the duration rather than bypassing it — going straight to the renderer
            # would leave the text out of `self.paragraphs`, and that is what the message's `text` reads.
            body_column = dpg.add_group(parent=row)
            outer_group, self.gui_text_group = self.gui_text_group, body_column
            outer_indent, self.text_indent_w = self.text_indent_w, self.text_indent_w + gui_config.toolbutton_w
            try:
                if maybe_collapse in ("document", "excerpt") and not expanded:
                    self._render_text_paragraphs(chatutil.excerpt(body, librarian_config.tool_result_preview_characters))
                elif maybe_collapse == "document":
                    self._render_text_paragraphs(body)
                else:
                    labels = {}  # one render's memo: a search often names one document several times
                    for index, one_text in enumerate(texts):  # one paragraph run per part, preserving the per-result separation
                        in_full = self._part_shown_in_full(index, maybe_collapse)
                        maybe_span = maybe_spans[index - 1] if (maybe_spans and index >= 1) else None
                        if maybe_span is None:
                            self._render_text_paragraphs(one_text if in_full else messagetext.collapse_docs_match(one_text))
                            continue
                        leading = ((lambda row, index=index: add_part_toggle(index, row))
                                   if maybe_collapse == "per_part" else None)
                        self._render_document_reference(maybe_span["document_id"], labels, leading=leading)
                        shown = one_text if in_full else messagetext.docs_match_snippet(one_text)
                        if shown:
                            self._render_text_paragraphs(shown)
            finally:
                self.gui_text_group = outer_group
                self.text_indent_w = outer_indent

    def _part_shown_in_full(self, index: int, maybe_collapse: str | None) -> bool:
        """Whether text part `index` of this tool result is shown whole, rather than shortened to a snippet.

        Whole when the reader opened it, by its own toggle or by the message's, which opens them all.
        """
        return maybe_collapse != "per_part" or index in self.expanded_parts

    def _change_and_rebuild(self, change: Callable[[], None]) -> None:
        """Apply `change` to this message's view state, and redraw the message without moving the conversation.

        Sampled *before* the rebuild: expanding grows the container and leaves the offset alone, but
        collapsing shrinks it, and DPG clamps the scroll to the smaller maximum at the next layout. Without
        putting it back, a collapse scrolls the conversation under the reader — the message they just
        collapsed jumps down the screen, which reads as a glitch rather than as an action.
        """
        y_scroll = dpg.get_y_scroll(self.parent_view.gui_parent)
        change()
        # Rebuild just this message rather than the whole view, and build the replacement before tearing
        # the original down — see `rebuild_in_place` for why the obvious order flickers. The button
        # running this callback is one of the widgets that goes away; that is the same thing the branch
        # and delete buttons already do through `parent_view.build()`, one level wider.
        self.rebuild_in_place()
        self.parent_view.hold_scroll_across_rebuild(y_scroll)

    def _render_injected_texts(self, texts: list[str]) -> None:
        """Render one block of per-turn texts, under the label saying they are added rather than stored.

        Both blocks a system message shows — the preamble ahead of the stored prompt, and the postamble
        after it — carry the same words, because they are the same category of thing. Saying it twice is
        what keeps each block legible where it sits, with stored prose between them.
        """
        self.add_paragraph("*Added to every request, not stored:*", is_thought=False)
        for text in texts:
            self.add_paragraph(text, is_thought=False)

    def _render_system_preamble(self) -> None:
        """Draw the per-turn texts that precede the stored prompt, ahead of a rendered system message.

        The mirror of `_render_system_postamble`, which see for why these are shown live rather than
        stored, and for what the display leaves out. Currently one text: the notice saying that the block
        below it is the model's setup rather than something the user said.

        It is drawn first because that is where the wire carries it (`scaffold.build_system_preamble`) —
        and a notice announcing what follows, printed after the thing it announces, would be describing
        the conversation instead.
        """
        llm_settings = self.parent_view.chat_controller.llm_settings
        if llm_settings is None:  # no backend connected yet; there is no settings object to ask
            return
        preamble = scaffold.build_system_preamble(llm_settings=llm_settings)
        if not preamble:
            return
        self.rendered_system_preamble = list(preamble)
        self._render_injected_texts(preamble)

    def _render_system_postamble(self) -> None:
        """Append the per-turn facts to a rendered system message, so the log shows what is sent.

        The chat log's promise is that it shows what was said, and these are said on every turn while
        appearing nowhere in it: the date, and the standing reminder about how to write.
        `scaffold.build_turn_prompt` folds them into the leading system message at send time and never
        stores them, so the node holds the standing prompt while the model reads that prompt *plus this*.

        Shown live rather than recorded, which matches what this node already is: `appstate` overwrites the
        stored system prompt at every app start instead of keeping a revision per session, so it has never
        been a record of a past turn. What is shown is therefore what the *next* turn will send. On a
        session running past midnight the date here catches up at the next view rebuild, while an earlier
        turn in the same log really did send yesterday's - the node cannot express that, and does not try.

        Two further injects are conditional on turn state - whether anything grounded the answer, whether
        the tool budget ran out - and are left out. Neither is knowable before the turn runs, and a line
        that came and went between rebuilds would read as instability rather than as information.

        The synthetic tool exchanges are deliberately not shown either, each for its own reason: the
        clock's call is staged for the model's benefit and would only raise the question of who made a call
        the user never saw, and retrieval runs at `k=50`, so its results would bury the conversation they
        were fetched to support.
        """
        llm_settings = self.parent_view.chat_controller.llm_settings
        if llm_settings is None:  # no backend connected yet; there is no settings object to ask
            return
        # `grounding_material_exists=False` selects exactly the unconditional ones; see above.
        postamble = scaffold.build_system_postamble(llm_settings=llm_settings,
                                                    grounding_material_exists=False)
        if not postamble:
            return
        # What was drawn, so `DPGChatController.refresh_system_injects_if_stale` can tell whether it still
        # matches what a request would carry. Comparing the texts rather than just the date also catches an
        # experiment that swapped a formatter mid-session.
        self.rendered_system_postamble = list(postamble)
        self._render_injected_texts(postamble)

    def _render_image_part(self, part: dict[str, Any], sidecars_meta: dict[str, Any]) -> None:
        """Render one `image_url` content-part: an inline thumbnail plus a per-image provenance cluster.

        In a stored message the URL is always a Raven-internal `sidecar:<filename>` reference (see
        `chatutil.image_content_part`); the thumbnail texture is resolved and cached by the controller. A
        non-sidecar URL (shouldn't occur in stored data) is skipped for forward-compat; an unresolvable sidecar
        renders a small placeholder rather than nothing, so the message still reads as "an image was here".

        Provenance for this image lives in `sidecars_meta[filename]` (see `imagestore.store_image_as_sidecar`).
        The thumbnail carries the original filename as a tooltip, and a small action row below it offers, per
        image (a message may hold several): show the stored original at full size, open the recorded source (a
        `file://` original or an `https://` page — disabled when there is nothing openable), and reveal the
        chat's image-sidecar directory."""
        url = (part.get("image_url") or {}).get("url", "")
        if not url.startswith(sidecarstore.SIDECAR_SCHEME):
            return  # only local sidecar refs are resolvable here; skip anything else (forward-compat)
        filename = url[len(sidecarstore.SIDECAR_SCHEME):]
        meta = sidecars_meta.get(filename) or {}
        texture = self.parent_view.chat_controller.attachment_textures.inline_image(filename)
        datastore = self.parent_view.chat_controller.datastore
        with self.paragraphs_lock:
            if texture is None:
                dpg.add_text("[image unavailable]", color=(180, 120, 120), parent=self.gui_text_group)
                return

            cluster = dpg.add_group(parent=self.gui_text_group)  # thumbnail + its provenance action row, stacked
            image_id = dpg.add_image(texture.texture_tag,  # tag
                                     width=texture.w,
                                     height=texture.h,
                                     parent=cluster)
            archival_filename = meta.get("original_sidecar") or filename
            open_saved_copy = lambda: common_utils.open_file(datastore.sidecar_path(archival_filename))  # noqa: E731 -- shared by the click shortcut and the button below
            # original filename, and that the thumbnail itself opens it
            dpg.add_text(f"{sidecarstore.provenance_filename_from_url(meta.get('url')) or 'attached image'}"
                         "\n(click to open)",
                         parent=dpg.add_tooltip(image_id))
            self._make_clickable([image_id], action=open_saved_copy)

            # Per-image provenance actions. "Show original" resolves to the archival copy — the verbatim original
            # kept as a second sidecar (case 2 of the image store), or the primary itself when that is the
            # verbatim original (case 1); a downsample-only image (case 3) has no archival original, so the
            # primary is the best copy stored. "Open source" targets the recorded provenance URL, which is
            # fragile (the file may have moved, the page may 404) and absent for some images — disabled up front
            # when there is nothing openable. "Open folder" reveals the datastore's image-sidecar directory.
            source_url = meta.get("url") or ""
            source_openable = bool(source_url) and not source_url.startswith("data:")
            actions = dpg.add_group(horizontal=True, parent=cluster)

            self._add_action_button(parent=actions,
                                    icon=fa.ICON_IMAGE,
                                    tooltip_text="Show full-size image\n(the saved copy, in the chat data folder)",
                                    ok_message="Opened image",
                                    action=open_saved_copy)
            if source_openable:
                source_tooltip = f"Open original source\n{urllib.parse.unquote(source_url)}"
            elif source_url.startswith("data:"):
                source_tooltip = "Open original source — unavailable\n(the image was embedded inline; no external source)"
            else:
                source_tooltip = "Open original source — unavailable\n(no source location was recorded)"
            self._add_action_button(parent=actions,
                                    icon=fa.ICON_LINK,
                                    tooltip_text=source_tooltip,
                                    ok_message="Opened source",
                                    enabled=source_openable,
                                    action=lambda: _open_source_url(source_url))
            self._add_action_button(parent=actions,
                                    icon=fa.ICON_FOLDER_OPEN,
                                    tooltip_text="Open the attachments folder\n(where attached files are stored)",
                                    ok_message="Opened folder",
                                    action=lambda: common_utils.open_in_file_manager(datastore.sidecar_dir))

    def _render_text_file_part(self, part: dict[str, Any], sidecars_meta: dict[str, Any]) -> None:
        """Render one `text_file` content-part: an inline file chip plus a per-document provenance cluster.

        The file counterpart of `_render_image_part`. A document has no thumbnail, so it renders as a chip — a
        document glyph and the original filename — followed by the same action row images get: show the stored
        copy (opens it in the OS default app for its type), open the recorded source (a `file://` original or an
        `https://` page — disabled when nothing openable), and reveal the datastore's sidecar directory. The
        document's text is *not* shown inline (it went to the model at wire-build, folded into the message text);
        this is the visible handle for it. Provenance lives in `sidecars_meta[filename]` (see
        `textfilestore.store_file_as_sidecar`). A non-sidecar URL (shouldn't occur in stored data) is skipped."""
        url = (part.get("text_file") or {}).get("url", "")
        if not url.startswith(sidecarstore.SIDECAR_SCHEME):
            return  # only local sidecar refs are resolvable here; skip anything else (forward-compat)
        filename = url[len(sidecarstore.SIDECAR_SCHEME):]
        meta = sidecars_meta.get(filename) or {}
        name = (part.get("text_file") or {}).get("name") or meta.get("name") or "attached file"
        datastore = self.parent_view.chat_controller.datastore
        open_saved_copy = lambda: common_utils.open_file(datastore.sidecar_path(filename))  # noqa: E731 -- shared by the click shortcut and the button below
        source_url = meta.get("url") or ""
        source_openable = bool(source_url) and not source_url.startswith("data:")
        with self.paragraphs_lock:
            # One row: the actions, then the name they act on. The buttons come first because they are the
            # fixed part — three glyphs in the same place on every attachment — while the name is arbitrary
            # length, so leading with it would leave the buttons at a different x on every chip. The name
            # carries no glyph of its own: the button beside it already shows the document icon, and repeating
            # it a few pixels away reads as two separate things rather than one. That button is the last one,
            # next to the name, because clicking the name does the same thing.
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)

            # "Open source" targets the recorded provenance URL, disabled when nothing is openable. "Open folder"
            # reveals the sidecar dir. "Show document" opens the stored sidecar (verbatim — documents are never
            # transformed, so the sidecar IS the original) in the OS default app.
            if source_openable:
                source_tooltip = f"Open original source\n{urllib.parse.unquote(source_url)}"
            else:
                source_tooltip = "Open original source — unavailable\n(no source location was recorded)"
            self._add_action_button(parent=row,
                                    icon=fa.ICON_LINK,
                                    tooltip_text=source_tooltip,
                                    ok_message="Opened source",
                                    enabled=source_openable,
                                    action=lambda: _open_source_url(source_url))
            self._add_action_button(parent=row,
                                    icon=fa.ICON_FOLDER_OPEN,
                                    tooltip_text="Open the attachments folder\n(where attached files are stored)",
                                    ok_message="Opened folder",
                                    action=lambda: common_utils.open_in_file_manager(datastore.sidecar_dir))
            self._add_action_button(parent=row,
                                    icon=fa.ICON_FILE_LINES,
                                    tooltip_text="Show the attached document\n(the saved copy, in the chat data folder)",
                                    ok_message="Opened document",
                                    action=open_saved_copy)

            name_id = self._add_clickable_text(name, parent=row, action=open_saved_copy)
            # The tooltip names where the document came from and when, which is what tells two same-titled
            # fetches apart.
            document_tooltip = dpg.add_tooltip(name_id)
            dpg.add_text("Click to open the attached document", parent=document_tooltip)
            if source_url:
                dpg.add_text(urllib.parse.unquote(source_url), color=(180, 180, 180), parent=document_tooltip)
            if meta.get("fetched_at"):
                dpg.add_text(f"saved {meta['fetched_at']}", color=(180, 180, 180), parent=document_tooltip)

    def _render_document_reference(self, document_id: str, labels: dict[str, str] | None = None,
                                   leading: Callable[[int | str], None] | None = None) -> None:
        """Render a handle on one knowledge-base document the AI fetched or found: a chip plus its two actions.

        `labels`: A memo of document ID to name, for a caller rendering many handles at once, which a document
                  search's result does — often naming one document several times. Made and dropped by that
                  caller within one render, so an edited document shows its new name on the next.
        `leading`: Called with the handle's row before anything is added to it, for a caller that puts a
                   control of its own at the start of the row — a search match's own expand toggle.

        The docs-DB counterpart of `_render_text_file_part`, and deliberately the same shape — a document
        glyph, a name, and a small action row — because to the reader these are the same kind of thing. What
        differs is where they point. An attachment has a saved copy and a recorded source; an indexed document
        *is* its source, so "open the saved copy" and "open the original" collapse into one action, and the
        folder to reveal is the documents folder rather than the sidecar directory.

        The name is `chatutil.document_label` (the document's own title, per its content), falling back to the
        ID, which is the handle `fetch_document` takes and so is worth showing when nothing better exists.

        A document that is no longer in the index renders with its ID and a disabled open button: the
        conversation did read it, and saying so with a dead handle is more honest than showing nothing.
        """
        retriever = self.parent_view.chat_controller.retriever
        path = llmclient.document_path(retriever, document_id)
        if labels is not None and document_id in labels:
            name = labels[document_id]
        else:
            text = llmclient.document_text(retriever, document_id)
            name = (chatutil.document_label(text) if text else "") or document_id
            if labels is not None:
                labels[document_id] = name
        with self.paragraphs_lock:
            # One row — actions, then the name they act on — matching `_render_text_file_part`. A
            # knowledge-base document gets the book glyph rather than the attachment's document glyph, since
            # the two point at different places (the user's documents folder, not the sidecar store).
            row = dpg.add_group(horizontal=True, parent=self.gui_text_group)
            if leading is not None:
                leading(row)
            if path is not None:
                open_document = lambda: common_utils.open_file(path)  # noqa: E731 -- shared by the click shortcut and the button below
                self._add_action_button(parent=row,
                                        icon=fa.ICON_FOLDER_OPEN,
                                        tooltip_text="Open the documents folder\n(the knowledge base the AI searches)",
                                        ok_message="Opened folder",
                                        action=lambda: common_utils.open_in_file_manager(librarian_config.llm_docs_dir))
                self._add_action_button(parent=row,
                                        icon=fa.ICON_BOOK_OPEN,
                                        tooltip_text=f"Open the document\n{path}",
                                        ok_message="Opened document",
                                        action=open_document)
                name_id = self._add_clickable_text(name, parent=row, action=open_document)
                path_tooltip = dpg.add_tooltip(name_id)
                dpg.add_text("Click to open the document", parent=path_tooltip)
                dpg.add_text(str(path), color=(180, 180, 180), parent=path_tooltip)
            else:
                self._add_action_button(parent=row,
                                        icon=fa.ICON_BOOK_OPEN,
                                        tooltip_text="Open the document — unavailable\n(no longer in the document database)",
                                        ok_message="Opened document",
                                        enabled=False,
                                        action=lambda: None)
                name_id = dpg.add_text(name, parent=row)
                dpg.add_text(f"Document '{document_id}'", parent=dpg.add_tooltip(name_id))


class DPGStreamingChatMessage(DPGChatMessage):
    renders_live_reply = True

    def __init__(self,
                 gui_parent: str | int,
                 parent_view: "DPGLinearizedChatView",
                 node_id: str):
        """A chat message being streamed live from the LLM, displayed in the linearized chat view.

        `gui_parent`: DPG tag or ID of the GUI widget (typically child window or group) to add the chat message to.
        `parent_view`: The linearized chat view widget this chat message is rendered in (and is owned by).
        `node_id`: The chat node this message is being written into. It exists already, carrying whatever
                   has streamed so far — the turn creates it before the reply starts (`scaffold.ai_turn`).

                   Holding the id rather than a floating identity is what lets a rebuilt view find this
                   message again: it is the rendering of a node, like every other message in the view, and
                   the node is where its text lives.

        Starts from whatever the node says, which is empty at the beginning of a reply and not empty for a
        view rebuilt mid-reply. Use `add_paragraph` and `replace_last_paragraph` to extend it as more text
        arrives.

        To replace the streaming message with a completed message, call the streaming message's
        `demolish` method first. Doing so removes its widgets from the GUI.
        """
        super().__init__(gui_parent=gui_parent,
                         parent_view=parent_view)
        # Between the base constructor and `build`: the base declares `node_id` as `None`, and `build`
        # reads it to find the message this renders. Super init fires first here as everywhere in Raven,
        # so a subclass assigning before that call has it silently undone.
        self.node_id = node_id
        # A reply being generated is exactly the case the preference speaks to.
        self.start_thinking_open = parent_view.chat_controller.app_state.get("show_thinking", False)
        # What the cloud is currently saying, or `None` while nothing has been said yet. See `set_thinking`.
        self._thinking_shown = None
        # The counter line as last written, so a per-frame redraw costs a string comparison. See
        # `set_thinking_progress`.
        self._thinking_readout_shown = None
        self.build()

    def build(self):
        persona = self.parent_view.chat_controller.llm_settings.personas.get("assistant", None)
        super().build(role="assistant",  # TODO: parameterize this?
                      persona=persona,
                      node_id=self.node_id)

        # Seed from what the node already holds. Empty at the start of a reply, which is the common case and
        # renders as nothing; not empty when a view is rebuilt mid-reply, which is the case this exists for —
        # the words that have arrived are in the node, so they come back with it.
        message = self.parent_view.chat_controller.datastore.get_payload(self.node_id)["message"]
        self._seed_streamed_paragraphs(message.get("reasoning_content") or "", is_thought=True)
        self._seed_streamed_paragraphs(
            chatutil.remove_persona_from_start_of_line(persona=persona,
                                                       text=chatutil.content_to_text(message.get("content", []))),
            is_thought=False)

    def _seed_streamed_paragraphs(self, text: str, is_thought: bool) -> None:
        """Lay `text` out in paragraphs exactly as streaming it would have, so rendering can carry on from here.

        Not `_render_text_paragraphs`, which is for a *finished* message and drops empty paragraphs. This
        one keeps them, and the difference is not cosmetic: the live renderer works on the last paragraph,
        replacing it as the current one grows and appending a fresh empty one at each newline. So the last
        paragraph seeded here has to be the one still being written — including when that is empty, which
        is what a reply whose text ends at a newline looks like.

        Seed it wrong and the next chunk to arrive replaces a *finished* paragraph with the partial one,
        which reads as the reply eating what it already said.
        """
        if not text:
            return
        for piece in text.split("\n"):
            self.add_paragraph(piece, is_thought=is_thought)

    def set_thinking_progress(self, dt: float, n_chunks: int) -> None:
        """Show how long the model has been reasoning, and roughly how much of it there is so far.

        `dt`: seconds since the first reasoning arrived.
        `n_chunks`: text-bearing deltas so far. During the thinking phase every one of them is reasoning,
                    and a streaming backend emits one per token — so it is the same estimate the stored
                    figure falls back to, and it is marked with a `~` for the same reason.

        With the trace collapsed there is otherwise nothing on screen but a pulsating cloud, which says
        *something is happening* and not *how long you have been waiting*.

        Called once per frame while the model reasons, so it writes only when the text it would write has
        changed — which at a tenth of a second's resolution is about ten times a second, whatever the frame
        rate happens to be.
        """
        if self.gui_thought_stats is None:
            return
        text = f"Thinking… {dt:0.1f}s, ~{n_chunks}t"
        if text == self._thinking_readout_shown:
            return
        self._thinking_readout_shown = text
        with guiutils.nonexistent_ok():
            dpg.set_value(self.gui_thought_stats, text)

    def set_thinking(self, is_thinking: bool) -> None:
        """Say whether the model is reasoning right now, by pulsating this message's cloud or settling it.

        Does nothing until a thinking paragraph has arrived, since until then there is no cloud to pulsate.

        With the trace collapsed, this is the only thing on screen saying the model is working — so it is
        not decoration: an app that showed nothing would look frozen for exactly as long as the reasoning
        takes, which on a thinking model is most of the turn. Pulsating carries that meaning already,
        from the INDEXING / DOCUMENTS / READING / SYSTEM / INTERNET indicators.
        """
        if self.gui_thought_button is None:  # nothing has been thought yet, so there is no cloud to mark
            return
        # Acts on the transition only. The caller says this per streamed event rather than per change —
        # it has to, since the bubble does not exist until the first thinking paragraph is rendered, which
        # is already past the transition that would have started the pulsation. Re-binding the theme every
        # event would be harmless; re-resetting the animation every event would not, since a cycle restarted
        # every few milliseconds never leaves its first frame, and a pulsation stuck at full alpha is
        # indistinguishable from a static color.
        if is_thinking == self._thinking_shown:
            return
        self._thinking_shown = is_thinking
        with guiutils.nonexistent_ok() as nok:
            if is_thinking:
                # Start every stint at full alpha, the way an appearing indicator does, rather than wherever
                # in the cycle a continuously-running animation happens to be.
                think_glow = self.parent_view.chat_controller.think_glow_animation
                if think_glow is not None:
                    think_glow.reset()
                dpg.bind_item_theme(self.gui_thought_button, "my_pulsating_think_theme")  # tag
            else:
                dpg.bind_item_theme(self.gui_thought_button, "my_steady_think_theme")  # tag
        if nok.errored:
            logger.info("DPGStreamingChatMessage.set_thinking: GUI widget does not exist, ignoring.")
