"""Chat controller.

This module renders a linearized chat view of the current branch, and contains the scaffold to GUI integration
that controls chatting with the AI.
"""

# TODO: check if we need to shuffle the abstraction levels around - e.g. if there are many references to `self.parent_view.chat_controller.something`, does `something` really belong to the controller level?

__all__ = ["readout_is_exact",
           "TailFollowSample",
           "DPGLinearizedChatView",
           "DPGChatController"]

import logging
logger = logging.getLogger(__name__)

import concurrent.futures
import dataclasses
import io
import threading
import time
from typing import Any, Callable, TYPE_CHECKING
import uuid

import dearpygui.dearpygui as dpg

from unpythonic import box, sym, unbox
from unpythonic.env import env

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders

# `raven.client.api` imports torch and spaCy at module scope, and `avatar_controller` reaches it, so a
# module-level import of either drags the whole ML stack in — and with it, the reason this module's tests
# skip themselves in the minimal-dependency CI job. Both are deferred instead: the avatar controller is only
# a type here, and the API is reached through the seam below. Same arrangement as `llmclient`.
if TYPE_CHECKING:
    from ..client.avatar_controller import DPGAvatarController


def _client_api():
    """Return `raven.client.api`, imported on first use.

    Not initialized here, unlike `llmclient`'s namesake: the only caller of this module is
    `raven.librarian.app`, which initializes the API at startup with its own executor, long before any
    avatar speech can start. Initializing again would be harmless but would log on every call.
    """
    from ..client import api  # noqa: PLC0415 -- deferred on purpose; see the note above
    return api

from ..common import bgtask
from ..common import netutil
from ..common import numutils
from ..common import text as common_text

from ..common.gui import animation as gui_animation
from ..common.gui import keyboardmark
from ..common.gui import layout_math
from ..common.gui import utils as guiutils
from ..common.gui import widgetfinder

from . import chatlog_search
from . import chatmessage
from . import chattextures
from . import chattree
from . import chatutil
from . import config as librarian_config
from . import hybridir
from . import llmclient
from . import messagetext
from . import scaffold
from . import textfilestore

gui_config = librarian_config.gui_config  # shorthand, this is used a lot

# Slack, in pixels, for the two comparisons in `DPGLinearizedChatView.should_follow_tail`: how close to the end
# still counts as being at the end, and how far the scroll position may sit from where we put it before we
# conclude that the user moved it.
#
# Not a config knob yet, because the right value is still being measured; see the diagnostics in that method.
# It is squeezed from both sides. Too small, and the view stops following immediately after the user sends a
# message: `dpg.set_y_scroll` is applied by the render loop, so a position sampled before the next frame can
# still report the pre-scroll value, leaving a gap the size of whatever was just added. Too large, and
# scrolling up a line or two from the end still counts as being at the end, so the arrow keys look broken.
#
# It is worth knowing that this is *not* two lines of text, though it reads as if it were: `font_size` is the
# glyph size, while a rendered line also carries the item spacing, and the chat panel measures 26 px per line
# against a font size of 20. So the value allows about one and a half lines. Deliberately left as it is:
# widening it to a true two lines would have covered both refusals recorded in
# `investigations/follow-tail-drift/`, but those had a cause, which is fixed where it happens instead — a
# bound that hides a defect is worth less than the defect being gone, and this one is squeezed from the other
# side by the arrow keys. If the cause recurs, this is the knob, and a real line height has to be *measured*
# rather than derived: the ratio to the font size is set by the theme's spacing and is not a constant.
_PIN_TOLERANCE_PX = 2 * gui_config.font_size  # about one and a half lines; see below

# A refusal to follow, within this many tolerances of the end, is reported at INFO as a near miss: that is the
# shape a wrong refusal takes, and the logged numbers say which comparison let it through.
_PIN_NEAR_MISS_FACTOR = 20

# Labels for the jump-to-latest pill. Each carries the state as well as the action, so that it informs
# during the turn rather than only announcing its end: a reader who has scrolled away wants to know whether
# there is any point waiting. The arrow says which way the button will take them.
_JUMP_TO_LATEST_WRITING_LABEL = "AI writing ↓"
_JUMP_TO_LATEST_FINISHED_LABEL = "AI finished ↓"

# The pill is the one widget in Librarian not drawn in the app font, and the arrow above is why. The UI font
# is OpenSans (`guiutils.bootup`'s default, chosen for scientific text — see the note there), whose cmap has
# no arrow or triangle glyphs at all: U+2193, U+25BC and U+25BE are all absent, so any of them renders as a
# blank box. InterTight, shipped alongside it, has them.
#
# Binding a second face to one small control is the cheaper of the two compromises available. The others
# were: spell the direction in words, which makes a pill into a sentence; or put the arrow in a separate
# icon-font widget beside the button, which splits one affordance into a clickable half and a decorative
# half. A DPG item draws its whole label in a single font, so mixing within the label is not on the menu —
# which is also why FontAwesome cannot supply the arrow here, having no letters to spell the state with.
_JUMP_TO_LATEST_FONT_BASENAME = "InterTight"
_JUMP_TO_LATEST_FONT_VARIANT = "Regular"

# The gap the pill keeps from the panel's inner bottom-right corner, in pixels — the same on both axes, so
# the corner reads as a corner. Small: this is a thing tucked against the edge, not a floating card.
_JUMP_TO_LATEST_MARGIN = 8

# One pulsation cycle for the pill while the AI is writing, in seconds. Matches the indicator glows, so the
# app breathes at one rate rather than several.
_JUMP_TO_LATEST_PULSE_SECONDS = 2.0

# How many consecutive frames the chat panel's scroll maximum must report the same value before "scroll to
# the end" believes it. One is not enough, and the difference is visible rather than theoretical: the panel's
# content is laid out in pieces — the Markdown renderer runs on its own worker — and the maximum stands still
# between them. Measured on a real chat at startup: 3051 for a frame, then 3497 a few frames later, then
# 4147, where it stopped. A scroll issued at the first standstill went to 3051 and left the reader 1096 px
# short of the message they had come back to.
#
# A heuristic, and worth naming as one: the renderer reports no "finished" event, so there is nothing to wait
# on that would make this exact. What it buys is that a lull has to last three frames to be mistaken for the
# end, and the measured lull was one.
_SCROLL_SETTLE_FRAMES = 3

# What a full rebuild allows for that settling — laying out a chat from nothing takes many more frames than
# appending one message to a chat already on screen. Measured growth above had stopped by frame 20; this is
# headroom over that, and it costs nothing when the content settles sooner, which is the ordinary case.
_BUILD_SCROLL_WAIT_FRAMES = 60

# The same gray the SYSTEM / DOCUMENTS / INTERNET indicators use, rather than a pure white. White would be the brightest
# thing on the panel and would read as an alert; this is one more quiet status light, and it belongs to that
# family both in what it means and in how it looks.
_JUMP_TO_LATEST_COLOR = (180, 180, 180)


# --------------------------------------------------------------------------------


# How long a status indicator stays up once shown, in seconds, however soon its work is over.
_INDICATOR_MIN_SHOW_TIME = 0.5

# How long an indicator saying "Done" stays up after its work ends, in seconds, however long the work took.
_INDICATOR_DONE_LINGER = 0.5

# The same for a closing line that reports something other than success ("No search needed"), which has more
# to say and so more to read.
_INDICATOR_NOTICE_LINGER = 2.0

# Built-in tools that reach out over the network -> light up the INTERNET (globe) indicator while they run.
# The set `llmtools.perform_tool_calls` runs one at a time, which is what keeps that light truthful.
web_access_tool_names = llmclient.NETWORK_TOOL_NAMES
# Tools that read the document database -> light up DOCUMENTS while they run, as the automatic search does.
# The set `llmtools.perform_tool_calls` runs one at a time, as for INTERNET above.
document_access_tool_names = llmclient.DOCUMENT_TOOL_NAMES


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


# The part of the context-fill figure that may be estimated while the readout still calls it exact: the tail
# after the last user message, which the prefill does not send (see `scaffold.build_prefill_prompt`).
_NEGLIGIBLE_TAIL_FRACTION = 0.02

def readout_is_exact(tail_tokens: int, total_tokens: int, tail: list[dict], tokenizer_loaded: bool) -> bool:
    """Whether the context-fill readout may show `total_tokens` as exact, the backend having counted all but `tail`.

    `tail_tokens`: the local count of `tail`, the messages after the last user message.
    `tail`: those messages, as `chatutil.linearize_chat` gives them.
    `tokenizer_loaded`: whether the local count came from a tokenizer rather than from the ratio.

    True when the tail is too small to matter, or was counted rather than estimated: with a tokenizer and no
    image in it, its count is exact short of the chat template's few tokens of framing per message.
    """
    if tail_tokens <= _NEGLIGIBLE_TAIL_FRACTION * total_tokens:
        return True
    tail_has_images = any(isinstance(part, dict) and part.get("type") == "image_url"
                          for message in tail
                          for part in (message.get("content") or []))
    return tokenizer_loaded and not tail_has_images

@dataclasses.dataclass(frozen=True)
class TailFollowSample:
    """What the view looked like just before some content was added or replaced.

    Produced by `DPGLinearizedChatView.sample_tail_follow` and handed straight back to `follow_tail` or
    `restore_scroll_after_swap` once the content has landed. One object rather than three loose values,
    because the three have to be read at the same instant to mean anything together, and because the act
    methods need all of them to notice that the instant has passed.

    `follow`: What `should_follow_tail` said. Sampled *before* the content arrived, since adding content moves
              the end and a view sitting at it is no longer at it a moment later.
    `y_scroll`: Where the panel was, for the swap case, which has to put a reader who was not following back
                where they were.
    `user_scroll_generation`: How many reader-initiated scrolls had happened when this was taken. Compared
                              again at act time: the gap between sample and act spans markdown rendering and
                              at least one `split_frame`, which is long enough for a keypress to land in, and
                              acting on the earlier answer would then undo a scroll the reader asked for after
                              it was given.
    """
    follow: bool
    y_scroll: int
    user_scroll_generation: int


class DPGLinearizedChatView:
    def __init__(self,
                 themes_and_fonts: env,
                 gui_parent: str | int,
                 chat_controller: "DPGChatController",
                 is_any_modal_window_visible: Callable[[], bool] | None = None):
        """A view of the current chat branch, displayed as a linear chat.

        `themes_and_fonts`: Obtain by calling `raven.common.gui.utils.bootup` at app start time.

        `gui_parent`: DPG tag or ID of the panel (child window) you want the chat to be rendered in.

        `chat_controller`: The controller this view belongs to. Managed internally;
                           the `DPGLinearizedChatView` is instantiated and owned by the `DPGChatController`.

        `is_any_modal_window_visible`: Zero-argument predicate, or `None` to skip the check. Consulted while
                                       the scroll-end flasher is fading, which it abandons if a modal opens
                                       — the flasher is drawn in borderless always-on-top windows, so it
                                       would otherwise sit over the dialog. Injected because the app layer
                                       is what knows its own dialogs; this layer must not import it.
        """
        self.themes_and_fonts = themes_and_fonts
        self._text_color_themes: dict[tuple[int, int, int], int | str] = {}
        self.gui_parent = gui_parent
        self.gui_uuid = str(uuid.uuid4())  # used in GUI widget tags
        self.chat_controller = chat_controller

        # TODO: We can later use the existence of this chat container group widget for double-buffering (can render a new group and then switch it in)
        self.chat_messages_container_group_widget = dpg.add_group(tag=f"chat_messages_container_group_{self.gui_uuid}",
                                                                  parent=gui_parent)

        # Where we last put the scroll position ourselves, and whether that was a scroll to the end. Needed to
        # tell our own scrolling apart from the user's; see `should_follow_tail`.
        #
        # A box rather than a plain attribute, because in smooth mode the writer is `SmoothScrolling`: it
        # writes a new position every frame, in the same breath as each `dpg.set_y_scroll`. This view owns
        # the storage because the animation does not outlive its own scroll — it deregisters itself on
        # finishing — and the comparison is needed precisely in the gaps when no animation exists: sitting
        # still after a reply has finished, or deciding whether the jump-to-latest affordance belongs on
        # screen. One writer at a time either way, so the value cannot drift.
        self._commanded_y_scroll: box = box(None)  # int | None inside
        self._commanded_scroll_was_to_end = False

        # Bumped by every reader-initiated scroll, so that a decision taken before one can be recognized as
        # stale afterwards. See `TailFollowSample`. A plain int is enough: reader-initiated scrolls all
        # originate from DPG's callback thread — key handlers and button callbacks alike — so there is one
        # writer, while the readers are the LLM task thread comparing a value it captured earlier.
        self._user_scroll_generation = 0

        # The message open for editing, if any, and what has been typed into it. Held here rather than on the
        # message, because a view rebuild — a window resize, for one — replaces every message instance, and an
        # edit should come through that with its text intact. One at a time.
        self.edit_node_id: str | None = None
        self.edit_draft: str | None = None  # `None` until something has been captured from the field
        self.gui_edit_field = None  # populated by the edited message's `build`
        self.gui_edit_save_button = None  # ...and these, which report a refused save
        self.gui_edit_save_tooltip = None

        # Flashes an arrow band at whichever end a scroll came to rest against. Attached per scroll rather
        # than owned by the animation, because whether an arrival is worth announcing depends on who asked
        # for it — see `_set_y_scroll`.
        #
        # The tag carries this view's UUID: DPG frees deleted items lazily, so a rebuilt view creating the
        # same tag again could collide with one not yet collected, and a tag collision takes the process
        # down rather than raising.
        # Jump-to-latest pill. Raised when content arrives while the reader is away from the end, and
        # cleared by arriving there — the condition that raises it is the condition that clears it, so there
        # is no timeout to tune and no dismiss button to add.
        #
        # Deliberately a *state* rather than an event. A toast or an indicator flash would announce "a reply
        # finished" once, and a reader who is mid-paragraph when it fires has missed it with no way to get it
        # back. What is actually true is "you are not looking at the end, and there is something down there
        # you have not seen", which stays true until it doesn't, so the affordance can simply persist.
        #
        # Note the *and*: this is not "the reader is not at the bottom". Someone paging back through an old
        # conversation is not waiting for anything, and a pill following them up the log would be noise. It
        # takes an arrival to raise it, which is also what makes the "AI finished" label always truthful.
        self._content_arrived_while_unpinned = False

        # Two themes, swapped by state, rather than one theme whose animation is started and stopped: a
        # `PulsatingColor` runs continuously once registered, and the steady variant is how the rest of the
        # app expresses "this is on, but not asking for attention" (cf. the DOCUMENTS indicator's steady/pulsating
        # pair). Pulsating while the AI writes, steady once it has finished, so the pill reports the state by
        # how it behaves as well as by what it says.
        with dpg.theme(tag=f"chat_jump_to_latest_pulsating_theme_{self.gui_uuid}") as self._jump_to_latest_pulsating_theme:  # tag
            with dpg.theme_component(dpg.mvAll):
                pulsating_color_widget = dpg.add_theme_color(dpg.mvThemeCol_Text, _JUMP_TO_LATEST_COLOR)
        self._jump_to_latest_glow = gui_animation.PulsatingColor(cycle_duration=_JUMP_TO_LATEST_PULSE_SECONDS,
                                                                 theme_color_widget=pulsating_color_widget)
        gui_animation.animator.add(self._jump_to_latest_glow)

        with dpg.theme(tag=f"chat_jump_to_latest_steady_theme_{self.gui_uuid}") as self._jump_to_latest_steady_theme:  # tag
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, _JUMP_TO_LATEST_COLOR)

        # Which of the two is currently bound. Tracked so the per-frame update can rebind only on a
        # transition — and so that entering the writing state can restart the pulsation from full alpha,
        # the way an appearing indicator does.
        self._jump_to_latest_is_pulsating: bool | None = None

        with dpg.window(tag=f"chat_jump_to_latest_window_{self.gui_uuid}",  # tag
                        show=False,
                        no_title_bar=True,
                        autosize=True,
                        # Without this the window is silently 100 px tall whatever it holds: `min_size`
                        # defaults to ~[100, 100] and autosize will not shrink past it (`mvStyleVar_
                        # WindowMinSize` does not override it — see `dpg-notes.md`, "Window sizing"). That
                        # is not merely cosmetic here. A DPG window captures mouse input across its whole
                        # rect, background or no background, so the surplus would sit over the chat log as
                        # an invisible patch that swallows the wheel — which is exactly the reason
                        # `ScrollEndFlasher` splits its overlay into two windows rather than covering the
                        # panel with one.
                        min_size=[1, 1],
                        no_collapse=True,
                        no_focus_on_appearing=True,  # a pill appearing must not take the keyboard from the reader
                        no_resize=True,
                        no_move=True,
                        no_background=True,  # the button draws the pill; this window only positions it
                        no_scrollbar=True,
                        no_scroll_with_mouse=True) as self._jump_to_latest_window:
            def jump_to_latest_callback(sender, app_data, user_data) -> None:
                """Take the reader to the end of the chat, and resume following it."""
                self.go_to_bottom()
            self._jump_to_latest_button = dpg.add_button(label=_JUMP_TO_LATEST_FINISHED_LABEL,
                                                         callback=jump_to_latest_callback)
            # Cached and shared by key, so asking for the same face at the same size twice costs nothing.
            _, jump_to_latest_font = guiutils.load_extra_font(themes_and_fonts=themes_and_fonts,
                                                              font_size=gui_config.font_size,
                                                              font_basename=_JUMP_TO_LATEST_FONT_BASENAME,
                                                              variant=_JUMP_TO_LATEST_FONT_VARIANT)
            dpg.bind_item_font(self._jump_to_latest_button, jump_to_latest_font)
            # The pill is the pointer's half of `End`, so it says so. Plain DPG rather than
            # `gui_tooltip.Tooltip`: the *label* changes as the turn proceeds, but this caption does not.
            dpg.add_text("Jump to the latest message [End]",
                         parent=dpg.add_tooltip(self._jump_to_latest_button))

        self._scroll_end_flasher = gui_animation.ScrollEndFlasher(target=gui_parent,
                                                                  tag=f"chat_scroll_end_flasher_{self.gui_uuid}",  # tag
                                                                  duration=gui_config.scroll_ends_here_duration,
                                                                  custom_finish_pred=(lambda _flasher: is_any_modal_window_visible()) if is_any_modal_window_visible is not None else None,
                                                                  font=themes_and_fonts.icon_font_solid,
                                                                  text_top=fa.ICON_ARROWS_UP_TO_LINE,
                                                                  text_bottom=fa.ICON_ARROWS_DOWN_TO_LINE)

    def text_color_theme(self, color: tuple[int, int, int]) -> int | str:
        """A theme setting the text colour to `color`, made once per colour and kept for the life of this view."""
        if color not in self._text_color_themes:
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvThemeCol_Text, color, category=dpg.mvThemeCat_Core)
            self._text_color_themes[color] = theme
        return self._text_color_themes[color]

    def note_wheel_scroll(self) -> None:
        """Announce this view's scroll ends when the mouse wheel reaches or presses against one.

        Call from a mouse-wheel handler, having checked the pointer is over this view. The wheel is the one
        movement path `SmoothScrolling` cannot see, DPG scrolling the child window internally, so without
        this a reader who wheels to the end of the log is told nothing while one who pages there is.

        No `user_initiated` gate here, unlike `_start_scroll_animation`: a wheel event *is* the reader.
        """
        if self._scroll_end_flasher is not None:
            self._scroll_end_flasher.note_wheel_scroll()

    def should_follow_tail(self, verbose: bool = True) -> bool:
        """Whether new content should pull the view along with it.

        `verbose`: Whether to log the decision and its numbers. Pass `False` from a per-frame caller — the
                   jump-to-latest pill asks this sixty times a second, and at DEBUG that buries the
                   once-per-chunk decisions this log exists to let you read. The answer is identical either
                   way; this method stores no state and has no other side effect.

        Not the same question as "is the view at the bottom", and the difference is the whole bug this exists
        to avoid. Two endpoints move here: the user moves the scroll *position*, and arriving content moves the
        *maximum*. A position-only test cannot tell those apart — both show up as a gap — so it reads new
        content as "the user scrolled away".

        Getting that wrong latches, which is what makes it severe rather than occasional. The answer is sampled
        once per streamed chunk, before that chunk is rendered; if a single transient displacement makes it
        `False`, the next sample is taken from a view that has fallen one chunk further behind, so it stays
        `False` and the gap only grows. Even a momentary displacement of a line or two — never mind a whole
        swapped-out paragraph — is enough to freeze the view for the rest of the turn, wherever it happened to
        be at that moment.

        So the position is compared against `self._commanded_y_scroll` — where *we* last put it. Content
        growing moves the maximum but not the position, so the position still matches what we commanded and
        following continues. The user scrolling moves the position away from what we commanded, which is the
        one thing content arrival cannot do. That distinction needs no scroll events, which is essential:
        of the three ways this panel moves — scrollbar drag, mouse wheel, navigation keys — the drag is
        handled inside ImGui and raises nothing we could hook.

        The comparison is only as good as the record it compares against, which is why `scroll_view` waits for
        its command to actually land rather than assuming it did. A command still in flight leaves the position
        disagreeing with the record, which is indistinguishable here from the user having scrolled.

        Each call decides on current evidence and stores no verdict. A wrong answer therefore costs one chunk
        rather than the remainder of the reply.

        Both questions are asked of where our own scrolling is *heading*, not of where it has got to. A scroll
        in flight has a reported position somewhere along the way, which answers for the movement's past
        rather than for the request that started it — so a scroll away from the end reads as still-at-the-end
        until enough of it has been carried out, and whether that has happened by the time the next chunk
        samples this is a matter of timing rather than of intent.

        The tolerance (`_PIN_TOLERANCE_PX`) absorbs the drift of a scroll that has effectively, but not
        exactly, arrived. It is a genuine trade-off in both directions, which is why it is instrumented rather
        than guessed: too small and the view stops following right after the user sends a message; too large
        and a deliberate scroll of one or two lines away from the end still counts as following, so the arrow
        keys appear not to work.

        While one of our own scroll animations is running, the tolerance widens to cover a single frame of it.
        The panel's report lags the last written value by exactly one step, so a gap that size is ours; and
        early in an exponential decay a step is hundreds of pixels, which a bound sized for a human's nudge
        would read as user input. With nothing animating the tight bound applies, which is the case where
        catching a real user scroll matters most.

        Diagnostics: every call logs the numbers and which branch decided, at DEBUG. Run with
        `logsetup.init(level=logging.DEBUG)` for the full trace. A refusal that is *near* the end additionally
        logs at INFO, because that is the shape a wrong answer takes: if it fires on a turn you expected to be
        followed, the reported numbers say which of the two branches let it through.
        """
        max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        if max_y_scroll <= 0:  # no scrollbar: the tail is always in view
            if verbose:
                logger.debug("DPGLinearizedChatView.should_follow_tail: no scrollbar -> True")
            return True

        y_scroll = dpg.get_y_scroll(self.gui_parent)
        scroll_animation = gui_animation.SmoothScrolling.instances.get(self.gui_parent)
        animation_slack = scroll_animation.last_step if scroll_animation is not None else 0.0
        commanded_y_scroll = unbox(self._commanded_y_scroll)
        decision = layout_math.decide_tail_follow(y_scroll=y_scroll,
                                                  max_y_scroll=max_y_scroll,
                                                  maybe_target_y_scroll=scroll_animation.target_y_scroll if scroll_animation is not None else None,
                                                  last_step=animation_slack,
                                                  maybe_commanded_y_scroll=commanded_y_scroll,
                                                  commanded_to_end=self._commanded_scroll_was_to_end,
                                                  tolerance=_PIN_TOLERANCE_PX)
        follow = decision.follow

        if verbose:
            logger.debug(f"DPGLinearizedChatView.should_follow_tail: y_scroll={y_scroll}, max_y_scroll={max_y_scroll}, "
                         f"gap={decision.gap}, settled_gap={decision.settled_gap} to y={decision.settled_y_scroll} "
                         f"(tolerance={_PIN_TOLERANCE_PX}) -> at_end={decision.at_end}; "
                         f"drift tolerance={decision.drift_tolerance} (animation slack={animation_slack}); "
                         f"commanded={commanded_y_scroll} (to_end={self._commanded_scroll_was_to_end}), "
                         f"expected={decision.maybe_expected_y_scroll}, drift={decision.maybe_drift} -> undisturbed={decision.undisturbed}; "
                         f"-> follow={follow}")
        if verbose and not follow and 0 < decision.settled_gap <= _PIN_NEAR_MISS_FACTOR * _PIN_TOLERANCE_PX:
            logger.info(f"DPGLinearizedChatView.should_follow_tail: NEAR MISS — settled_gap={decision.settled_gap}px "
                        f"exceeds tolerance={_PIN_TOLERANCE_PX}px and the position has drifted {decision.maybe_drift}px from the "
                        f"{commanded_y_scroll} we last commanded (to_end={self._commanded_scroll_was_to_end}, "
                        f"drift tolerance={decision.drift_tolerance}px including {animation_slack}px of animation slack), "
                        "so the view will not follow. If you expected it to follow, the drift is the number to "
                        "look at: a drift above the tolerance with no user scrolling and no animation running "
                        "means something moved the position behind our back.")

        # Deliberately *not* recording this refusal anywhere. Making it sticky looks like the careful choice —
        # it would stop one ambiguous frame from resuming the drag — but it is both unnecessary and harmful. A
        # reader who really has scrolled away keeps failing the drift test on every later sample all by itself,
        # because they stay where they put themselves and we issue no further commands. What stickiness adds is
        # amplification: any single wrong refusal becomes permanent for the rest of the reply. Observed exactly
        # that way, so the state stays where it is and each sample decides on current evidence.
        return follow

    def sample_tail_follow(self) -> TailFollowSample:
        """Read everything `follow_tail` / `restore_scroll_after_swap` will need, as of right now.

        Call this *before* adding or replacing content, and hand the result back to whichever of the two
        applies once the content has landed. Sampling the pieces separately at the call site is what this
        exists to prevent: they are only meaningful as a set taken at one instant.
        """
        return TailFollowSample(follow=self.should_follow_tail(),
                                y_scroll=dpg.get_y_scroll(self.gui_parent),
                                user_scroll_generation=self._user_scroll_generation)

    def _reader_scrolled_since(self, sample: TailFollowSample) -> bool:
        """Whether a reader-initiated scroll has landed since `sample` was taken.

        When it has, the sample describes a view the reader has since moved on from, and acting on it would
        take back a scroll they asked for. Both act methods therefore do nothing at all in that case, rather
        than falling back to their non-following branch: the reader's own scroll is already in flight and
        will carry the view where they wanted it.
        """
        if sample.user_scroll_generation == self._user_scroll_generation:
            return False
        logger.info(f"DPGLinearizedChatView._reader_scrolled_since: the reader scrolled while this content was "
                    f"being laid out (generation {sample.user_scroll_generation} -> "
                    f"{self._user_scroll_generation}), so the follow decision taken before it "
                    f"(follow={sample.follow}) is stale and will not be acted on.")
        return True

    def restore_scroll_after_swap(self, sample: TailFollowSample) -> None:
        """Put the view back after content was *replaced* — deleted and re-added — rather than appended.

        `sample`: what `sample_tail_follow` reported **before** the swap.

        Appending only ever grows the container, so a reader below the fold keeps their offset for free and
        `follow_tail` is enough. A swap briefly *shrinks* it, and DPG clamps the scroll position to the
        smaller maximum at the next layout — which the render loop can perform mid-swap, since these
        callbacks run on the LLM task thread rather than the main one. A reader who was following recovers
        from that on their own (the scroll to the new end happens afterwards); one who was not does not, so
        their offset is restored explicitly.
        """
        if not sample.follow:
            self._content_arrived_while_unpinned = True  # raises the jump-to-latest pill
        guiutils.split_frame(operation="restore_scroll_after_swap: lay out the replacement content")
        if self._reader_scrolled_since(sample):
            return
        if sample.follow:
            self.scroll_view(abort_if_reader_scrolled_since=sample)
        else:
            self._set_y_scroll(sample.y_scroll, to_end=False)

    def hold_scroll_across_rebuild(self, y_scroll: int) -> None:
        """Put the view back at `y_scroll` after one message rebuilt itself in place.

        `y_scroll`: what `dpg.get_y_scroll(self.gui_parent)` reported **before** the rebuild.

        The narrow sibling of `restore_scroll_after_swap`, for a rebuild the *reader* asked for rather than
        one that content arrival forced. Neither of that one's two behaviours is right here: no content
        arrived, so raising the jump-to-latest pill would be a lie, and a reader who happened to be at the
        end did not ask to be taken there — they asked to expand a message and expect to still be looking
        at it.

        Nothing above the rebuilt message changes, so its offset in content coordinates is the same before
        and after; restoring the viewport offset therefore restores its *screen* position exactly. The wait
        is not optional — DPG clamps the scroll to the smaller maximum at the next layout, so reading or
        writing the position before the replacement has been laid out reads a number that is about to change.

        The thinking-trace toggle needs none of this, and the difference is *rebuilding*, not the toggling:
        it renders both states up front and flips `hide_item` / `show_item`, so DPG's layout engine reflows
        around the change and the position stays consistent by construction. That trade is not available
        here — it would mean laying out tens of thousands of characters of markdown on every chat-view
        rebuild to keep a copy hidden — so this restores by hand what that gets for free.

        One case cannot be honoured, and it is arithmetic rather than a bug: collapsing a document that was
        most of the conversation can leave less content than `y_scroll` scrolls past, and the view then sits
        at the new maximum with the message lower on screen than it was. There is nowhere else for it to be.

        **Instant, not animated, and written twice.** This is a correction rather than a navigation: the
        reader asked to expand a message, not to travel, so animating the fix shows them a wrong position and
        then makes them watch it being undone — the jump reads as a glitch and the glide reads as the app
        changing its mind. Writing it before the wait as well as after narrows how long the wrong position is
        on screen: the rebuild lays out its markdown over several frames, and the clamp lands on the first of
        them, so waiting for the whole layout before correcting means showing the clamped position for all of
        them. The write before the wait is the one that usually holds; the one after is what catches the
        clamp when the layout moved the maximum under it.
        """
        self._set_y_scroll(y_scroll, to_end=False, smooth=False)
        guiutils.split_frame(operation="hold_scroll_across_rebuild: lay out the rebuilt message")
        self._set_y_scroll(y_scroll, to_end=False, smooth=False)

    def follow_tail(self, sample: TailFollowSample) -> None:
        """Scroll the view to the end of the chat, but only if it was following *before* the content grew.

        `sample`: what `sample_tail_follow` reported **before** whatever just added content.

        That ordering is the whole point, and is why this takes the answer instead of asking for itself.
        Appending text grows the container, so `max_y_scroll` rises and a view that was at the bottom is no
        longer at the bottom the instant the new content lands. A version that sampled here would read "the
        user has scrolled away" on every chunk, never follow, and leave the view frozen where the stream
        began — failing in the opposite direction from the bug it fixes, while looking entirely reasonable.

        The same ordering opens the window this then has to close. Between the sample and the write sit the
        markdown render and two `split_frame` waits — around a tenth of a second, with the reader's keyboard
        live throughout. An arrow key landing in there was erased rather than outvoted: `scroll_view`
        retargeted the reader's in-flight upward scroll back to the end and re-asserted tail-following, so
        roughly one press in fifteen vanished.

        The window closes at the write, not here. Testing the sample at this end of it catches only presses
        that arrived before the call, which is why `scroll_view` re-checks after its own settle wait — that
        wait turned out to be where the surviving losses were landing.
        """
        if not sample.follow:
            self._content_arrived_while_unpinned = True  # raises the jump-to-latest pill
            return
        if self._reader_scrolled_since(sample):
            return
        # Re-checked inside, after the settle wait: this early test only saves the trip, it does not close
        # the window — the wait is where a keypress actually lands.
        self.scroll_view(abort_if_reader_scrolled_since=sample)  # waits for the new content to lay out, so this reaches the *new* end

    def _set_y_scroll(self, y_scroll: int, *, to_end: bool, user_initiated: bool = False,
                      smooth: bool | None = None) -> None:
        """Move the scroll position, remembering what we asked for.

        `y_scroll`: The target position in content coordinates — a non-negative offset from the top. It is
                    recorded and compared later against the position the panel reports, so it has to be the
                    same number that ends up applied. "Go to the end" is therefore expressed as `to_end`
                    plus the concrete maximum, which the caller has already computed in order to clamp.
        `to_end`: Whether this scroll was a scroll to the end of the content, i.e. whether the view should
                  keep following the tail as more content arrives.
        `user_initiated`: Whether a human asked for this scroll, as opposed to the view following a growing
                          reply on its own. Decides whether reaching an end is worth signalling: the
                          scroll-end flasher asserts *"you tried to go further and could not"*, which is a
                          statement about a thwarted intent. Tail-following has none — arriving at the end
                          is its whole purpose — so a flasher on that path would strobe once per streamed
                          chunk for the length of a reply.

                          It cannot be derived from `to_end`: clicking jump-to-latest is also a scroll to
                          the end, and there the flash is *wanted*, because it confirms arrival. What
                          separates the two is provenance, not destination.

        Every scroll this class performs goes through here, so that `should_follow_tail` can tell our own
        scrolling from the user's. A bare `dpg.set_y_scroll` elsewhere would look exactly like a user scroll
        and silently stop the view following.

        Animated or instant is `SmoothScrolling`'s own `smooth` flag rather than two code paths here: that
        class jumps straight to the target when told not to animate, explicitly so that both behaviours wear
        one API. Routing the instant case through it too is what keeps the commanded-position bookkeeping,
        the retargeting and the end-of-content signalling identical in both modes, instead of one of them
        quietly growing a second set of rules.

        `smooth`: `None` (the default) takes `config.smooth_scrolling`, which is what every *navigation*
                  wants — a reader who pressed a key or a button is going somewhere, and the animation is
                  what tells them where from. Pass `False` for a **correction**: a scroll that exists only to
                  undo a position the layout engine imposed, where the reader asked to go nowhere at all.
                  Animating one of those shows them the wrong position and then makes them watch it being
                  fixed, which reads as the app changing its mind.

        If a scroll is already in flight on this panel it is *retargeted* rather than replaced, keeping its
        subpixel position so the movement bends toward the new target instead of restarting. The retarget
        adopts this request wholesale, so a follow scroll correctly takes the flasher back off a scroll the
        user had started.

        The commanded position is handed over as a box because the animation writes a new value every frame,
        in the same breath as each `dpg.set_y_scroll` — which is what keeps `should_follow_tail`'s
        comparison meaningful while a scroll is in flight.
        """
        if y_scroll < 0:
            raise ValueError(f"_set_y_scroll: expected a non-negative position, got {y_scroll}")
        self._commanded_scroll_was_to_end = to_end

        # Announce a reader-initiated scroll to any follow decision that was taken before it and has not been
        # acted on yet. `user_initiated` is already the right discriminator: it marks the scrolls that express
        # someone's intent — keys, and the jump-to-message buttons — as against the view chasing a stream.
        if user_initiated:
            self._user_scroll_generation += 1

        # The flasher rides on the scroll request, and only on a reader's. `user_initiated` is the whole
        # gate: the flash asserts *"you asked to go further and there is no further"*, which is a statement
        # about someone's intent, and tail-following has none — arriving at the end is what it is for, so a
        # flasher on that path would strobe once per streamed chunk for the length of a reply.
        #
        # Retargeting keeps the gate honest without any help here, because a retarget adopts the incoming
        # request wholesale: a follow scroll landing on a reader's in-flight one carries `None` and so takes
        # the flasher back off, and a reader's scroll landing on a follow puts one on. Latest asker wins.
        gui_animation.SmoothScrolling.scroll(target_child_window=self.gui_parent,
                                             target_y_scroll=y_scroll,
                                             smooth=(gui_config.smooth_scrolling if smooth is None else smooth),
                                             smooth_step=gui_config.smooth_scrolling_step_parameter,
                                             flasher=(self._scroll_end_flasher if user_initiated else None),
                                             commanded_y_scroll=self._commanded_y_scroll)

    def find_message(self, node_id: str) -> "chatmessage.DPGChatMessage | None":
        """Return the message widget showing chat node `node_id`, or `None` if it is not in this view.

        `None` is the ordinary answer, not an error: the view holds one branch, and a node on any other one
        has no widget here by construction.
        """
        with self.chat_controller.current_chat_history_lock:  # `build` refills this from another thread
            for dpg_chat_message in self.chat_controller.current_chat_history:
                if dpg_chat_message.node_id == node_id:
                    return dpg_chat_message
        return None

    # --------------------------------------------------------------------------------
    # Editing a message

    def capture_edit_draft(self) -> None:
        """Remember what is in the edit field, so the edited message can be rebuilt without losing it."""
        if self.gui_edit_field is None:
            return
        with guiutils.nonexistent_ok():
            maybe_value = dpg.get_value(self.gui_edit_field)
            if maybe_value is not None:
                self.edit_draft = maybe_value

    def start_editing(self, node_id: str) -> str | None:
        """Open the message showing `node_id` for editing, and put the caret in it.

        Ask `chatutil.is_editable` first. Asking for the message already open puts the caret back in it.

        Returns `None` when done, or, when refused, a short reason for the caller to show.
        """
        if self.edit_node_id == node_id:
            if self.gui_edit_field is not None:
                self.chat_controller.give_caret(self.gui_edit_field)
            return None
        if self.edit_node_id is not None:
            return "Another message is open for editing."
        maybe_refusal = self.chat_controller.edit_refusal()
        if maybe_refusal is not None:
            return maybe_refusal
        message = self.find_message(node_id)
        if message is None:
            return "This message is not in the chat log."
        self.edit_node_id = node_id
        self.edit_draft = None
        message.rebuild_in_place()
        self.chat_controller.give_caret(self.gui_edit_field)
        return None

    def finish_editing(self, save: bool) -> str | None:
        """Close the message open for editing, saving what was typed as a new revision if `save`.

        Saving text identical to the message's is the same as cancelling: no revision is made.

        Returns `None` when done, or, when the save is refused, a short reason, which is also flashed on the
        Save button. A refused save leaves the message open, with what was typed still in it.
        """
        node_id = self.edit_node_id
        if node_id is None:
            return None
        if save:
            self.capture_edit_draft()
            unused_role, unused_persona, old_text = chatutil.get_node_message_text_without_persona(self.chat_controller.datastore, node_id)
            new_text = (self.edit_draft if self.edit_draft is not None else old_text).strip()
            if new_text != old_text.strip():
                maybe_refusal = self.chat_controller.revise_message(node_id, new_text)
                if maybe_refusal is not None:
                    if self.gui_edit_save_button is not None:
                        gui_animation.flash_button(button=self.gui_edit_save_button, tooltip=self.gui_edit_save_tooltip,
                                                   ok=False, message=maybe_refusal,
                                                   duration=gui_config.acknowledgment_duration)
                    return maybe_refusal
        self.edit_node_id = None
        self.edit_draft = None
        self.gui_edit_field = None
        self.gui_edit_save_button = None
        self.gui_edit_save_tooltip = None
        maybe_message = self.find_message(node_id)
        if maybe_message is not None:
            maybe_message.rebuild_in_place()
        self.chat_controller.give_keyboard_to_log()
        return None

    def handle_edit_key(self, key: int, ctrl: bool) -> bool:
        """Offer a key to the message open for editing. Returns whether it was taken.

        While the edit field holds the caret, every key is taken, so no hotkey acts on the chat mid-edit.
        The commit chord — the composer's send key — saves, and Esc cancels.
        """
        field = self.gui_edit_field
        if self.edit_node_id is None or field is None:
            return False
        active = focused = False
        with guiutils.nonexistent_ok():
            active = dpg.is_item_active(field)
            focused = dpg.is_item_focused(field)
        if not (active or focused):  # also the answer when the field has just gone away
            return False
        # Both chords reach the field before they reach this handler, and both deactivate it — the commit
        # chord validates the edit, and Esc reverts it — so the key that has just ended an edit reads
        # focused and not active. Measured for the commit chord in
        # `investigations/dpg-focus/commit_chord_dispatch_probe.py`.
        commit_needs_ctrl = (librarian_config.send_message_key == "ctrl+enter")
        if key == dpg.mvKey_Return and ctrl == commit_needs_ctrl:
            self.finish_editing(save=True)
            return True
        if key == dpg.mvKey_Escape:
            self.finish_editing(save=False)
            return True
        return active

    def jump_to_node(self, node_id: str) -> int | None:
        """Scroll to the message showing chat node `node_id` and flash it.

        Returns the scroll position the view is heading for, or `None` if `node_id` is not in this view.

        The pair `_make_jump_to_tool_call` uses, for a whole message rather than a sub-element: scrolling
        alone lands the reader somewhere without saying which of the messages now on screen was the answer.
        """
        message = self.find_message(node_id)
        if message is None:
            logger.info(f"DPGLinearizedChatView.jump_to_node: chat node '{node_id}' is not in this view")
            return None
        y_scroll = self.scroll_view(scroll_target_node_id=node_id, user_initiated=True)
        gui_animation.highlight_widget(widget=f"chat_message_timestamp_{message.gui_uuid}",  # tag
                                       duration=gui_config.acknowledgment_duration)
        return y_scroll

    def scroll_view(self,
                    max_wait_frames: int = 10,
                    scroll_target_node_id: str | None = None,
                    user_initiated: bool = False,
                    abort_if_reader_scrolled_since: TailFollowSample | None = None) -> int | None:
        """Scroll this linearized chat view to the end.

        Returns the scroll position the view is heading for, or `None` if the scroll was abandoned (see
        `abort_if_reader_scrolled_since`).

        `abort_if_reader_scrolled_since`: A sample whose currency is re-checked at the last moment, just
                                          before the scroll is committed, and which cancels the scroll if a
                                          reader-initiated one has landed meanwhile. For the automatic paths
                                          (`follow_tail`, `restore_scroll_after_swap`), which must not
                                          override a reader.

                                          Checking once at the caller is not enough, and the reason is in
                                          this method: the settle wait below is itself part of the window.
                                          Observed with the callers already guarded — the sample was current
                                          when `follow_tail` tested it, an arrow key landed during the two
                                          frames spent waiting for the maximum to settle, and the scroll to
                                          the end was committed on top of it. The guard has to sit next to
                                          the write, not next to the decision to write.

        `user_initiated`: Whether a human asked for this scroll, rather than the view following a growing
                          reply on its own. Only affects presentation — see `_start_scroll_animation`, where
                          it decides whether hitting an end is worth signalling. Defaults to `False` because
                          the automatic path is the frequent one, and because a wrong `True` is the noisy
                          failure (a flash per streamed chunk) while a wrong `False` merely omits a
                          confirmation.

        `max_wait_frames`: If `max_wait_frames > 0`, wait at most for that many frames for the chat panel
                           (`self.gui_parent`) to report a `max_y_scroll` that has *settled*: nonzero, and
                           unchanged from the previous frame.

                           Some waiting is usually needed at least at app startup before the GUI settles.

                           Settling, rather than merely waiting for nonzero, because `get_y_scroll_max` lags
                           a content change by more than one frame — the same lag `SmoothScrolling` budgets
                           four frames for. Reading it too early returns the maximum from *before* the
                           content was added, and then "scroll to the end" lands where the previous message
                           ended, so the view visibly fails to reach the message the user just sent.

        `target_y`: y coordinate to scroll to, in coordinate system of `self.gui_parent`.
                    If not provided (default), scroll to end.

        NOTE: When called from the render loop thread, `max_wait_frames` must be 0, as any attempt to
              wait would hang that loop. Enforced below rather than merely asked for, because the
              penalty is a hang with no traceback — the one DPG failure mode that tells you nothing.

              When called from any other thread (also event handlers), waiting is fine. All current
              callers qualify: the LLM task thread, `bgtask` workers, DPG event callbacks and frame
              callbacks are all dispatched off the render loop.
        """
        # Waiting below goes through `guiutils.split_frame`, which would deadlock in the render loop.
        # Degrade rather than raise: the scroll then lands on a possibly stale maximum, which is a visible
        # imperfection rather than a dead app.
        if max_wait_frames > 0 and guiutils.is_render_thread():
            logger.warning("DPGLinearizedChatView.scroll_view: called from the render loop thread with "
                           f"max_wait_frames={max_wait_frames}; waiting would deadlock it, so proceeding "
                           "without waiting. Pass max_wait_frames=0 explicitly at this call site.")
            max_wait_frames = 0

        # Settling takes at least one frame by construction: a single sample cannot tell a settled value from
        # a stale one, so there is always a second sample to compare against. One frame on a background thread
        # is a cheap price for the scroll landing where it was asked to.
        elapsed_frames = 0
        stable_frames = 0
        max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        for elapsed_frames in range(1, max_wait_frames + 1):
            guiutils.split_frame(operation="scroll_view: settle the chat panel's scroll maximum")
            previous_max_y_scroll, max_y_scroll = max_y_scroll, dpg.get_y_scroll_max(self.gui_parent)
            if max_y_scroll > 0 and max_y_scroll == previous_max_y_scroll:  # TODO: The nonzero requirement fails when the content is less than one screenful in length: a legitimately zero maximum is indistinguishable from a panel that has not laid out yet, so we wait out `max_wait_frames`. Think of a better way.
                stable_frames += 1
                if stable_frames >= _SCROLL_SETTLE_FRAMES:
                    break
            else:
                stable_frames = 0  # it moved again; whatever we saw was a lull, not the end
        plural_s = "s" if elapsed_frames != 1 else ""
        waited_str = f" (after waiting for {elapsed_frames} frame{plural_s})" if elapsed_frames > 0 else " (no waiting was needed)"
        # Logging the frame number only when we waited is deliberate but no longer explained. It used to cite
        # `dpg.get_frame_count()` needing the render thread mutex (DearPyGui#2366) — which is wrong twice over:
        # that issue is about holding `dpg.mutex()` for a long time inside a frame callback, and every Raven app
        # calls `get_frame_count()` from the animator on the render thread, every frame, without trouble. Some
        # real problem was being named here and its actual cause is unidentified, so the condition stays as
        # written rather than being "simplified" on the strength of not being able to reproduce it. If a hang
        # or a stall ever surfaces at this line, this is the note to start from — and to replace.
        frames_str = f" frame {dpg.get_frame_count()}" if max_wait_frames > 0 else ""

        if scroll_target_node_id is not None:
            logger.info(f"DPGLinearizedChatView.scroll_view: Scroll target chat node is '{scroll_target_node_id}'")
            def get_target_widget() -> str | int | None:
                with self.chat_controller.current_chat_history_lock:  # `build` refills this from another thread
                    for dpg_chat_message in self.chat_controller.current_chat_history:
                        if dpg_chat_message.node_id == scroll_target_node_id:  # found?
                            return dpg_chat_message.gui_container_group
                return None
            if (target_message_widget := get_target_widget()) is not None:
                # `get_widget_pos` reports *viewport* (on-screen) coordinates, while `set_y_scroll` wants an
                # offset into the panel's scrollable content. The two coincide only when the panel happens to
                # be scrolled to the top — which is why this went unnoticed for so long: the only previous
                # caller scrolls immediately after a full rebuild, when it is. From an already-scrolled view
                # the target is on screen by definition, so its viewport y is small and the panel jumped to
                # the top instead. Convert: undo the panel's own origin, then add back where we already are.
                # Same transformation as `raven.visualizer.info_panel.scroll_to_item`, deliberately: the
                # panel origin is offset by the content area's outer + inner padding, so that the target
                # lands at the top of the *content* rather than 11 px below it.
                _, target_viewport_y = guiutils.get_widget_pos(target_message_widget)
                _, panel_viewport_y = guiutils.get_widget_pos(self.gui_parent)
                content_origin_y = panel_viewport_y + guiutils.DPG_WINDOW_PADDING + guiutils.DPG_FRAME_PADDING_Y
                y0 = (target_viewport_y - content_origin_y) + dpg.get_y_scroll(self.gui_parent)
                logger.info(f"DPGLinearizedChatView.scroll_view: Scroll target chat node is at content y = {y0} (viewport y = {target_viewport_y}, panel origin y = {panel_viewport_y}).")
            else:
                y0 = max_y_scroll
                logger.warning(f"DPGLinearizedChatView.scroll_view: Scroll target chat node '{scroll_target_node_id}' not found in view, scrolling to end instead.")
            y_scroll = min(max(0, y0), max_y_scroll)
            to_end = False  # a jump to a specific message: the reader wants to be *there*, not at the tail
        else:
            logger.info("DPGLinearizedChatView.scroll_view: No scroll target chat node specified, scrolling to end.")
            y_scroll = max_y_scroll
            to_end = True
        logger.info(f"DPGLinearizedChatView.scroll_view:{frames_str}{waited_str}: max_y_scroll = {max_y_scroll}, scrolling to y = {y_scroll}")

        # Last check before the write, with no wait left between the two. Everything above this line — the
        # settle wait especially — is time in which a keypress can arrive.
        if abort_if_reader_scrolled_since is not None and self._reader_scrolled_since(abort_if_reader_scrolled_since):
            return None

        self._set_y_scroll(y_scroll, to_end=to_end, user_initiated=user_initiated)

        # There used to be a verification loop here, waiting for the panel to report the position we asked
        # for and re-issuing until it did. It is gone because `SmoothScrolling` now owns the whole job, and
        # the reasoning is worth keeping because deleting a careful mechanism deserves an argument.
        #
        # `dpg.get_y_scroll` does not reflect a `dpg.set_y_scroll` for more than one frame: a single
        # `split_frame` afterwards still reads the *previous* position. Measured over a session of streaming
        # replies, one extra frame sufficed 114 times out of 115, and two were needed once. Waiting mattered
        # because `should_follow_tail` compares the position against what we commanded, so a command that has
        # not landed yet is indistinguishable from the user having scrolled away — and that latches, freezing
        # the view for the rest of the reply.
        #
        # Three jobs were tangled in that loop, and each has a better home now:
        #
        #   - *Making the record true.* `SmoothScrolling`'s per-frame guard is the same device — it refuses to
        #     advance until DPG reports back the value it last wrote — and it writes the commanded-position box
        #     in the same breath, so the record is never more than one frame stale. `_PIN_TOLERANCE_PX` already
        #     absorbs a frame. The loop was a coarser hand-rolled version of that guard, run from another
        #     thread.
        #   - *Chasing a target that moved while we waited.* Retargeting covers it, on a better trigger: the
        #     target moves when content arrives, and content arriving is exactly when `follow_tail` fires. Event
        #     driven, rather than polled against a fixed attempt budget.
        #   - *Recovering from a DPG clamp.* Same event. The clamp sources are the streaming chunk handler's:
        #     `reclassify_all_paragraphs_as_thought`, which deletes a reply's paragraphs and re-renders them
        #     into the thought bubble, and `replace_last_paragraph`'s delete-then-add fallback (it otherwise
        #     swaps through `WidgetSwap`) — so a clamp can only happen while streaming, which is precisely
        #     when `follow_tail` retargets per chunk.
        #
        # None of that depends on the scroll being *animated*: it depends on retargeting, which works the same
        # when `smooth` is off. That is why there is one path here rather than two.
        #
        # The consequence to protect: `follow_tail`'s retarget is now load-bearing for *correctness*, not only
        # for smoothness. Rate-limiting it, or gating it on the view having visibly moved, would silently bring
        # back the last two failures.
        return y_scroll

    # ------------------------------------------------------------
    # Reader-driven scrolling (hotkeys, and later the on-screen controls)

    def scroll_to_position(self, target_y_scroll: int | None) -> None:
        """Scroll to an absolute position, clamped into range.

        `target_y_scroll`: Offset from the top, in content coordinates. `None` means the end of the content —
                           spelled as a distinct value rather than as a large number, because "the end" has
                           to keep meaning the end as the content grows, and because only that case should
                           re-engage tail-following.

        For reader-initiated scrolling. `scroll_view` remains the entry point for the program's own scrolls
        (following a stream, landing on a message after a rebuild), and does the content-settling wait those
        need; a reader pressing a key is not waiting for anything to lay out.
        """
        max_y_scroll = dpg.get_y_scroll_max(self.gui_parent)
        to_end = (target_y_scroll is None)
        y_scroll = max_y_scroll if to_end else int(numutils.clamp(target_y_scroll, 0, max_y_scroll))
        logger.debug(f"DPGLinearizedChatView.scroll_to_position: to y = {y_scroll} (max = {max_y_scroll}, to_end = {to_end})")
        self._set_y_scroll(y_scroll, to_end=to_end, user_initiated=True)

    def go_to_top(self) -> None:
        """Scroll to the start of the chat."""
        self.scroll_to_position(0)

    def go_to_bottom(self) -> None:
        """Scroll to the end of the chat, and resume following the tail."""
        self.scroll_to_position(None)

    def update_jump_to_latest_pill(self) -> None:
        """Show, hide, label and position the jump-to-latest pill. Call once per frame, from the render loop.

        Polled rather than event-driven, and it has to be: of the ways this panel moves, the mouse wheel and
        the scrollbar are handled inside ImGui and raise nothing to hook. A reader who wheels away from the
        end must see the pill appear, so the only reliable trigger is looking every frame.

        Cheap enough for that: two scroll queries and a dict lookup, with the logging suppressed — see
        `should_follow_tail`'s `verbose`.
        """
        # Arriving at the end is what takes the pill down, whether the reader got there by clicking it, by
        # pressing End, or by scrolling back by hand. Asking `should_follow_tail` rather than comparing
        # positions here keeps one definition of "at the end" for the whole view; a second one would drift
        # from it and show a pill while the view was in fact following.
        if self.should_follow_tail(verbose=False):
            self._content_arrived_while_unpinned = False

        if not self._content_arrived_while_unpinned:
            if dpg.is_item_shown(self._jump_to_latest_window):
                dpg.hide_item(self._jump_to_latest_window)
            return

        is_writing = self.chat_controller.is_generating()

        label = _JUMP_TO_LATEST_WRITING_LABEL if is_writing else _JUMP_TO_LATEST_FINISHED_LABEL
        if dpg.get_item_label(self._jump_to_latest_button) != label:
            dpg.set_item_label(self._jump_to_latest_button, label)

        if is_writing != self._jump_to_latest_is_pulsating:
            self._jump_to_latest_is_pulsating = is_writing
            if is_writing:
                self._jump_to_latest_glow.reset()  # start the cycle at full alpha, as an appearing indicator does
                dpg.bind_item_theme(self._jump_to_latest_button, self._jump_to_latest_pulsating_theme)
            else:
                dpg.bind_item_theme(self._jump_to_latest_button, self._jump_to_latest_steady_theme)

        # Position it against the panel every frame, so it follows a window resize without a resize hook —
        # same approach `ScrollEndFlasher` takes, and for the same reason: the geometry is cheap to read and
        # a hook is one more thing to remember to call.
        #
        # Bottom-right rather than bottom-centre: centred, it sat over the text a reader is in the middle of.
        # The corner is out of the way, and it is also where the eye already is, since reaching this state
        # means having just worked the scrollbar. Hence the extra clearance on the right — landing under the
        # scrollbar would put the pill exactly where the pointer is.
        # Measured from the *button*, not from the window holding it, because the button is the pill a
        # reader sees: the window adds a padding ring around it, and measuring the gap to the window's edge
        # would quietly make the visible gap twice what it says. So the arithmetic places the button and
        # then backs out the window's content origin, one window padding in from its corner.
        panel_x, panel_y = dpg.get_item_pos(self.gui_parent)  # child windows report `pos`, not `rect_min`
        panel_w, panel_h = dpg.get_item_rect_size(self.gui_parent)
        button_w, button_h = dpg.get_item_rect_size(self._jump_to_latest_button)
        button_right = panel_x + panel_w - guiutils.DPG_SCROLLBAR_SIZE - _JUMP_TO_LATEST_MARGIN
        button_bottom = panel_y + panel_h - _JUMP_TO_LATEST_MARGIN
        dpg.set_item_pos(self._jump_to_latest_window,
                         [button_right - button_w - guiutils.DPG_WINDOW_PADDING,
                          button_bottom - button_h - guiutils.DPG_WINDOW_PADDING])

        if not dpg.is_item_shown(self._jump_to_latest_window):
            dpg.show_item(self._jump_to_latest_window)

    def _page_extent(self) -> float:
        """How far one page-up/page-down moves, in pixels.

        Less than a full panel height on purpose: the overlap leaves a couple of lines of the previous view
        on screen, which is what lets a reader stitch the pages together instead of having to re-find their
        place. Same fraction the Visualizer's info panel uses, so the two apps page alike.
        """
        _, panel_h = dpg.get_item_rect_size(self.gui_parent)
        return 0.7 * panel_h

    def scroll_by_font_heights(self, delta: int) -> None:
        """Scroll by `delta` font heights; negative is up.

        The fine-adjustment gesture, for a reader whose hands are on the keyboard — which in a chat app is
        the default posture, since typing is the primary activity. (The Visualizer reaches for the mouse
        instead, because there the map *is* the interaction.)

        The unit is the font height rather than a rendered line, and the distinction is why it is named that
        way: a line box also carries the item spacing, so a line runs about a quarter taller. Callers wanting
        "a couple of lines" should ask for a couple more of these.

        What matters is not the count but that the caller's step **clears the follow-tail floor**:
        `should_follow_tail` treats anything within `_PIN_TOLERANCE_PX` of the end as still at the end, so a
        smaller scroll is undone by the next arriving chunk during a streaming reply. That floor is counted
        in the same unit, so the margin holds at any font size. See `_SCROLL_FONT_HEIGHTS_PER_ARROW` in
        `app.py` for the caller's side of it.
        """
        self.scroll_to_position(dpg.get_y_scroll(self.gui_parent) + delta * gui_config.font_size)

    def page_up(self) -> None:
        """Scroll up by one page."""
        self.scroll_to_position(dpg.get_y_scroll(self.gui_parent) - self._page_extent())

    def page_down(self) -> None:
        """Scroll down by one page.

        Deliberately does *not* pass "to the end" even when the page lands there. Reaching the end by paging
        is the reader arriving, not the reader asking to be pinned — and if they did land exactly at the end,
        `should_follow_tail` says yes on position alone, so following resumes anyway without being asserted
        here.
        """
        self.scroll_to_position(dpg.get_y_scroll(self.gui_parent) + self._page_extent())

    def get_chatlog_as_markdown(self, include_metadata: bool) -> str | None:
        """Format this linearized chat as Markdown, for e.g. copying to the clipboard or saving to a file.

        `include_metadata`: If `True`, the output will contain the node IDs, revision timestamps (ISO format), and revision numbers.

        Returns the chatlog as Markdown. If the view is empty, returns `None`.
        """
        with self.chat_controller.current_chat_history_lock:
            if not self.chat_controller.current_chat_history:
                return None

            # Read the payloads up front: the disclosure manifest describes the whole export, so it has to be
            # built before any message text is written, and it must land first in the output for a front-matter
            # parser to see it at all.
            node_payloads = [self.chat_controller.datastore.get_payload(dpg_chat_message.node_id)  # auto-selects active revision
                             for dpg_chat_message in self.chat_controller.current_chat_history]

            output_text = io.StringIO()
            output_text.write(chatutil.format_disclosure_manifest(node_payloads))
            output_text.write(f"\n# Raven-librarian chatlog\n\n- *HEAD node ID*: `{self.chat_controller.current_chat_history[-1].node_id}`\n- *Log generated*: {chatutil.format_chatlog_datetime_now()}\n\n{'-' * 80}\n\n")
            for message_number, (dpg_chat_message, node_payload) in enumerate(zip(self.chat_controller.current_chat_history, node_payloads)):
                message = node_payload["message"]
                role = message["role"]
                persona = node_payload["general_metadata"]["persona"]  # stored persona for this chat message
                text = messagetext.export_text(self.chat_controller.datastore, node_payload)
                formatted_message = messagetext.format_chat_message_for_clipboard(message_number=message_number,
                                                                                  role=role,
                                                                                  persona=persona,
                                                                                  text=text,
                                                                                  add_heading=True,  # In the full chatlog, the message numbers and role names are important, so always include them.
                                                                                  tool_name=chatutil.tool_name_of(node_payload))
                if include_metadata:
                    payload_datetime = node_payload["general_metadata"]["datetime"]  # of the active payload revision!
                    node_active_revision = self.chat_controller.datastore.get_revision(dpg_chat_message.node_id)
                    header = f"- *Node ID*: `{dpg_chat_message.node_id}`\n- *Revision date*: {payload_datetime}\n- *Revision number*: {node_active_revision}\n\n"  # yes, it'll say `None` when no node ID is available (incoming streaming message), which is exactly what we want.
                else:
                    header = ""
                output_text.write(f"{header}{formatted_message}\n\n{'-' * 80}\n\n")

            return output_text.getvalue()

    def add_streaming_message(self, node_id: str) -> "chatmessage.DPGStreamingChatMessage":
        """Append a live message rendering `node_id` to the end of the view, and return it.

        The node exists and carries whatever has streamed so far, so this is the same act as
        `add_complete_message` — rendering a node — differing only in which of the two message classes
        does it, and therefore in what the reader gets: live paragraph updates and a thinking cloud rather
        than a button row and sibling navigation.
        """
        with self.chat_controller.current_chat_history_lock:
            message = chatmessage.DPGStreamingChatMessage(gui_parent=self.chat_messages_container_group_widget,
                                                          parent_view=self,
                                                          node_id=node_id)
            self.chat_controller.current_chat_history.append(message)
        return message

    def streaming_message_for(self, node_id: str) -> "chatmessage.DPGStreamingChatMessage | None":
        """The live message rendering `node_id`, or `None` if this view is not showing one.

        Asked rather than remembered, because a rebuild replaces the widget: whoever is streaming into a
        reply holds the node's id, not the widget, and looks it up each time. `None` is the ordinary answer
        when the user is looking at another branch — there is nothing to draw into, and nothing wrong.

        Like any lookup here, the answer can go stale before the caller is done with it, and the renderer
        does *not* hold the lock while drawing — a render is a long sequence of DPG calls, and holding the
        view's lock across it would stall every rebuild behind it. What covers that is EAFP instead:
        `_render_text` runs inside `guiutils.nonexistent_ok(parent_gone_ok=True)`, so the first call to find
        its parent gone abandons the rest, which is what the render would have done had it known. Nothing is
        lost by giving up, the text having gone into the node either way.

        Deliberately unlocked, over a snapshot: `tuple(list)` is one C-level pass that cannot observe a
        half-mutated list, and that is the whole guarantee the lock would buy a reader that already accepts
        a stale answer. It has to be unlocked, too — `update_thinking_readout` calls this once per frame
        from the render thread, and a render thread that can wait on a lock a worker holds while waiting for
        a frame is a circular wait.
        """
        for message in tuple(self.chat_controller.current_chat_history):
            if isinstance(message, chatmessage.DPGStreamingChatMessage) and message.node_id == node_id:
                return message
        return None

    def rewind_to(self, node_id: str) -> int:
        """Take the message rendering `node_id` off screen along with every message after it. Returns how many went.

        For the actions that undo part of the conversation on screen before regenerating it — reroll, and
        the approve-and-retry override. Both used to walk backwards popping by index; the position is
        derived and the removal performed in one step here instead, because `build` empties and refills
        this list from a background task (the debounced resize rebuild among others), and an index derived
        before that lands somewhere else afterwards.

        Returns `0` when the message is not on screen, which is what losing that race looks like from here.

        The widgets are destroyed inside the lock, with the list update, so that the two are never observed
        disagreeing: a reader that snapshots the list between them would hold messages whose widgets are
        about to be deleted, and ask DPG about them.
        """
        with self.chat_controller.current_chat_history_lock:
            history = self.chat_controller.current_chat_history
            index = next((i for i, message in enumerate(history) if message.node_id == node_id), None)
            if index is None:
                logger.info(f"DPGLinearizedChatView.rewind_to: node '{node_id}' is not on screen; nothing to rewind.")
                return 0
            removed = history[index:]
            del history[index:]
            for message in removed:
                message.demolish()
            return len(removed)

    def _remove_first_message(self, matches: Callable[["chatmessage.DPGChatMessage"], bool]) -> None:
        """Take the first message satisfying `matches` off screen. No-op when there is none.

        By search rather than by position, and under the lock, because the caller is usually a background
        turn while the thing it is removing belongs to a view a *rebuild* may be replacing underneath it.
        Indexing into the list from a turn is what makes that a crash: `build` clears it, so "the last
        message" is briefly nothing at all.
        """
        with self.chat_controller.current_chat_history_lock:
            message = next((m for m in self.chat_controller.current_chat_history if matches(m)), None)
            if message is None:
                return
            self.chat_controller.current_chat_history.remove(message)
            message.demolish()

    def remove_message_for(self, node_id: str) -> None:
        """Take whichever message renders `node_id` off screen, if this view is showing one.

        Idempotent, and a no-op when the user is looking at another branch.
        """
        self._remove_first_message(lambda message: message.node_id == node_id)

    def remove_streaming_message_for(self, node_id: str) -> None:
        """Take the *live* message rendering `node_id` off screen, if this view is showing one.

        The narrow sibling of `remove_message_for`, and the one a turn cleaning up after its own round
        wants: a node outlives the widget that was streaming into it. `on_done` swaps the live message for
        a stored one rendering the same node, so a removal by node alone reaches past the widget it meant
        and takes the finished reply off screen — which looks from the outside exactly like the reply never
        arriving.

        Idempotent, and a no-op when the user is looking at another branch.
        """
        self._remove_first_message(lambda message: isinstance(message, chatmessage.DPGStreamingChatMessage) and message.node_id == node_id)

    def add_complete_message(self,
                             node_id: str,
                             scroll_view: bool = True,
                             start_thinking_open: bool = False) -> chatmessage.DPGCompleteChatMessage:
        """Append the chat node with `node_id` to the end of the linearized chat view in the GUI.

        `scroll_view`: If `True`, then once the message has been added, wait for it to render and scroll the
                       chat view to the end.

                       This is *unconditional*, so pass it only where jumping to the new message is the
                       expected answer to something the user just did. For a message that appears on its
                       own — an AI reply finalizing, a tool result arriving — pass `False`, and instead
                       sample `should_follow_tail` before the call and hand it to `follow_tail` after, so
                       a reader who has scrolled up is left where they put themselves.

        `start_thinking_open`: See `DPGCompleteChatMessage`. Pass `True` only for the reply that has just
                               finished generating, so that a trace the user was watching does not shut
                               itself the moment the message finalizes.
        """
        with self.chat_controller.current_chat_history_lock:
            # A linearized view shows each node once, so a node already on screen is a duplicate rather than
            # a second occurrence. Two paths append: `build`, walking the branch out of the datastore, and a
            # turn's own `on_done` / `on_tool_done` as each node is written. They race over a window that is
            # narrow but real — a rebuild landing between the node being stored and the callback firing
            # draws it, and the callback then draws it again — and flicking between branches while a turn
            # runs is exactly how a user hits it.
            #
            # Guarded here rather than at each caller because this is the one place both paths pass through.
            already_shown = next((message for message in self.chat_controller.current_chat_history
                                  if message.node_id == node_id), None)
            if already_shown is not None:
                logger.info(f"DPGLinearizedChatView.add_complete_message: node '{node_id}' is already in the view; not adding it twice.")
                if scroll_view:  # the caller still asked to be taken there, and it is on screen to be taken to
                    self.scroll_view()
                return already_shown

            dpg_chat_message = chatmessage.DPGCompleteChatMessage(gui_parent=self.chat_messages_container_group_widget,
                                                                  parent_view=self,
                                                                  node_id=node_id,
                                                                  start_thinking_open=start_thinking_open)
            self.chat_controller.current_chat_history.append(dpg_chat_message)

            # Disable the "continue generation" and "show chat continuation" buttons on the old messages.
            # The latest message already has them *enabled* if it should.
            for dpg_old_message in self.chat_controller.current_chat_history[:-1]:
                if dpg_old_message.role == "assistant":  # only AI messages have a continue button
                    dpg.disable_item(f"message_continue_button_{dpg_old_message.gui_uuid}")
                dpg.disable_item(f"message_show_chat_continuation_button_{dpg_old_message.gui_uuid}")

        self.chat_controller.search.add_matches_for(node_id)
        if scroll_view:
            self.scroll_view()
        return dpg_chat_message

    # TODO: does this `build` really belong in `DPGLinearizedChatView` or in `DPGChatController`?
    def build(self,
              head_node_id: str | None = None,
              scroll_target_node_id: str | None = None) -> None:
        """Build the linearized chat view in the GUI, linearizing up from `head_node_id`.

        `scroll_target_node_id`: If provided, scroll to this node instead of to the end.
                                 Must be the chat node ID of a message shown in the view,
                                 i.e. either `head_node_id`, or one of its ancestors.

        As side effects:

          - Update the `current_chat_history` of the chat controller this view is bound to.
          - If `head_node_id` is an AI message, update the avatar's emotion from that
            (using the node's current payload revision).
        """
        # Shutdown guard (catch-all). `build` creates chat-message widgets, and several callers reach it on
        # background threads — the startup frame callback, but also the debounced resize-rebuild task, which
        # can be *submitted* after teardown has begun and so slip past the cancel. Creating widgets once the
        # app is tearing down races `destroy_context` → segfault. `gui_updates_safe` goes False as the very
        # first action of shutdown, so bailing on it here covers every path.
        if not self.chat_controller.gui_updates_safe:
            return
        if head_node_id is None:  # use current HEAD from app_state?
            head_node_id = self.chat_controller.app_state["HEAD"]
        node_id_history = self.chat_controller.datastore.linearize_up(head_node_id)
        # An edit survives a rebuild of the branch it is on, a resize for one, and ends with a branch it is not.
        self.capture_edit_draft()
        self.gui_edit_field = None
        if self.edit_node_id is not None and self.edit_node_id not in node_id_history:
            logger.info(f"DPGLinearizedChatView.build: the message open for editing, '{self.edit_node_id}', is not on the new branch; discarding the edit.")
            self.edit_node_id = None
            self.edit_draft = None
        with self.chat_controller.current_chat_history_lock:
            # Demolished one by one rather than only dropped, although the wholesale delete below would take
            # their widgets anyway: a caller may still hold one of these instances (`get_current_message`
            # says so), and a demolished message answers `None` for its widgets instead of stale ids.
            for message in self.chat_controller.current_chat_history:
                message.demolish()
            self.chat_controller.current_chat_history.clear()
            self.chat_controller.search.clear_matches()  # refilled message by message, as the branch is added below
            dpg.delete_item(self.chat_messages_container_group_widget,
                            children_only=True)  # clear old content from GUI
            for node_id in node_id_history:
                # A reply still being written renders as a live message, carrying everything that has
                # arrived so far. It is the same act as rendering any other node — the text is in the tree
                # — which is what lets returning to a branch mid-reply look like never having left, with
                # no navigation path having to know a turn is running. Only the tail can be unfinished: a
                # node with a child was finished before the child was made.
                if node_id == node_id_history[-1] and messagetext.node_is_unfinished(self.chat_controller.datastore, node_id):
                    self.add_streaming_message(node_id)
                else:
                    self.add_complete_message(node_id=node_id,
                                              scroll_view=False)  # we scroll just once, when done
        # Update avatar emotion from the final message text (use only non-thought message content)
        role, persona, text = chatutil.get_node_message_text_without_persona(self.chat_controller.datastore, head_node_id)
        if role == "assistant":
            logger.info("DPGLinearizedChatView.build: linearized chat view new HEAD node is an AI message; updating avatar emotion from (non-thought) message content")
            text = chatutil.scrub(persona=persona,
                                  text=text,
                                  thoughts_mode="discard",
                                  markup=None,
                                  add_persona=False)
            self.chat_controller.avatar_controller.update_emotion_from_text(config=self.chat_controller.avatar_record,
                                                                            text=text)
        self.chat_controller.avatar_controller.ping(config=self.chat_controller.avatar_record)  # wake up the AI avatar when the chat view is re-rendered
        self.chat_controller.update_context_fill_indicator()  # HEAD changed (rebuild / branch switch / initial load)
        # Skip the final settle-and-scroll during shutdown: once the render loop has stopped, `split_frame`
        # blocks forever (it waits for a frame that will never come). `gui_updates_safe` goes False as the very
        # first action of teardown, so a startup `build()` that races the close bails here instead of parking.
        if self.chat_controller.gui_updates_safe:
            dpg.split_frame()
            self.scroll_view(scroll_target_node_id=scroll_target_node_id,
                             max_wait_frames=_BUILD_SCROLL_WAIT_FRAMES)

# --------------------------------------------------------------------------------
# Scaffold to GUI integration

class DPGChatController:
    def __init__(self,
                 llm_settings: env,
                 datastore: chattree.Forest,
                 retriever: hybridir.HybridIR | None,
                 app_state: env,
                 avatar_controller: "DPGAvatarController",
                 avatar_record: env,
                 themes_and_fonts: env,
                 chat_panel_widget: str | int,
                 chat_stop_generation_button_widget: str | int,
                 indicator_glow_animation: gui_animation.PulsatingColor | None,
                 docs_indexing_glow_animation: gui_animation.PulsatingColor | None,
                 think_glow_animation: gui_animation.PulsatingColor | None,
                 attachment_read_indicator_widget: str | int,
                 llm_indicator_widget: str | int,
                 docs_indexing_indicator_widget: str | int,
                 docs_indexing_progress_text_widget: str | int,
                 docs_access_indicator_widget: str | int,
                 docs_access_progress_text_widget: str | int,
                 web_indicator_widget: str | int,
                 web_progress_text_widget: str | int,
                 is_any_modal_window_visible: Callable[[], bool] | None = None,
                 avatar_panel_covered: Callable[[], bool] | None = None,
                 on_search_results_changed: Callable[[], None] | None = None,
                 on_navigate: Callable[[], None] | None = None,
                 give_caret: Callable[[str | int], None] | None = None,
                 give_keyboard_to_log: Callable[[], None] | None = None,
                 open_revision_history: Callable[[str], None] | None = None,
                 executor: concurrent.futures.Executor | None = None):
        """Controller for LLM scaffold to GUI integration.

        Owns a `DPGLinearizedChatView`, which displays the current branch of the chat.

        `llm_settings`: Obtain this by calling `raven.librarian.llmclient.setup` at app start time.

        `datastore`: The chat datastore.

        `retriever`: A `raven.librarian.hybridir.HybridIR` retriever connected to the document database.

        `app_state`: The chat's HEAD node ID, plus some persistent option flags.
                     See `raven.librarian.appstate`.

        `avatar_controller`: For TTS, and for controlling the "data eyes" effect of the avatar.

                             NOTE: In case of multiple avatars in the same app, there is still just one controller (to serialize TTS correctly).
                                   Each avatar instance has its own `avatar_record`.

        `avatar_record`: Control data for the avatar instance of the AI in this chat view.

                         See the `register_avatar_instance` method of `raven.client.avatar_controller.DPGAvatarController`.

        `themes_and_fonts`: Obtain by calling `raven.common.gui.utils.bootup` at app start time.

        `chat_panel_widget`: DPG tag or ID of the panel (child window) you want the chat to be rendered in.

        `chat_stop_generation_button_widget`: DPG tag or ID of the GUI button to interrupt the LLM (stop generating text).
                                              Will be auto-enabled only while the LLM is generating.

        `indicator_glow_animation`: When an indicator icon appears, the cycle of this animation will be reset,
                                    so that the glow always starts at the first animation frame.

                                    See `PulsatingColor` in `raven.common.gui.animation`.

        `docs_indexing_glow_animation`: Pulsator for the INDEXING indicator. Phase-reset on transition
                                        into the indexing state, so the glow always starts at the first
                                        animation frame when the indicator appears.

        `think_glow_animation`: Pulsator for the thought bubble's cloud while the model is reasoning.
                                Phase-reset when the reasoning starts, for the same reason as the two above.

                                Its own pulsator rather than a shared one, so that another owner resetting
                                theirs cannot make this one jump mid-thought.

        `attachment_read_indicator_widget`: DPG tag or ID of the widget to show while an attached document's
                                            text is being extracted. That is local work — pypdf, a couple of
                                            seconds on a branch of unread papers — and it happens *before*
                                            the backend sees anything, so it gets its own row rather than
                                            borrowing the one that means "the backend is busy".

        `llm_indicator_widget`: DPG tag or ID of the widget to show while the prompt is being processed by
                                the LLM backend. Typically, a DPG group with items bound to the theme whose
                                color `indicator_glow_animation` pulsates.

        `docs_indexing_indicator_widget`: DPG tag or ID of the widget to show while the RAG database is
                                          being *indexed*. Independent from the search indicator —
                                          indexing and search can run concurrently, so they're separate
                                          stacked rows rather than two states of one widget.

        `docs_indexing_progress_text_widget`: DPG tag or ID of a text widget inside the indexing indicator;
                                              mirrors `retriever.get_indexing_progress_text()`.

        `docs_access_indicator_widget`: DPG tag or ID of the widget to show while the database is being
                                        read: searched, automatically or by the LLM, or a document fetched.

        `docs_access_progress_text_widget`: DPG tag or ID of a text widget inside that indicator; shows the
                                            step the search or the document tool call reports.

        `is_any_modal_window_visible`: Zero-argument predicate, or `None` to skip the check. Passed to the
                                       chat view, whose scroll-end flasher abandons its fade if a modal
                                       opens. The app layer owns the list of its own dialogs, and this layer
                                       must not import it, so it arrives as a callable.

        `avatar_panel_covered`: Zero-argument predicate: has the user put something else in the avatar's
                                panel? `None` means never. When it answers `True`, the avatar and the
                                subtitles drawn in its rect are both off screen, and a reply that would be
                                captioned does not start speaking of its own accord. An explicit request
                                — `Ctrl+S`, or a message's speak button — still speaks.

                                Something standing in for an avatar that is merely asleep should not count:
                                speaking wakes the avatar, and the captions come back with its video.

                                Like `is_any_modal_window_visible`, a callable rather than a value: the app
                                layer owns its panels and this layer must not import it.

        `on_search_results_changed`: Called with no arguments when the search's match count or the current
                                     match's position changes, so the app can update its search row.

        `on_navigate`: Called with no arguments after the chat log navigates — a branch, a sibling switch, a
                       jump to a continuation, a new chat through `navigated` — as opposed to HEAD moving
                       because the conversation grew.

        `give_caret`: Called with a text field's DPG tag or ID, to put the caret there — as when a message
                      opens for editing. `None` means `raven.common.gui.animation.give_caret`. The app passes
                      its own, which also releases whatever else was holding the keyboard.

        `give_keyboard_to_log`: Called with no arguments to hand the keyboard back to the chat log, as when an
                                edit is saved or cancelled. `None` means nothing is done.

        `open_revision_history`: Called with a chat node ID when the reader clicks a message's revision
                                 number, to show that message's revisions. `None` means nothing is done.

        `web_indicator_widget`: DPG tag or ID of the widget to show while a web tool call is in progress.

        `web_progress_text_widget`: DPG tag or ID of a text widget inside the web indicator; shows the step
                                    the running web tool call reports.

        `executor`: A `ThreadPoolExecutor` or something duck-compatible with it. Used for background tasks.
        """
        # Whose glyph each message wears. Asked by both views — the chat log here, the chat graph through the
        # app — so that they cannot disagree about who wrote a message.
        self.speaker_glyphs = chattextures.SpeakerGlyphs(llm_settings)

        self.llm_settings = llm_settings
        self.datastore = datastore
        self.retriever = retriever
        self.app_state = app_state
        self.avatar_controller = avatar_controller
        self.avatar_record = avatar_record
        self.avatar_panel_covered = avatar_panel_covered if avatar_panel_covered is not None else (lambda: False)
        self.chat_stop_generation_button_widget = chat_stop_generation_button_widget
        self.indicator_glow_animation = indicator_glow_animation
        self.think_glow_animation = think_glow_animation
        self.docs_indexing_glow_animation = docs_indexing_glow_animation
        self.attachment_read_indicator_widget = attachment_read_indicator_widget
        self.llm_indicator_widget = llm_indicator_widget
        self.docs_indexing_indicator_widget = docs_indexing_indicator_widget
        self.docs_indexing_progress_text_widget = docs_indexing_progress_text_widget
        self.docs_access_indicator_widget = docs_access_indicator_widget
        # The indicators go up and down through this, so that a search over in a frame still reads as a signal.
        self.indicator_hold = guiutils.MinimumShowTime(_INDICATOR_MIN_SHOW_TIME)
        self.docs_access_progress_text_widget = docs_access_progress_text_widget
        self.web_indicator_widget = web_indicator_widget
        self.web_progress_text_widget = web_progress_text_widget

        # Indicator wiring. Show/hide events are pushed via callbacks (symmetric across all four
        # indicators: on_docs_start/done from the chat scaffold drive DOCUMENTS / SYSTEM / INTERNET; the new
        # on_indexing_start/done on the retriever drive INDEXING). So are the progress texts of DOCUMENTS
        # and INTERNET, which a turn's callbacks report. INDEXING's is polled: a commit runs in the
        # background, outside any turn, with no caller to report to.
        self._docs_indexing_progress_last = ""
        if self.retriever is not None:
            self.retriever.set_indexing_callbacks(on_start=self._on_indexing_start,
                                                  on_done=self._on_indexing_done)
        self.current_chat_history = []
        self.current_chat_history_lock = threading.RLock()

        self.on_navigate = on_navigate
        self.give_caret = give_caret if give_caret is not None else gui_animation.give_caret
        self.give_keyboard_to_log = give_keyboard_to_log if give_keyboard_to_log is not None else (lambda: None)
        self.open_revision_history = open_revision_history if open_revision_history is not None else (lambda node_id: None)

        # The keyboard mark on the current message's button row, built on first use by
        # `update_current_message_mark`. One mark that moves, rather than one per message: a chat has as
        # many messages as the user has written, so a theme apiece would grow with the conversation.
        self._current_message_mark = None

        self.gui_updates_safe = True  # At app shutdown, they aren't.

        # Sync the INDEXING indicator to any commit already in progress. The startup rescan
        # (`hybridir.setup`) can begin re-indexing before this controller exists to wire its callbacks, so
        # the 0→1 edge that fires `on_indexing_start` passes unheard — belongs with the indicator wiring
        # above, but must run after `gui_updates_safe`, which `_on_indexing_start` gates on.
        if self.retriever is not None and self.retriever.is_indexing():
            self._on_indexing_start()

        self.view = DPGLinearizedChatView(themes_and_fonts=themes_and_fonts,
                                          gui_parent=chat_panel_widget,
                                          chat_controller=self,
                                          is_any_modal_window_visible=is_any_modal_window_visible)

        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor()

        # What the live "Thinking…" readout should say: `(node_id, t0, n_chunks)` for the reply currently
        # reasoning, or `None` when none is. Published by the turn as the stream arrives and drawn once per
        # frame by `update_thinking_readout` — see there for why the drawing is not done at the same place.
        #
        # Rebound whole, never mutated, and read without the lock: a tuple swap is one bytecode, so a frame
        # sees either the old triple or the new one. Being one frame behind costs a thirtieth of a second
        # on a counter showing tenths.
        self._thinking_readout = None

        self.task_manager = bgtask.TaskManager(name="librarian_chat_controller",  # for most tasks
                                               mode="concurrent",
                                               executor=executor)
        # The pictures attachments show as, in the chat log and in the chat graph.
        self.attachment_textures = chattextures.AttachmentTextures(datastore, self.task_manager)
        # Its own manager so that a send counts as in flight from the moment it is accepted: the exchange runs the
        # user's turn before it submits the AI's, and a second send arriving in between would otherwise pass the
        # gate and start a second AI turn on top of the first.
        self.chat_exchange_task_manager = bgtask.TaskManager(name="librarian_chat_controller_chat_exchange",
                                                             mode="concurrent",
                                                             executor=executor)  # same thread pool
        self.ai_turn_task_manager = bgtask.TaskManager(name="librarian_chat_controller_ai_turn",  # for running the AI's turn, specifically (so that we can easily cancel just that one task when needed)
                                                       mode="concurrent",
                                                       executor=executor)  # same thread pool
        self.context_prefill_task_manager = bgtask.TaskManager(name="librarian_chat_controller_context_prefill",  # its own manager so a HEAD change cancels just the prefill
                                                               mode="sequential",  # only the latest HEAD's prefill matters; submitting a new one auto-cancels the previous
                                                               executor=executor)  # same thread pool
        self.search_task_manager = bgtask.TaskManager(name="librarian_chat_controller_search",
                                                      mode="sequential",  # a new search's re-highlight cancels the previous one's
                                                      executor=executor)  # same thread pool
        # The chat log's search. Asked by the app's search row, and told by the view as messages come and go.
        self.search = chatlog_search.DPGChatLogSearch(datastore=datastore,
                                                      view=self.view,
                                                      history=self.current_chat_history,
                                                      history_lock=self.current_chat_history_lock,
                                                      task_manager=self.search_task_manager,
                                                      gui_updates_safe=lambda: self.gui_updates_safe,
                                                      on_search_results_changed=on_search_results_changed)
        # The debounced idle context-prefill. `ManagedTask` supplies the pending-wait debounce (cancellable in
        # `running_poll_interval` chunks) and the single-in-flight guarantee; we just submit one per HEAD change.
        # Created only when the feature is enabled (`config.context_prefill_idle_delay is not None`).
        self.context_prefill_task = None
        if librarian_config.context_prefill_idle_delay is not None:
            self.context_prefill_task = bgtask.ManagedTask(category="raven_librarian_chat_controller_context_prefill",
                                                           entrypoint=self._context_prefill_entrypoint,
                                                           running_poll_interval=0.25,
                                                           pending_wait_duration=librarian_config.context_prefill_idle_delay)

    def mark_discontinuity(self) -> None:
        """Run the configured visual effect over the avatar, to mark that the conversation on screen changed.

        For the four places where what the user is reading is replaced by something else: stepping to a
        sibling branch, jumping to where a branch continues, starting a new chat, and rerolling a reply.

        Does nothing when `librarian_config.avatar_discontinuity_effect_enabled` is off. Call it before the
        rebuild rather than after — the rebuild is what takes the time, so the effect wants to be up while
        it happens.
        """
        if not librarian_config.avatar_discontinuity_effect_enabled:
            return
        self.avatar_controller.mark_discontinuity(config=self.avatar_record,
                                                  effect=librarian_config.avatar_discontinuity_effect,
                                                  floor=librarian_config.avatar_discontinuity_effect_floor,
                                                  ceiling=librarian_config.avatar_discontinuity_effect_ceiling)

    def find_tool_call_origin(self, tool_call_id: str) -> tuple[chatmessage.DPGChatMessage, int] | None:
        """Find the assistant message that made the tool call `tool_call_id`.

        Returns `(dpg_chat_message, index_among_that_message's_tool_calls)`, or `None` if the current branch
        holds no such call. The index is what distinguishes one call from another when an assistant turn made
        several, which is exactly when a navigation link is worth having.

        Searches `current_chat_history`, which *is* the HEAD lineage by construction — so a branched alternate's
        calls are correctly invisible here, without any filtering. Resolved per lookup rather than from a map
        built at render time: the answer depends on what is in the branch *now*, and a message's own render
        happens before the rest of the turn exists.
        """
        with self.current_chat_history_lock:
            for dpg_chat_message in self.current_chat_history:
                if dpg_chat_message.node_id is None:  # a live streaming message is not in the datastore yet
                    continue
                message = self.datastore.get_payload(dpg_chat_message.node_id)["message"]
                if message.get("role") != "assistant":
                    continue
                for index, tool_call in enumerate(message.get("tool_calls") or []):
                    if tool_call.get("id") == tool_call_id:
                        return dpg_chat_message, index
        return None

    def find_tool_response(self, tool_call_id: str) -> chatmessage.DPGChatMessage | None:
        """Find the tool-role message answering the tool call `tool_call_id`, or `None` if there is none.

        The reverse of `find_tool_call_origin`, with the same branch scoping. `None` is an ordinary outcome
        rather than an error: the call may still be in flight, its result may live on a branch other than the
        one being viewed, or an interrupted turn may have left it genuinely unanswered.
        """
        with self.current_chat_history_lock:
            for dpg_chat_message in self.current_chat_history:
                if dpg_chat_message.node_id is None:
                    continue
                message = self.datastore.get_payload(dpg_chat_message.node_id)["message"]
                if message.get("role") == "tool" and message.get("tool_call_id") == tool_call_id:
                    return dpg_chat_message
        return None

    def navigated(self) -> None:
        """Report that HEAD was moved by a navigation rather than by the conversation growing. See `on_navigate`."""
        if self.on_navigate is not None:
            self.on_navigate()

    def disable_gui_updates(self) -> None:
        """Stop the controller from firing GUI events.

        After this call:
          - `gui_updates_safe` is `False`, so any callback that gates on it (the on_docs_*,
            on_llm_*, on_tools_*, on_indexing_* handlers) becomes a no-op.
          - The retriever's indexing-lifecycle callbacks are cleared, so a cancelled `commit()`'s
            `finally` won't even reach the controller.

        Idempotent. Use as the first phase of app shutdown — run *before* `hybridir.shutdown()`
        and DPG teardown. The cancelled commit's `finally` block fires `on_indexing_done` from a
        worker thread, and any in-flight chat task can fire `on_docs_done` similarly; if those
        run while DPG widgets are already being torn down, `dpg.show/hide_item` raises against
        deleted widgets. Disabling the GUI-side hooks first sidesteps that race.

        The second phase is `shutdown()`, which drains background tasks. That has to run *after*
        `hybridir.shutdown()` because chat tasks blocked in `retriever.search` need
        `datastore_lock` to be released first.
        """
        self.gui_updates_safe = False
        if self.retriever is not None:
            self.retriever.set_indexing_callbacks(on_start=None, on_done=None)

    def cancel_tasks(self) -> None:
        """Signal all background tasks to stop, WITHOUT waiting. Idempotent.

        The non-blocking first phase of shutdown, meant to run from the app's DPG exit callback — i.e.
        from inside a render frame. A task parked in `dpg.split_frame` (e.g. the chat-streaming updater)
        can only be released by the render loop completing one more frame; waiting for it *here* would
        deadlock, because the render loop is currently sitting in the exit callback. So we only signal
        cancellation now (so the final frame releases the `split_frame` waiters, which then observe the
        flag and exit), and leave the blocking drain to `shutdown()`, called from the render loop's
        `finally` once the loop has exited.
        """
        self.disable_gui_updates()
        self.task_manager.clear(wait=False)
        self.chat_exchange_task_manager.clear(wait=False)  # before the AI turns, since an exchange submits one
        self.ai_turn_task_manager.clear(wait=False)
        self.context_prefill_task_manager.clear(wait=False)
        self.search_task_manager.clear(wait=False)

    def shutdown(self):
        """Prepare module for app shutdown.

        Second phase of shutdown: signal the background tasks to exit and wait for them.
        Calls `disable_gui_updates()` first (idempotent), so callers that haven't already
        invoked the first phase still get safe semantics.
        """
        self.disable_gui_updates()
        self.task_manager.clear(wait=True)
        self.chat_exchange_task_manager.clear(wait=True)  # before the AI turns, since an exchange submits one
        self.ai_turn_task_manager.clear(wait=True)
        self.context_prefill_task_manager.clear(wait=True)
        self.search_task_manager.clear(wait=True)

    def _on_indexing_start(self) -> None:
        """Show the INDEXING indicator. Called from `HybridIR.commit()`'s worker thread."""
        # TEMP INSTRUMENTATION: INDEXING indicator debugging (2026-04-28)
        logger.info(f"DPGChatController._on_indexing_start: INSTR entered: gui_updates_safe={self.gui_updates_safe}, widget={self.docs_indexing_indicator_widget!r}, exists={dpg.does_item_exist(self.docs_indexing_indicator_widget)}")
        if self.gui_updates_safe:
            if self.docs_indexing_glow_animation is not None:
                self.docs_indexing_glow_animation.reset()  # crisp phase on appear
            self.indicator_hold.show(self.docs_indexing_indicator_widget)
            logger.info(f"DPGChatController._on_indexing_start: INSTR after show: visible={dpg.is_item_shown(self.docs_indexing_indicator_widget)}")

    def _on_indexing_done(self) -> None:
        """Hide the INDEXING indicator. Called from `HybridIR.commit()`'s worker thread."""
        # TEMP INSTRUMENTATION: INDEXING indicator debugging (2026-04-28)
        logger.info(f"DPGChatController._on_indexing_done: INSTR entered: gui_updates_safe={self.gui_updates_safe}, widget={self.docs_indexing_indicator_widget!r}, exists={dpg.does_item_exist(self.docs_indexing_indicator_widget)}")
        if self.gui_updates_safe:
            self.indicator_hold.hide(self.docs_indexing_indicator_widget)

    def update_indexing_progress_text(self) -> None:
        """Poll the retriever's indexing progress text; mirror a change to INDEXING's text widget.

        Intended to be called once per frame from the app's `update_animations` tick. Cheap when nothing
        is changing (one string comparison), only does GUI work on change.

        Indicator visibility is push-driven via callbacks — `on_indexing_start`/`on_indexing_done` from the
        retriever. The text is polled, a commit running in the background with no caller to report to.
        """
        if self.retriever is None:
            return
        if not self.gui_updates_safe:
            return

        indexing_progress = self.retriever.get_indexing_progress_text()
        if indexing_progress != self._docs_indexing_progress_last:
            dpg.set_value(self.docs_indexing_progress_text_widget, indexing_progress)
            self._docs_indexing_progress_last = indexing_progress

    def _show_docs_access_indicator(self) -> None:
        """Show DOCUMENTS, with its progress text cleared, so the previous query's "Done" does not flash first."""
        dpg.set_value(self.docs_access_progress_text_widget, "")
        self.indicator_hold.show(self.docs_access_indicator_widget)

    def _hide_docs_access_indicator(self, maybe_notice: str | None = None) -> None:
        """Hide DOCUMENTS, saying "Done" on its way out, or `maybe_notice` for a longer while if given."""
        dpg.set_value(self.docs_access_progress_text_widget, maybe_notice or "Done")
        self.indicator_hold.hide(self.docs_access_indicator_widget,
                                 linger=(_INDICATOR_NOTICE_LINGER if maybe_notice else _INDICATOR_DONE_LINGER))

    def is_generating(self) -> bool:
        """Return whether an AI turn is currently in flight (LLM streaming or tool calls), or a send has been
        accepted and its turn has not started yet.

        Intended for GUI clients that gate an idle-throttle predicate on "something is happening", and for
        refusing a send while one is already under way.
        """
        return self.chat_exchange_task_manager.has_tasks() or self.ai_turn_task_manager.has_tasks()

    def delete_refusal(self) -> str | None:
        """Return why a delete would be refused right now, as a short reason for a button, or `None` if it would not.

        Asked by a delete button on its *first* press, so the refusal comes before the reader is asked to
        confirm something that would not happen. `delete_subtree` asks again, the answer being able to
        change between the two presses.
        """
        # Refused while a turn is in flight, and for data integrity rather than compute: the turn is writing
        # into the tree, and a subtree holding the node it writes into would be deleted from under it. A turn
        # records that node only as it starts each round, so answering exactly for one particular delete
        # would have to reason about the windows where it has not yet — a queued turn, a user message not
        # yet written. A reply is a moment away, so refusing outright is the cheap and safe answer.
        if self.is_generating():
            return "Not while a reply is being written."
        return None

    def delete_subtree(self, node_id: str) -> str | None:
        """Delete the message at `node_id` with everything below it, moving HEAD off it if it was there.

        The one route to a delete, for every view that offers one. Ask `chatutil.is_deletable` first; this
        checks only what can change between building a button and pressing it, which is `delete_refusal`.

        Rebuilds the chat log when the deletion touched the branch it shows, and leaves it alone otherwise.

        Returns `None` when done, or, when refused, a short reason for the caller to show on its button.
        """
        maybe_refusal = self.delete_refusal()
        if maybe_refusal is not None:
            logger.info(f"DPGChatController.delete_subtree: refusing to delete '{node_id}': {maybe_refusal}")
            return maybe_refusal
        new_head_node_id, branch_changed = chatutil.delete_subtree(self.datastore, node_id, self.app_state["HEAD"])
        self.app_state["HEAD"] = new_head_node_id
        if branch_changed:
            self.view.build()
        return None

    def edit_refusal(self) -> str | None:
        """Return why an edit would be refused right now, as a short reason for a button, or `None` if it would not.

        The edit counterpart of `delete_refusal`: asked when editing starts, and again by `revise_message`.
        """
        # A turn finishes the node it is writing into by replacing that node's active revision in place
        # (`overwrite_active_revision`), so an edit landing there mid-reply would be the revision replaced.
        # Refused outright rather than per node, as a delete is: a reply is a moment away.
        if self.is_generating():
            return "Not while a reply is being written."
        return None

    def revise_message(self, node_id: str, text: str) -> str | None:
        """Replace the text of the message at `node_id` by adding a revision holding `text`, and make it active.

        The one route to an edit. Ask `chatutil.is_editable` first; this checks `edit_refusal`, and refuses
        an empty `text` on a message with no attachment or tool call to keep it from being empty.

        `text`: the new text, without the persona prefix (see `chatutil.revise_message_text`).

        Redraws nothing: the message being edited is the caller's to rebuild.

        Returns `None` when done, or, when refused, a short reason for the caller to show.
        """
        maybe_refusal = self.edit_refusal()
        if maybe_refusal is not None:
            logger.info(f"DPGChatController.revise_message: refusing to edit '{node_id}': {maybe_refusal}")
            return maybe_refusal
        old_payload = self.datastore.get_payload(node_id)
        old_message = old_payload["message"]
        if not text.strip() and not (old_message.get("tool_calls") or
                                     any(part.get("type") != "text" for part in old_message.get("content") or [])):
            logger.info(f"DPGChatController.revise_message: refusing to edit '{node_id}': nothing would be left of it")
            return "Nothing would be left. To remove the message, delete it."
        revision_id = self.datastore.add_revision(node_id, chatutil.revise_message_text(old_payload, text))
        logger.info(f"DPGChatController.revise_message: node '{node_id}' is now at revision {revision_id}.")
        self.update_context_fill_indicator()  # the branch's text changed
        return None

    def _revision_refusal(self, node_id: str) -> str | None:
        """Why a revision of `node_id` may not be shown or deleted right now, or `None`. See `edit_refusal`."""
        maybe_refusal = self.edit_refusal()
        if maybe_refusal is not None:
            return maybe_refusal
        if self.view.edit_node_id == node_id:  # the field would go on holding text from the old revision
            return "Close the editor first."
        return None

    def show_revision(self, node_id: str, revision_id: int) -> str | None:
        """Make `revision_id` the revision of `node_id` the chat shows, and redraw that message.

        Refused as an edit is, and while that message is open for editing. Returns `None` when done, or,
        when refused, a short reason for the caller to show.
        """
        maybe_refusal = self._revision_refusal(node_id)
        if maybe_refusal is not None:
            logger.info(f"DPGChatController.show_revision: refusing to show revision {revision_id} of '{node_id}': {maybe_refusal}")
            return maybe_refusal
        self.datastore.set_revision(node_id, revision_id)
        self._revision_changed(node_id)
        return None

    def delete_revision(self, node_id: str, revision_id: int) -> str | None:
        """Delete revision `revision_id` of `node_id`, permanently, and redraw that message.

        Deleting the revision on screen shows the next newer one, or the newest if it was the newest. The
        only revision of a message cannot be deleted; delete the message instead.

        Refused as `show_revision` is. Returns `None` when done, or, when refused, a short reason.
        """
        maybe_refusal = self._revision_refusal(node_id)
        if maybe_refusal is None and len(self.datastore.get_revisions(node_id)) == 1:
            maybe_refusal = "The only revision. To remove the message, delete it."
        if maybe_refusal is not None:
            logger.info(f"DPGChatController.delete_revision: refusing to delete revision {revision_id} of '{node_id}': {maybe_refusal}")
            return maybe_refusal
        self.datastore.delete_revision(node_id, revision_id)
        self._revision_changed(node_id)
        return None

    def _revision_changed(self, node_id: str) -> None:
        """Redraw the message showing `node_id`, if it is on screen, after its active revision may have changed."""
        maybe_message = self.view.find_message(node_id)
        if maybe_message is not None:
            maybe_message.rebuild_in_place()
        self.update_context_fill_indicator()

    def get_current_message(self) -> chatmessage.DPGChatMessage | None:
        """Return the `DPGChatMessage` the per-message hotkeys act on, or `None` if the view is empty.

        **The bottommost message whose button row is fully on screen**, and failing that, the bottommost
        message that is on screen at all. For a chat scrolled to the end that is the last message; once the
        reader has scrolled back it is not, which is exactly when the difference matters — a reroll aimed
        at a message off the bottom of the screen is an edit nobody can see happening.

        **The answer can go stale before you use it.** `DPGLinearizedChatView.build` may demolish the
        returned message a moment later, from a background task, and no lock held inside here outlives the
        return. A caller for whom that matters should hold `current_chat_history_lock` across both this
        call and what it does with the answer; the lock is reentrant, so this taking it again is free.

        Today's callers need no such thing, and for a reason worth stating rather than relying on: a
        demolished message answers `None` for its widgets and `{}` for its button callbacks, so reading
        either out of a stale one is a no-op rather than a stale action on a dead widget.
        """
        # **The button row is the criterion rather than the message**, because the mark that says which
        # message this is lives *in* that row. "The bottommost partially visible message" was the first
        # rule here, and reading a long one put its row below the fold — so the hotkeys had a target and
        # the screen said nothing about which it was.
        #
        # The fallback is that same first rule, and it is not a leftover: a message taller than the panel
        # covers the whole view, so no button row is on screen at all and there is nothing else the keys
        # could sensibly act on. The mark is then invisible, which is honest — there is no row to put it in
        # — and it reappears as soon as one comes into view.
        # A copy, and deliberately **without the lock**, because this runs once per frame from the render
        # thread. Taking the lock here deadlocks the app: `DPGLinearizedChatView.build` holds it while doing
        # DPG work that needs frames to complete, so a render thread waiting on it is a render thread not
        # completing the frame that would release it. Measured the hard way — the GUI came up with both
        # panels blank, and `py-spy` put the main thread on that `with` line.
        #
        # No lock is needed for the copy to be coherent: building a tuple from a list is one C-level pass
        # that never releases the GIL, so no other thread can mutate the list part-way through it. What the
        # copy can be is an instant out of date, which costs a frame of the mark sitting on a message that
        # was just replaced.
        #
        # Iterating the live list instead would not crash — the list iterator bounds-checks, so a concurrent
        # `clear` ends the loop early rather than raising — but it can be read *torn*, half of it from before
        # a rebuild and half from after.
        history = tuple(self.current_chat_history)
        if not history:
            return None

        # The snapshot is a moment old, and the widgets it names can be deleted while this reads them — a
        # branch switch rebuilds the view from another thread, and every widget in the old list goes. Asking
        # DPG where a deleted widget is raises, and this runs on the render thread once per frame, so an
        # unguarded raise takes the app down with it. (Live: a sibling switch, 2026-08-27.)
        #
        # EAFP rather than a lock, deliberately: a lock here is what deadlocked startup, the render thread
        # waiting on a rebuild that was itself waiting for a frame. But a snapshot is only half the remedy —
        # it makes the *list* safe to iterate and says nothing about the widgets in it — and this is the half
        # that was missing.
        #
        # Giving up costs one frame of the keyboard mark, and the rebuild that invalidated the answer is
        # about to ask again anyway.
        with guiutils.nonexistent_ok() as nok:
            return self._current_message_in(history)
        if nok.errored:
            logger.debug("DPGChatController.get_current_message: the view was rebuilt while looking; no answer this frame.")
        return None

    def _current_message_in(self, history: tuple) -> chatmessage.DPGChatMessage | None:
        """The body of `get_current_message`, over a snapshot its caller has taken. See there."""
        _, panel_y = guiutils.get_widget_pos(self.view.gui_parent)
        _, panel_h = guiutils.get_widget_size(self.view.gui_parent)
        top_y = panel_y
        bottom_y = panel_y + panel_h

        by_row = {message.gui_buttons_group: message for message in history if message.gui_buttons_group is not None}

        # A binary search needs its criterion to go false→true down the list, and visibility goes the other
        # way — so each step below asks the complement, which has the same threshold, and takes the last
        # widget that fails it.
        #
        # *Partially below the bottom edge* is the complement of *ends at or above it*, so `direction="left"`
        # gives the bottommost row that fits entirely above the fold. That is the row a mark can be drawn in
        # whole, which is the point of choosing it.
        def hangs_past_the_bottom(widget):
            return widgetfinder.is_partially_below_target_y(widget, target_y=bottom_y)

        row = widgetfinder.binary_search_widget(widgets=list(by_row.keys()),
                                                accept=hangs_past_the_bottom,
                                                consider=None,  # every entry is a button row; no confounders to step over
                                                skip=None,
                                                direction="left")
        # A row that clears the bottom edge may still be above the *top* one, and then it is not on screen
        # either — which is the case where a single message covers the view, since every row is then either
        # above it or below it.
        if row is not None and widgetfinder.is_completely_above_target_y(row, target_y=top_y) is None:
            return by_row[row]

        def is_below_the_fold(widget):
            return widgetfinder.is_completely_below_target_y(widget, target_y=bottom_y)

        widget = widgetfinder.binary_search_widget(widgets=[message.gui_container_group for message in history],
                                                   accept=is_below_the_fold,
                                                   consider=None,
                                                   skip=None,
                                                   direction="left")
        if widget is None:  # every message is below the fold, which a clamped scroll position should prevent
            return history[-1]
        for message in history:
            if message.gui_container_group == widget:
                return message
        return history[-1]

    def update_thinking_readout(self) -> None:
        """Advance the live "Thinking… 12.4s, ~480t" counter. Call once per frame, from the render loop.

        Per frame because it is a *clock*, and a clock is judged by its cadence rather than by its accuracy:
        one that jumps by a second and then sits still for four looks broken even while every value it shows
        is correct. The stream cannot supply that cadence — the model decides when chunks arrive, and there
        may be seconds between them precisely when the reader most wants to see the counter move — so the
        turn publishes the numbers and the frame clock draws them.

        Cheap: a tuple read, and a `set_value` only when the tenth of a second on display actually changes.
        """
        readout = self._thinking_readout
        if readout is None:
            return
        node_id, t0, n_chunks = readout
        message = self.view.streaming_message_for(node_id)
        if message is not None:
            message.set_thinking_progress(time.monotonic() - t0, n_chunks)

    def update_current_message_mark(self) -> None:
        """Move the keyboard mark onto the current message's button row. Call once per frame.

        Per frame rather than on a scroll event, because the current message changes with the scroll
        position however that position came about — a wheel, a drag, a keypress, a streamed reply growing
        the content, or a rebuild — and the mark has to agree with `get_current_message` at the instant a
        hotkey is pressed rather than shortly afterwards.
        """
        if self._current_message_mark is None:
            # The tooltip goes on the dot, which is a widget built for the mark and has none of its own —
            # not on the button row, where it would be a second tooltip over buttons that each carry one.
            self._current_message_mark = keyboardmark.Mark(None,
                                                           kind=keyboardmark.MarkKind.DOT,
                                                           tooltip="Message-specific hotkeys go to this message")
        message = self.get_current_message()
        target = message.gui_keyboard_mark_widget if message is not None else None
        self._current_message_mark.target = target
        self._current_message_mark.lit = (target is not None)

    def _render_context_fill(self, count: int, is_exact: bool) -> None:
        """Set the bottom-toolbar context-fill readout text from a token `count`. Low-level; does no scheduling.

        `is_exact` drives the typography: `X%` when the count is exact (a local tokenizer, ooba's token-count
        endpoint, or a backend-reported `prompt_tokens` from `_context_prefill_task`), `~X%` when it is a
        calibrated estimate.
        """
        if not self.gui_updates_safe:
            return
        context_length = self.llm_settings.context_length
        percent = round(100 * count / context_length) if context_length else 0
        prefix = "" if is_exact else "~"
        with guiutils.nonexistent_ok():  # the readout widget may vanish under a shutdown race (background prefill caller)
            dpg.set_value("context_fill_text", f"{prefix}{percent}%  ({count} / {context_length})")  # tag

    def refresh_system_injects_if_stale(self) -> None:
        """Redraw the system message if the injects it shows no longer match what a request would carry.

        The system message displays both kinds of system inject live — the preamble ahead of the stored
        prompt and the postamble after it (see `DPGCompleteChatMessage._render_system_preamble` and
        `_render_system_postamble`) — and one of the postamble's is the date. A session left open across
        midnight would otherwise send the new date on the wire while the log still showed the old one -
        the exact divergence that displaying them at all is meant to remove.

        Called at the start of a turn, which is when the wire value is recomputed, so the two change
        together. Between turns the display can lag a rollover; nothing is being sent then, and the next
        turn or view rebuild corrects it.
        """
        if self.llm_settings is None:
            return
        with self.current_chat_history_lock:
            if not self.current_chat_history:
                return
            message = self.current_chat_history[0]  # the system prompt is the branch root
            if message.rendered_system_postamble is None:  # not a system message, or drawn before connecting
                return
            current_preamble = scaffold.build_system_preamble(llm_settings=self.llm_settings)
            current_postamble = scaffold.build_system_postamble(llm_settings=self.llm_settings)
            if (current_preamble == message.rendered_system_preamble and
                    current_postamble == message.rendered_system_postamble):
                return
            logger.info("DPGChatController.refresh_system_injects_if_stale: system injects changed since they were drawn (most likely the date rolled over); redrawing the system message.")
            message.rebuild_in_place()

    def update_context_fill_indicator(self) -> None:
        """Refresh the bottom-toolbar context-fill readout: the current chat's token size vs the loaded window.

        Two-stage: this immediate pass counts the branch locally via `llmclient.count_branch_tokens` (which see
        for what is and is not counted, and when the figure is exact), and then schedules a debounced background
        prefill (`_schedule_context_prefill`) that, once the chat settles, replaces the estimate with the
        backend's exact full-prompt `prompt_tokens` — and warms the KV cache on the way.
        """
        if not self.gui_updates_safe:
            return
        try:
            # **`extract_attachments=False` is what keeps this off the critical path.** This runs on every
            # HEAD change, and a HEAD change happens inside a DPG callback — so extracting an attached PDF
            # here (pypdf, seconds for a large one) holds the callback thread, and every key pressed
            # meanwhile queues behind it. Measured 2026-08-21: switching to a branch with three PDFs took
            # 3038 ms, against 38 ms for one with nothing to extract, and the app read as frozen throughout.
            #
            # The cost of skipping is an undercount for a moment, shown as `~X%`, which the debounced
            # prefill below then replaces with the backend's exact figure. Trading a transient wrong number
            # for a transient dead keyboard is the right way round: the number corrects itself and says it
            # is approximate while it is wrong, where the freeze reads as the app having crashed.
            count, is_exact = llmclient.count_branch_tokens(self.llm_settings, self.datastore, self.app_state["HEAD"],
                                                            extract_attachments=False)
            self._render_context_fill(count, is_exact)
        except Exception:  # noqa: BLE001 -- a status readout must never break the GUI or a chat turn
            logger.exception("DPGChatController.update_context_fill_indicator: failed to update the context-fill readout")
        self._schedule_context_prefill()

    def _schedule_context_prefill(self) -> None:
        """(Re)arm the debounced background context-prefill for the current HEAD.

        Submits a `ManagedTask`; the sequential `TaskManager` auto-cancels the previous pending/in-flight prefill
        (a HEAD change invalidates it), so this is safe to call from every HEAD-change site — it's driven from
        `update_context_fill_indicator`. The actual backend round-trip happens only after the `ManagedTask`'s
        pending wait (`config.context_prefill_idle_delay` seconds of quiet); see `_context_prefill_entrypoint`.
        No-op when the feature is disabled (the task wasn't created).
        """
        if not self.gui_updates_safe:
            return
        if self.context_prefill_task is None:  # feature disabled (config.context_prefill_idle_delay is None)
            return
        # The abort handle is what makes a superseded prefill actually stop. The `TaskManager` already
        # cancels the previous one whenever a new HEAD supersedes it, but that cancellation is a flag, and
        # a prefill blocked in the backend read cannot look at a flag until the read returns — up to a
        # minute on a heavy branch, during which the user's next turn queues behind work whose only product
        # was a warm cache for a branch they have left. `on_cancel` fires the handle, which ends the read.
        maybe_abort = netutil.Abort()
        self.context_prefill_task_manager.submit(self.context_prefill_task,
                                                 env(wait=True,
                                                     head_node_id=self.app_state["HEAD"],
                                                     maybe_abort=maybe_abort,
                                                     on_cancel=lambda task_env: task_env.maybe_abort.abort()))

    def _context_prefill_entrypoint(self, task_env: env) -> None:
        """`ManagedTask` entrypoint: after the idle debounce, ask the backend for the exact prompt size of the captured branch.

        The pending-wait debounce and cancel-on-resubmit are handled by the `ManagedTask` / sequential-`TaskManager`
        machinery; we reach here only once the wait has elapsed without a newer HEAD superseding us. Sends the
        linearized branch to the backend via `llmclient.prefill` (generates ~nothing, but reports the exact templated
        `prompt_tokens` and warms the KV cache). On success, upgrades the indicator to `X%`.

        Bails (leaving the estimate in place) if cancelled, if the app is shutting down, if a real generation is in
        flight (that turn warms the cache and reports its own exact count), or if HEAD has moved off the branch this
        task captured — including a final re-check after the round-trip, so a late reply can't overwrite a newer
        branch's readout.

        Those checks happen between steps, so they cannot end a round-trip already under way. `task_env.maybe_abort`
        is what does that: cancelling this task fires it, the backend read returns at once, and `prefill` answers
        `None` like any other unanswered prefill.
        """
        if task_env.cancelled or not self.gui_updates_safe or self.is_generating():
            return
        if self.app_state["HEAD"] != task_env.head_node_id:  # HEAD moved during the idle wait
            return

        history = chatutil.linearize_chat(datastore=self.datastore,
                                          node_id=task_env.head_node_id)

        # Read the attachments and re-estimate *before* asking the backend anything, and show that. Until
        # this point the readout is whatever the immediate count could say without waiting for pypdf, which
        # on a branch of unread PDFs is a small fraction of the truth — measured at ~1% for a branch that is
        # two-thirds full. Extraction has to happen for the prompt below in any case and `sidecar_to_text`
        # memoizes it, so doing it here costs nothing and buys the honest figure a whole round-trip earlier:
        # against an 88500-token prompt that round-trip was ~5 s, and the extraction ahead of it is the only
        # part the user now spends looking at a wrong number.
        #
        # It also stands in for the exact figure when the backend never answers, which is the case that used
        # to leave the readout stuck at the immediate count until HEAD moved.
        # Only *say* we are reading if there is something to read. This runs on the idle prefill after every
        # reply, and the counting below happens either way - so signalling it unconditionally lit READING and
        # the avatar's data eyes for a moment on every turn, including in chats with no attachments at all.
        # Reported from the running app 2026-08-25: "a stray data eyes light-up after the model replied".
        #
        # `sidecar_text_if_extracted` is the question asked without paying for the answer, which is what
        # makes this affordable here: `None` means not extracted yet.
        reading_something = any(textfilestore.sidecar_text_if_extracted(part["text_file"]["url"]) is None
                                for message in history
                                for part in message.get("content", [])
                                if isinstance(part, dict) and part.get("type") == "text_file")

        if reading_something and self.gui_updates_safe:
            if self.indicator_glow_animation is not None:
                self.indicator_glow_animation.reset()  # start a new pulsation cycle
            self.indicator_hold.show(self.attachment_read_indicator_widget)  # tag
            # Reading an attached document is the system consulting an external source, the same as a web
            # fetch or a document search - so the avatar shows it the same way. The effect nests, which
            # matters here specifically: this runs on a background task and can overlap a turn's tool call.
            self.avatar_controller.start_data_eyes(config=self.avatar_record)
        try:
            estimate, estimate_is_exact = llmclient.count_branch_tokens(self.llm_settings, self.datastore, task_env.head_node_id)
        finally:
            if reading_something and self.gui_updates_safe:
                self.indicator_hold.hide(self.attachment_read_indicator_widget)  # tag
                self.avatar_controller.stop_data_eyes(config=self.avatar_record)

        if task_env.cancelled or not self.gui_updates_safe:
            return
        if self.app_state["HEAD"] != task_env.head_node_id:  # HEAD moved while we were reading the attachments
            return
        self._render_context_fill(estimate, estimate_is_exact)

        # The tool settings must match what the next turn will send, so the tool definitions are counted and
        # cached identically. They sit in the system block at the very front of the prompt, so warming a
        # different list warms a prefix that turn never sends — the whole prompt gets reprocessed anyway.
        maybe_tool_names = llmclient.maybe_tool_names_for_turn(
            self.llm_settings,
            documents_available=(self.app_state["docs_enabled"] and self.retriever is not None),
            internet_available=self.app_state["internet_enabled"])
        # SYSTEM means "the backend is reading a prompt and has emitted nothing yet" — that is what the turn
        # path uses it for (`on_llm_start` raises it, the first content chunk drops it). A prefill is the same
        # activity on a different trigger, so it says so too (Juha, 2026-08-25). Only around the request: the
        # extraction above is local work, and claiming the backend is busy during it would be a lie about
        # where the time goes.
        if self.gui_updates_safe:
            if self.indicator_glow_animation is not None:
                self.indicator_glow_animation.reset()  # start a new pulsation cycle
            self.indicator_hold.show(self.llm_indicator_widget)  # tag
        try:
            out = llmclient.prefill(self.llm_settings,
                                    # As the next turn will begin, up to its last user message: see
                                    # `scaffold.build_prefill_prompt` for why it stops there.
                                    scaffold.build_prefill_prompt(self.llm_settings, history),
                                    # All the per-group gating is in `maybe_tool_names` now, so this coarser
                                    # switch has nothing left to decide and stays on. It is not redundant at
                                    # its own layer: `ai_turn` still sets it `False` to withdraw the tools
                                    # outright when the round budget is spent — which cannot happen at prefill
                                    # time, since what is being warmed is the *first* round of the next turn.
                                    tools_enabled=True,
                                    tool_names=maybe_tool_names,
                                    datastore=self.datastore,  # resolve any sidecar: image refs so the exact prompt size counts image tokens
                                    maybe_abort=task_env.maybe_abort)
        finally:
            # Not if a turn started while we were waiting: it raised the same indicator for its own prompt,
            # and dropping it here would report that turn as further along than it is.
            if self.gui_updates_safe and not self.is_generating():
                self.indicator_hold.hide(self.llm_indicator_widget)  # tag

        if task_env.cancelled or not self.gui_updates_safe:
            return
        if out is None or out.usage is None or out.usage.get("prompt_tokens") is None:
            return  # backend didn't report usage; keep the estimate
        if self.app_state["HEAD"] != task_env.head_node_id:  # branch switched while we were waiting on the backend
            return
        # Checked against the local estimate before it is believed, because a backend may be reporting the
        # tokens it had to *process* rather than the size of the prompt — see `prompt_size_report_looks_whole`.
        # The estimate is the one already computed and shown above, so a refused figure simply leaves that
        # standing rather than replacing it with an identical recount.
        # What was sent stops at the last user message, so the backend counted the branch up to there; the
        # rest, usually the last reply, is added from the local estimate. Shown as exact while that tail is a
        # small part of the whole, which it is unless the last turn brought in a lot - fetched documents, say.
        reported = out.usage["prompt_tokens"]
        node_ids = self.datastore.linearize_up(task_env.head_node_id)
        cut_index = scaffold.prefill_cut_index(history)
        prefix_estimate = (llmclient.count_branch_tokens(self.llm_settings, self.datastore, node_ids[cut_index])[0]
                           if cut_index >= 0 else 0)
        if not llmclient.prompt_size_report_looks_whole(reported, prefix_estimate):
            return  # `prompt_size_report_looks_whole` logs why; the estimate is already on screen, so leave it there
        tail_estimate = max(0, estimate - prefix_estimate)
        total = reported + tail_estimate
        is_exact = readout_is_exact(tail_estimate, total, history[cut_index + 1:],
                                    tokenizer_loaded=(self.llm_settings.tokenizer is not None))
        logger.info(f"DPGChatController._context_prefill_entrypoint: prompt size for HEAD '{task_env.head_node_id}': {reported} tokens "
                    f"counted by the backend up to the last user message, plus ~{tail_estimate} estimated after it")
        self._render_context_fill(total, is_exact=is_exact)

    def chat_exchange(self, user_message_text: str, staged_images: list[env] | None = None,
                      staged_files: list[env] | None = None) -> None:
        """Run one exchange: the user's turn, then the AI's.

        `user_message_text`: What the user wrote.

                             If `user_message_text` is the empty string *and* nothing is attached (no images and
                             no documents), the AI will generate another message without the user writing in
                             between — if `librarian_config.llm_allow_empty_send` is on. Otherwise such an
                             exchange does nothing.

        `staged_images`: Images the user attached to this message, or `None`. Each entry is an `env` with `raw`
                         (image bytes), `provenance_url`, and `provenance_source` (see `scaffold.user_turn`).
                         An attachment counts as user content: with images present, an exchange runs even when
                         the text is empty (rather than being treated as "let the AI take another turn").

        `staged_files`: Documents (plain text / PDF) the user attached, or `None` — the file counterpart of
                        `staged_images` (see `scaffold.user_turn`). Also counts as user content: an exchange runs
                        with attachments present even when the text is empty.

        The RAG query (for document database search) is taken from the latest available user message:

          - `user_message_text` if not the empty string.
          - Otherwise, the latest user message on the branch this turn answers on
            (`chatutil.latest_user_message_text`). The *branch*, not the view: a reader who has navigated
            away is looking at a different conversation, and this turn is still answering theirs.

        This spawns a background task to avoid hanging GUI event handlers,
        since the typical use case is to call `chat_exchange` from a GUI event handler.
        """
        if not (user_message_text or staged_images or staged_files) and not chatutil.empty_send_allowed(
                self.datastore, self.app_state["HEAD"], librarian_config.llm_allow_empty_send):
            logger.info("chat_exchange: empty message and nothing attached, HEAD is neither a user message nor a tool result, and `llm_allow_empty_send` is off; ignoring.")
            return

        def chat_exchange_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            # Add the user's message to the chat if the user entered any text or attached anything.
            if user_message_text or staged_images or staged_files:
                self.user_turn(text=user_message_text, staged_images=staged_images, staged_files=staged_files)
                # NOTE: Rudimentary approach to RAG search, using the user's message text as the query. (Good enough to demonstrate the functionality. Improve later.)
                docs_query = user_message_text or None  # image-only message: no text to search docs with
            else:
                # Handle the RAG query: find the latest existing user message. Asked of the *branch* rather
                # than of the view — this turn answers on the branch HEAD names, and the view shows whatever
                # branch the reader is on, which during a turn need not be the same one.
                docs_query = chatutil.latest_user_message_text(self.datastore, self.app_state["HEAD"])
                if docs_query is None:
                    # Taking another turn needs a user turn to take it *about*. With nothing said yet, the only
                    # user-role content reaching the model would be our own temporary injects — so it answers
                    # those, discussing its own instructions instead of talking to anyone. A stray Enter in an
                    # untouched chat is enough to land here, so do nothing, which is what a stray Enter should do.
                    logger.info("chat_exchange: empty message, nothing attached, and no user message in this chat. Nothing to continue from; ignoring.")
                    return
            if task_env.cancelled:  # during user turn
                return
            self.ai_turn(docs_query=docs_query,
                         continue_=False)
        self.chat_exchange_task_manager.submit(chat_exchange_task, env())

    def user_turn(self, text: str, staged_images: list[env] | None = None,
                  staged_files: list[env] | None = None) -> str:
        """Run the user's turn: create the user message node, update HEAD, append it to the view.

        Returns the new HEAD node id.

        Runs **synchronously on the caller's thread** — deliberately not as a task of its own, and deliberately
        asymmetric with `ai_turn`, which *is* task-based (see its docstring for why that one must be). The AI
        turn that follows in the same exchange must observe the completed user turn (its message node as the new
        HEAD, its sidecar images already written, and the message already in the view); if the two ran as
        separate concurrent tasks, that ordering would be a race — invisible while the AI turn takes seconds to
        reach its first output, but wrong the instant the backend errors immediately (the AI's error message
        would append before the user's message, and could even be parented to the pre-user HEAD). So
        `chat_exchange` calls this inline, then submits the AI turn.

        Call from a background thread (as `chat_exchange` does), never directly from a GUI event handler — it does
        datastore and (with attachments) image work. That constraint is exactly why this needs no task of its
        own: unlike `ai_turn`, it is never invoked straight from the GUI, so there is no GUI thread to keep free.

        `staged_images`: Images the user attached, or `None`. Passed through to `scaffold.user_turn`, which
                         stores each as a datastore sidecar (decode/downsample happens here, off the GUI thread).
        `staged_files`: Documents (plain text / PDF) the user attached, or `None`. Passed through to
                        `scaffold.user_turn`, which stores each verbatim as a datastore sidecar.
        """
        new_head_node_id = scaffold.user_turn(llm_settings=self.llm_settings,
                                              datastore=self.datastore,
                                              head_node_id=self.app_state["HEAD"],
                                              user_message_text=text,
                                              staged_images=staged_images,
                                              staged_files=staged_files)
        self.app_state["HEAD"] = new_head_node_id  # update HEAD before the AI turn reads it as the parent
        self.view.add_complete_message(new_head_node_id)
        self.update_context_fill_indicator()  # user message added -> context grew
        return new_head_node_id

    def ai_turn(self,
                docs_query: str | None,
                continue_: bool,
                _retry_tool_node_id: str | None = None) -> None:
        """Run the AI's turn: the reply, including the whole tool loop.

        Spawns a background task (on its own `ai_turn_task_manager`) — deliberately, and deliberately asymmetric
        with `user_turn`, which runs synchronously. Three reasons this one must be tasked, none of which apply to
        `user_turn`:
          1. It is invoked *directly from GUI event handlers* — reroll, continue, and "approve denied host &
             retry" all call `ai_turn` from the DPG callback thread, which must return at once. (`user_turn` is
             only ever called from inside `chat_exchange`'s task, already off the GUI thread.)
          2. It needs *independent cancellation* — the Stop button clears just `ai_turn_task_manager`
             (`stop_ai_turn`), interrupting the AI turn without disturbing any other task.
          3. It is *long-running* — LLM streaming, tool calls, web fetches — the actual reason GUI responsiveness
             is at stake here.
        The underlying `scaffold.ai_turn` is itself synchronous; the tasking is the controller's concern (the CLI
        client `minichat` calls `scaffold.ai_turn` straight, and blocks, which is right for a REPL).

        `docs_query`: Query for RAG document database, or `None` for no search. Search results are auto-injected before the LLM replies.

        `continue_`: If `False`, create a new AI message. Most of the time, this is what you want.
                     If `True`, continue the AI's current message.

        `_retry_tool_node_id`: Internal. If set, this is the "approve denied host & retry" override: instead
                               of a normal AI turn, re-run the previously-denied tool call at this node on a
                               new branch (`scaffold.retry_tool_calls`) and continue from there. The same GUI
                               callback bundle is reused; `docs_query`/`continue_` are ignored in this mode.
        """

        def ai_turn_task(task_env: env) -> None:
            if task_env.cancelled:  # while the task was in the queue
                return

            # The branch this turn is answering on. Captured here rather than at submit time because the
            # task may have waited in the queue, and it is the branch we are about to *read* that this turn
            # belongs to. Every later comparison is against this, updated as the turn writes (`advance_head`).
            task_env.expected_head = self.app_state["HEAD"]
            # The node this turn's current round is writing into, set by `on_llm_start`. `None` until the
            # first round starts, which is the window a turn cancelled while still queued dies in.
            task_env.ai_node_id = None
            # That node's parent, remembered alongside it so HEAD has somewhere to go if the node is taken
            # back. See `on_llm_start` and the `Aborted` handler.
            task_env.round_parent_node_id = None

            # A live turn supersedes any pending idle-prefill: it warms the KV cache itself and reports its own
            # exact `prompt_tokens`, so a concurrent prefill round-trip would be wasted (and would contend with
            # the real request on a single-model backend).
            self.context_prefill_task_manager.clear()

            if self.gui_updates_safe:
                dpg.enable_item(self.chat_stop_generation_button_widget)

            # Whether this reply speaks itself. Grabbed once, in case the user toggles something while the
            # turn is being processed.
            #
            # Subtitles are a text widget positioned in the avatar's rect, so while another panel holds
            # that rect they are simply not on screen — and a reply that speaks with its captions missing
            # is precisely what someone who switched captions on cannot use. So it waits, silently, rather
            # than speaking uncaptioned. With captions off there is nothing to lose and it speaks as usual,
            # the only channel the panel covers then being lipsync, which is paused anyway.
            #
            # This is about speech *starting on its own*. An explicit request — `Ctrl+S`, a message's speak
            # button — is honoured whatever holds the panel: it was asked for, and the reader can see for
            # themselves where the captions went.
            captions_would_be_hidden = self.app_state["avatar_subtitles_enabled"] and self.avatar_panel_covered()
            speak_this_turn = self.app_state["avatar_speech_enabled"] and not captions_would_be_hidden

            try:
                def streaming_widget() -> "chatmessage.DPGStreamingChatMessage | None":
                    """The widget this round is streaming into, or `None` when the view is not showing it.

                    Looked up per use rather than held, which is the whole point of the reply being a node:
                    a rebuild replaces the widget, and the next chunk simply finds the new one. `None` means
                    the user is looking at another branch — nothing to draw into, and nothing wrong.
                    """
                    if task_env.ai_node_id is None:  # no round has started, so there is no node to render
                        return None
                    return self.view.streaming_message_for(task_env.ai_node_id)

                def drop_streaming_widget() -> None:
                    """Take this round's live message off screen, wherever it ended up. Safe before one exists.

                    Only the live one: once `on_done` has swapped in the stored rendering of the same node,
                    there is nothing here left to drop.
                    """
                    if task_env.ai_node_id is None:
                        return
                    self.view.remove_streaming_message_for(task_env.ai_node_id)

                def turn_owns_the_view() -> bool:
                    """Whether the chat on screen is the branch this turn is writing to.

                    A turn is allowed to keep running when the user navigates away — it finishes on its own
                    branch, and the reply is there when they come back — but what it does to the *view* has
                    to stop while they are elsewhere.

                    "HEAD has not moved" would be the wrong question, because this turn is itself the thing
                    that moves HEAD. The comparison is against where *this turn* last left it.
                    """
                    return self.app_state["HEAD"] == task_env.expected_head

                def advance_head(node_id: str) -> None:
                    """Move HEAD to a node this turn has just written, and keep the guard in step with it."""
                    self.app_state["HEAD"] = node_id
                    task_env.expected_head = node_id

                # The turn's own data-eyes uses, counted so that teardown can release exactly those.
                #
                # The effect is reference-counted across the app, so a bare "make sure it is off" at the end
                # of a turn would decrement whatever else is holding it - an attachment being read on a
                # background task, most likely - and switch the eyes off under it. `scaffold` calls the
                # `..._done` callbacks outside a `finally`, so a turn that raises really can leak a use, and
                # this is what lets teardown clean up after itself without reaching into anyone else's.
                #
                # A plain int: every one of these callbacks runs on the turn's own thread.
                turn_data_eyes_uses = 0

                def start_turn_data_eyes() -> None:
                    nonlocal turn_data_eyes_uses
                    turn_data_eyes_uses += 1
                    self.avatar_controller.start_data_eyes(config=self.avatar_record)

                def stop_turn_data_eyes() -> None:
                    nonlocal turn_data_eyes_uses
                    if turn_data_eyes_uses > 0:
                        turn_data_eyes_uses -= 1
                        self.avatar_controller.stop_data_eyes(config=self.avatar_record)

                def on_docs_start() -> None:
                    task_env.docs_notice = None
                    if self.gui_updates_safe:
                        start_turn_data_eyes()
                        if self.indicator_glow_animation is not None:
                            self.indicator_glow_animation.reset()  # crisp phase on appear
                        self._show_docs_access_indicator()

                def on_docs_progress(text: str) -> None:
                    if self.gui_updates_safe:
                        dpg.set_value(self.docs_access_progress_text_widget, text)

                def on_docs_query(status: str, maybe_query: str | None) -> None:
                    # What DOCUMENTS says on its way out, when there was no search to report "Done" for.
                    task_env.docs_notice = {"not_needed": "No search needed",
                                            "failed": "No search: query failed"}.get(status)

                def on_docs_done(matches: list[dict]) -> None:
                    if self.gui_updates_safe:
                        self._hide_docs_access_indicator(getattr(task_env, "docs_notice", None))
                        stop_turn_data_eyes()

                def on_llm_start(node_id: str) -> None:
                    # The node this round writes into. It exists already, carrying an empty message; the
                    # widget below renders it, and every later lookup goes through this id.
                    task_env.ai_node_id = node_id
                    # Where HEAD goes if this node is taken back — asked now, while the node still exists to
                    # be asked about. See the `Aborted` handler.
                    task_env.round_parent_node_id = self.datastore.get_parent(node_id)

                    # HEAD moves to it now, rather than when the reply is finished. A view is built from
                    # `linearize_up(HEAD)`, so a reply whose node is not yet on that path cannot be
                    # rendered by a rebuild however complete the node is — which is exactly what a resize
                    # mid-reply used to demonstrate, the message vanishing until the turn ended.
                    #
                    # `expected_head` follows even when the user is elsewhere, so the guard keeps naming
                    # the node this turn is actually writing; HEAD itself only moves for a reader who is
                    # here to see it. Navigating back lands on this node anyway, `descend_to_latest`
                    # finding it as the newest child — so the two agree again at that moment.
                    if turn_owns_the_view():
                        advance_head(node_id)
                    else:
                        task_env.expected_head = node_id

                    # Per round, not per turn: what this arms is "the backend is reading the prompt and has
                    # sent nothing back yet", which is true again at the start of every round of the agent
                    # loop. See `abort_if_nothing_to_lose`.
                    task_env.round_has_streamed = False

                    if not turn_owns_the_view():  # the user is elsewhere; nothing to put a new widget into
                        return

                    if self.gui_updates_safe:
                        # When continuing, take the message's previous rendering off screen: the live one
                        # added below replaces it. By node rather than by position — this runs on the turn's
                        # thread, and a rebuild may be rewriting the view's list as we look at it.
                        if continue_:
                            self.view.remove_message_for(node_id)

                        # Sampled before the new message widget exists — creating it is itself a content change.
                        follow_sample = self.view.sample_tail_follow()
                        self.view.add_streaming_message(node_id)
                        self.view.follow_tail(follow_sample)

                        if self.indicator_glow_animation is not None:
                            self.indicator_glow_animation.reset()  # start new pulsation cycle
                        self.indicator_hold.show(self.llm_indicator_widget)  # show prompt processing indicator

                task_env.text = io.StringIO()  # incoming, in-progress paragraph
                task_env.t0 = time.monotonic()  # timestamp of last GUI update
                task_env.n_chunks0 = 0  # chunks received since last GUI update

                task_env.current_is_thought = False  # which channel the in-progress paragraph belongs to (thought bubble vs visible answer)
                task_env.seen_content = False  # whether any visible-answer content has arrived yet (to fire the talking animation once)
                task_env.thinking_t0 = None  # when reasoning first arrived, for the live count on the thought bubble
                task_env.first_chunk_t = None  # when the first generated text arrived on any channel; what `thinking_t0` becomes if it turns out all of it was reasoning

                task_env.emotion_window = common_text.EmotionWindow()
                def _update_avatar_emotion_from_incoming_text(new_paragraph: str) -> None:
                    if (text := task_env.emotion_window.add(new_paragraph)) is not None:
                        logger.info(f"ai_turn.ai_turn_task._update_avatar_emotion_from_incoming_text: updating emotion from {len(text)} characters of recent text")
                        self.avatar_controller.update_emotion_from_text(config=self.avatar_record,
                                                                        text=text)

                def on_llm_progress(event: dict[str, Any]) -> sym | None:
                    """Render one streaming event, tolerating the widget disappearing mid-render.

                    `turn_owns_the_view` is a check-then-act, so it leaves a window: the user can navigate
                    away between the check and any of the DPG calls below it, and each of those is a
                    separate opportunity. `nonexistent_ok` closes the window from the other side — the first
                    call to find its item gone abandons the rest of the render, which is what the render
                    would have done anyway had it known.

                    Losing a render this way costs nothing beyond the frame: the paragraph records still
                    hold the text, so the rebuild that took the widgets away puts all of it back.
                    """
                    with guiutils.nonexistent_ok() as nok:
                        action = _render_llm_progress(event)
                    if nok.errored:
                        logger.info("ai_turn.ai_turn_task.on_llm_progress: the widget being rendered into is gone; abandoning this render.")
                        return llmclient.action_ack  # the turn continues; only the drawing stopped
                    return action

                def _render_llm_progress(event: dict[str, Any]) -> sym | None:
                    # `invoke` is the single parser; this handler is a pure renderer dispatching on the typed
                    # event. No regex-sniffing of the text stream; the event type *is* the state.

                    task_env.round_has_streamed = True  # the backend is answering, so co-operative stop can reach it

                    # Keep generating — an abandoned reader is not an abandoned turn — but stop drawing
                    # while the user is looking at a different branch. Nothing is lost by not drawing: the
                    # text is going into the node either way, so a view rebuilt later renders it from there
                    # and this picks up from whatever widget that rebuild made.
                    streaming_chat_message = streaming_widget()
                    if streaming_chat_message is None:
                        return llmclient.action_ack

                    # If the task is cancelled (`stop_ai_turn` was called), interrupt the LLM, keeping the content received so far.
                    # The scaffold will automatically send the content to `on_llm_done`.
                    if task_env.cancelled or not self.gui_updates_safe:  # the EAFP half is in the caller's `nonexistent_ok`
                        reason = "Cancelled" if task_env.cancelled else "App is shutting down"
                        logger.info(f"ai_turn.ai_turn_task.on_llm_progress: {reason}, stopping text generation.")
                        return llmclient.action_stop

                    event_type = event["type"]
                    if event_type == "tool_call":
                        # Structured tool-call invocations render when the completed message reloads. Nothing to stream live.
                        return llmclient.action_ack

                    if event_type == "reasoning_retcon":
                        # None of what we have shown as the answer was the answer: the model was inside its
                        # thinking block from the first token, and only the close arrived to say so. Move the
                        # text, then undo everything the wrong reading caused.
                        logger.info("ai_turn.ai_turn_task.on_llm_progress: reasoning arrived with no opening tag; moving the reply so far into the thought bubble.")
                        streaming_chat_message.reclassify_all_paragraphs_as_thought()
                        task_env.current_is_thought = True  # ...including the paragraph still being accumulated
                        # The thinking began with the first token, which is what the live count should have
                        # been measuring all along.
                        task_env.thinking_t0 = task_env.first_chunk_t
                        if task_env.seen_content:
                            # Clearing this re-arms the "the answer has started" trigger below, which is the
                            # half that matters: the animation is not merely stopped here, it starts again
                            # on the first chunk that really is the answer — which is the next one, the
                            # close tag having ended the thinking block.
                            task_env.seen_content = False
                            if not speak_this_turn:
                                # The generic talking animation — randomized mouth, no audio, used only when
                                # TTS is off, since otherwise lipsync drives the mouth. It says the AI is
                                # writing the visible answer, and was started on that claim. It has not
                                # started writing one yet.
                                _client_api().avatar_stop_talking(self.avatar_record.avatar_instance_id)
                        return llmclient.action_ack

                    chunk_text = event["text"]
                    n_chunks = event.get("n_chunks", 0)
                    is_thought = (event_type == "reasoning")  # reasoning -> thought bubble; content -> visible answer

                    # Sampled here, before anything in this callback can add content: the view follows the
                    # reply only for a reader who is already at the end of it. Someone who scrolled up to
                    # re-read an earlier message stays where they put themselves, instead of being dragged
                    # back down by every chunk — which, on a thinking model, meant waiting out the whole turn.
                    follow_sample = self.view.sample_tail_follow()

                    if self.gui_updates_safe and chunk_text:  # avoid triggering on an empty event
                        self.indicator_hold.hide(self.llm_indicator_widget)  # hide prompt processing indicator

                    if chunk_text and task_env.first_chunk_t is None:
                        task_env.first_chunk_t = time.monotonic()

                    # Fire the generic talking animation once, when the model transitions from thinking to the
                    # visible answer (replaces the old "</think> seen" trigger).
                    if is_thought and task_env.thinking_t0 is None:
                        task_env.thinking_t0 = time.monotonic()

                    if not is_thought and not task_env.seen_content:
                        task_env.seen_content = True
                        logger.info("ai_turn.ai_turn_task.on_llm_progress: AI started writing the visible answer.")
                        if not speak_this_turn:  # If TTS is not speaking this turn, show the generic talking animation while the LLM is writing
                            _client_api().avatar_start_talking(self.avatar_record.avatar_instance_id)

                    # If the channel changed mid-paragraph (thought <-> answer), commit the in-progress paragraph
                    # and start a fresh one in the new channel — the renderer colors per paragraph, so a thought
                    # and the answer must never share one.
                    if task_env.text.getvalue() and (is_thought != task_env.current_is_thought):
                        streaming_chat_message.replace_last_paragraph(task_env.text.getvalue(),
                                                                      is_thought=task_env.current_is_thought)
                        streaming_chat_message.add_paragraph("", is_thought=is_thought)
                        task_env.text = io.StringIO()
                        task_env.t0 = time.monotonic()
                        task_env.n_chunks0 = n_chunks
                        self.view.follow_tail(follow_sample)
                    task_env.current_is_thought = is_thought
                    # The cloud pulsates while the reasoning is arriving and settles when the answer starts.
                    # Set on every event rather than only on the transition: the bubble does not exist until
                    # the first thinking paragraph has been rendered, which is after the transition that
                    # would have started it.
                    streaming_chat_message.set_thinking(is_thought)

                    # Accumulate the chunk, then render. Write *before* reading the paragraph so the chunk is
                    # never lost when it carries the paragraph-break newline (the trailing newline is stripped at render time).
                    task_env.text.write(chunk_text)
                    paragraph_text = task_env.text.getvalue()
                    time_now = time.monotonic()
                    dt = time_now - task_env.t0  # seconds since last GUI update
                    dchunks = n_chunks - task_env.n_chunks0  # chunks since last GUI update
                    # The counter is *published* here and *drawn* once per frame, by
                    # `update_thinking_readout`. Drawing it here tied its cadence to the arrival of chunks,
                    # and through the rate limiter below to the model's newline pattern — so the seconds
                    # advanced in steps of one to five of them, and a clock that stutters reads as a clock
                    # that has stopped.
                    self._thinking_readout = (task_env.ai_node_id, task_env.thinking_t0, n_chunks) if is_thought else None
                    if "\n" in chunk_text:  # start new paragraph?
                        task_env.t0 = time_now
                        task_env.n_chunks0 = n_chunks
                        # NOTE: The last paragraph of the AI's reply - for thinking models, commonly the final response - often never gets a "\n", and must be handled in `on_done`.
                        # With speech on, the listener has not heard this yet, so the emotion waits for the voice.
                        if not speak_this_turn:
                            _update_avatar_emotion_from_incoming_text(paragraph_text)  # update emotion from recent received text (thoughts too)
                        streaming_chat_message.replace_last_paragraph(paragraph_text,
                                                                      is_thought=is_thought)
                        streaming_chat_message.add_paragraph("",
                                                             is_thought=is_thought)
                        task_env.text = io.StringIO()
                        self.view.follow_tail(follow_sample)
                    # - update at least every 0.5 sec, even if the LLM is slow
                    # - update after every 10 chunks, but with a rate limit
                    elif dt >= 0.5 or (dt >= 0.25 and dchunks >= 10):  # commit changes to in-progress last paragraph
                        task_env.t0 = time_now
                        task_env.n_chunks0 = n_chunks
                        streaming_chat_message.replace_last_paragraph(paragraph_text,
                                                                      is_thought=is_thought)  # at first paragraph, will auto-create the paragraph if not created yet
                        self.view.follow_tail(follow_sample)

                    # Let the LLM keep generating (if it wants to).
                    return llmclient.action_ack

                def on_done(node_id: str) -> None:
                    task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                    task_env.seen_content = False  # re-arm the talking animation for it too, since the animation is stopped below
                    if not turn_owns_the_view():
                        # The user has navigated away. The reply is written and stays where it belongs, on
                        # the branch it was generated for; what must not happen is this turn dragging the
                        # user back to it, or drawing into the chat they are now looking at.
                        logger.info(f"ai_turn.ai_turn_task.on_done: HEAD has moved off this turn's branch; leaving node '{node_id}' where it is.")
                        # The streaming widget still goes, though: it belongs to the round that just ended,
                        # not to the view. Leaving it published outlives its content — the stored node is
                        # what the branch shows now — and `DPGLinearizedChatView.build` would faithfully
                        # re-attach the empty husk the next time the user came back to this branch.
                        drop_streaming_widget()
                        return
                    advance_head(node_id)  # update just in case of Ctrl+C or crash during tool calls
                    if self.gui_updates_safe:
                        if not speak_this_turn:  # If TTS is not speaking this turn, stop the generic talking animation now that the LLM is done
                            _client_api().avatar_stop_talking(self.avatar_record.avatar_instance_id)

                        unused_role, persona, text = chatutil.get_node_message_text_without_persona(self.datastore, node_id)

                        # Keep only non-thought content for TTS and final emotion update
                        text = chatutil.scrub(persona=persona,
                                              text=text,
                                              thoughts_mode="discard",
                                              markup=None,
                                              add_persona=False)

                        # Avatar speech and subtitling
                        if speak_this_turn:  # send final message text to TTS preprocess queue (this always uses lipsync)
                            logger.info("ai_turn.ai_turn_task.on_done: sending final (non-thought) message content for translation, TTS, and subtitling")
                            self.avatar_controller.send_text_to_tts(config=self.avatar_record,
                                                                    text=text,
                                                                    video_offset=librarian_config.avatar_config.video_offset,
                                                                    update_emotion=True)
                        else:
                            # Update avatar emotion one last time, from the final message text
                            logger.info("ai_turn.ai_turn_task.on_done: updating emotion from final (non-thought) message content")
                            self.avatar_controller.update_emotion_from_text(config=self.avatar_record,
                                                                            text=text)

                        # Update linearized chat view
                        logger.info("ai_turn.ai_turn_task.on_done: updating chat view with final message")
                        # The streaming message finalizing is an automatic step, not something the user asked
                        # for, so it must not move a reader who has scrolled away any more than the chunks did.
                        # This one replaces rather than appends, so the offset is restored too, not just the pin.
                        follow_sample = self.view.sample_tail_follow()
                        drop_streaming_widget()  # no-ops when there is no in-progress message in the GUI
                        # The reply that has just finished is the one case the `show_thinking` preference
                        # speaks to, so it survives the swap from streaming widget to stored one. Without
                        # this the trace would shut itself at the exact moment the reader reached the end
                        # of it.
                        self.view.add_complete_message(node_id, scroll_view=False,
                                                       start_thinking_open=self.app_state.get("show_thinking", False))
                        self.view.restore_scroll_after_swap(follow_sample)
                        self.update_context_fill_indicator()  # AI message completed -> context grew

                        logger.info("ai_turn.ai_turn_task.on_done: all done.")

                # def _parse_toolcall(request_record: dict[str, Any]) -> tuple[str | None, str | None]:
                #     """Given a tool call request record in OpenAI format, return tool call ID and function name."""
                #     tool_call_id = request_record["id"] if "id" in request_record else None
                #     function_name = None
                #     if "type" in request_record and request_record["type"] == "function":
                #         if "function" in request_record:
                #             function_record = request_record["function"]
                #             if "name" in function_record:
                #                 function_name = function_record["name"]
                #     return tool_call_id, function_name

                def _reaches_outside(tool_calls: list[dict]) -> bool:
                    """Whether any of these tools consults something beyond this conversation."""
                    names = {call.get("function", {}).get("name") for call in tool_calls}
                    return bool(names & llmclient.EXTERNAL_SOURCE_TOOL_NAMES)

                def on_tools_start(tool_calls: list[dict]) -> None:
                    task_env.in_tool_round = True  # a Stop now abandons the calls; see `abort_if_nothing_to_lose`
                    if self.gui_updates_safe:
                        # Only for tools that actually reach outside the conversation. A clock read or an
                        # arithmetic evaluation answers from nothing, and lighting the avatar for those
                        # spends a signal whose whole value is that it means something.
                        if _reaches_outside(tool_calls):
                            start_turn_data_eyes()

                        # # HACK: If websearch is present *anywhere* among the tool calls in this message,
                        # #       light up the web access indicator for the whole tool call processing step.
                        # #       Often there is just one tool call, so it's fine.
                        # ids_and_names = [_parse_toolcall(request_record) for request_record in tool_calls]
                        # names = [name for _id, name in ids_and_names]
                        # if "websearch" in names:
                        #     if self.indicator_glow_animation is not None:
                        #         self.indicator_glow_animation.reset()  # start new pulsation cycle
                        #     dpg.show_item(self.web_indicator_widget)

                def on_call_lowlevel_start(tool_call_id: str, function_name: str, arguments: dict[str, Any]) -> None:
                    if self.gui_updates_safe:
                        if function_name in web_access_tool_names:
                            if self.indicator_glow_animation is not None:
                                self.indicator_glow_animation.reset()  # start new pulsation cycle
                            dpg.set_value(self.web_progress_text_widget, "")  # not the previous call's "Done"
                            self.indicator_hold.show(self.web_indicator_widget)
                        elif function_name in document_access_tool_names:
                            if self.indicator_glow_animation is not None:
                                self.indicator_glow_animation.reset()
                            self._show_docs_access_indicator()

                def on_call_lowlevel_progress(tool_call_id: str, function_name: str, text: str) -> None:
                    if self.gui_updates_safe:
                        if function_name in web_access_tool_names:
                            dpg.set_value(self.web_progress_text_widget, text)
                        elif function_name in document_access_tool_names:
                            dpg.set_value(self.docs_access_progress_text_widget, text)

                def on_call_lowlevel_done(tool_call_id: str, function_name: str, status: str, text: str) -> None:
                    if self.gui_updates_safe:
                        if function_name in web_access_tool_names:
                            # Says so on its way out, as DOCUMENTS does.
                            dpg.set_value(self.web_progress_text_widget, "Done")
                            self.indicator_hold.hide(self.web_indicator_widget, linger=_INDICATOR_DONE_LINGER)
                        elif function_name in document_access_tool_names:
                            self._hide_docs_access_indicator()

                def on_tool_done(node_id: str) -> None:
                    task_env.text = io.StringIO()  # for next AI message (in case of tool calls)
                    if not turn_owns_the_view():  # same as `on_done`: keep the node, leave the view alone
                        return
                    advance_head(node_id)  # update just in case of Ctrl+C or crash during tool calls
                    if self.gui_updates_safe:
                        follow_sample = self.view.sample_tail_follow()  # a tool result also arrives on its own
                        drop_streaming_widget()  # it shouldn't exist when this triggers, but robustness.
                        self.view.add_complete_message(node_id, scroll_view=False)
                        self.view.restore_scroll_after_swap(follow_sample)
                        self.update_context_fill_indicator()  # tool result added -> context grew

                def on_tools_done(tool_calls: list[dict]) -> None:
                    task_env.in_tool_round = False
                    if self.gui_updates_safe and _reaches_outside(tool_calls):
                        # dpg.hide_item(self.web_indicator_widget)
                        stop_turn_data_eyes()

                def on_prompt_ready(history) -> None:
                    # logger.info("DPGChatController.ai_turn.on_prompt_ready: full prompt (message history) that will be sent to the LLM:")
                    # logger.info("=" * 80)
                    # for item in history:
                    #     logger.info(item)
                    # logger.info("=" * 80)
                    pass

                # `scaffold.ai_turn` / `scaffold.retry_tool_calls` are synchronous calls, which lets us use
                # the context manager for the idle-off override. The same callback bundle serves both: the
                # override re-runs one denied tool call on a new branch, then continues via `ai_turn`.
                common_callbacks = dict(on_docs_start=on_docs_start,
                                        on_docs_progress=on_docs_progress,
                                        on_docs_query=on_docs_query,
                                        on_docs_done=on_docs_done,
                                        on_llm_start=on_llm_start,
                                        on_prompt_ready=on_prompt_ready,  # debug/info hook
                                        on_llm_progress=on_llm_progress,
                                        on_llm_done=on_done,
                                        on_tools_start=on_tools_start,
                                        on_call_lowlevel_start=on_call_lowlevel_start,
                                        on_call_lowlevel_done=on_call_lowlevel_done,
                                        on_call_lowlevel_progress=on_call_lowlevel_progress,
                                        on_tool_done=on_tool_done,
                                        on_tools_done=on_tools_done)
                # The turn is about to recompute the injects for the wire; keep the log's copy in step, so a
                # session that ran past midnight does not show yesterday's date beside today's request.
                self.refresh_system_injects_if_stale()
                with self.avatar_controller.idle_override(config=self.avatar_record):
                    if _retry_tool_node_id is None:
                        new_head_node_id = scaffold.ai_turn(llm_settings=self.llm_settings,
                                                            datastore=self.datastore,
                                                            retriever=self.retriever,
                                                            head_node_id=self.app_state["HEAD"],
                                                            internet_enabled=self.app_state["internet_enabled"],
                                                            continue_=continue_,
                                                            docs_enabled=self.app_state["docs_enabled"],
                                                            docs_query=(docs_query if self.app_state["autosearch_enabled"] else None),
                                                            write_docs_query=librarian_config.docs_query_written_by_model,
                                                            docs_num_results=librarian_config.docs_num_results,
                                                            thinking_enabled=self.app_state["thinking_enabled"],
                                                            maybe_abort=task_env.maybe_abort,
                                                            markup="markdown",  # TODO: check if we actually use the `markup` argument for anything but thought blocks - those are in any case emitted as-is (and formatted at render time).
                                                            **common_callbacks)
                    else:
                        new_head_node_id = scaffold.retry_tool_calls(llm_settings=self.llm_settings,
                                                                     datastore=self.datastore,
                                                                     retriever=self.retriever,
                                                                     tool_node_id=_retry_tool_node_id,
                                                                     internet_enabled=self.app_state["internet_enabled"],
                                                                     docs_enabled=self.app_state["docs_enabled"],
                                                                     markup="markdown",
                                                                     docs_num_results=librarian_config.docs_num_results,
                                                                     thinking_enabled=self.app_state["thinking_enabled"],
                                                                     maybe_abort=task_env.maybe_abort,
                                                                     **common_callbacks)
                if turn_owns_the_view():
                    advance_head(new_head_node_id)
            except netutil.Aborted:
                # The user cancelled before the backend had sent anything, or during tool calls. Either way
                # there is no reply to keep and nothing to finalize. After tool calls, HEAD is already on the
                # last result — `on_tool_done` moved it there, the unfinished calls included, each answered
                # as cancelled — so the fix-up below finds nothing to do, and an empty send resumes from it.
                logger.info("ai_turn.ai_turn_task: turn abandoned before the backend answered, or during tool calls.")
                # `on_llm_start` has already put an empty streaming message in the view, and the callback
                # that would normally take it away is `on_done`, which is not going to run.
                if self.gui_updates_safe:
                    drop_streaming_widget()

                # HEAD cannot stay where `on_llm_start` put it: `ai_turn` takes back a node it never wrote
                # into, so HEAD would be naming something that no longer exists.
                #
                # It goes one step down from that node's parent, not to the parent itself. After a cancelled
                # reroll the parent is the user message and the reply being rerolled is its child, so
                # stopping at the parent would leave the view ending one message short — the reply present
                # in the tree, correct, and not on screen. One step puts it back.
                #
                # One step and no further: descending greedily would follow the newest child at every level,
                # which is a guess about which continuation the reader was in rather than a fact.
                #
                # The single step is not exact either, and cannot be. It lands on the newest surviving
                # sibling, which is the one being rerolled only when the reroll was launched from the newest
                # one — reroll from 3 of 5 and cancelling leaves the reader on 5. Nothing here can do better:
                # which sibling the reader was on is not recorded anywhere, HEAD being the whole of the
                # app's memory of where it is (Juha, 2026-08-27). Tracked in `TODO_DEFERRED.md`.
                #
                # Left alone, HEAD on a deleted node is not merely a blank view: the next thing the user does
                # builds under it. Sending a message there put a fresh exchange beneath a node that no longer
                # existed, stranding the rerolled reply two levels up on a branch nobody was on — with a
                # sibling counter reading 1 / 1, which was true and looked like data loss.
                #
                # Asked of the datastore rather than assumed, because whether the node survives is
                # `ai_turn`'s decision and depends on whether anything arrived before the abort landed.
                if task_env.ai_node_id is not None and task_env.ai_node_id not in self.datastore.nodes:
                    advance_head(chatutil.descend_to_latest(self.datastore,
                                                            task_env.round_parent_node_id,
                                                            recursive=False))
                    if self.gui_updates_safe:
                        # The branch on screen ended at the node that has just gone. Rebuilding is what puts
                        # the reply the reroll was replacing back where it was.
                        self.view.build()
            finally:
                # A live message left over from a round that ended without `on_done` running — an abort, or
                # a failure on a path that does not finalize. Reached through the view rather than a held
                # reference, so it finds whichever widget a rebuild left behind, and does nothing when the
                # user is elsewhere or the swap already happened.
                #
                # Not a matter of tidiness: the node stops being unfinished when the turn writes its final
                # payload, so a live widget outliving that would render a message the tree now says is
                # done, and go on saying it is being written.
                drop_streaming_widget()
                self._thinking_readout = None  # no reply is reasoning any more, whatever ended this turn
                if self.gui_updates_safe:
                    dpg.disable_item(self.chat_stop_generation_button_widget)
                    while turn_data_eyes_uses:  # release anything this turn started and did not finish
                        stop_turn_data_eyes()
                    if not speak_this_turn:  # make sure the generic talking animation ends (if we invoked it)
                        _client_api().avatar_stop_talking(self.avatar_record.avatar_instance_id)
                    # Also make sure that the AI-turn-scoped processing indicators hide. The INDEXING
                    # indicator is intentionally *not* touched here — it has its own polling-driven
                    # lifecycle (background commits run independent of any AI turn).
                    self.indicator_hold.hide(self.docs_access_indicator_widget)
                    self.indicator_hold.hide(self.web_indicator_widget)
                    self.indicator_hold.hide(self.llm_indicator_widget)
        def abort_if_nothing_to_lose(task_env: env) -> None:
            """`on_cancel` hook: end a backend read that co-operative cancellation cannot reach.

            Fires only while the current round has streamed nothing. That distinction is what keeps Stop's
            promise intact: once text is arriving, `on_llm_progress` runs per chunk and answers `action_stop`,
            which finishes the turn tidily and *keeps the partial reply* — the behaviour the Stop button has
            always had. Aborting there would throw that text away to save a moment.

            Before the first chunk there is no such handler to run and nothing to keep: the backend is
            processing the prompt, which on a heavy branch is tens of seconds of a Stop button that appears
            to do nothing. That is the case this exists for.

            Also during a round of tool calls, which nothing co-operative can reach either: a web tool can
            wait on a slow site for a minute. The results that had arrived are kept, and the unfinished
            calls are answered as cancelled (`scaffold.ai_turn`, which see).
            """
            if not task_env.round_has_streamed:
                logger.info("ai_turn.abort_if_nothing_to_lose: cancelled with nothing streamed yet; abandoning the backend request.")
                task_env.maybe_abort.abort()
            elif task_env.in_tool_round:
                logger.info("ai_turn.abort_if_nothing_to_lose: cancelled during tool calls; abandoning the unfinished ones.")
                task_env.maybe_abort.abort()

        self.ai_turn_task_manager.submit(ai_turn_task,
                                         env(maybe_abort=netutil.Abort(),
                                             # True until `on_llm_start` arms it for the first round, so a
                                             # cancellation landing before the backend is even called takes
                                             # the co-operative path — which the queued-task check handles.
                                             round_has_streamed=True,
                                             in_tool_round=False,
                                             on_cancel=abort_if_nothing_to_lose))

    def stop_ai_turn(self) -> None:
        """Interrupt the AI, i.e. stop ongoing text generation.

        Useful to have in case you (as the user) see the AI has misunderstood your question,
        so that there's no need to wait for a complete response.
        """
        if self.gui_updates_safe:
            dpg.disable_item(self.chat_stop_generation_button_widget)
        # Cancelling all background tasks from the AI turn specific task manager stops the task (co-operatively, so it shuts down gracefully).
        # A send whose turn has not started yet is stopped too: its exchange checks `cancelled` before submitting the turn.
        self.chat_exchange_task_manager.clear()
        self.ai_turn_task_manager.clear()
