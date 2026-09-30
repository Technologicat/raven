"""The textures the chat views draw: whose glyph a message wears, and the pictures its attachments show as.

`SpeakerGlyphs` answers which icon a message is drawn with, for the chat log and the chat graph alike.
`AttachmentTextures` turns attachment sidecars into textures — the chat log's inline thumbnails, and the
chat graph's mip chains — decoding each once and caching it for the life of the instance.

Both are held by `chat_controller.DPGChatController`, which hands them to whatever draws.
"""

__all__ = ["SpeakerGlyphs",
           "AttachmentTextures"]

import logging
logger = logging.getLogger(__name__)

import itertools
import os
import pathlib
import threading

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..avatar import characters as avatar_characters  # who the shipped characters are, by name
from ..common import bgtask
from ..vendor.file_dialog import fdialog  # for the file-type icons a document attachment is drawn as

from . import chattree
from . import config as librarian_config
from . import userprofile

gui_config = librarian_config.gui_config

# Marks a chat-graph thumbnail cache key as "a file-type icon" rather than "this image's own
# pixels". Every document of one type then shares a texture, where images are content-addressed
# and are their own identity. See `AttachmentTextures.graph_thumbnail_identity`.
_DOCUMENT_ICON_PREFIX = "icon:"

# Short edge, in pixels, at which a chat-graph thumbnail's mip chain stops halving. Far below the
# resampler's default of sixty-four, because the coarsest level has to cover the *smallest* a card is
# drawn at: zoom-to-fit on a real conversation lands around 0.2, which puts a 55-unit card at some 11
# pixels, and a graph wider than that goes lower still.
#
# **Gated on the short edge, which is why this is 4 rather than 8.** A 4:1 screenshot's chain stops when
# its short edge would go under the bound, so the long edge -- the one a card is drawn to -- is still
# four times it. The levels this buys are together well under a percent of the finest one's pixels; what
# they cost is a DPG texture apiece.
_GRAPH_THUMBNAIL_MIN_MIP = 4

_ICONS_DIR = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons")).expanduser().resolve()


class SpeakerGlyphs:
    """The glyph each chat message is drawn with: a role icon, or the face of whoever spoke.

    The generic icons are loaded once per process, into a registry shared by every instance; an instance
    adds the configured character's and user's own icons where they declare one.
    """
    class_lock = threading.RLock()
    _class_initialized = False

    @classmethod
    def _load_class_textures(cls):
        """Load textures common to all instances of this class."""
        with cls.class_lock:
            if cls._class_initialized:
                return
            with dpg.texture_registry(tag="librarian_chat_controller_textures"):
                w, h, c, data = dpg.load_image(str(_ICONS_DIR / "system.png"))
                cls.icon_system_texture = dpg.add_static_texture(w, h, data, tag="icon_system_texture")

                w, h, c, data = dpg.load_image(str(_ICONS_DIR / "tool.png"))
                cls.icon_tool_texture = dpg.add_static_texture(w, h, data, tag="icon_tool_texture")

                w, h, c, data = dpg.load_image(str(_ICONS_DIR / "user.png"))
                cls.icon_user_texture = dpg.add_static_texture(w, h, data, tag="icon_user_texture")

                w, h, c, data = dpg.load_image(str(_ICONS_DIR / "ai.png"))   # generic AI icon
                cls.icon_ai_texture = dpg.add_static_texture(w, h, data, tag="icon_ai_texture_generic")
            cls._class_initialized = True

    def __init__(self, llm_settings: env):
        """Load the glyphs.

        `llm_settings`: The LLM settings, as from `raven.librarian.llmclient.setup`. Its `personas` say who is
                        configured: the character (`raven.avatar.characters`) and the user
                        (`raven.librarian.userprofile`), whose declared icons are loaded here. Either one
                        without an icon of its own gets the generic glyph for its role. The personas are read
                        again on every `icon_texture_for`, so a stored message is drawn with the configured
                        icon only when it was written under the configured name.

        **Both sides, symmetrically.** The AI's face and the user's are found the same way — looked up by
        name, as an `_icon.png` beside the thing that declares them.
        """
        type(self)._load_class_textures()
        self.llm_settings = llm_settings

        # Where the configured speaker declares an icon of their own, it shadows the class's generic one.
        character = avatar_characters.find(llm_settings.personas.get("assistant"))
        if character is not None and character.icon_path is not None:
            w, h, c, data = dpg.load_image(str(character.icon_path))
            self.icon_ai_texture = dpg.add_static_texture(w, h, data, tag=f"icon_ai_texture_0x{id(self):x}", parent="librarian_chat_controller_textures")  # tag
        profile = userprofile.find(llm_settings.personas.get("user"))
        if profile is not None and profile.icon_path is not None:
            w, h, c, data = dpg.load_image(str(profile.icon_path))
            self.icon_user_texture = dpg.add_static_texture(w, h, data, tag=f"icon_user_texture_0x{id(self):x}", parent="librarian_chat_controller_textures")  # tag

        # The glyphs for the roles whose speaker cannot vary. Private, and deliberately holding neither an
        # "assistant" nor a "user" entry: a table keyed by role has exactly one slot per role, so every
        # stored message drawn from it wears whoever is configured now. Ask `icon_texture_for`, which
        # takes the message's own persona as well.
        self._role_icon_textures = {"system": self.icon_system_texture,
                                    "tool": self.icon_tool_texture,
                                    }

    def icon_texture_for(self,
                         role: str,
                         persona: str | None) -> int | str | None:
        """Return the speaker glyph for a message written by `persona` in `role`. `None` draws no glyph.

        `role`: One of "assistant", "system", "tool", "user".
        `persona`: The name *stored with that message*, or `None` where the role has none. Not the
                   configured one: a chat may hold turns by several characters, and by a user under
                   another name, and whoever is configured now is not who wrote the older messages.

        Satisfies `chatgraph.IconFor`, and is what both views ask — the chat log draws one message's glyph,
        the graph draws a branch of them at once, and a rule applied in only one of them would be visible
        as a disagreement between the two.

        **Two of the four roles have a speaker and two do not.** A system prompt and a tool result are
        nobody's, so one glyph each is the whole answer; an assistant message and a user message are
        somebody's, and are resolved the same way below.
        """
        if role not in ("assistant", "user"):
            return self._role_icon_textures.get(role)

        # Whoever is *configured* gets their own icon where they have declared one; everybody else gets
        # the generic glyph for their role. Only the configured pair can be placed, because their icons
        # are loaded once at construction — and drawing a stored message as somebody it was not written
        # by is the one outcome that must not happen, which is what the fallback is for.
        if role == "assistant":
            configured, own, generic = (self.llm_settings.personas.get("assistant"),
                                        self.icon_ai_texture, type(self).icon_ai_texture)
        else:
            configured, own, generic = (self.llm_settings.personas.get("user"),
                                        self.icon_user_texture, type(self).icon_user_texture)
        # `own` is the instance attribute, which `__init__` shadows over the class one when a
        # per-character or per-user icon was found; where it did not, the two are the same object and
        # this comes out as the generic glyph either way.
        return own if (persona is not None and persona == configured) else generic


class AttachmentTextures:
    """Textures of attachment sidecars, decoded once and cached for the life of the instance.

    Two caches over the same sidecars at different sizes: the chat log's inline thumbnails, and the chat
    graph's mip chains. Each has a lock of its own, since the chat log's is read from a message build and
    the graph's from the render thread, and there is no reason for either to wait on the other.

    The textures are never deleted while the instance lives, which also sidesteps the Nvidia/Linux
    texture-delete segfault.
    """
    # Every texture tag an instance creates carries its serial, so two instances in one process cannot
    # collide: a duplicate DPG tag crashes the process rather than raising, and deleted tags are freed
    # lazily. The app has one instance; the tests make many.
    _serials = itertools.count()

    def __init__(self,
                 datastore: chattree.Forest,
                 task_manager: bgtask.TaskManager):
        """`datastore`: The datastore whose sidecars are drawn.

        `task_manager`: Where the chat graph's thumbnails are prepared, off the render thread.
        """
        self.datastore = datastore
        self.task_manager = task_manager
        self._tag_prefix = f"chat_attachment_{next(type(self)._serials)}"
        self._registry = dpg.add_texture_registry()

        # The lock serializes get-or-create so two concurrent message builds can't both try to create the
        # same-tagged texture.
        self._inline_textures = {}  # {sidecar_filename: env(texture_tag, w, h)}
        self._inline_lock = threading.RLock()

        self._graph_textures = {}  # {(identity, size): env(levels)}
        self._graph_pending = set()  # keys a background task is currently preparing
        self._graph_failed = set()  # keys that could not be prepared, so nothing retries forever
        self._graph_lock = threading.Lock()

    def destroy(self) -> None:
        """Delete every texture this instance made. The app never calls this; its textures live as long as it does."""
        dpg.delete_item(self._registry)

    def inline_image(self, filename: str) -> env | None:
        """Return the chat log's texture for the image sidecar `filename`, creating it on first use.

        Reads the sidecar bytes, downsamples to a thumbnail that fits the inline display box
        (`gui_config.chat_inline_image_h` × `chat_inline_image_w`, aspect preserved, never upscaled), uploads a
        static texture, and caches it by filename — so the same image referenced by several messages, or
        re-encountered on a view rebuild, decodes once. Returns an `env(texture_tag, w, h)`, or `None` if the
        sidecar is missing or can't be decoded.

        Safe to call from a message-build background thread: texture creation is serialized (a duplicate DPG tag
        would crash the process), and two `split_frame`s after a fresh upload let DPG process the new texture
        before it is first drawn. (DPG defers the OpenGL upload to a render frame; a single wait empirically
        isn't enough — see dpg-notes.md "Texture upload ordering". A `static_texture` is correct here because
        these thumbnails are permanent — cached for the instance's lifetime, never deleted.)
        """
        with self._inline_lock:
            cached = self._inline_textures.get(filename)
            if cached is not None:
                return cached
            try:
                from ..common.image import codec  # deferred: pulls torch / Pillow only when an image is shown
                from ..common.image import utils as image_utils
                raw = self.datastore.read_sidecar(filename)
                arr = image_utils.ensure_rgba(codec.decode(raw))  # (H, W, 4) uint8
                tensor = image_utils.np_to_tensor(arr, device="cpu")  # (1, 4, H, W) float32
                tensor = image_utils.fit_contain(tensor,  # no upscale: a small image shows at native size
                                                 gui_config.chat_inline_image_h,
                                                 gui_config.chat_inline_image_w)
                disp_h, disp_w = int(tensor.shape[2]), int(tensor.shape[3])
                flat = image_utils.tensor_to_dpg_flat(tensor)  # flat float32 RGBA in [0, 1]
                texture_tag = f"{self._tag_prefix}_inline_{filename}"  # tag  # filename is a content-addressed sha256.ext, so unique
                dpg.add_static_texture(disp_w, disp_h, flat,
                                       tag=texture_tag,  # tag
                                       parent=self._registry)
                dpg.split_frame()  # trigger the deferred OpenGL upload...
                dpg.split_frame()  # ...and ensure it completed before the image widget draws it (single wait isn't enough; dpg-notes.md)
                result = env(texture_tag=texture_tag, w=disp_w, h=disp_h)
                self._inline_textures[filename] = result
                return result
            except Exception as exc:  # noqa: BLE001 -- a broken sidecar must not break rendering the rest of the chat
                logger.error(f"AttachmentTextures.inline_image: failed to load sidecar '{filename}': {type(exc)}: {exc}")
                return None

    @staticmethod
    def graph_thumbnail_identity(filename: str) -> str:
        """What two attachments must share for one prepared thumbnail to serve both.

        A picture is its own subject, and sidecar names are content-addressed, so an image is its own
        identity and the same bytes attached twice decode once. A *document* has no picture: it is drawn
        as its file type's icon, so every PDF in the datastore is one texture rather than one each.

        The type icons and the mapping onto them are the file dialog's — the same picture for the same
        kind of file, wherever in Raven it appears. `.pdf` maps to nothing there on purpose (there is no
        presentation icon either, and the generic document is the right picture for all of them), which is
        what the fallback is for.
        """
        from ..common.image import codec  # deferred, as the decode below is
        if os.path.splitext(filename)[1].lower() in codec.IMAGE_EXTENSIONS:
            return filename
        return f"{_DOCUMENT_ICON_PREFIX}{fdialog.icon_name_for_extension(filename) or 'document'}"

    def graph_thumbnail(self, filename: str, size: float) -> env | None:
        """Return the chat graph's thumbnail of sidecar `filename`, or `None` if it is not ready.

        `size`: The longest edge to prepare the *finest* level at, in pixels. Part of the cache key, so
                the same image can be held at the chat log's inline size and at the graph's at once.

        Returns an `env(levels)`: the mip chain as `(width, height, texture_tag)` triples, finest first.
        A chain rather than one texture because the graph zooms continuously, and dimensions per level
        because the graph draws the picture at its own proportions and cannot ask a texture how big it
        is — see `_prepare_graph_thumbnail`.

        **Never blocks, and `None` is an ordinary answer rather than a failure.** The graph rebuilds from
        its animator hook, which runs on the render thread, and preparing a texture needs `split_frame` --
        which deadlocks there. So a miss queues the work and answers `None`; the graph draws an empty frame
        meanwhile, and the panel notices when the answer changes.

        An attachment that cannot be prepared is remembered as such, so a broken sidecar costs one attempt
        rather than one per rebuild for the life of the session.
        """
        key = (self.graph_thumbnail_identity(filename), size)
        with self._graph_lock:
            cached = self._graph_textures.get(key)
            if cached is not None or key in self._graph_failed:
                return cached
            if key in self._graph_pending:
                return None
            self._graph_pending.add(key)
        self.task_manager.submit(lambda task_env: self._prepare_graph_thumbnail(key, filename, size, task_env),
                                 env())
        return None

    def _prepare_graph_thumbnail(self, key: tuple, filename: str, size: float, task_env: env) -> None:
        """Turn one attachment into a mip chain of graph textures. Runs on a background thread.

        A chain rather than one texture because the graph zooms continuously, and DPG does not average
        when it draws a texture smaller (no mipmaps), so the size a card is drawn at is not known here: the
        finest level is prepared at `size` and the renderer draws whichever level suits the card on screen.

        Uploading the whole chain before publishing any of it is what keeps the graph from drawing a
        half-arrived picture — the shape reads its levels without a lock, one rebuild at a time.
        """
        try:
            if task_env.cancelled:  # shutdown, most likely; the pending mark is cleared in `finally`
                return
            from ..common.image import codec  # deferred: pulls torch / Pillow only when an image is shown
            from ..common.image import lanczos
            from ..common.image import utils as image_utils
            identity = key[0]
            if identity.startswith(_DOCUMENT_ICON_PREFIX):
                # A document has no picture, so it gets its type's icon. Read from the file dialog's own
                # assets rather than copied, so the two views cannot come to disagree about what a `.bib`
                # file looks like.
                icon_path = os.path.join(fdialog.IMAGES_DIR,
                                         f"{identity[len(_DOCUMENT_ICON_PREFIX):]}.png")
                with open(icon_path, "rb") as icon_file:
                    raw = icon_file.read()
            else:
                raw = self.datastore.read_sidecar(filename)
            arr = image_utils.ensure_rgba(codec.decode(raw))  # (H, W, 4) uint8
            tensor = image_utils.np_to_tensor(arr, device="cpu")  # (1, 4, H, W) float32
            # Aspect preserved, and never upscaled: `fit_contain` scales the whole image to fit the box
            # and hands back its own dimensions, which the graph then draws it at. Squaring it here would
            # stretch a wide photograph into a square one, which is a lie about the picture and looks like
            # one -- the *card* is square, and the picture is letterboxed inside it.
            tensor = image_utils.fit_contain(tensor, int(size), int(size))
            # Down to a level small enough for the card at the zooms a whole conversation is read at: a
            # card is some 55 graph units across, so an overview at 0.25 draws it at 14 pixels.
            levels = lanczos.mipchain(tensor, min_size=_GRAPH_THUMBNAIL_MIN_MIP)
            del tensor
            # Uploaded as a set, with no cancellation check in between, so that the tags this run claims
            # are either all registered or none: a duplicate DPG tag crashes the process rather than
            # raising, which makes a half-registered chain the expensive kind of leftover. The loop is a
            # few array conversions -- the decode and the resize, which are what a shutdown wants to cut
            # short, are already done above.
            uploaded = []
            for index, level in enumerate(levels):
                level_h, level_w = int(level.shape[2]), int(level.shape[3])
                flat = image_utils.tensor_to_dpg_flat(level)  # flat float32 RGBA in [0, 1]
                # One tag per (cache key, level). The suffix is the level index rather than its size:
                # two levels of a very wide picture can share a short edge, and a size-named tag would
                # then collide -- which crashes the process rather than raising.
                texture_tag = f"{self._tag_prefix}_graph_{int(size)}_{identity}_mip{index}"  # tag
                dpg.add_static_texture(level_w, level_h, flat,
                                       tag=texture_tag,  # tag
                                       parent=self._registry)
                uploaded.append((level_w, level_h, texture_tag))
            del levels
            dpg.split_frame()  # trigger the deferred OpenGL upload...
            dpg.split_frame()  # ...and ensure it completed before the graph draws it (dpg-notes.md, "Texture upload ordering")
            with self._graph_lock:
                self._graph_textures[key] = env(levels=tuple(uploaded))
        except Exception as exc:  # noqa: BLE001 -- a broken sidecar must not break the graph
            logger.error(f"AttachmentTextures._prepare_graph_thumbnail: failed to prepare '{filename}' at {size}: {type(exc)}: {exc}")
            with self._graph_lock:
                self._graph_failed.add(key)
        finally:
            with self._graph_lock:
                self._graph_pending.discard(key)
