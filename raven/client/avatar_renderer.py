"""DPG GUI driver for Raven-avatar.

This allows easily integrating the avatar to your DPG GUI apps.

`DPGAvatarRenderer` connects to a given avatar instance running on Raven-server (using `raven.client.api`),
and blits the live video into a DPG texture.

The texture is attached to a DPG image widget, thus rendering the live video in the GUI.
"""

__all__ = ["DPGAvatarRenderer"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
import os
import pathlib
import platform
import qoi
import time
import traceback
from typing import Optional, Tuple, Union

import PIL.Image

from unpythonic.env import env as envcls

import numpy as np

import dearpygui.dearpygui as dpg

from ..common import bgtask
from ..common.gui import utils as guiutils
from ..common.running_average import RunningAverage
from ..common.numutils import si_prefix

from . import api

# WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
# https://github.com/hoffstadt/DearPyGui/issues/554
if platform.system().upper() == "LINUX":
    os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

# --------------------------------------------------------------------------------
# helpers

class ResultFeedReader:
    def __init__(self, avatar_instance_id: str):
        """Helper class for `DPGAvatarRenderer`."""
        self.avatar_instance_id = avatar_instance_id
        self.gen = None

    def start(self) -> None:
        if self.gen is None:
            self.gen = api.avatar_result_feed(self.avatar_instance_id)

    def is_running(self) -> bool:
        return self.gen is not None

    def get_frame(self) -> Tuple[Optional[str], bytes]:
        """-> (received_mimetype, payload)"""
        return next(self.gen)  # next-gen lol

    def stop(self) -> None:
        if self.gen is not None:
            self.gen.close()
            self.gen = None

# --------------------------------------------------------------------------------
# API

# TODO: manage our own texture registry?

class DPGAvatarRenderer:
    def __init__(self,
                 texture_registry: Union[str, int],
                 gui_parent: Union[str, int],
                 avatar_x_center: int,
                 avatar_y_bottom: int,
                 paused_text: Optional[str],
                 task_manager: bgtask.TaskManager):
        """DPG GUI driver for Raven-avatar.

        This can only be instantiated after DPG bootup is complete, because the constructor sets up some GUI widgets.

        `texture_registry`: DPG tag or ID; where to keep the DPG texture object.
        `gui_parent`: DPG tag or ID; where to put the GUI image widget that displays the live texture.
        `avatar_x_center`: x center position of avatar video feed, in pixels, in the coordinate system of `gui_parent`.
        `avatar_y_bottom`: y bottom (one past end) of avatar video feed, in pixels, in the coordinate system of `gui_parent`.
        `paused_text`: Text to show when the animator is not running, or `None` to leave empty.
        `task_manager`: For submitting the background task when you call the `start` method.

        Once you have instantiated `DPGAvatarRenderer`, you can call the `configure_*` methods at any time,
        and as many times as you need.

        After you have called `configure_live_texture` (at least once), you can connect to Raven-server and start
        displaying the avatar video stream by calling `start`. You will need to provide the avatar instance ID of
        the avatar session to connect to; you get this ID by calling `raven.client.api.load_avatar`.

        For a complete usage example, see `raven.avatar.settings_editor.app`.
        """
        self.texture_registry = texture_registry
        self.gui_parent = gui_parent
        self.task_manager = task_manager
        self._task_env = None  # This holds the local namespace (`unpythonic.env.env`) of the background task, so we can cancel the task if needed.

        self.live_texture = None  # The raw texture object
        self.blank_texture = None  # Raw blank texture
        self.live_texture_id_counter = 0  # For creating unique DPG IDs when the size changes on the fly, since the delete might not take immediately.
        self.live_image_widget = None  # GUI widget the texture renders to
        self.last_image_rgba = None  # For rescaling last received frame on upscaler size change before we get new data

        self.backdrop_image = None  # PIL image
        self.backdrop_texture = None  # raw texture
        self.backdrop_texture_id_counter = 0
        self.backdrop_width = None
        self.backdrop_height = None
        self.backdrop_blur_state = None  # whether to blur background (by calling Raven-server's `imagefx` module)
        # tracking for backdrop's previous state so that `configure_backdrop` knows whether it needs to do anything
        self.backdrop_old_image = None
        self.backdrop_old_width = None
        self.backdrop_old_height = None
        self.backdrop_old_blur_state = None

        self.avatar_instance_id = None  # initialized at `start`
        self.animator_running = False
        self.image_size = None  # initialized at first call of `configure_live_texture`
        self.avatar_x_center = avatar_x_center
        self.avatar_y_bottom = avatar_y_bottom

        self.fps_statistics = RunningAverage()
        self.frame_size_statistics = RunningAverage()

        self.backdrop_drawlist_gui_widget = dpg.add_drawlist(tag="avatar_backdrop_drawlist", width=1024, height=1024, pos=(0, 0))  # for backdrop image (bottommost GUI item in z-order)

        # For displaying current video FPS arriving from the server
        self.fps_text_gui_widget = dpg.add_text("FPS counter will appear here",
                                                color=(0, 255, 0),
                                                pos=(8, 0),
                                                show=False,
                                                tag="avatar_fps_text",
                                                parent=gui_parent)
        # Text to show while paused. This will be positioned when shown.
        paused_str = paused_text if paused_text is not None else ""
        self.paused_text_gui_widget = dpg.add_text(paused_str,
                                                   show=False,
                                                   tag="paused_text",
                                                   parent=gui_parent)

    def configure_fps_counter(self, show: Optional[bool]) -> None:
        """Show or hide the FPS counter.

        If `show is None`, toggle the state.
        """
        try:
            if show is None:
                show = not dpg.is_item_visible(self.fps_text_gui_widget)

            if show:
                dpg.show_item(self.fps_text_gui_widget)
            else:
                dpg.hide_item(self.fps_text_gui_widget)
        except SystemError:  # DPG widget does not exist (can happen at app shutdown)
            pass
        except AttributeError:  # GUI instance went bye-bye (can happen at app shutdown)
            pass

    def load_backdrop_image(self, filename: Optional[Union[pathlib.Path, str]]):
        """Load a backdrop image. To clear the background (no image), use `filename=None`.

        The backdrop change takes effect upon the next call to `configure_backdrop`, which see.
        """
        if filename is not None:
            self.backdrop_image = PIL.Image.open(filename)
        else:
            self.backdrop_image = None

    def configure_backdrop(self,
                           new_width: int,
                           new_height: int,
                           new_blur_state: bool) -> None:
        """Configure the size and blur state of the current backdrop image.

        The backdrop always starts at the upper left corner of the GUI parent widget.

        If the size of the loaded backdrop image does not match `new_width x new_height`,
        the image is rescaled with Lanczos on CPU, and then cropped to fit the aspect ratio
        `new_width / new_height`.

        This method has no effect when no backdrop image is loaded. Call `load_backdrop_image` first!

        NOTE: This needs to wait for a frame to eliminate GUI flicker when the texture is replaced.
              Thus, this method CANNOT be called from the main thread that runs the render loop
              (doing so will hang the app).

              Calling from any other thread (including GUI event handlers) is fine.
        """
        old_width = self.backdrop_old_width
        old_height = self.backdrop_old_height
        old_blur_state = self.backdrop_old_blur_state
        old_texture_id = self.backdrop_texture_id_counter
        if self.backdrop_image is not None and (self.backdrop_image != self.backdrop_old_image or new_width != old_width or new_height != old_height or new_blur_state != old_blur_state):
            new_texture_id = old_texture_id + 1

            image_width, image_height = self.backdrop_image.size

            # TODO: If the backdrop image is small and/or has a wild aspect ratio, would be more efficient to cut first, then scale.
            #
            # Scale image, preserving aspect ratio, to cover the whole backdrop region (1024 x h)
            # https://stackoverflow.com/questions/1373035/how-do-i-scale-one-rectangle-to-the-maximum-size-possible-within-another-rectang
            scale = max(new_width / image_width, new_height / image_height)  # max(dst.w / src.w, dst.h / src.h)
            pil_image = self.backdrop_image.resize((int(scale * image_width), int(scale * image_height)),
                                                   resample=PIL.Image.LANCZOS)
            # Then cut the part we need
            pil_image = pil_image.crop(box=(0, 0, new_width, new_height))  # (left, upper, right, lower), in pixels

            image_rgba = pil_image.convert("RGBA")
            image_rgba = np.asarray(image_rgba, dtype=np.float32) / 255

            if new_blur_state:
                image_rgba = api.imagefx_process_array(image_rgba,
                                                       filters=[["analog_lowres", {"sigma": 3.0}],  # maximum sigma is 3.0 due to convolution kernel size
                                                                ["analog_lowres", {"sigma": 3.0}],  # how to blur more: unrolled loop
                                                                ["analog_lowres", {"sigma": 3.0}],
                                                                ["analog_lowres", {"sigma": 3.0}],
                                                                ["analog_lowres", {"sigma": 3.0}]]
                                                       )

            raw_data = image_rgba.ravel()

            logger.info(f"DPGAvatarRenderer.configure_backdrop: Creating new GUI item avatar_backdrop_texture_{new_texture_id}")
            self.backdrop_texture = dpg.add_raw_texture(width=new_width,
                                                        height=new_height,
                                                        default_value=raw_data,
                                                        format=dpg.mvFormat_Float_rgba,
                                                        tag=f"avatar_backdrop_texture_{new_texture_id}",
                                                        parent=self.texture_registry)
            self.backdrop_texture_id_counter += 1
            dpg.split_frame()  # For some reason, waiting for a frame here eliminates flicker when the background image is replaced.
            dpg.delete_item("avatar_backdrop_drawlist", children_only=True)  # delete old draw items
            dpg.configure_item("avatar_backdrop_drawlist", width=new_width, height=new_height)
            dpg.draw_image(f"avatar_backdrop_texture_{new_texture_id}", (0, 0), (new_width, new_height), uv_min=(0, 0), uv_max=(1, 1), parent="avatar_backdrop_drawlist")
            guiutils.maybe_delete_item(f"avatar_backdrop_texture_{old_texture_id}")
        elif self.backdrop_image is None:  # clear backdrop?
            dpg.delete_item("avatar_backdrop_drawlist", children_only=True)  # delete old draw items
            dpg.configure_item("avatar_backdrop_drawlist", width=new_width, height=new_height)
            guiutils.maybe_delete_item(f"avatar_backdrop_texture_{old_texture_id}")

        self.backdrop_old_image = self.backdrop_image
        self.backdrop_old_width = new_width
        self.backdrop_old_height = new_height
        self.backdrop_old_blur_state = new_blur_state

    def configure_live_texture(self, new_image_size: int) -> None:
        """Set up (or re-set-up) the texture and image widgets for rendering the video stream of the live AI avatar.

        `new_image_size`: The image size to receive. We use this to make the texture pixel-perfect.
                          The image has square aspect ratio; this is the length of one side.

        If the server's video stream has a different size, it will be software-upscaled (slowly!) on the fly to `new_image_size`.

        You can call this method again whenever you want to switch to a different image size (e.g. when changing upscaler settings).
        """
        old_texture_id = self.live_texture_id_counter
        new_texture_id = old_texture_id + 1

        logger.info(f"DPGAvatarRenderer.configure_live_texture: Creating new GUI item avatar_live_texture_{new_texture_id} for new size {new_image_size}x{new_image_size}")
        self.blank_texture = np.zeros([new_image_size,  # height
                                       new_image_size,  # width
                                       4],  # RGBA
                                      dtype=np.float32).ravel()
        if self.last_image_rgba is not None:
            # To reduce flicker when the texture is replaced: take the last frame we have,
            # rescale it, and use that as the initial content of the new texture.
            image_rgba = self.last_image_rgba  # from the background thread
            pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
            if image_rgba.shape[2] == 4:
                alpha_channel = image_rgba[:, :, 3]
                pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
            pil_image = pil_image.resize((new_image_size, new_image_size),
                                         resample=PIL.Image.LANCZOS)
            image_rgba = pil_image.convert("RGBA")
            image_rgba = np.asarray(image_rgba, dtype=np.float32) / 255
            default_image = image_rgba.ravel()
        else:
            default_image = self.blank_texture
        self.live_texture = dpg.add_raw_texture(width=new_image_size,
                                                height=new_image_size,
                                                default_value=default_image,
                                                format=dpg.mvFormat_Float_rgba,
                                                tag=f"avatar_live_texture_{new_texture_id}",
                                                parent=self.texture_registry)
        self.live_texture_id_counter += 1  # now the new texture exists so it's safe to write to (in the background thread)
        self.image_size = new_image_size

        first_time = (self.live_image_widget is None)
        logger.info(f"DPGAvatarRenderer.configure_live_texture: Creating new GUI item avatar_live_image_{new_texture_id}")
        self.live_image_widget = dpg.add_image(f"avatar_live_texture_{new_texture_id}",  # tag
                                               show=self.animator_running,  # if paused, leave it hidden (note that we're initially paused when the `DPGAvatarRenderer` instance is created!)
                                               tag=f"avatar_live_image_{new_texture_id}",
                                               parent=self.gui_parent,
                                               before=self.fps_text_gui_widget)
        self.reposition()

        if not first_time:
            dpg.split_frame()  # For some reason, waiting for a frame here eliminates flicker when the texture object is replaced.
        try:
            dpg.hide_item(f"avatar_live_image_{old_texture_id}")  # tag
        except SystemError:  # does not exist
            pass
        else:
            dpg.split_frame()  # Only safe after startup, once the GUI render loop is running. At startup, the old image widget doesn't exist, so we detect the situation from that.
        # Now the old image widget is guaranteed to be hidden, so we can delete it without breaking GUI render
        guiutils.maybe_delete_item(f"avatar_live_image_{old_texture_id}")  # tag
        guiutils.maybe_delete_item(f"avatar_live_texture_{old_texture_id}")  # tag

        logger.info("DPGAvatarRenderer.configure_live_texture: done.")

    def reposition(self,
                   new_x_center: Optional[int] = None,
                   new_y_bottom: Optional[int] = None) -> None:
        """Position the image widget within its parent.

        Optionally, change the position.

        `configure_live_texture` calls this automatically.
        """
        self.avatar_x_center = new_x_center if new_x_center is not None else self.avatar_x_center
        self.avatar_y_bottom = new_y_bottom if new_y_bottom is not None else self.avatar_y_bottom
        logger.info(f"DPGAvatarRenderer.reposition: Updating position to x_center = {self.avatar_x_center}, y_bottom = {self.avatar_y_bottom}")

        x0, y0 = guiutils.get_widget_pos(self.gui_parent)
        w, h = guiutils.get_widget_size(self.gui_parent)
        logger.info(f"DPGAvatarRenderer.reposition: Gui parent is at ({x0}, {y0}), and has size {w}x{h}")

        x_left = self.avatar_x_center - (self.image_size // 2)
        y_top = self.avatar_y_bottom - self.image_size
        try:
            dpg.set_item_pos(self.live_image_widget, (x_left, y_top))
        except SystemError:  # window or live image widget does not exist
            logger.info("DPGAvatarRenderer.reposition: Live image GUI widget doesn't exist; ignoring. (This is normal at app shutdown.)")
        else:
            logger.info("DPGAvatarRenderer.reposition: success")

    def pause(self, action: str) -> None:
        """Pause or resume the animator.

        `action`: One of "pause", "resume", or "toggle".

        This also pauses/resumes the avatar instance on the server.

        When paused, if a `paused_text` was set in `__init__`, it is shown in the center of the video feed area.

        When resumed, the `paused_text` (if any) is hidden, and the video feed resumes.

        NOTE: If you need to query the current state, it is in the `animator_running` (bool) attribute.
              It is part of the public API, but consider it read-only.

              Querying the state and acting explicitly can be convenient, instead of just toggling, if you need to
              do custom GUI actions (such as changing the text on a pause/resume button) when you pause/resume.
        """
        if action not in ("pause", "resume", "toggle"):
            raise ValueError(f"DPGAvatarRenderer.pause: Unknown `action` '{action}'; valid actions: 'pause', 'resume', 'toggle'.")
        if self.avatar_instance_id is None:
            raise RuntimeError(f"DPGAvatarRenderer.pause (action '{action}'): The renderer must be started first by calling `start` before the `pause` method can be called.")

        if action == "toggle":
            if self.animator_running:
                action = "pause"
            else:
                action = "resume"
        assert action in ("pause", "resume")

        if action == "pause":
            # center the paused indicator on the video feed in the GUI
            try:
                # position offscreen and render, to compute size
                dpg.set_item_pos(self.paused_text_gui_widget, (0, -100))
                dpg.show_item(self.paused_text_gui_widget)
                dpg.split_frame()
                w, h = guiutils.get_widget_size(self.paused_text_gui_widget)
                dpg.set_item_pos(self.paused_text_gui_widget, (((self.image_size - w) // 2), (self.image_size // 2)))  # TODO: account for font size / height
                dpg.hide_item(f"avatar_live_image_{self.live_texture_id_counter}")
                api.avatar_stop(self.avatar_instance_id)
                self.animator_running = False
            except SystemError:  # window or live image widget does not exist
                logger.info(f"DPGAvatarRenderer.pause (avatar instance '{self.avatar_instance_id}', action '{action}'): Pause text GUI widget doesn't exist.")
        else:  # action == "resume":
            api.avatar_start(self.avatar_instance_id)
            dpg.hide_item(self.paused_text_gui_widget)
            dpg.show_item(f"avatar_live_image_{self.live_texture_id_counter}")
            self.animator_running = True

        logger.info(f"DPGAvatarRenderer.pause (avatar instance '{self.avatar_instance_id}', action '{action}'): success")

    def start(self,
              avatar_instance_id: str) -> None:
        """Start receiving the live video stream.

        `avatar_instance_id`: Avatar instance to receive. You get this from `raven.client.api.avatar_load`.

        There is currently no function to stop receiving. You can just close the session (`raven.client.api.avatar_unload`);
        the background task then shuts down gracefully.
        """
        if self._task_env is not None:
            raise RuntimeError("DPGAvatarRenderer.start: already running, cannot start again. If you need to connect to a different avatar session, `stop` first.")

        logger.info(f"DPGAvatarRenderer.start: Setting up background task for avatar instance '{avatar_instance_id}'.")

        self.avatar_instance_id = avatar_instance_id  # store for pause/resume

        # We must continuously retrieve new frames as they become ready, so this runs in the background.
        def update_live_texture(task_env) -> None:
            """Live texture update task.

            Receives video frames from Raven-server and renders them in the GUI.
            """
            assert task_env is not None
            logger.info(f"DPGAvatarRenderer.start.update_live_texture: Background task for avatar instance '{avatar_instance_id}' starting.")

            def describe_performance(video_format: str, video_height: int, video_width: int):  # actual received video height/width of the frame being described
                if not self.animator_running:
                    return "RX (avg) -- B/s @ -- FPS; avg -- B per frame (--x--, -- px, --)"

                avg_fps = self.fps_statistics.average()
                avg_bytes = int(self.frame_size_statistics.average())
                pixels = video_height * video_width

                return f"RX (avg) {si_prefix(avg_fps * avg_bytes)}B/s @ {avg_fps:0.2f} FPS; avg {si_prefix(avg_bytes)}B per frame ({video_width}x{video_height}, {si_prefix(pixels)}px, {video_format})"

            def maybe_set_fps_counter(text):
                try:
                    dpg.set_value(self.fps_text_gui_widget,
                                  text)
                except SystemError:  # DPG widget does not exist (can happen at app shutdown)
                    pass
                except AttributeError:  # GUI instance went bye-bye (can happen at app shutdown)
                    pass

            reader = ResultFeedReader(avatar_instance_id)
            reader.start()
            try:
                while not task_env.cancelled:
                    frame_start_time = time.time_ns()  # for FPS measurement

                    # sync `ResultFeedReader` state from `animator_running` state
                    if not self.animator_running and reader.is_running():
                        reader.stop()
                        maybe_set_fps_counter(describe_performance(None, None, None))
                    elif self.animator_running and not reader.is_running():
                        reader.start()

                    if reader.is_running():
                        mimetype, image_data = reader.get_frame()
                        self.frame_size_statistics.add_datapoint(len(image_data))

                    # If our `ResultFeedReader` isn't running, there's nothing to do at the moment.
                    if not reader.is_running():
                        time.sleep(0.04)   # 1/25 s
                        continue

                    try:  # EAFP to avoid TOCTTOU
                        # Before blitting, make sure the texture is of the expected size. When an upscale change is underway, it will be temporarily of the wrong size.
                        tex = self.live_texture  # Get the reference only once, since it could change in another thread at any time (by the user calling `configure_live_texture`), if the user changes the upscaler settings.
                        config = dpg.get_item_configuration(tex)
                        expected_w = config["width"]
                        expected_h = config["height"]
                    except SystemError:  # does not exist (can happen at least during app shutdown, or during a texture object swap)
                        time.sleep(0.04)   # 1/25 s
                        continue  # can't do anything without a texture to blit to, so discard this frame

                    if mimetype == "image/qoi":
                        image_rgba = qoi.decode(image_data)  # -> uint8 array of shape (h, w, c)
                        # Don't crash if we get frames at a different size from what is expected. But log a warning, as software rescaling is slow.
                        h, w = image_rgba.shape[:2]
                        if w != expected_w or h != expected_h:
                            logger.warning(f"update_live_texture: Got frame at wrong (old?) size {w}x{h}; slow CPU resizing to {expected_w}x{expected_h}")
                            pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                            if image_rgba.shape[2] == 4:
                                alpha_channel = image_rgba[:, :, 3]
                                pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
                            pil_image = pil_image.resize((expected_w, expected_h),
                                                         resample=PIL.Image.LANCZOS)
                            image_rgba = np.asarray(pil_image.convert("RGBA"))
                    else:  # use PIL
                        image_file = io.BytesIO(image_data)
                        pil_image = PIL.Image.open(image_file)
                        # Don't crash if we get frames at a different size from what is expected. But log a warning, as software rescaling is slow.
                        w, h = pil_image.size
                        if w != expected_w or h != expected_h:
                            logger.warning(f"update_live_texture: Got frame at wrong (old?) size {w}x{h}; slow CPU resizing to {expected_w}x{expected_h}")
                            pil_image = pil_image.resize((expected_w, expected_h),
                                                         resample=PIL.Image.LANCZOS)
                        image_rgba = np.asarray(pil_image.convert("RGBA"))
                    self.last_image_rgba = image_rgba  # for reducing flicker when upscaler settings change
                    image_rgba = np.array(image_rgba, dtype=np.float32) / 255
                    raw_data = image_rgba.ravel()  # shape [h, w, c] -> linearly indexed
                    try:  # EAFP to avoid TOCTTOU
                        dpg.set_value(tex, raw_data)  # to GUI
                    except SystemError:  # does not exist (might have gone bye-bye while we were decoding)
                        continue  # can't do anything without a texture to blit to, so discard this frame

                    # Update FPS counter.
                    # NOTE: Since we wait on the server to send a frame, the refresh is capped to the rate that data actually arrives at, i.e. the server's TARGET_FPS.
                    #       If the machine could render faster, this just means less than 100% CPU/GPU usage.
                    elapsed_time = time.time_ns() - frame_start_time
                    fps = 1.0 / (elapsed_time / 10**9)
                    self.fps_statistics.add_datapoint(fps)
                    maybe_set_fps_counter(describe_performance(mimetype, h, w))
            except EOFError:  # `result_feed` has shut down (normal at app exit, after we call `api.avatar_unload`)
                logger.info("DPGAvatarRenderer.start.update_live_texture: Stream closed, shutting down.")
            except Exception as exc:
                traceback.print_exc()
                logger.error(f"DPGAvatarRenderer.start.update_live_texture: {type(exc)}: {exc}")

                # TODO: recovery if the server comes back online
                dpg.set_value(self.paused_text_gui_widget, "[Connection lost]")
                self.pause(action="pause")
                maybe_set_fps_counter(describe_performance(None, None, None))
            finally:
                reader.stop()  # Close the stream to ensure that the server's network send thread serving our request exits.
                self.avatar_instance_id = None
                self._task_env = None
            logger.info(f"DPGAvatarRenderer.start.update_live_texture: Background task for avatar instance '{avatar_instance_id}' exiting.")

        logger.info(f"DPGAvatarRenderer.start: Submitting background task to task manager for avatar instance '{avatar_instance_id}'.")
        self._task_env = envcls()
        self.task_manager.submit(update_live_texture, self._task_env)
        self.animator_running = True  # start in animator running state
        dpg.show_item(f"avatar_live_image_{self.live_texture_id_counter}")  # and show the image  # tag

    def stop(self):
        """The opposite of `start`.

        This disconnects the `DPGAvatarRenderer` instance from the avatar instance.
        """
        if self._task_env is None:
            raise RuntimeError("DPGAvatarRenderer.stop: not running, nothing to stop.")
        self._task_env.cancelled = True
