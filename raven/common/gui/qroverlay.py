"""A "Get Raven" QR code in the corner of the window, for showing the app at an exhibit.

A visitor watches a demo for a minute and then walks away; nobody writes down a URL. A QR code in the
corner lets them point a phone at it and read about the project later, which is the whole point.

Off unless an app is started with `--qr`, so it costs nothing in ordinary use.

**One overlay, every app.** Installed with a single call, the way `filedrop` is::

    qroverlay.install()                      # the project's own repository
    qroverlay.install(url="https://...", label="Ask me about Raven")

The code is generated at runtime from the URL recorded in the package metadata, rather than shipped as an
image: a baked-in PNG would be a second copy of the URL that nothing keeps honest, and regenerating it is a
step someone forgets.
"""

__all__ = ["Corner",
           "get_project_url",
           "matrix_to_pixels",
           "QRCodeOverlay", "install", "uninstall"]

import enum
import importlib.metadata
import logging
from typing import List, Optional, Tuple, Union

import dearpygui.dearpygui as dpg

import numpy as np

import segno

from unpythonic import sym

from . import animation as gui_animation

logger = logging.getLogger(__name__)

# The distribution name, which is not the package name. `importlib.metadata` wants the former.
_DISTRIBUTION = "raven-visualizer"

# A QR code is only reliably scannable with a clear margin around it, four modules wide by the standard.
_QUIET_ZONE_MODULES = 4


class Corner(enum.Enum):
    """Which corner of the viewport the overlay sits in."""
    TOP_LEFT = "top left"
    TOP_RIGHT = "top right"
    BOTTOM_LEFT = "bottom left"
    BOTTOM_RIGHT = "bottom right"


def get_project_url(distribution: str = _DISTRIBUTION) -> Optional[str]:
    """Return the project's repository URL, from the installed package's own metadata.

    `distribution`: the *distribution* name, e.g. "raven-visualizer" - not the importable package name.

    Returns `None` if the distribution is not installed, or declares no repository. Callers are expected to
    fall back to a URL of their own rather than treat that as fatal: an overlay is a demo convenience, and
    a source checkout that was never installed is a normal way to run Raven.
    """
    try:
        project_urls = importlib.metadata.metadata(distribution).get_all("Project-URL") or []
    except importlib.metadata.PackageNotFoundError:
        logger.warning(f"get_project_url: distribution '{distribution}' is not installed; no URL to show.")
        return None

    # Each entry is "Label, URL". The label is what `[project.urls]` was keyed by in `pyproject.toml`.
    for entry in project_urls:
        label, _, url = entry.partition(",")
        if label.strip().lower() == "repository":
            return url.strip()

    logger.warning(f"get_project_url: distribution '{distribution}' declares no Repository URL.")
    return None


def matrix_to_pixels(matrix: List[List[bool]],
                     module_size: int,
                     quiet_zone: int,
                     foreground: Tuple[int, int, int, int],
                     background: Tuple[int, int, int, int]) -> Tuple[int, int, np.array]:
    """Rasterize a QR matrix. Returns `(width, height, rgba)`, with `rgba` as DPG wants it: floats in [0, 1].

    `matrix`: rows of columns, as `segno` produces. `matrix[row][column]`.
    `module_size`: pixels per module, in each direction.
    `quiet_zone`: width of the clear margin, in modules. Included in the returned size.
    `foreground`, `background`: RGBA, each component in [0, 255].

    A texture rather than one rectangle per module. Which is *faster* has not been measured and probably
    does not matter at this size - DPG and the GPU are both happy with a couple of hundred triangles, and a
    texture trades them for a little memory bandwidth. What it buys for certain is that the drawlist holds
    one item instead of a couple of hundred, rebuilt on every window resize, for an image that never
    changes.
    """
    modules = np.array(matrix, dtype=bool)
    padded = np.pad(modules, quiet_zone, constant_values=False)
    # Nearest-neighbour upscale. A QR module is a hard-edged square, so this is exact rather than an
    # approximation - and doing it here means DPG is handed the texture at its final size and never
    # resamples it, which it would do nearest-neighbour anyway.
    pixels = np.repeat(np.repeat(padded, module_size, axis=0), module_size, axis=1)

    colors = np.array([background, foreground], dtype=np.float32) / 255.0
    rgba = colors[pixels.astype(np.uint8)]
    height, width = pixels.shape
    return (width, height, rgba.ravel())


class QRCodeOverlay(gui_animation.Animation):
    def __init__(self,
                 url: str,
                 label: str = "Get Raven",
                 corner: Corner = Corner.BOTTOM_RIGHT,
                 module_size: int = 3,
                 margin: int = 16,
                 foreground: Tuple[int, int, int, int] = (0, 0, 0, 255),
                 background: Tuple[int, int, int, int] = (255, 255, 255, 255),
                 label_size: int = 15):
        """A QR code drawn over the viewport, in one corner.

        `url`: what the code encodes.
        `label`: short line of text above the code. Empty string for none.
        `corner`: which corner of the viewport to sit in.
        `module_size`: pixels per QR module. The code's pixel size is this times its module count,
                       which depends on how long `url` is.
        `margin`: pixels between the overlay and the edges of the viewport.
        `foreground`, `background`: RGBA. The defaults are the black-on-white a scanner expects; inverting
                                    them is possible and many scanners refuse it.
        `label_size`: font size for `label`, in pixels.

        Drawn into a *viewport* drawlist rather than a window, which is what makes it safe to leave on top
        of a live app: a DPG window swallows the mouse across its whole rect, so an overlay window would
        make its corner unclickable, while a viewport drawlist captures no input at all.
        """
        # Ambient, and that is load-bearing rather than a label. This animation never finishes, and the
        # apps' idle-framerate throttles ask `Animator.transient_count` whether anything is happening - so
        # a non-ambient permanent animation would hold every app that installs one at full framerate for
        # as long as it runs, which at an exhibit is all evening.
        super().__init__(ambient=True)
        self.url = url
        self.label = label
        self.corner = corner
        self.module_size = module_size
        self.margin = margin
        self.foreground = foreground
        self.background = background
        self.label_size = label_size

        self.matrix: List[List[bool]] = [[bool(module) for module in row]
                                         for row in segno.make(url, error="m").matrix]

        width, height, rgba = matrix_to_pixels(self.matrix, module_size, _QUIET_ZONE_MODULES,
                                               foreground, background)
        self.code_size = (width, height)
        # Its own registry, so that `uninstall` can drop the texture without knowing what else an app keeps
        # in its registries. DPG frees deleted items lazily, so the registry outliving the texture briefly
        # is fine; another overlay's registry is a separate item either way.
        self.texture_registry = dpg.add_texture_registry()
        self.texture = dpg.add_static_texture(width=width, height=height, default_value=rgba,
                                              parent=self.texture_registry)

        self.drawlist = dpg.add_viewport_drawlist(front=True)
        self._last_viewport_size: Optional[Tuple[int, int]] = None

    def _get_size(self) -> Tuple[int, int]:
        """Return the (width, height) the overlay occupies, in pixels."""
        code_width, code_height = self.code_size
        label_height = (self.label_size + self.module_size) if self.label else 0
        return (code_width, code_height + label_height)

    def _redraw(self, viewport_width: int, viewport_height: int) -> None:
        """Place the overlay for a viewport of the given size. Two draw items, or one without a label."""
        dpg.delete_item(self.drawlist, children_only=True)

        width, height = self._get_size()
        left = self.margin if self.corner in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT) else viewport_width - width - self.margin
        top = self.margin if self.corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT) else viewport_height - height - self.margin

        if self.label:
            dpg.draw_text((left, top), self.label, size=self.label_size,
                          color=self.background, parent=self.drawlist)
            top += self.label_size + self.module_size

        code_width, code_height = self.code_size
        dpg.draw_image(self.texture, (left, top), (left + code_width, top + code_height),
                       parent=self.drawlist)

    def render_frame(self, t: int) -> sym:
        """Reposition the overlay if the viewport has been resized. Never finishes."""
        viewport_size = (dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
        if viewport_size != self._last_viewport_size:
            self._last_viewport_size = viewport_size
            self._redraw(*viewport_size)
        return gui_animation.action_continue


def install(url: Optional[str] = None,
            label: str = "Get Raven",
            corner: Corner = Corner.BOTTOM_RIGHT,
            **kwargs) -> Optional[QRCodeOverlay]:
    """Put a QR code in a corner of the viewport, and keep it there.

    `url`: what to encode. `None` (default) reads the project's repository URL from the package metadata.
    `label`: short line of text above the code.
    `corner`: which corner to sit in.

    Any further keyword arguments go to `QRCodeOverlay`; see it for sizing and colours.

    Returns the overlay, already registered with Raven's GUI animator so that it follows a window resize -
    or `None` if no URL was given and none could be discovered, which is not treated as an error.

    Call once, after the viewport exists. Intended for a `--qr` command-line flag.
    """
    if url is None:
        url = get_project_url()
        if url is None:
            logger.warning("install: no URL given and none found in the package metadata; overlay not installed.")
            return None

    overlay = QRCodeOverlay(url=url, label=label, corner=corner, **kwargs)
    gui_animation.animator.add(overlay)
    logger.info(f"install: QR overlay for '{url}' installed in the {corner.value} corner.")
    return overlay


def uninstall(overlay: Union[QRCodeOverlay, None]) -> None:
    """Remove an overlay installed by `install`. Accepts `None`, so an uninstalled overlay needs no guard."""
    if overlay is None:
        return
    gui_animation.animator.cancel(overlay)
    # Reverse of the order they were created in.
    for item in (overlay.drawlist, overlay.texture_registry):
        if dpg.does_item_exist(item):
            dpg.delete_item(item)
