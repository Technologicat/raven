"""Server-side image postprocessing and upscaling.

If you want these features locally, just directly use `raven.common.video.postprocessor` and `raven.common.video.upscaler`;
they operate on Torch tensors and can run on a local GPU.

This module is intended for situations where it is preferred to use the server's GPU.
"""

__all__ = ["init_module", "is_available", "process", "upscale"]

import logging
logger = logging.getLogger(__name__)

import threading
import traceback
from typing import Any, BinaryIO, Dict, List, Union

from colorama import Fore, Style

import torch

from ...common.image import codec as imagecodec
from ...common.image import utils as imageutils
from ...common.video.postprocessor import Postprocessor
from ...common.video.upscaler import Upscaler

postprocessor = None
postprocessor_lock = threading.Lock()

upscaler = None
upscaler_settings = {}
upscaler_lock = threading.Lock()

def init_module(device_string: str, dtype: Union[str, torch.dtype]) -> None:
    global postprocessor
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}imagefx{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        postprocessor = Postprocessor(device_string, dtype, chain=[])
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        postprocessor = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (postprocessor is not None)

# --------------------------------------------------------------------------------

def process(input_stream: BinaryIO,
            output_format: str,
            postprocessor_chain: List[Dict[str, Any]]) -> bytes:
    """Process an image through Raven-avatar's postprocessor.

    `input_stream`: a `flask.request.file.stream` (werkzeug `FileStorage.stream`, typed as `BinaryIO`) containing an image file
    `output_format`: format to encode output to (e.g. "png")
    `postprocessor_chain`: formatted as in `raven.server.config.postprocessor_defaults`

    Returns a `bytes` object containing the processed image, encoded in `output_format`.
    """
    try:
        image_rgba = imageutils.ensure_rgba(imagecodec.decode(input_stream))

        logger.info("process: processing image")
        with torch.inference_mode():
            image_rgba = imageutils.np_to_tensor(image_rgba, device=postprocessor.device,
                                                 dtype=postprocessor.dtype, batch=False)

            with postprocessor_lock:  # because we replace the chain
                postprocessor.chain = postprocessor_chain
                postprocessor.render_into(image_rgba)

            image_rgba = imageutils.tensor_to_np(image_rgba)

        encoded_image_bytes = imagecodec.encode(image_rgba, output_format)

        logger.info("process: all done")
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during `process` in module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"process: failed: {type(exc)}: {exc}")
        raise

    return encoded_image_bytes

def upscale(input_stream: BinaryIO,
            output_format: str,
            upscaled_width: int,
            upscaled_height: int,
            preset: str,
            quality: str) -> bytes:
    """Upscale an image with Anime4K.

    `input_stream`: a `flask.request.file.stream` (werkzeug `FileStorage.stream`, typed as `BinaryIO`) containing an image file
    `output_format`: format to encode output to (e.g. "png")
    `upscaled_width`, `upscaled_height`: desired output image resolution.

    For `preset` and `quality`, see `raven.common.video.upscaler`.

    Returns a `bytes` object containing the upscaled image, encoded in `output_format`.
    """
    global upscaler

    try:
        image_rgba = imageutils.ensure_rgba(imagecodec.decode(input_stream))

        with torch.inference_mode():
            image_rgba = imageutils.np_to_tensor(image_rgba, device=postprocessor.device,
                                                 dtype=postprocessor.dtype, batch=False)

            with upscaler_lock:
                # Instantiate upscaler at first run, and re-instantiate when settings change
                old_upscaled_width = upscaler_settings.get("upscaled_width", None)
                old_upscaled_height = upscaler_settings.get("upscaled_height", None)
                old_preset = upscaler_settings.get("preset", None)
                old_quality = upscaler_settings.get("quality", None)
                if upscaled_width != old_upscaled_width or upscaled_height != old_upscaled_height or preset != old_preset or quality != old_quality:
                    logger.info("upscale: instantiating upscaler")
                    device = postprocessor.device
                    dtype = postprocessor.dtype
                    upscaler = Upscaler(device=device,
                                        dtype=dtype,
                                        upscaled_width=upscaled_width,
                                        upscaled_height=upscaled_height,
                                        preset=preset,
                                        quality=quality)
                    upscaler_settings["upscaled_width"] = upscaled_width
                    upscaler_settings["upscaled_height"] = upscaled_height
                    upscaler_settings["preset"] = preset
                    upscaler_settings["quality"] = quality
                logger.info("upscale: upscaling")
                image_rgba = upscaler.upscale(image_rgba)

            image_rgba = imageutils.tensor_to_np(image_rgba)

        encoded_image_bytes = imagecodec.encode(image_rgba, output_format)

        logger.info("upscale: all done")
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during `upscale` in module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"upscale: failed: {type(exc)}: {exc}")
        raise

    return encoded_image_bytes
