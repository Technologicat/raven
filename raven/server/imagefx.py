"""Server-side image postprocessing and upscaling.

If you want these features locally, just directly use `raven.common.postprocessor` and `raven.common.upscaler`;
they operate on Torch tensors and can run on a local GPU.

This module is intended for situations where it is preferred to use the server's GPU.
"""

__all__ = ["init_module", "is_available", "process", "upscale"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import io
import threading
import traceback
from typing import Any, Dict, List, Union

from colorama import Fore, Style

import PIL.Image
import qoi

import numpy as np

import torch

from ..common.video.postprocessor import Postprocessor
from ..common.video.upscaler import Upscaler

postprocessor = None
postprocessor_lock = threading.Lock()

upscaler = None
upscaler_settings = {}
upscaler_lock = threading.Lock()

def init_module(device_string: str, torch_dtype: Union[str, torch.dtype]) -> None:
    global postprocessor
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}imagefx{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device_string}{Style.RESET_ALL}'...")
    try:
        postprocessor = Postprocessor(device_string, torch_dtype, chain=[])
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")
        postprocessor = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return (postprocessor is not None)

# --------------------------------------------------------------------------------

def decode_image(stream) -> np.array:
    """Read an image from a Flask stream or anything duck-compatible with it.

    The whole stream will be read into a temporary buffer to guarantee seekability,
    needed for format detection.

    Returns the image data as an `np.array` of size [h, w, c], type uint8.
    """
    logger.info("decode_image: detecting input format")

    # Detect if the input is QOI and use the separate fast decoder if so, otherwise delegate to Pillow.
    img_data = io.BytesIO()
    img_data.write(stream.read())  # copy input into our own in-memory buffer, just in case the actual `input_stream` isn't rewindable
    img_data.seek(0)

    magic = img_data.read(4)  # file magic
    img_data.seek(0)

    if magic == b"qoif":
        logger.info("decode_image: decoding from QOI format")
        image_rgba = qoi.decode(img_data.getvalue())  # -> uint8 array of shape (h, w, c)
    else:
        logger.info("decode_image: decoding via Pillow")
        input_pil_image = PIL.Image.open(img_data)
        image_rgba = np.asarray(input_pil_image)  # maybe we don't need to convert regardless of whether `input_pil_image.mode` is "RGB" or "RGBA"?
    logger.info("decode_image: done")

    return image_rgba

def encode_image(image_rgba: np.array, output_format: str) -> bytes:
    """Encode an image into `output_format`.

    The input is an `np.array` of size [h, w, c], type uint8.

    Returns the encoded image as a `bytes` object.
    """
    if output_format.upper() == "QOI":
        logger.info("encode_image: encoding to QOI format")
        encoded_image_bytes = qoi.encode(image_rgba.copy(order="C"))
    else:
        logger.info(f"encode_image: encoding to {output_format.upper()} via Pillow")
        output_pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
        if image_rgba.shape[2] == 4:
            alpha_channel = image_rgba[:, :, 3]
            output_pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))

        buffer = io.BytesIO()
        if output_format.upper() == "PNG":
            kwargs = {"compress_level": 1}
        elif output_format.upper() == "TGA":
            kwargs = {"compression": "tga_rle"}
        else:
            kwargs = {}
        output_pil_image.save(buffer,
                              format=output_format.upper(),
                              **kwargs)
        encoded_image_bytes = buffer.getvalue()
    logger.info("encode_image: done")
    return encoded_image_bytes

# --------------------------------------------------------------------------------

# TODO: the input is a flask.request.file.stream; what's the type of that?
def process(input_stream,
            output_format: str,
            postprocessor_chain: List[Dict[str, Any]]) -> bytes:
    """Process an image through Raven-avatar's postprocessor.

    `input_stream`: a `flask.request.file.stream` containing an image file
    `output_format`: format to encode output to (e.g. "png")
    `postprocessor_chain`: formatted as in `raven.avatar.common.config.postprocessor_defaults`

    Returns a `bytes` object containing the processed image, encoded in `output_format`.
    """
    try:
        image_rgba = decode_image(input_stream)

        logger.info("process: processing image")
        image_rgba = np.array(image_rgba, dtype=np.float32) / 255  # uint8 -> float [0, 1]
        with torch.no_grad():
            h, w, c = image_rgba.shape
            image_rgba = torch.tensor(image_rgba).to(postprocessor.device).to(postprocessor.dtype)
            image_rgba = torch.transpose(image_rgba.reshape(h * w, c), 0, 1).reshape(c, h, w)

            with postprocessor_lock:  # because we replace the chain
                postprocessor.chain = postprocessor_chain
                postprocessor.render_into(image_rgba)

            image_rgba = torch.transpose(image_rgba.reshape(c, h * w), 0, 1).reshape(h, w, c)
            image_rgba = (255.0 * image_rgba).byte()  # float [0, 1] -> uint8
            image_rgba = image_rgba.detach().cpu().numpy()

        encoded_image_bytes = encode_image(image_rgba, output_format)

        logger.info("process: all done")
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during `process` in module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"process: failed: {type(exc)}: {exc}")
        raise

    return encoded_image_bytes

# TODO: the input is a flask.request.file.stream; what's the type of that?
def upscale(input_stream,
            output_format: str,
            upscaled_width: int,
            upscaled_height: int,
            preset: str,
            quality: str) -> bytes:
    """Upscale an image with Anime4K.

    `input_stream`: a `flask.request.file.stream` containing an image file
    `output_format`: format to encode output to (e.g. "png")
    `upscaled_width`, `upscaled_height`: desired output image resolution.

    For `preset` and `quality`, see `raven.avatar.common.upscaler`.

    Returns a `bytes` object containing the upscaled image, encoded in `output_format`.
    """
    global upscaler

    try:
        image_rgba = decode_image(input_stream)

        image_rgba = np.array(image_rgba, dtype=np.float32) / 255  # uint8 -> float [0, 1]
        with torch.no_grad():
            h, w, c = image_rgba.shape
            image_rgba = torch.tensor(image_rgba).to(postprocessor.device).to(postprocessor.dtype)
            image_rgba = torch.transpose(image_rgba.reshape(h * w, c), 0, 1).reshape(c, h, w)

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

            c, h, w = image_rgba.shape  # after upscale (in Torch layout)
            image_rgba = torch.transpose(image_rgba.reshape(c, h * w), 0, 1).reshape(h, w, c)
            image_rgba = (255.0 * image_rgba).byte()  # float [0, 1] -> uint8
            image_rgba = image_rgba.detach().cpu().numpy()

        encoded_image_bytes = encode_image(image_rgba, output_format)

        logger.info("upscale: all done")
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during `upscale` in module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"upscale: failed: {type(exc)}: {exc}")
        raise

    return encoded_image_bytes
