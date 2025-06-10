__all__ = ["init_module", "is_available", "process"]

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

from ..common.postprocessor import Postprocessor

postprocessor = None
postprocessor_lock = threading.Lock()

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
        logger.info("process: processing image")

        # Detect if the input is QOI and use the separate fast decoder if so, otherwise delegate to Pillow.
        img_data = io.BytesIO()
        img_data.write(input_stream.read())  # copy input into our own in-memory buffer, just in case the actual `input_stream` isn't rewindable
        img_data.seek(0)

        magic = img_data.read(4)  # file magic
        img_data.seek(0)

        if magic == b"qoif":
            logger.info("process: loading from QOI format")
            image_rgba = qoi.decode(img_data.getvalue())  # -> uint8 array of shape (h, w, c)
        else:
            logger.info("process: loading via Pillow")
            input_pil_image = PIL.Image.open(img_data)
            image_rgba = np.asarray(input_pil_image)  # maybe we don't need to convert regardless of whether `input_pil_image.mode` is "RGB" or "RGBA"?
        logger.info("process: image loaded, processing")

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

        if output_format.upper() == "QOI":
            logger.info("process: encoding to QOI format")
            encoded_image_bytes = qoi.encode(image_rgba.copy(order="C"))
        else:
            logger.info(f"process: encoding to {output_format} via Pillow")
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

        logger.info("process: all done")
    except Exception as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during `process` in module 'imagefx'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"process: failed: {type(exc)}: {exc}")
        raise

    return encoded_image_bytes
