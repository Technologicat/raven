"""A very simple PyTorch GPU configuration validator with automatic CPU fallback."""

__all__ = ["get_device_and_dtype",
           "validate"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Any, Dict

import torch

def get_device_and_dtype(record: Dict[str, Any]) -> (str, torch.dtype):
    """Validate a single GPU configuration record.

    Input format is::

        {"device_string": "cuda:0",
         "dtype": torch.float16}

    where::

      device_string is a Torch device string, and
      dtype is a Torch dtype or None (to indicate that the software component this record belongs to has no configurable dtype).

    Return the tuple `device_string, torch_dtype` after validation.
    """
    device_string = record["device_string"]
    torch_dtype = record.get("dtype", None)  # not all components have a specifiable dtype

    if device_string.startswith("cuda"):  # Nvidia
        if not torch.cuda.is_available():
            logger.warning(f"CUDA backend specified in config (device string '{device_string}'), but CUDA not available. Using CPU instead.")
            device_string = "cpu"

    elif device_string.startswith("mps"):  # Mac, Apple Metal Performance Shaders
        if not torch.backends.mps.is_available():
            logger.warning(f"MPS backend specified in config (device string '{device_string}'), but MPS not available. Using CPU instead.")
            device_string = "cpu"

    if device_string == "cpu":  # no "elif" because also as fallback if CUDA/MPS wasn't available
        if torch_dtype is torch.float16:
            logger.warning("dtype is set to torch.float16, but device 'cpu' does not support half precision. Using torch.float32 instead.")
            torch_dtype = torch.float32

    return device_string, torch_dtype

def validate(device_config):
    """Validate a GPU configuration.

    The GPU configuration records are modified in-place.

    Input format is::

        {"my_component_name": {"device_string": "cuda:0",
                               "dtype": torch.float16},
         ...}

    where `"my_component_name"` is arbitrary.

    This checks that CUDA/MPS is available (if it was requested), and if not, falls back to CPU.

    The dtype is checked so that if, after the fallback logic, a component is running on CPU
    but `torch.float16` was requested, the dtype is set to `torch.float32` (because CPU
    does not support `float16`).

    Additionally, each record gets the field "device_name" injected. For CUDA devices, this is
    the human-readable GPU name. For MPS devices, it is always "MPS", and for CPU, "CPU".
    """
    unique_cuda_gpus = set()
    for component, record in device_config.items():
        if "device_string" not in record:  # skip software components that have no device configuration
            continue

        device_string, torch_dtype = get_device_and_dtype(record)

        record["device_string"] = device_string
        record["torch_dtype"] = torch_dtype

        if device_string.startswith("cuda") and torch.cuda.is_available():
            record["device_name"] = torch.cuda.get_device_name(record["device_string"])
        elif device_string.startswith("cuda") and torch.backends.mps.is_available():
            record["device_name"] = "MPS"
        else:
            record["device_name"] = "CPU"

        if record["device_string"].startswith("cuda"):
            unique_cuda_gpus.add((record["device_string"], record["device_name"]))

        dtype_string = f", dtype {record['torch_dtype']}" if record['torch_dtype'] is not None else ""
        logger.info(f"Compute device for '{component}' is '{record['device_string']}'{dtype_string}")

    if torch.cuda.is_available():
        for device_string, device_name in sorted(unique_cuda_gpus):
            logger.info(f"Device info for CUDA GPU {device_string} ({device_name}):")
            logger.info(f"    {torch.cuda.get_device_properties(device_string)}")
            logger.info(f"    Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(device_string))}")
            logger.info(f"    Detected CUDA version {torch.version.cuda}")
    # TODO: Torch MPS backend info? There doesn't seem to be anything useful in `torch.backends.mps`.
