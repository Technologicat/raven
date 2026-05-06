"""A very simple PyTorch GPU configuration validator with automatic CPU fallback."""

__all__ = ["get_device_and_dtype",
           "validate",
           "cuda_sanity_check"]

import logging
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

    Return the tuple `device_string, dtype` after validation.
    """
    device_string = record["device_string"]
    dtype = record.get("dtype", None)  # not all components have a specifiable dtype

    if device_string.startswith("cuda"):  # Nvidia
        if not torch.cuda.is_available():
            logger.warning(f"CUDA backend specified in config (device string '{device_string}'), but CUDA not available. Using CPU instead.")
            device_string = "cpu"

    elif device_string.startswith("mps"):  # Mac, Apple Metal Performance Shaders
        if not torch.backends.mps.is_available():
            logger.warning(f"MPS backend specified in config (device string '{device_string}'), but MPS not available. Using CPU instead.")
            device_string = "cpu"

    if device_string == "cpu":  # no "elif" because also as fallback if CUDA/MPS wasn't available
        if dtype is torch.float16:
            logger.warning("dtype is set to torch.float16, but device 'cpu' does not support half precision. Using torch.float32 instead.")
            dtype = torch.float32

    return device_string, dtype

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

        device_string, dtype = get_device_and_dtype(record)

        record["device_string"] = device_string
        record["dtype"] = dtype

        if device_string.startswith("cuda") and torch.cuda.is_available():
            record["device_name"] = torch.cuda.get_device_name(record["device_string"])
        elif device_string.startswith("cuda") and torch.backends.mps.is_available():
            record["device_name"] = "MPS"
        else:
            record["device_name"] = "CPU"

        if record["device_string"].startswith("cuda"):
            unique_cuda_gpus.add((record["device_string"], record["device_name"]))

        dtype_string = f", dtype {record['dtype']}" if record['dtype'] is not None else ""
        logger.info(f"Compute device for '{component}' is '{record['device_string']}'{dtype_string}")

    if torch.cuda.is_available():
        for device_string, device_name in sorted(unique_cuda_gpus):
            logger.info(f"Device info for CUDA GPU {device_string} ({device_name}):")
            logger.info(f"    {torch.cuda.get_device_properties(device_string)}")
            logger.info(f"    Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(device_string))}")
            logger.info(f"    Detected CUDA version {torch.version.cuda}")
    # TODO: Torch MPS backend info? There doesn't seem to be anything useful in `torch.backends.mps`.

def cuda_sanity_check() -> bool:
    """Probe CUDA + NVRTC at startup; warn loudly on a known silent-failure mode.

    `torch.cuda.is_available()` returns True even when JIT-compilation paths are
    broken — most commonly a misconfigured or missing NVRTC runtime (e.g.
    `libnvrtc-builtins.so` not on the loader path, or a CUDA-version skew between
    the bundled and host NVRTC). The error then surfaces only when something
    triggers JIT compilation (postprocessor warm-up, `torch.compile`, jiterator
    paths, …), typically well after the server has reported itself ready.

    Compile a trivial element-wise kernel via the jiterator path, which exercises
    NVRTC end-to-end. ~300 ms one-shot at the first call in a process; cheap on
    subsequent calls.

    Returns True if CUDA + NVRTC are healthy, or if no CUDA backend was reported
    available (CPU-only setups bypass the NVRTC stack entirely and need no
    probe). Returns False if a probe was attempted and failed; the caller may
    use this to abort or downgrade.

    There is no MPS counterpart to this probe — not because MPS failures are eager
    (they aren't; they surface when the offending op runs, same as the NVRTC class
    of CUDA failures) but because there's no analogous infra-layer split that a
    startup probe could exercise. CUDA splits cleanly into "is the device there"
    (`cuda.is_available()`) and "does JIT compilation work" (NVRTC), and a tiny
    fixed kernel exercises NVRTC for any CUDA workload, so a single startup probe
    catches the gap. On MPS there's no equivalent shared component that can be
    silently broken while `mps.is_available()` returns True; the failures that
    matter (this op has no MPS implementation, this dtype is unsupported) depend
    entirely on which ops the actual workload calls, so a generic probe wouldn't
    catch anything `validate()` doesn't already catch — only running the real
    workload does.
    """
    if not torch.cuda.is_available():
        return True
    try:
        fn = torch.cuda.jiterator._create_jit_fn(
            "template <typename T> T raven_probe(T a) { return a + T(1); }"
        )
        x = torch.zeros(1, device="cuda")
        fn(x).cpu()  # `.cpu()` forces a sync so a launch failure is observed here, not later
    except Exception as exc:
        logger.warning(
            f"cuda_sanity_check: CUDA reports available, but the NVRTC smoke test failed. "
            f"JIT-compiled code paths will likely error later. Reason: {type(exc)}: {exc}"
        )
        return False
    logger.info("cuda_sanity_check: CUDA + NVRTC OK.")
    return True
