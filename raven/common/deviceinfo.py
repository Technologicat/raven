"""PyTorch device configuration validator with `gpu` autodetect alias and CPU fallback."""

__all__ = ["get_device_and_dtype",
           "validate",
           "cuda_sanity_check",
           "VRAMLedger"]

import logging
logger = logging.getLogger(__name__)

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

# Recognized GPU backends, in the priority order tried by the `"gpu"` autodetect alias.
# Each entry: (prefix used in device strings, human-readable label, probe callable).
#
# Every probe begins with a `hasattr` check on the namespace before calling its
# `is_available()`. `torch.cuda` and `torch.backends.mps` are reliably present in
# Raven's supported PyTorch range (>=2.4) even on builds without those backends
# compiled in, but the gates are kept for symmetry — a future PyTorch could rename
# or reorganize, and gating all four lines costs essentially nothing. `torch.xpu`
# (Intel Arc) and `torch.is_vulkan_available` are newer / build-conditional, so the
# gate matters there. ROCm (AMD) presents to PyTorch as `cuda` and is covered by
# that entry. `xla` (TPU) is intentionally absent: its lazy-graph placement model
# doesn't slot into Raven's eager-mode architecture, so silently mapping `"gpu"`
# onto it would be misleading.
_GPU_BACKENDS: List[Tuple[str, str, Callable[[], bool]]] = [
    ("cuda",   "CUDA",   lambda: hasattr(torch, "cuda") and torch.cuda.is_available()),
    ("mps",    "MPS",    lambda: hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    ("xpu",    "XPU",    lambda: hasattr(torch, "xpu") and torch.xpu.is_available()),
    ("vulkan", "Vulkan", lambda: hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available()),
]

def _backend_for(device_string: str) -> Optional[Tuple[str, str, Callable[[], bool]]]:
    """If `device_string` names a recognized GPU backend (with or without index), return
    its `_GPU_BACKENDS` row. Else `None` (covers `"cpu"`, `"gpu"`, and unrecognized strings)."""
    for entry in _GPU_BACKENDS:
        prefix = entry[0]
        if device_string == prefix or device_string.startswith(prefix + ":"):
            return entry
    return None

def _autodetect_gpu() -> str:
    """Resolve `device_string == "gpu"`: pick the single available GPU backend, or fall back to CPU.

    Probes each entry in `_GPU_BACKENDS` in declaration order. Single match wins. Multiple
    matches raise `RuntimeError` — the only way to land here is a machine with two distinct
    GPU vendors active simultaneously (e.g. NVIDIA + Intel Arc), which is rare enough that
    forcing an explicit pick is the right move. No match → CPU with an info-level log.

    Returns the resolved literal device string.
    """
    available = [(prefix, label) for prefix, label, probe in _GPU_BACKENDS if probe()]
    if len(available) == 0:
        logger.info("get_device_and_dtype: 'gpu' autodetect found no GPU backend; using CPU.")
        return "cpu"
    if len(available) > 1:
        labels = ", ".join(label for _, label in available)
        prefixes = ", ".join(prefix for prefix, _ in available)
        raise RuntimeError(
            f"get_device_and_dtype: 'gpu' autodetect found multiple GPU backends ({labels}). "
            f"Specify exactly one in the config (one of: {prefixes})."
        )
    prefix, label = available[0]
    # Only CUDA carries a meaningful index — pin to `:0` so the resolved string is unambiguous,
    # matching the existing CUDA_VISIBLE_DEVICES + always-`cuda:0` convention. Other backends
    # have no per-device index in PyTorch's device-string vocabulary, so they stay bare.
    resolved = f"{prefix}:0" if prefix == "cuda" else prefix
    logger.info(f"get_device_and_dtype: 'gpu' autodetect resolved to {label} (device_string='{resolved}').")
    return resolved

def _device_label(device_string: str) -> str:
    """Return a human-readable label for a (post-resolution) device string."""
    if device_string == "cpu":
        return "CPU"
    backend = _backend_for(device_string)
    if backend is None:
        return device_string  # unrecognized; surface as-is so the user can see what got through
    prefix, label, probe = backend
    if not probe():
        # Shouldn't happen post-resolution (the resolver would've fallen back to CPU), but if
        # the caller passed an unresolved record straight in, don't lie about the device.
        return "CPU"
    if prefix == "cuda":
        return torch.cuda.get_device_name(device_string)
    return label

def get_device_and_dtype(record: Dict[str, Any]) -> Tuple[str, torch.dtype]:
    """Validate and resolve a single device-configuration record.

    Input format::

        {"device_string": "gpu",
         "dtype": torch.float16}

    `device_string` is one of:

      - `"gpu"`: autodetect — pick whichever GPU backend is available. Tries CUDA, MPS,
        XPU, Vulkan in that order. Single match wins; multiple distinct backends present
        raises `RuntimeError` (specify one explicitly). No GPU available falls back to CPU.
      - `"cuda"` / `"cuda:N"` / `"mps"` / `"xpu"` / `"vulkan"`: use that backend if
        available; else fall back to CPU with a warning. Explicit choices are honored —
        no cross-backend fallback (an explicit `"mps"` on a non-Mac with CUDA goes to
        CPU, not CUDA, because the user picked MPS deliberately).
      - `"cpu"`: unchanged.

    `dtype` is a Torch dtype, or `None` for components without a configurable dtype.
    If the resolved device is CPU but `torch.float16` was requested, falls back to
    `torch.float32` (CPU does not support half precision).

    Returns `(resolved_device_string, possibly_coerced_dtype)`.
    """
    device_string = record["device_string"]
    dtype = record.get("dtype", None)  # not all components have a specifiable dtype

    if device_string == "gpu":
        device_string = _autodetect_gpu()
    else:
        backend = _backend_for(device_string)
        if backend is not None:
            prefix, label, probe = backend
            if not probe():
                logger.warning(f"get_device_and_dtype: {label} backend specified in config (device string '{device_string}'), but {label} not available. Using CPU instead.")
                device_string = "cpu"
        # Else: `"cpu"` or an unrecognized string — pass through. Unrecognized strings
        # will surface immediately when `torch.device()` is called downstream.

    if device_string == "cpu" and dtype is torch.float16:
        logger.warning("get_device_and_dtype: dtype is set to torch.float16, but device 'cpu' does not support half precision. Using torch.float32 instead.")
        dtype = torch.float32

    return device_string, dtype

def validate(device_config: Dict[str, Dict[str, Any]]) -> None:
    """Validate every device-configuration record in `device_config`. Modifies in-place.

    Input format::

        {"my_component_name": {"device_string": "gpu",
                               "dtype": torch.float16},
         ...}

    where `"my_component_name"` is arbitrary.

    For each record:

      - Resolves `device_string` per `get_device_and_dtype` (autodetect / backend-availability
        check / CPU fallback).
      - Coerces `dtype` to `torch.float32` if the resolved device is CPU and `torch.float16`
        was requested (CPU has no half precision).
      - Injects a `device_name` field with a human-readable label: the GPU model name for
        CUDA, the backend name (`"MPS"`, `"XPU"`, `"Vulkan"`) for the rest, `"CPU"` otherwise.
    """
    unique_cuda_gpus = set()
    for component, record in device_config.items():
        if "device_string" not in record:  # skip software components that have no device configuration
            continue

        device_string, dtype = get_device_and_dtype(record)

        record["device_string"] = device_string
        record["dtype"] = dtype
        record["device_name"] = _device_label(device_string)

        if device_string.startswith("cuda"):
            unique_cuda_gpus.add((device_string, record["device_name"]))

        dtype_string = f", dtype {dtype}" if dtype is not None else ""
        logger.info(f"Compute device for '{component}' is '{device_string}'{dtype_string}")

    if torch.cuda.is_available():
        for device_string, device_name in sorted(unique_cuda_gpus):
            logger.info(f"Device info for CUDA GPU {device_string} ({device_name}):")
            logger.info(f"    {torch.cuda.get_device_properties(device_string)}")
            logger.info(f"    Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(device_string))}")
            logger.info(f"    Detected CUDA version {torch.version.cuda}")
    # TODO: per-backend info for MPS / XPU / Vulkan? Their public APIs don't seem to expose anything analogous to `torch.cuda.get_device_properties` yet.

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


# --------------------------------------------------------------------------------
# VRAM accounting

_GIB = 1024 ** 3

class VRAMLedger:
    """Record how much GPU memory each of a sequence of load steps consumes.

    Answers "will this set of models fit on that card, and which one is the expensive
    one" - a question that has to be settled per machine, because the answer depends on
    the card, on the model choices, and on what else is resident.

    Usage::

        ledger = VRAMLedger()
        with ledger.measure("avatar", "cuda:0"):
            avatar.init_module(...)
        with ledger.measure("stt", "cuda:0"):
            stt.init_module(...)
        print(ledger.format_report())

    **Two numbers per step, and the gap between them is the interesting part.**
    `driver` is taken from `torch.cuda.mem_get_info`, i.e. straight from the driver, so it
    counts every byte the step took regardless of who allocated it. `torch` is PyTorch's
    own caching-allocator figure, which sees only what PyTorch allocated. A step that loads
    a model through some other CUDA allocator - spaCy on GPU goes through cupy, for instance
    - shows up in the first and is invisible to the second, and that divergence is worth
    seeing rather than averaging away.

    **Three caveats, all of which affect how the numbers should be read:**

      - *The driver figure is device-global.* Anything else using the same GPU during a
        measurement lands in the step that happened to be running. Measure on an otherwise
        idle card - in particular with no LLM backend holding it.
      - *This measures loading, not running.* Models that allocate lazily on first inference
        contribute nothing here, so these figures are a floor on the real footprint. Peak
        usage needs a warm-up call per module, which this does not do.
      - *Shared models are attributed to whoever loaded first.* When one step reuses a model
        another already loaded, the cost lands entirely on the earlier step and the later one
        looks free. The totals stay right; the per-step split is a function of load order.

    Non-CUDA devices are recorded as unmeasured rather than as zero: MPS, XPU and Vulkan
    expose no equivalent of `mem_get_info`, and reporting 0 would read as "free".
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._baseline: Dict[str, Dict[str, int]] = {}  # device_string -> {"free", "total"} at first touch

    def _snapshot(self, device_string: str) -> Optional[Tuple[int, int]]:
        """Return `(free_bytes, allocated_by_torch_bytes)` for a CUDA device, or `None` if not measurable."""
        if not device_string.startswith("cuda") or not torch.cuda.is_available():
            return None
        torch.cuda.synchronize(device_string)  # let queued work land, or it lands inside the next step
        free, total = torch.cuda.mem_get_info(device_string)
        if device_string not in self._baseline:
            self._baseline[device_string] = {"free": free, "total": total}
        return free, torch.cuda.memory_allocated(device_string)

    @contextlib.contextmanager
    def measure(self, label: str, device_string: str) -> Iterator[None]:
        """Context manager: attribute whatever happens in the block to `label`, on `device_string`."""
        before = self._snapshot(device_string)
        try:
            yield
        finally:
            after = self._snapshot(device_string)
            if before is None or after is None:
                self._entries.append({"label": label, "device_string": device_string,
                                      "driver_bytes": None, "torch_bytes": None})
            else:
                free_before, torch_before = before
                free_after, torch_after = after
                self._entries.append({"label": label, "device_string": device_string,
                                      "driver_bytes": free_before - free_after,
                                      "torch_bytes": torch_after - torch_before})

    def entries(self) -> List[Dict[str, Any]]:
        """Return the recorded steps, in load order. Byte counts; `None` where not measurable."""
        return list(self._entries)

    def totals(self) -> Dict[str, Dict[str, Optional[int]]]:
        """Per device: total VRAM, and how much of it the measured steps consumed.

        `consumed` is measured against the device's state when the ledger first touched it, so it
        excludes whatever was already resident - including the CUDA context, which is created before
        any module loads and belongs to no module.
        """
        out: Dict[str, Dict[str, Optional[int]]] = {}
        for device_string, base in self._baseline.items():
            snapshot = self._snapshot(device_string)
            free_now = snapshot[0] if snapshot is not None else None
            out[device_string] = {"total_bytes": base["total"],
                                  "free_at_start_bytes": base["free"],
                                  "free_now_bytes": free_now,
                                  "consumed_bytes": (base["free"] - free_now) if free_now is not None else None}
        return out

    def format_report(self) -> str:
        """Return the ledger as a human-readable table."""
        if not self._entries:
            return "VRAM: nothing measured."

        def gib(n: Optional[int]) -> str:
            return "     -" if n is None else f"{n / _GIB:6.2f}"

        lines = ["VRAM used by module loading (GiB; driver = all allocators, torch = PyTorch's only)",
                 f"  {'module':<14} {'device':<10} {'driver':>7} {'torch':>7}"]
        divergent = []
        for entry in self._entries:
            lines.append(f"  {entry['label']:<14} {entry['device_string']:<10} "
                         f"{gib(entry['driver_bytes']):>7} {gib(entry['torch_bytes']):>7}")
            driver, torch_bytes = entry["driver_bytes"], entry["torch_bytes"]
            # A step whose driver cost is well above what PyTorch accounts for allocated through some
            # other path. Worth naming: it is the case the torch-only figure would have missed entirely.
            if driver is not None and torch_bytes is not None and driver - torch_bytes > 0.25 * _GIB:
                divergent.append(entry["label"])

        for device_string, totals in self.totals().items():
            consumed, total = totals["consumed_bytes"], totals["total_bytes"]
            free_now = totals["free_now_bytes"]
            lines.append(f"  {'TOTAL':<14} {device_string:<10} {gib(consumed):>7}"
                         f"   of {gib(total)} GiB on device, {gib(free_now)} GiB free")

        if divergent:
            lines.append(f"  Note: {', '.join(divergent)} allocated substantially outside PyTorch's allocator "
                         f"(spaCy on GPU uses cupy, for instance).")
        lines.append("  Loading only - models that allocate on first inference are not counted here.")
        return "\n".join(lines)
