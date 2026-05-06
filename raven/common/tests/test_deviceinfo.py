"""Tests for raven.common.deviceinfo — device-string resolution and labeling.

Exercises the resolution rules in `get_device_and_dtype` and the labeling pass
in `validate` across mocked backend permutations. Real-hardware probes (the
NVRTC sanity check) are out of scope for unit tests; they live with the apps
that call them.
"""

from unittest.mock import patch

import pytest
import torch

from raven.common import deviceinfo


def _mock_backends(*, cuda: bool = False, mps: bool = False, xpu: bool = False, vulkan: bool = False):
    """Context manager that pins all four backend probes to fixed boolean states.

    Patches the probe functions on the `torch` namespace plus the CUDA-side metadata
    lookups that `validate` invokes in its post-loop "Device info for CUDA GPU ..."
    section. The metadata patches matter on CPU-only PyTorch builds (e.g. CI runners),
    where unmocked `get_device_properties` / `get_device_capability` raise
    `AssertionError("Torch not compiled with CUDA enabled")` even when
    `cuda.is_available()` has been mocked to return True.
    """
    patches = [
        patch.object(torch.cuda, "is_available", return_value=cuda),
        patch.object(torch.backends.mps, "is_available", return_value=mps),
        patch.object(torch.cuda, "get_device_name", return_value="MOCK GPU"),
        patch.object(torch.cuda, "get_device_properties", return_value="MOCK GPU PROPERTIES"),
        patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
    ]
    if hasattr(torch, "xpu"):
        patches.append(patch.object(torch.xpu, "is_available", return_value=xpu))
    if hasattr(torch, "is_vulkan_available"):
        patches.append(patch.object(torch, "is_vulkan_available", return_value=vulkan))

    class _MultiPatch:
        def __enter__(self):
            for p in patches:
                p.start()
            return self
        def __exit__(self, *args):
            for p in patches:
                p.stop()
    return _MultiPatch()


# ---------------------------------------------------------------------------
# Tests: get_device_and_dtype — explicit backend strings
# ---------------------------------------------------------------------------

class TestExplicitBackend:
    """Explicit backend names are honored if available, else fall back to CPU. No cross-backend fallback."""

    def test_cuda_when_available(self):
        with _mock_backends(cuda=True):
            ds, dt = deviceinfo.get_device_and_dtype({"device_string": "cuda:0", "dtype": torch.float16})
            assert ds == "cuda:0"
            assert dt is torch.float16

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        with _mock_backends(cuda=False):
            ds, dt = deviceinfo.get_device_and_dtype({"device_string": "cuda:0", "dtype": torch.float16})
            assert ds == "cpu"
            assert dt is torch.float32  # half precision coerced on CPU

    def test_mps_when_available(self):
        with _mock_backends(mps=True):
            ds, dt = deviceinfo.get_device_and_dtype({"device_string": "mps", "dtype": torch.float32})
            assert ds == "mps"

    def test_mps_falls_back_to_cpu_when_unavailable(self):
        with _mock_backends(mps=False):
            ds, _ = deviceinfo.get_device_and_dtype({"device_string": "mps", "dtype": torch.float32})
            assert ds == "cpu"

    def test_explicit_mps_does_not_promote_to_cuda(self):
        """Explicit choices are deliberate — `mps` on a CUDA-only box goes to CPU, not CUDA."""
        with _mock_backends(cuda=True, mps=False):
            ds, _ = deviceinfo.get_device_and_dtype({"device_string": "mps", "dtype": torch.float32})
            assert ds == "cpu"

    def test_cpu_passes_through(self):
        with _mock_backends(cuda=True):
            ds, dt = deviceinfo.get_device_and_dtype({"device_string": "cpu", "dtype": torch.float32})
            assert ds == "cpu"
            assert dt is torch.float32


# ---------------------------------------------------------------------------
# Tests: get_device_and_dtype — `gpu` autodetect
# ---------------------------------------------------------------------------

class TestGpuAutodetect:
    """`gpu` picks the single available backend, errors on ambiguity, falls back to CPU on none."""

    def test_resolves_to_cuda_with_index(self):
        with _mock_backends(cuda=True):
            ds, _ = deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": torch.float16})
            assert ds == "cuda:0"  # CUDA gets the implicit `:0`

    def test_resolves_to_mps_without_index(self):
        with _mock_backends(mps=True):
            ds, _ = deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": torch.float32})
            assert ds == "mps"  # non-CUDA backends stay bare

    def test_no_gpu_falls_back_to_cpu(self):
        with _mock_backends():  # all False
            ds, dt = deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": torch.float16})
            assert ds == "cpu"
            assert dt is torch.float32

    def test_multiple_gpus_raises(self):
        with _mock_backends(cuda=True, mps=True):
            with pytest.raises(RuntimeError, match="multiple GPU backends"):
                deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": torch.float16})

    def test_priority_cuda_over_mps_does_not_apply_when_one_is_present(self):
        """The priority order matters only for ambiguity — a single backend wins outright."""
        with _mock_backends(cuda=False, mps=True):
            ds, _ = deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": torch.float32})
            assert ds == "mps"


# ---------------------------------------------------------------------------
# Tests: validate — labeling and in-place mutation
# ---------------------------------------------------------------------------

class TestValidate:
    """`validate` injects a `device_name` label that matches the resolved backend."""

    def test_cuda_label_is_device_name(self):
        with _mock_backends(cuda=True):
            config = {"avatar": {"device_string": "cuda:0", "dtype": torch.float16}}
            deviceinfo.validate(config)
            assert config["avatar"]["device_name"] == "MOCK GPU"

    def test_mps_label(self):
        with _mock_backends(mps=True):
            config = {"avatar": {"device_string": "mps", "dtype": torch.float32}}
            deviceinfo.validate(config)
            assert config["avatar"]["device_name"] == "MPS"

    def test_cpu_label_after_fallback(self):
        with _mock_backends():  # nothing available
            config = {"avatar": {"device_string": "cuda:0", "dtype": torch.float16}}
            deviceinfo.validate(config)
            assert config["avatar"]["device_string"] == "cpu"
            assert config["avatar"]["device_name"] == "CPU"

    def test_explicit_mps_no_longer_labeled_cpu(self):
        """Regression: previously a working MPS run was labeled 'CPU' due to a typo in the labeling block."""
        with _mock_backends(mps=True):
            config = {"avatar": {"device_string": "mps", "dtype": torch.float32}}
            deviceinfo.validate(config)
            assert config["avatar"]["device_name"] == "MPS"  # not "CPU"

    def test_records_without_device_string_are_skipped(self):
        config = {"sw_only": {"some_setting": 42}}
        deviceinfo.validate(config)
        assert "device_name" not in config["sw_only"]

    def test_dtype_coerced_to_float32_on_cpu(self):
        with _mock_backends():
            config = {"avatar": {"device_string": "cuda:0", "dtype": torch.float16}}
            deviceinfo.validate(config)
            assert config["avatar"]["dtype"] is torch.float32
