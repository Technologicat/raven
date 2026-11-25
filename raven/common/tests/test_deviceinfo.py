"""Tests for raven.common.deviceinfo — device-string resolution and labeling.

Exercises the resolution rules in `get_device_and_dtype` and the labeling pass
in `validate` across mocked backend permutations. Real-hardware probes (the
NVRTC sanity check) are out of scope for unit tests; they live with the apps
that call them.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from raven.common import deviceinfo


@pytest.fixture(autouse=True)
def isolate_rocm_compiler_setup(monkeypatch):
    """Keep `get_device_and_dtype`'s once-per-process ROCm compiler setup from leaking between tests.

    On a ROCm build of Torch, any test that resolves a `cuda` device runs it, and it writes `os.environ`.
    """
    monkeypatch.setattr(deviceinfo, "_rocm_compiler_prepared", False)
    monkeypatch.delenv(deviceinfo._HIPRTC_OPTIONS_VARIABLE, raising=False)


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


class TestDtypeByName:
    """The configs name a dtype as Torch names it, so that reading one does not import Torch."""

    def test_a_name_resolves_to_the_dtype(self):
        with _mock_backends(cuda=True):
            config = {"embeddings": {"device_string": "gpu", "dtype": "bfloat16"}}
            deviceinfo.validate(config)
            assert config["embeddings"]["dtype"] is torch.bfloat16

    def test_a_named_float16_is_coerced_on_cpu_as_the_dtype_is(self):
        """The CPU fallback compares against `torch.float16`, so it must see the resolved dtype, not the name."""
        with _mock_backends():
            config = {"avatar": {"device_string": "cuda:0", "dtype": "float16"}}
            deviceinfo.validate(config)
            assert config["avatar"]["dtype"] is torch.float32

    @pytest.mark.parametrize("bad_name", ["float17",  # no such name
                                          "cuda"])    # a name Torch has, which is not a dtype
    def test_a_name_that_is_not_a_dtype_is_refused(self, bad_name):
        with _mock_backends(cuda=True):
            with pytest.raises(ValueError, match=bad_name):
                deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": bad_name})

    def test_the_configs_import_without_torch(self):
        """The point of naming dtypes: a CLI tool's `--help` reads config defaults, and should not wait for Torch.

        In a fresh interpreter, since `validate` rewrites the records in place and this process has Torch.
        """
        code = ("import sys\n"
                "import raven.client.config, raven.server.config, raven.cherrypick.config\n"
                "print('torch' in sys.modules)\n")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "False", "importing a config module imported Torch"


class TestVRAMLedger:
    """`VRAMLedger` accounting, on the paths that need no GPU.

    The CUDA path is exercised by running raven-server with `--vram-report`; what is pinned here is
    the behaviour that must hold on every machine, CI included: a device it cannot measure is
    reported as *unmeasured*, never as zero. Zero would read as "this module is free", which is the
    one wrong answer that looks like a right one.
    """

    def test_non_cuda_device_records_unmeasured_not_zero(self):
        ledger = deviceinfo.VRAMLedger()
        with ledger.measure("somemodule", "cpu"):
            pass
        entry = ledger.entries()[0]
        assert entry["label"] == "somemodule"
        assert entry["driver_bytes"] is None
        assert entry["torch_bytes"] is None

    def test_entries_preserve_load_order(self):
        # The per-step split is order-dependent when models are shared, so the order is part of the
        # data rather than a presentational detail.
        ledger = deviceinfo.VRAMLedger()
        for name in ("first", "second", "third"):
            with ledger.measure(name, "cpu"):
                pass
        assert [entry["label"] for entry in ledger.entries()] == ["first", "second", "third"]

    def test_step_is_recorded_even_when_it_raises(self):
        # A module that dies partway through has usually already allocated something, and that is
        # exactly when someone wants to know how much.
        ledger = deviceinfo.VRAMLedger()
        with pytest.raises(RuntimeError):
            with ledger.measure("exploding", "cpu"):
                raise RuntimeError("model failed to load")
        assert [entry["label"] for entry in ledger.entries()] == ["exploding"]

    def test_empty_ledger_reports_nothing_measured(self):
        assert "nothing measured" in deviceinfo.VRAMLedger().format_report()

    def test_report_mentions_every_measured_module(self):
        ledger = deviceinfo.VRAMLedger()
        for name in ("avatar", "stt", "tts"):
            with ledger.measure(name, "cpu"):
                pass
        report = ledger.format_report()
        for name in ("avatar", "stt", "tts"):
            assert name in report

    def test_totals_are_empty_without_a_measurable_device(self):
        ledger = deviceinfo.VRAMLedger()
        with ledger.measure("somemodule", "cpu"):
            pass
        assert ledger.totals() == {}


# ---------------------------------------------------------------------------
# Tests: the ROCm run-time compiler setup
# ---------------------------------------------------------------------------

# What `g++ -E -x c++ -v /dev/null` prints on stderr on a Debian-family system, trimmed.
_COMPILER_STDERR = """\
ignoring nonexistent directory "/usr/local/include/x86_64-linux-gnu"
#include "..." search starts here:
#include <...> search starts here:
 /usr/include/c++/13
 /usr/include/x86_64-linux-gnu/c++/13
 /usr/include/c++/13/backward
 /usr/lib/gcc/x86_64-linux-gnu/13/include
 /usr/local/include
 /usr/include
End of search list.
"""


def _compiler_reports(stderr: str):
    """Patch a C++ compiler into existence that prints `stderr` when asked for its search path."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=stderr)
    return (patch.object(deviceinfo.shutil, "which", return_value="/usr/bin/g++"),
            patch.object(deviceinfo.subprocess, "run", return_value=completed))


class TestLibstdcxxIncludeDirs:
    """The C++ library's header directories, read from the compiler's own search path."""

    def test_keeps_only_the_cxx_library_directories(self):
        which, run = _compiler_reports(_COMPILER_STDERR)
        with which, run:
            assert deviceinfo._libstdcxx_include_dirs() == ["/usr/include/c++/13",
                                                            "/usr/include/x86_64-linux-gnu/c++/13",
                                                            "/usr/include/c++/13/backward"]

    def test_finds_headers_kept_directly_in_a_cxx_directory(self):
        # MinGW's layout: the headers are in `.../include/c++` itself, with the version higher up the path.
        stderr = ("#include <...> search starts here:\n"
                  " C:/mingw64/lib/gcc/x86_64-w64-mingw32/15.2.0/include/c++\n"
                  " C:/mingw64/lib/gcc/x86_64-w64-mingw32/15.2.0/include/c++/x86_64-w64-mingw32\n"
                  " C:/mingw64/lib/gcc/x86_64-w64-mingw32/15.2.0/include\n"
                  "End of search list.\n")
        which, run = _compiler_reports(stderr)
        with which, run:
            assert deviceinfo._libstdcxx_include_dirs() == [
                "C:/mingw64/lib/gcc/x86_64-w64-mingw32/15.2.0/include/c++",
                "C:/mingw64/lib/gcc/x86_64-w64-mingw32/15.2.0/include/c++/x86_64-w64-mingw32"]

    def test_no_compiler_gives_nothing(self):
        with patch.object(deviceinfo.shutil, "which", return_value=None):
            assert deviceinfo._libstdcxx_include_dirs() == []

    def test_a_compiler_that_does_not_answer_gives_nothing(self):
        timeout = subprocess.TimeoutExpired(cmd="g++", timeout=30)
        with patch.object(deviceinfo.shutil, "which", return_value="/usr/bin/g++"), \
             patch.object(deviceinfo.subprocess, "run", side_effect=timeout):
            assert deviceinfo._libstdcxx_include_dirs() == []

    def test_unreadable_output_gives_nothing(self):
        which, run = _compiler_reports("clang: error: unknown argument\n")
        with which, run:
            assert deviceinfo._libstdcxx_include_dirs() == []

    @pytest.mark.skipif(not (deviceinfo.shutil.which("g++") or deviceinfo.shutil.which("c++")),
                        reason="no C++ compiler on this machine")
    def test_the_real_compiler_names_a_directory_with_type_traits(self):
        # Against the installed compiler rather than a fixture, so a change in its output format shows here. With
        # a long timeout: a CI runner's first compiler start can be slow, and this is not a test of the timeout.
        include_dirs = deviceinfo._libstdcxx_include_dirs(timeout=120)
        assert any(os.path.isfile(os.path.join(path, "type_traits")) for path in include_dirs), include_dirs


class TestPrepareRocmCompiler:
    """On a ROCm build, the C++ headers are appended to hiprtc's options, once."""

    VARIABLE = deviceinfo._HIPRTC_OPTIONS_VARIABLE

    def test_appends_to_what_the_user_set(self, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", "7.2.0")
        monkeypatch.setenv(self.VARIABLE, "-O3")
        with patch.object(deviceinfo, "_libstdcxx_include_dirs", return_value=["/a/c++/13", "/b/c++/13"]):
            deviceinfo._prepare_rocm_compiler()
        assert os.environ[self.VARIABLE] == "-O3 -I/a/c++/13 -I/b/c++/13"

    def test_runs_once_per_process(self, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", "7.2.0")
        with patch.object(deviceinfo, "_libstdcxx_include_dirs", return_value=["/a/c++/13"]) as include_dirs:
            deviceinfo._prepare_rocm_compiler()
            deviceinfo._prepare_rocm_compiler()
        assert include_dirs.call_count == 1
        assert os.environ[self.VARIABLE] == "-I/a/c++/13"

    def test_does_nothing_on_a_build_without_hip(self, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", None)
        with patch.object(deviceinfo, "_libstdcxx_include_dirs", return_value=["/a/c++/13"]) as include_dirs:
            deviceinfo._prepare_rocm_compiler()
        include_dirs.assert_not_called()
        assert self.VARIABLE not in os.environ

    def test_reports_an_error_and_leaves_the_variable_alone_without_a_compiler(self, monkeypatch, caplog):
        monkeypatch.setattr(torch.version, "hip", "7.2.0")
        with patch.object(deviceinfo, "_libstdcxx_include_dirs", return_value=[]):
            with caplog.at_level("ERROR", logger=deviceinfo.logger.name):
                deviceinfo._prepare_rocm_compiler()
        assert self.VARIABLE not in os.environ
        assert "g++" in caplog.text

    def test_resolving_a_cuda_device_prepares_it(self, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", "7.2.0")
        with _mock_backends(cuda=True):
            with patch.object(deviceinfo, "_libstdcxx_include_dirs", return_value=["/a/c++/13"]):
                deviceinfo.get_device_and_dtype({"device_string": "gpu", "dtype": torch.float16})
        assert os.environ[self.VARIABLE] == "-I/a/c++/13"
