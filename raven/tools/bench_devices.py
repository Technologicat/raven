"""Time Raven's speech models on each device this machine has, to choose where each should run.

Which device is faster is not obvious from the hardware: a model can run faster on a strong CPU than on the
GPU, and CPUs differ severalfold. This times the configured TTS and STT models on the GPU and on the CPU,
and prints the settings that put each on the faster one.
"""

__all__ = ["main"]

import argparse
import json
import logging
import sys
import time
from typing import Callable, NamedTuple

from .. import __version__

# Long enough that a timing is not dominated by per-call overhead, short enough to keep the run quick.
_PASSAGE = ("Raven is a constellation of local research tools. This passage is read aloud to time speech "
            "synthesis, and the recording is then transcribed to time speech recognition.")


class _Setting(NamedTuple):
    """Where a module's device is configured: a config module, and a dotted path into it, as `overrides.json` writes it."""
    module: str
    path: str


class _Bench(NamedTuple):
    """One module to time."""
    name: str  # as the command line names it
    label: str
    settings: list[_Setting]  # every place this module's device is configured


_BENCHES = [
    _Bench("tts", "Speech synthesis (TTS)",
           [_Setting("raven.server.config", "enabled_modules.tts.device_string"),
            _Setting("raven.client.config", "devices.tts.device_string")]),  # Librarian's local mode, without Raven-server
    _Bench("stt", "Speech recognition (STT)",
           [_Setting("raven.server.config", "enabled_modules.stt.device_string")]),
]


def _current_value(setting: _Setting) -> str:
    """Return the value `setting` has in Raven's configuration, overrides included."""
    import importlib
    value = importlib.import_module(setting.module)
    first, *rest = setting.path.split(".")
    value = getattr(value, first)
    for key in rest:
        value = value[key]
    return value


def _best_time(f: Callable[[], object], repeats: int, synchronize: Callable[[], None]) -> float:
    """Run `f` once to warm up, then `repeats` times; return the fastest, in seconds."""
    f()
    times = []
    for _ in range(repeats):
        synchronize()
        t0 = time.perf_counter()
        f()
        synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="""Time Raven's speech models on each device this machine has: the GPU, and the CPU. Prints how fast each runs, and the settings that put each on the faster device. Other GPU work running at the same time, such as a local LLM, slows the GPU timings; close it first for a fair comparison.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--modules", nargs="+", choices=[bench.name for bench in _BENCHES],
                        default=[bench.name for bench in _BENCHES],
                        help="Which modules to time.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="How many timed runs per module and device; the fastest counts.")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="Mirror the log to this file (overwritten each run).")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Root logger level.")
    opts = parser.parse_args()

    # Heavy imports after parsing, so that `--help` answers at once.
    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)

    import torch
    from ..common import deviceinfo
    from ..common.audio.speech import stt, tts
    from ..server import config as server_config

    gpu, _ = deviceinfo.get_device_and_dtype({"device_string": "gpu"})
    devices = [gpu, "cpu"] if gpu != "cpu" else ["cpu"]
    if gpu == "cpu":
        print("No GPU found, so there is nothing to compare: everything runs on the CPU.")
        return
    gpu_name = torch.cuda.get_device_name(gpu) if gpu.startswith("cuda") else gpu

    # The libraries log to stderr while the models load and run, so progress goes there too, and the report is
    # collected and printed on stdout at the end, where it is not buried in their output.
    def progress(message: str) -> None:
        print(f"raven-bench-devices: {message}", file=sys.stderr, flush=True)
    report = [f"Timed on {gpu} ({gpu_name}) and on the CPU, best of {opts.repeats}. Higher is faster.", ""]

    def synchronize_for(device: str) -> Callable[[], None]:
        if device.startswith("cuda"):
            return torch.cuda.synchronize
        return lambda: None

    # The recording STT transcribes, made once on the CPU, so that it is the same audio on every device.
    clip = tts.prepare(tts.load_tts_pipeline(server_config.kokoro_models, "cpu"), "af_nova", _PASSAGE)

    stt_dtype = server_config.enabled_modules["stt"].get("dtype")
    measurements = {}  # bench name -> {device: real-time factor}
    for bench in (b for b in _BENCHES if b.name in opts.modules):
        speeds = {}
        for device in devices:
            progress(f"timing {bench.label} on {device}...")
            if bench.name == "tts":
                model_name = server_config.kokoro_models
                pipeline = tts.load_tts_pipeline(model_name, device)
                seconds = _best_time(lambda: tts.prepare(pipeline, "af_nova", _PASSAGE, get_metadata=True),
                                     opts.repeats, synchronize_for(device))
            else:  # "stt"
                model_name = server_config.speech_recognition_model
                device_string, dtype = deviceinfo.get_device_and_dtype({"device_string": device, "dtype": stt_dtype})
                model = stt.load_stt_model(model_name, device_string, dtype)
                seconds = _best_time(lambda: stt.transcribe(model, clip.audio, clip.sample_rate),
                                     opts.repeats, synchronize_for(device))
            speeds[device] = clip.duration / seconds  # real-time factor: seconds of audio per second of work
        measurements[bench.name] = speeds

        faster = max(speeds, key=speeds.get)
        report.append(f"{bench.label}, {model_name}:")
        for device, speed in speeds.items():
            marker = "  <- faster" if device == faster else ""
            report.append(f"    {device:8s} {speed:6.1f}x real time{marker}")
        report.append("")

    # Suggest only what differs from the current configuration.
    overrides: dict[str, dict[str, str]] = {}
    for bench in (b for b in _BENCHES if b.name in opts.modules):
        speeds = measurements[bench.name]
        faster = max(speeds, key=speeds.get)
        wanted = "cpu" if faster == "cpu" else "gpu"
        for setting in bench.settings:
            # Any GPU device string counts as the GPU, so that a GPU the user pinned (`"cuda:1"`) is left alone.
            if (_current_value(setting) == "cpu") != (wanted == "cpu"):
                overrides.setdefault(setting.module, {})[setting.path] = wanted
    if overrides:
        report.append("To run each on its faster device, put these in ~/.config/raven/overrides.json, merging them into what")
        report.append("is already there (see the README, under Configuration), and restart Raven:")
        report.append("")
        report.append(json.dumps(overrides, indent=4))
    else:
        report.append("The current settings already run each of these on its faster device.")

    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
