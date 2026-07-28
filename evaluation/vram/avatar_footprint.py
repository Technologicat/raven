#!/usr/bin/env python3
"""Measure what the avatar actually costs in VRAM while it is *running*.

`raven-server --vram-report` measures module loading, and for the avatar that number is close to
meaningless: THA3's posing engine loads eagerly at ~0.03 GiB, but the per-session render buffers, the
upscaler and the postprocessor chain are allocated when a session starts and frames begin to flow. The
avatar is the demo's centrepiece and the load-time figure understates it by an unknown factor, so it has
to be measured with the thing running.

**How it measures, and why not the obvious way.** Sampling is done by shelling out to `nvidia-smi`, not
by importing torch. A torch import here would create a second CUDA context in *this* process - a few
hundred MB - and then report it as part of what it was measuring. `nvidia-smi` reads the driver without
allocating anything.

That makes the reading device-global, which is the usual trade: **run this on an otherwise idle GPU.**
Anything else on the card lands in the numbers. In particular, no LLM backend.

**What it reports.** VRAM used at each stage of the session lifecycle, so the cost can be attributed:

    baseline        server up, no avatar session
    after load      character loaded, before animation starts
    running (peak)  highest sample seen while frames are being consumed
    after stop      animation stopped, session still loaded
    after unload    session destroyed - the gap against baseline is what a session leaks, if any

The peak is what the demo config has to budget for. The load/unload pair is a bonus: a session that
does not give its memory back is a different bug, and this is the cheapest place it would show up.

**Filters are not exercised** beyond whatever the default animator settings enable. The postprocessor
chain is where `crt` and `atmospheric_dust` will live, and each filter adds its own working buffers, so
re-run this after those land rather than assuming this figure still holds.

Usage:
    python avatar_footprint.py [character.png] [n_frames]

Requires a running raven-server with the `avatar` module enabled.
"""

import pathlib
import subprocess
import sys
import threading
import time

from raven.client import api as client_api
from raven.client import config as client_config

DEFAULT_CHARACTER = (pathlib.Path(__file__).parent.parent.parent
                     / "raven" / "avatar" / "assets" / "characters" / "other" / "aria1.png")
DEFAULT_FRAMES = 100

# The animator runs at ~25 FPS server-side, so 100 frames is about four seconds of animation - long
# enough for lazily-allocated buffers to have appeared, short enough to keep the probe interactive.
POLL_INTERVAL = 0.1


def vram_used_mib() -> int | None:
    """Total VRAM in use on GPU 0, in MiB, straight from the driver. `None` if `nvidia-smi` is absent."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return int(out.stdout.strip().split("\n")[0])


class Sampler:
    """Poll VRAM in the background, remembering the peak."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Sampler":
        def poll() -> None:
            while not self._stop.is_set():
                used = vram_used_mib()
                if used is not None:
                    self.peak = max(self.peak, used)
                self._stop.wait(POLL_INTERVAL)
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def report(label: str, used: int | None, baseline: int | None) -> None:
    if used is None:
        print(f"  {label:<22} (nvidia-smi unavailable)")
        return
    delta = f"{used - baseline:+6d}" if baseline is not None else "      "
    print(f"  {label:<22} {used:6d} MiB   {delta} vs baseline")


def main() -> None:
    character = pathlib.Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_CHARACTER
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FRAMES

    if vram_used_mib() is None:
        print("nvidia-smi not available; this probe measures nothing without it.")
        return
    if not character.is_file():
        print(f"character image not found: {character}")
        return

    client_api.initialize(raven_server_url=client_config.raven_server_url,
                          raven_api_key_file=client_config.raven_api_key_file,
                          executor=None)

    print(f"character: {character.name}, frames: {n_frames}")
    print("measuring GPU 0, device-global - anything else on this card lands in these numbers\n")

    baseline = vram_used_mib()
    report("baseline", baseline, None)

    instance_id = client_api.avatar_load(character)
    time.sleep(1.0)  # allocation lags the call returning
    report("after load", vram_used_mib(), baseline)

    with Sampler() as sampler:
        client_api.avatar_start(instance_id)
        feed = client_api.avatar_result_feed(instance_id)
        started = time.monotonic()
        received = 0
        try:
            for _mimetype, _headers, _payload in feed:
                received += 1
                if received >= n_frames:
                    break
        finally:
            feed.close()
        elapsed = time.monotonic() - started

    report("running (peak)", sampler.peak, baseline)
    print(f"  {'':<22} ({received} frames in {elapsed:.1f}s, {received / elapsed:.1f} FPS)")

    client_api.avatar_stop(instance_id)
    time.sleep(1.0)
    report("after stop", vram_used_mib(), baseline)

    client_api.avatar_unload(instance_id)
    time.sleep(1.0)
    after_unload = vram_used_mib()
    report("after unload", after_unload, baseline)

    if after_unload is not None and baseline is not None:
        retained = after_unload - baseline
        # Deliberately not called a leak, and deliberately given no threshold. `nvidia-smi` reports what
        # the *driver* has handed out, and PyTorch's caching allocator does not return freed blocks to
        # the driver - it keeps them reserved for reuse. So memory staying resident after a clean unload
        # is the expected outcome, and this probe has no way to tell that apart from a genuine leak.
        # Telling them apart needs `torch.cuda.memory_allocated` from *inside* the server process, which
        # is a different measurement than this one makes.
        print(f"\n  {retained} MiB still resident after unload. Expected: the caching allocator keeps")
        print("  freed blocks reserved rather than returning them. This probe reads the driver, so it")
        print("  cannot distinguish that from a leak - for which you would need the server's own")
        print("  allocator stats. What it does tell you is the ceiling a running session reaches.")

    print("\nNote: the postprocessor filter chain is where crt and atmospheric_dust will live.")
    print("Re-run this after those land; each filter adds its own working buffers.")


if __name__ == "__main__":
    main()
