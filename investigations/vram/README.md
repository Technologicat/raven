# What Raven-server costs in VRAM

Measured 2026-07-28 on the 16 GB machine, to derive the server's config variants from numbers rather than
from trial and error on demo day.

## At load: `raven-server --vram-report PATH`

The instrumentation lives in `deviceinfo.VRAMLedger`. All nine modules resident cost **2.27 GiB** at load,
leaving 13.0 GiB of 15.6 free:

| module | GiB | | module | GiB |
|---|---|---|---|---|
| embeddings | 0.83 | | tts | 0.34 |
| sanitize | 0.63 | | translate | 0.25 |
| classify | 0.14 | | stt | 0.04 |
| avatar | 0.03 | | natlang | 0.02 |
| imagefx | 0.00 | | | |

So the module set is not the constraint on either card; the LLM is.

**This is a floor, not the operating footprint**, and two rows say so. `avatar` genuinely loads THA3 eagerly
(0.03 is the posing engine), but per-session render buffers and the upscaler are allocated when a session
starts, not at init. `imagefx` reads 0.00 because it is constructed with an empty filter chain. Inference
activations are absent throughout.

## The avatar running: `avatar_footprint.py`

Drives a real session over the client API and samples `nvidia-smi` from outside, so it adds no CUDA context
of its own; its docstring has the method. **+386 MiB peak while animating**, against 30 MiB at load — 13×,
and the reason load-time figures cannot be trusted for this module. Session creation alone is +80 MiB; the
rest arrives once frames flow. It hit 25.4 FPS over 100 frames, the server's target rate.

So the whole server side, all nine modules with the avatar running, is **~2.9 GiB**. On 8 GB that leaves
~5 GiB for the LLM, which fits the on-the-road model with its KV cache.

Memory stays resident after unloading a session, which is expected — PyTorch's caching allocator keeps
freed blocks reserved rather than returning them to the driver. `nvidia-smi` cannot distinguish that from a
leak; doing so needs the server's own allocator stats.

## Not yet measured

- **`imagefx` with a filter chain in it.** `crt` and `atmospheric_dust` are in the default chain since
  2026-08-31, so the 0.00 above no longer describes what ships.
- **`embeddings`**, which goes stale at the Nomic switch.
- **Peak during use for the other eight modules** — each needs a warm-up request. Only worth doing if the
  LLM budget turns out tight.
