# Avatar Render Pipeline: Pose/Postproc Splitting

*2026-04-09, Claude Opus 4.6 — design brief for discussion*

## Result: Not Viable

Benchmarked 2026-04-09 on 3070 Ti. Neither threading nor CUDA streams produce overlap:

| Approach | ms/frame | Notes |
|---|---|---|
| Sequential (current) | 30.5ms | Baseline |
| Two threads | 68.8ms | **2.3× slower** — GIL contention |
| CUDA streams | 30.0ms | No overlap — GPU fully contended |
| Theoretical (perfect overlap) | 18.5ms | Unreachable |

The GPU cannot run THA3 and upscale+postproc concurrently. Despite nvtop reporting 30–70% utilization, the hardware is fully committed when kernels are running — the "idle" percentage reflects Python dispatch gaps between kernel launches, not available compute.

**Conclusion:** Pipelining is a dead end for this workload on this class of GPU. The remaining optimization paths are: faster GPU, smaller models (retraining), lower resolution, fewer filters, or resolving the torch.compile server hang.

---

## Original Motivation (retained for reference)

On a 3070 Ti (throttled clock), the avatar render pipeline takes ~52ms per frame:

| Stage | Time | Share |
|---|---|---|
| THA3 pose | ~28ms | 54% |
| Upscale (Anime4K) | ~9ms | 17% |
| Postprocessor | ~11ms | 21% |
| Other (cel, norm, animefx, gamma, format, CPU) | ~4ms | 8% |

Currently all stages run sequentially on the animator thread. GPU utilization is 30–70% — not saturated. The proposal: split the animator thread into two, pipelining pose with upscale+postproc.

## Current 3-Thread Architecture

```
Animator thread         Encoder thread         Network thread
─────────────────       ──────────────         ──────────────
animation drivers
cel blending (~0.5ms)
THA3 pose (~28ms)
normalize (~1ms)
animefx (~0.5ms)
upscale (~9ms)
crop (~0ms)
postproc (~11ms)
gamma (~0.5ms)
CHW→HWC (~0.5ms)
to CPU (~0.7ms)
      │
      ├──[output_lock]──→ QOI encode (~17ms) ──→ HTTP send (FPS-paced)
      │    flag+ref          frame tuple            id() feedback
```

Coordination: shared `result_image` + `new_frame_available` flag (protected by `output_lock`). No explicit queues. Backpressure from network→encoder via `latest_frame_sent` polling.

## Proposed 4-Thread Architecture

```
Pose thread             Postproc thread         Encoder thread      Network thread
───────────             ───────────────         ──────────────      ──────────────
animation drivers
cel blending
THA3 pose
normalize
animefx
      │
      ├──[Queue(1)]──→  upscale
                         crop
                         postproc
                         gamma
                         CHW→HWC
                         to CPU
                               │
                               ├──[output_lock]──→ QOI encode ──→ HTTP send
```

The intermediate `Queue(maxsize=1)` gives exactly one frame of pipeline overlap: while the postproc thread processes frame N, the pose thread is already working on frame N+1.

### Why split at the animefx/upscale boundary

- **AnimeFX stays with pose.** It needs the current celstack (emotion state), which is computed by the animation drivers on the pose thread. Moving it to the postproc thread would require passing the celstack through the queue. And it's cheap (~0.5ms).
- **Upscale stays with postproc.** It's stateless (only depends on the image tensor) and is the second-heaviest stage.
- **Postprocessor stays with postproc.** Also stateless per-frame (caches are internal).

### What goes through the queue

```python
# Pose thread pushes:
intermediate_queue.put({
    "image": output_image,          # torch.Tensor [c, h, w], GPU, float16, range [0, 1]
    "frame_timestamp_ns": int,      # time.monotonic_ns() at start of this frame's render
})
```

The postproc thread uses `frame_timestamp_ns` to compute the postprocessor's `frame_no`, keeping dynamic effects (banding, scanlines) time-synchronized with the pose, not with the postproc processing time.

### Queue behavior

`Queue(maxsize=1)`:
- Pose thread calls `put()` — blocks if postproc hasn't consumed the previous frame yet
- Postproc thread calls `get()` — blocks if no new frame is available
- This automatically rate-matches the two threads

At steady state, the pipeline throughput is `max(pose_time, postproc_time)` instead of `pose_time + postproc_time`.

### Latency impact

Adds one frame of pipeline latency (~40ms at 25 FPS). Total pipeline depth becomes 3 frames:
- Frame N being post-processed
- Frame N+1 being posed
- Frame N+2 being encoded / sent

Lipsync already has a configurable AV sync offset (currently several hundred ms), so the extra 40ms is absorbed by adjusting `video_offset` in `avatar_config`.

## Implementation Plan

### 1. New class: `PostprocessWorker`

Lives in `avatar.py` alongside `Animator` and `Encoder`. Mirrors the Encoder's lifecycle pattern (daemon thread, `_terminated` flag, `start()`/`exit()` methods).

**Owns:**
- `self.upscaler` (currently on Animator)
- `self.postprocessor` (currently on Animator)
- `self.intermediate_queue` — `queue.Queue(maxsize=1)`

**Main loop:**
```python
def _run(self):
    while not self._terminated:
        try:
            item = self.intermediate_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        image = item["image"]
        frame_timestamp_ns = item["frame_timestamp_ns"]

        with torch.inference_mode():
            if self.upscaler is not None:
                image = self.upscaler.upscale(image)
            # ... crop ...
            self.postprocessor.frame_no = self._compute_frame_no(frame_timestamp_ns)
            self.postprocessor.last_frame_no = self._last_frame_no
            self.postprocessor.render_into(image)
            # ... gamma, CHW→HWC, to CPU ...

        with self.animator.output_lock:
            self.animator.result_image = output_numpy
            self.animator.new_frame_available = True

        self._last_frame_no = self.postprocessor.frame_no
```

### 2. Slim down `render_animation_frame`

Remove from the animator's render loop (lines ~1506–1540):
- Upscale
- Crop
- Postprocessor
- Gamma correction
- CHW→HWC conversion
- CPU transfer

Replace the `output_lock` frame publish (lines 1559–1562) with:
```python
self.postproc_worker.intermediate_queue.put({
    "image": output_image,
    "frame_timestamp_ns": time_render_start,
})
```

### 3. Adjust backpressure

Currently, the animator blocks if `new_frame_available` is True (encoder hasn't consumed the previous frame). With the split:

- **Pose thread**: blocks on `intermediate_queue.put()` if postproc hasn't consumed. No longer checks `new_frame_available` directly.
- **Postproc thread**: blocks on `output_lock` / `new_frame_available` if encoder hasn't consumed. This preserves the existing encoder backpressure.

The `new_frame_available` early-return at line 1393 should be removed from `render_animation_frame` — the queue provides the backpressure instead.

### 4. Warmup

Keep warmup synchronous on the pose thread. The warmup already runs the full pipeline (pose → upscale → postproc). After warmup, start the postproc worker thread. This ensures:
- CUDA kernels are primed on the correct thread (pose thread for THA3, postproc thread for upscaler/postprocessor)

Wait — CUDA context is per-process, not per-thread (since PyTorch 2.x). But the warmup for the *upscaler and postprocessor* needs to happen on the *postproc thread* to prime any thread-local state. So:

**Revised warmup plan:**
1. Pose thread warmup: pose only (prime THA3)
2. Push warmup image through the queue
3. Postproc thread warmup: upscale + postproc (prime Anime4K + postprocessor)
4. Discard the warmup frame (don't push to encoder)

Or simpler: keep the existing warmup as-is (runs full pipeline on pose thread), then let the postproc thread do its own warmup on first frame. The CUDA context is shared, so kernel caches are warm regardless of which thread triggered them.

### 5. Shutdown sequence

```
1. pose_thread._terminated = True
2. pose_thread.join()
   → pose thread finishes current pose, stops
   → intermediate_queue may have one final frame

3. postproc_worker._terminated = True
4. postproc_worker.join()
   → postproc thread drains queue (processes remaining frame if any), stops
   → sets new_frame_available for the final frame

5. encoder._terminated = True
6. encoder.join()
   → encoder processes final frame, stops
```

### 6. Metrics

The per-stage timing currently uses `maybe_sync_cuda()` between each stage within `render_animation_frame`. With the split:

- **Pose thread metrics**: cel blending, pose, normalize, animefx — stays as-is
- **Postproc thread metrics**: upscale, crop, postproc, gamma, CHW→HWC, CPU — new per-stage timing in the postproc worker

Both threads log their own timing independently. The "total render time" becomes less meaningful (it's two overlapping threads), but the per-stage numbers stay accurate.

### 7. Settings changes

`PostprocessWorker` needs access to:
- `self._settings` (for crop, upscale, metrics_enabled, etc.) — pass reference from Animator
- `self.postprocessor` — move ownership from Animator to PostprocessWorker
- `self.upscaler` — move ownership from Animator to PostprocessWorker

The `load_animator_settings()` method currently configures upscaler and postprocessor on the Animator. These should move to PostprocessWorker, or the Animator can forward the settings.

## Expected Impact

**Theoretical best case:** Frame time drops from ~52ms to ~28ms (pose-bound), yielding ~36 FPS at throttled clock — well above the 25 FPS target.

**Realistic estimate:** GPU memory bandwidth contention + CUDA stream scheduling overhead mean some interference. Likely 35–42ms per frame, or **24–29 FPS at throttled clock**. At full clock (currently ~37ms sequential), this would give comfortable headroom above 25 FPS.

**Worst case:** If the GPU is actually compute-saturated (the 30–70% utilization is misleading), the overlap provides no benefit and we just add one frame of latency. We can measure and revert if needed.

## Risks

1. **CUDA stream contention.** Two CPU threads submit GPU work on separate CUDA streams. If the GPU can't overlap them, no speedup. Measurable before committing to the full implementation: just run pose + upscale+postproc on two threads with dummy data and measure throughput.

2. **Thread-safety of postprocessor state.** The postprocessor's caches (noise textures, meshgrids, zoom grids) are accessed from one thread only (the new postproc thread), so no new contention. But `load_animator_settings()` writes to `self.postprocessor.chain` from the main HTTP handler thread — this is already racy in the current design (the comment at `render_into` line 496 acknowledges it: "read just once; other threads might reassign it while we're rendering").

3. **Postprocessor `frame_no` override.** Currently `render_into()` computes `frame_no` from wall time. We need to either: (a) pass the timestamp and let render_into use it, or (b) set `frame_no` and `last_frame_no` on the postprocessor before calling `render_into`. Option (b) is simpler but requires render_into to skip its own frame_no computation when externally set.

4. **Complexity budget.** Adding a fourth thread increases the coordination surface. The existing 3-thread model was "arrived at as a result of much iteration." The benefit must clearly outweigh the added complexity.

## Design Decisions

1. **Dual mode.** Keep the current 3-thread mode as default (minimum latency). Add 4-thread pipelined mode as opt-in via config. Some GPUs are fast enough with 3 threads — no need to penalize those users with a one-frame delay.

2. **Queue type.** `queue.Queue(maxsize=1)`. Don't drop frames.

3. **Crop** goes with the postproc thread (currently unused by client, but configurable via `animator.json`).

4. **Naming.** Keep "Animator" for the pose thread — it still drives animation. The new thread is `PostprocessWorker`.

5. **Validation.** Run a concurrent overlap benchmark first. If infeasible (the isolated-vs-server discrepancy suggests GPU scheduling is subtle), build and measure the real thing. Git branch for easy revert if it doesn't pan out.
