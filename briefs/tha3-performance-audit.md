# THA3 Performance Audit

*2026-04-09, Claude Opus 4.6*

## What's already optimized

Raven already uses the fastest available THA3 variant:
- **Separable convolutions** (`separable_half`): ~10× fewer FLOPs per conv layer vs standard
- **FP16**: halves memory bandwidth, ~1.5× faster on tensor cores
- **`torch.no_grad()`** wrapping inference — no autograd overhead
- **Eyebrow decomposer cache**: skips stage 1 when the source image hasn't changed (it usually hasn't — only cel overlays change)
- **GridChangeApplier**: caches the identity matrix across frames

Total model: **14.5M parameters, 28 MB** on disk. This is small — the bottleneck isn't parameter count.

## Architecture recap

Five strictly sequential stages, each a small encoder-decoder network:

| Stage | Network | Resolution | Params | Bottleneck |
|-------|---------|-----------|--------|------------|
| 1 | Eyebrow Decomposer | 128×128 | 3.3M | 16×16, 512ch |
| 2 | Eyebrow Combiner | 128×128 | 3.3M | 16×16, 512ch |
| 3 | Face Morpher | 192×192 | 3.3M | 24×24, 512ch |
| 4 | Body Rotator | 256×256 | 3.6M | 32×32, 512ch |
| 5 | Editor (UNet) | 512×512 | 0.9M | 64×64, 512ch |

Between stages: image region extraction, clone+splice, bilinear upsampling.

The editor has the fewest parameters despite processing the largest resolution — it uses `start_channels=32` (vs 64 for others) and `bottleneck_image_size=64` (vs 16–32).

## Where the time goes

The fundamental cost is **many small CUDA kernel launches**. Each separable conv is 2 kernels (depthwise + pointwise), each followed by InstanceNorm (2–3 kernels: mean, variance, normalize) and an activation (1 kernel). A single conv block = ~5–6 kernel launches. Across 5 networks with ~8 conv blocks each plus 6 ResNet bottleneck blocks, that's roughly **300–400 kernel launches per frame**.

On a modern GPU, each kernel launch has ~5–10 µs overhead regardless of the actual compute. At 400 launches, that's 2–4 ms of pure launch overhead alone, before any actual tensor math.

## Optimization opportunities (ranked by expected impact)

### 1. `torch.compile()` — expected 1.5–3× overall speedup

The single biggest win. THA3 is pure eager PyTorch — no custom CUDA kernels, no exotic ops. `torch.compile()` (Inductor backend) would:
- **Fuse kernel chains**: conv → instance_norm → relu becomes one kernel launch instead of 3–4
- **Fuse elementwise ops**: the `apply_color_change` (alpha blend) and grid_change addition are currently separate kernels
- **Optimize memory layout**: reduce intermediate tensor allocations
- **Reduce launch overhead**: the ~400 launches could collapse to ~50–80

Each of the 5 stage modules can be compiled independently. The inter-stage splice/clone/interpolate code stays in eager Python — that's fine, it's lightweight.

Caveat: `grid_sample` and `affine_grid` may not fuse well with Inductor. If they graph-break, the gain narrows, but the conv chains still benefit heavily.

**Effort**: low-to-medium. Wrap each module with `torch.compile()` after loading. May need `torch.compile(mode="reduce-overhead")` for maximum kernel fusion. Watch for shape dynamism issues (batch size is always 1, so static shapes should work).

### 2. CUDA Graphs — expected 1.3–2× on top of compile

Since batch size is always 1 and tensor shapes are fixed across frames, CUDA graphs can capture the entire execution trace and replay it with zero launch overhead. This eliminates the per-kernel scheduling cost entirely.

Can be combined with `torch.compile(mode="reduce-overhead")`, which uses CUDA graphs internally.

**Effort**: low if using torch.compile's built-in CUDA graph support. Medium if doing it manually (need to handle the inter-stage Python logic).

### 3. `torch.inference_mode()` instead of `torch.no_grad()` — expected ~2–5% speedup

Free win. `inference_mode()` additionally disables version counting and view tracking that `no_grad()` still maintains. One-line change in `avatar.py`.

### 4. Cache `affine_grid` base grids — expected ~1–3% speedup

`GridChangeApplier.apply()` caches the identity matrix but still calls `affine_grid()` every time to generate the base grid. Since the grid dimensions are fixed (always batch=1 at 192×192, 256×256, or 512×512), the base grids could be computed once and reused.

This saves 3 `affine_grid` calls per frame (stages 3, 4, 5). `affine_grid` at 512×512 generates a (1, 512, 512, 2) coordinate mesh — a full grid of (x, y) pairs covering normalized device coordinates, with bilinear interpolation of the affine transform at each pixel.

### 5. Eliminate pose spatial expansion waste — expected ~1% speedup, significant memory savings

The body rotator and editor both do:
```python
pose = pose.view(n, c, 1, 1).repeat(1, 1, H, H)  # 6 → 6×256×256 or 6×512×512
```

This creates 393K (rotator) or 1.6M (editor) redundant copies of 6 floats, then concatenates into the feature map. The pose is then convolved as if it were a spatial signal — but it's constant across space.

The *proper* fix (injecting pose at the bottleneck, like stages 1–3 do) would require retraining. But without retraining: use `expand()` instead of `repeat()`. `expand()` creates a view without copying memory:
```python
pose = pose.view(n, c, 1, 1).expand(-1, -1, H, H)
```

`torch.compile()` may also optimize the `view + repeat + cat` pattern by never materializing the expanded tensor.

### 6. Pre-allocate per-frame tensors — expected <1% speedup

Every frame creates:
- `torch.tensor(self.current_pose, ...)` — fresh pose tensor
- Two `.clone()` calls for image splicing (192×192 and 512×512 regions)

Pre-allocating and reusing with `tensor.copy_()` avoids CUDA malloc overhead. Marginal, but free.

### 7. Skip unused intermediate outputs — negligible

The pipeline returns 29 output tensors, but Raven only uses `output[0]`. These intermediates are byproducts of the forward passes — computed anyway. Main waste is list concatenation and memory retention, not extra compute.

## What's NOT worth pursuing

- **InstanceNorm → BatchNorm/GroupNorm swap**: Would require retraining. InstanceNorm is mathematically necessary for this style-transfer-like architecture.
- **Model pruning / distillation**: 14.5M params is already lean. Requires training pipeline.
- **TensorRT export**: The 5-stage pipeline with inter-stage Python logic makes end-to-end TensorRT awkward. `torch.compile()` gets ~70% of TensorRT's benefit with ~10% of the effort.
- **Reducing resolution**: The editor bottleneck is already 64×64 for 512×512 output — aggressive. Would visibly degrade quality.
- **Batching**: Raven renders one frame at a time for low-latency streaming. Batching would add latency.

## Recommended implementation order

1. **`torch.inference_mode()`** — one line, zero risk
2. **`torch.compile()` on each module** — medium effort, biggest payoff
3. **Cache `affine_grid` base grids** — small effort, reliable win
4. **`expand()` vs `repeat()`** for pose — one-line fix each
5. **CUDA graphs** via `torch.compile(mode="reduce-overhead")` — may come free with step 2

Realistic expectation: steps 1–2 should yield **1.5–2.5× faster** inference.

## Model sizes

| Variant | Size | Notes |
|---------|------|-------|
| standard_float | 518 MB | Reference quality, training/validation |
| standard_half | 259 MB | FP16 reference |
| separable_float | 56 MB | Deployment variant, FP32 |
| separable_half | 28 MB | Deployment variant, FP16 — **used by Raven** |

The separable variants have ~10× fewer parameters — designed as the deployment option by the THA3 author (Pramook Khungurn, @pkhungurn).

## Postprocessor side

The `raven.common.video.postprocessor` has similar optimization opportunities:
- 20–30 kernel launches per frame (grid_sample, GaussianBlur, elementwise ops), up to 50–60 with Anime4K upscaling
- No `torch.compile()` currently active (Anime4K has compile infrastructure but disabled in production)
- `torch.no_grad()` used, but not `inference_mode()`
- Expected speedup from compile: 2–4× for postprocessor alone
