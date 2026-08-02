# Anime4K Upscaler Performance Audit

*2026-04-09, Claude Opus 4.6*

## Architecture

Vendored PyTorch port of Anime4K GLSL shaders. Core model: small CNN with CReLU activations (doubles channels), skip connections via depth list, and PixelShuffle for 2× upscaling.

**Presets** (Raven uses Preset C by default):
- **C (fast)**: ClampHighlight → Upscale_Denoise → AutoDownscalePre → Upscale → final interpolate (~25 kernel launches)
- **A/B (quality)**: ClampHighlight → Restore → Upscale → AutoDownscalePre → Upscale → final interpolate (~48 kernel launches)

Model sizes range from S (3 layers, 4 features) to UL (7 layers, 12 features).

## Key Findings

### 1. Missing `inference_mode` in upscaler.py

`Upscaler.upscale()` runs the pipeline without `inference_mode` or `no_grad`. Models have `requires_grad=False` on parameters, but autograd metadata is still tracked. Free win — same as what we did for THA3.

### 2. `torch.compile` readiness: POOR

Multiple graph-break patterns in the forward path:
- **List mutation** in `anime4k.forward()`: `depth_list.append(out)` per layer
- **List comprehension gather**: `torch.cat([depth_list[i] for i in self.stack_list], 1)`
- **String-based type dispatch** in pipeline: `model.__class__.__name__ == "ClampHighlight"`
- **Dynamic shape check** in ClampHighlight: `if statsmax.shape != curr_luma.shape`

To make compile-friendly, would need to refactor depth tracking to use tensors, replace string dispatch with isinstance or positional logic, and make the shape check unconditional.

### 3. In-place opportunities

- `torch.clamp(out, ...)` → `out.clamp_(...)` in anime4k.forward()
- Residual addition: `self.ps(out) + F.interpolate(...)` → `out = self.ps(out); out.add_(...)`
- Minor savings — these are small tensors relative to the convolution cost.

### 4. Pipeline clone

`Anime4KPipeline.forward()` clones the input for ClampHighlight's deferred application. This is ~2.8 MB at 1280×720 FP32. Could be avoided if ClampHighlight is restructured or if the input is guaranteed immutable (which it is in Raven's usage — the postprocessor doesn't need the pre-upscale image).

### 5. Alpha channel handling

Alpha is upscaled separately via bilinear interpolation after the RGB pipeline completes. Could be batched with the pipeline's final interpolation, but the CNN models only accept 3-channel input, so this is a design limitation.

### 6. Chained `.to()` calls

```python
data = rgb_image_tensor.unsqueeze(0).to(self.device).to(self.dtype)
```
Could combine into `.to(device=self.device, dtype=self.dtype)` to avoid intermediate.

## Optimization Priority

| # | Item | Impact | Effort |
|---|------|--------|--------|
| 1 | Add `inference_mode` to `Upscaler.upscale()` | ~2-5% | Trivial |
| 2 | In-place `clamp_` + `add_` in anime4k.forward() | ~1-2% | Low |
| 3 | Eliminate pipeline clone (skip or restructure ClampHighlight) | ~10% memory | Medium |
| 4 | Combine `.to()` calls | <1% | Trivial |
| 5 | Refactor for torch.compile (depth list → tensor, dispatch cleanup) | Potentially significant but untested | High |

## What's NOT worth pursuing

- **Batching alpha with RGB**: models are 3-channel only, architectural constraint
- **CReLU optimization**: already efficient for its purpose, doubling is intentional
- **Reducing kernel launches**: dominated by Conv2d which are the actual compute
- **Custom CUDA kernels**: the models are small, overhead is in the convolutions

## Comparison to THA3

Unlike THA3 where kernel launch overhead dominated (many small separable convs), Anime4K's convolutions are fewer but larger (standard Conv2d with CReLU doubling the channels). The bottleneck is more likely actual compute than overhead. torch.compile *might* help more here (by fusing Conv+CReLU), but the graph-break issues need to be fixed first.
