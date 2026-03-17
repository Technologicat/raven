# Loader pipeline benchmark results — 2026-03-18

GPU: RTX 4090 (cuda:0), Lanczos order 4, tile size 128.

## Test datasets

| Dataset | Files | Resolution | File format | Total on disk |
|---------|-------|-----------|-------------|---------------|
| ECCOMAS conference photos | 171 | 4624×3472 | JPEG | 613 MB |
| SD txt2img experiment | 133 | ~1000×1000 | PNG | 219 MB |
| Avatar recording frames | 200 | 1024×1024 | QOI | 166 MB |

## Stage breakdown

| Stage | ECCOMAS JPG | SD PNG | Avatar QOI |
|-------|------------|--------|------------|
| Raw disk read (total) | 113 ms | 51 ms | 134 ms |
| **Decode** (per image) | **196 ms** | **59 ms** | **7 ms** |
| CPU→GPU transfer (total) | 23.3 s | 1.5 s | 2.3 s |
| GPU Lanczos resize (per image) | 138 ms | 11 ms | 12 ms |
| Resize + GPU→CPU (total) | 24.2 s | 1.6 s | 2.5 s |

## End-to-end

| Mode | ECCOMAS JPG | SD PNG | Avatar QOI |
|------|------------|--------|------------|
| Sequential (ms/image) | 367 | 71 | 19 |
| Batched bs=32 (ms/image) | 318 | 69 | 18 |
| Sequential (images/s) | 2.7 | 14.1 | 52.6 |
| Batched bs=32 (images/s) | 3.1 | 14.5 | 55.3 |

## Analysis

1. **Decode dominates** — PIL JPEG at 196 ms/image for 16 MP photos is the
   bottleneck. PNG is 59 ms. QOI is 7 ms (fast C decoder).

2. **Raw I/O is negligible** — files are in Linux disk cache after first access.
   NVMe throughput (~2 GB/s) is not the limiting factor.

3. **GPU resize scales with input size** — 11 ms for ~1 MP, 138 ms for 16 MP.
   Multi-stage halving dominates for large images.

4. **Batching barely helps** without pipelining (3.1 vs 2.7 img/s for ECCOMAS).
   The real win will come from overlapping CPU decode with GPU work.

## Implications for loader.py

- Triple-buffer pipeline: decode thread(s) feed a queue, GPU thread consumes.
- For JPEG-heavy workloads, consider `pillow-simd` or `turbojpeg` as faster
  decoders (future optimization, decoder-agnostic pipeline design).
- QOI workloads are GPU-bound (decode is faster than resize) — pipeline still
  helps by keeping the GPU fed without idle gaps.

## Disk throughput reference (hdparm)

```
/dev/nvme0n1: cached 9021 MB/s, buffered 1995 MB/s
/dev/nvme1n1: cached 8665 MB/s, buffered 2146 MB/s
```
