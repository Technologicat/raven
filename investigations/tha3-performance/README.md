# THA3 performance

Where the avatar's inference time goes, and what can be done about it.

**The write-up is [`tha3-performance-audit.md`](tha3-performance-audit.md)** (2026-04-09).

## The scripts

| Script | What it answers |
|---|---|
| `bench_pipeline_overlap.py` | Can THA3 and upscale+postprocess overlap on the GPU? Measures sequential against concurrent execution of the two pipeline halves on separate threads. |
| `debug_torch_compile.py` | Diagnostic for `torch.compile` against THA3's modules — which ones compile, and what breaks. |

Neither is named by the audit; both are grouped here because they profile the same subsystem. Treat the link
as editorial rather than as something the audit asserts.

## Related

- `../anime4k-performance/` audits the upscaler that forms the other half of the pipeline measured by
  `bench_pipeline_overlap.py`.
- The vendored engine itself lives in `raven/vendor/tha3/`, where the `no_grad` → `inference_mode` change from
  this audit landed.
