# Benchmark Methodology

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BENCHMARK_METHODOLOGY.vi.md)

- Dtype: BF16 transformer weights/activations.
- Seed: 42.
- Scheduler: `FlowMatchEulerDiscreteScheduler` from the attached model.
- Default acceptance: 28 steps, CFG 3.0.
- XLA execution is explicitly synchronized before timing boundaries.
- First/cold steps are preserved rather than hidden.
- Warm forward median excludes the first two steps when at least three steps are available.
- VAE decode time is measured separately on CPU.
- Compile/cold and warm performance are not mixed into a single unlabeled metric.

See `evidence/reference/resolution.json` for frozen results.

## Two-image throughput comparison

The topology experiment measures **pair wall time**, not isolated per-device kernel time. single-mesh two-image control and replicated-model two-image workload use the same resolution, 28 steps, CFG 3.0, sequential CFG, frozen prompt embeddings, and two fixed seeds. Both samples must complete denoising and decode to plausible PNGs before a topology can be accepted.

Measured pair results:

| Resolution | 1×8 model mesh, batch=2 | 2×4 replica×model, batch=2 | replicated-model two-image workload speedup vs single-mesh two-image control |
|---|---:|---:|---:|
| 512×512 | 126.264 s | 95.001 s | 1.329× |
| 1024×1024 | 288.645 s | 127.218 s | **2.269×** |

At 1024, warm transformer-forward median was 8.266 s for the 1×8 batch=2 control and 1.145 s for replica×model, a 7.223× warm-forward improvement. Cold compile steps remain included in pair denoise time.

The reference showcase provides six checksum-verified images at 1024×1024. Published checksums are available in `evidence/reference/showcase.json`.

See `evidence/reference/throughput.json` and `evidence/reference/showcase.json`.
