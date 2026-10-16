# Limitations

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](LIMITATIONS.vi.md)

- The project validates inference, not training or fine-tuning.
- Very-high-resolution query chunking is substantially slower at 2560×2048.
- XLA compile latency can be large on the first steps and varies by tensor shape.
- Latency across different shapes must not be interpreted as a simple pixels-per-second curve.
- The generative model can produce prompt-adherence or visual-quality failures even when execution succeeds technically; visual review remains required for showcase acceptance.
- Independent 2-chip and 4-chip multi-chip PJRT subsets did not bootstrap on the measured Kaggle runtime. The accepted pair-throughput path is one PJRT client over all eight devices with `(replica=2, model=4)`.
- single-mesh two-image control and replicated-model two-image workload outputs are close but not bit-identical; no exact-equivalence claim is made.
- Published reference images have SHA256 checksums; generative image quality and prompt adherence still require visual inspection.
- Benchmark numbers apply to the tested Kaggle runtime, model build, prompts, seeds, scheduler, execution modes, and shapes; they are not universal TPU performance claims.
