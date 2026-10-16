# Project Guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PROJECT_GUIDE.vi.md)

This guide keeps audience-specific navigation out of the main README so the README can focus on the engineering result itself. Use the path below that matches what you want to verify or accomplish.

## If you only want to understand the project quickly

Start with the [README](../README.md), especially **Why this project exists**, **What this repository delivers**, **Verified at a glance**, and **Visual results**. Those sections summarize the engineering problem, the production topology, the measured boundary, and the accepted outputs without requiring you to read the implementation.

Then open:

- [Architecture](ARCHITECTURE.md) for the one-PJRT-process 4×2 design.
- [Production Reference Metrics](../evidence/reference/production.json) for published TPU reference measurements.
- [v1.0.0 Release Notes](releases/v1.0.0.md) for the frozen first-public-release contract.

## If you want to run the model on Kaggle

Read the [Kaggle Runtime Guide](KAGGLE.md) first, then use the clean notebook template under `notebooks/` as a guided execution surface. The reusable implementation remains in `zimage_tpu/`, the top-level runtime entry points, and `scripts/`.

The expected high-level flow is:

1. Attach the Kaggle Z-Image model mirror.
2. Confirm the pinned runtime environment.
3. Run a real BF16 PyTorch/XLA health operation.
4. Probe all eight TPU devices and the `(replica=4, model=2)` mesh.
5. Run the 512 smoke workload.
6. Run the 1024 global-batch-4 production workload.
7. Compare runtime evidence and output hashes.

High-resolution experiments are intentionally opt-in because they are substantially more expensive than the normal production path.

## If you are reviewing the engineering implementation

Focus on these components:

- `run_zimage_spmd.py` — baseline SPMD inference entry point.
- `chunked_zimage_attention.py` — full-context query-chunked attention.
- `zimage_tpu/replica_mesh.py` — replica/model mesh abstractions.
- `zimage_tpu/dual_runtime.py` — multi-prompt / topology support used during qualification.
- `scripts/run_zimage_tpu_batch4.py` — four-replica production runner.
- `scripts/probe_tpu_partition.py` — topology and device probing.
- `scripts/encode_prompt_cpu.py` and `scripts/decode_vae.py` — CPU-side pipeline stages.

Read [Architecture](ARCHITECTURE.md) before evaluating individual files; it explains why prompt encoding and VAE decode remain CPU-side and why the transformer is split over the TPU mesh.

## If you are reviewing performance claims

Do not compare only wall-clock numbers. XLA graph compilation is shape-sensitive, so the project separates cold compilation, warm transformer execution, VAE time, and end-to-end acceptance.

Use:

- [Benchmark Methodology](BENCHMARK_METHODOLOGY.md)
- [Production Reference Metrics](../evidence/reference/production.json)
- [Benchmark / Acceptance Table](../evidence/reference/resolution.json)
- [Throughput Comparison](../evidence/reference/throughput.json)

The current production topology is `(replica=4, model=2)`. The older `(replica=2, model=4)` result is preserved only as a matched historical comparison.

## If you want to reproduce or audit the evidence

Start with [Reproducibility](REPRODUCIBILITY.md). The publication model is evidence-first: prompts, seeds, runtime identity, topology, timings, acceptance decisions, and output SHA256 values are versioned so they can be checked independently from screenshots or prose claims.

Useful evidence directories:

```text
evidence/reference/     # public reference metrics, throughput, resolution, image checksums
assets/                 # accepted images exposed in the README/release
```

The repository audit and CPU-safe tests validate publication invariants without requiring a TPU runner.

## If you are evaluating high-resolution capability

Read [Architecture](ARCHITECTURE.md), [Benchmark Methodology](BENCHMARK_METHODOLOGY.md), and [Limitations](LIMITATIONS.md).

The key distinction is that query chunking splits only the query dimension while preserving full K/V context. It is a memory-management technique for full attention, not a switch to local/window attention. The accepted stretch boundary is 2560×2048 with sequential CFG, query chunk 4096, and explicit XLA synchronization.

## If you plan to extend the project

Read [Contributing](../CONTRIBUTING.md), [Security](../SECURITY.md), and [Support](../SUPPORT.md). Keep new benchmark claims reproducible, preserve the separation between measured evidence and interpretation, and do not silently replace failed or rejected evidence.

Potential extension areas include new topology studies, additional shape qualification, runtime optimization, or stronger automated image-quality checks. Training and fine-tuning are outside the current project scope.

## Document index

| Topic | Primary document |
|---|---|
| Project overview | [README](../README.md) |
| Runtime architecture | [Architecture](ARCHITECTURE.md) |
| Kaggle execution | [Kaggle Guide](KAGGLE.md) |
| Benchmark rules | [Benchmark Methodology](BENCHMARK_METHODOLOGY.md) |
| Reproduction | [Reproducibility](REPRODUCIBILITY.md) |
| Troubleshooting | [Troubleshooting](TROUBLESHOOTING.md) |
| Boundaries / non-goals | [Limitations](LIMITATIONS.md) |
| Model licensing | [Model License](MODEL_LICENSE.md) |
| First public release | [v1.0.0 Release Notes](releases/v1.0.0.md) |
