# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ARCHITECTURE.vi.md)

## Runtime

Kaggle TPU v5e-8 is used through PyTorch/XLA SPMD. The Z-Image transformer is loaded in BF16 and 521 parameter tensors are manually sharded across the 8 TPU devices.

## Pipeline

1. Prompt and negative prompt are encoded on CPU with the official `ZImagePipeline.encode_prompt()` path.
2. Prompt embeddings are moved to TPU.
3. The diffusion transformer performs denoising under one of three frozen execution modes.
4. The final latent is moved back to CPU.
5. VAE decoding runs separately, avoiding simultaneous transformer + VAE HBM pressure.

## Frozen inference modes

- **Baseline:** batched CFG, standard attention. Proven at 512×512 and 1024×1024.
- **High-resolution:** sequential CFG, standard attention. Proven through 2048×1904.
- **Very-high-resolution:** sequential CFG + full-context query chunking. Proven at 2048×2048 and 2560×2048.

## Query-chunked attention

Only the query dimension is split into chunks. Each query chunk still attends to the full key/value sequence. Outputs are concatenated in original query order. For the 2560×2048 path, explicit XLA synchronization between chunks prevents multiple large attention-score temporaries from remaining live in the same HBM schedule.

## Throughput topology experiment

The accepted single-image production path remains the original one-process 8-device SPMD mesh. A separate throughput experiment tested whether the Kaggle v5e-8 allocation could be used more efficiently for two-image workloads.

Independent 4-chip PJRT clients were **not** adopted: independent TPU partition discovery proved that this runtime can initialize a one-chip subset and the full eight-chip allocation, but tested 2-chip and 4-chip multi-chip subsets failed during libtpu/PJRT bootstrap before model loading. The experiment therefore failed closed instead of claiming a dual-process split.

The successful fallback uses one PJRT client over all eight devices with a 2D mesh:

```text
mesh_shape = (2, 4)
axis_names = ("replica", "model")
```

Transformer parameters are replicated across the `replica` axis and sharded four ways across the `model` axis. A two-sample batch is partitioned across `replica`. Sequential CFG is retained, so each CFG forward evaluates both samples concurrently across the two replicas.

This topology is additive: it does not change the accepted 1×8 single-image defaults in `run_zimage_spmd.py`.
