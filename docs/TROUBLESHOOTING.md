# Troubleshooting

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](TROUBLESHOOTING.vi.md)

## `metadata.google.internal` / TPU discovery failure

Desktop Commander child shells may not inherit the Kaggle Jupyter kernel TPU environment. Use `run_with_kernel_env.py` so subprocesses inherit the kernel environment before importing/initializing `torch_xla`.

## `/dev/vfio/*: Device or resource busy`

A prior XLA process or heartbeat still owns the TPU computation client. Stop the old process/process group, wait for VFIO release, and only then start the next XLA process. Do not treat this as a model failure.

## Auto-SPMD solver failure

XLA auto-sharding failed on the Z-Image graph. The accepted path uses explicit manual parameter sharding over an 8-device mesh.

## HBM OOM with CFG

Batched CFG doubles the effective attention batch and can create very large FP32 attention-score temporaries. Use sequential CFG for high resolution.

## HBM OOM at very high resolution

Use full-context query chunking. `2048×2048` is proven with chunk 8192. `2560×2048` is proven with chunk 4096 plus explicit XLA synchronization between chunks.

## Do not benchmark with heartbeat running

Heartbeat is only for host-heavy gaps. Stop it before measured TPU inference and restart it only after the TPU client has been released.

## Multi-chip subset bootstrap fails before Python evidence

On the measured Kaggle v5e-8 runtime, a one-chip subset and the full eight-device allocation initialized successfully, while tested 2-chip and 4-chip subsets failed during libtpu/PJRT bootstrap. Do not interpret that as a Z-Image model OOM and do not continue into model loading. The accepted throughput fallback is one PJRT client with a `(replica=2, model=4)` mesh over all eight devices.

## Do not start a second PJRT client during replica×model inference

The replica×model path already owns all eight TPU devices. A global heartbeat, probe, or second inference client can create VFIO/PJRT ownership conflicts. Keep heartbeat disabled during measured inference and use host-only monitoring for progress.

## Showcase output passes technically but violates the prompt

Verify generated PNG checksums and inspect image quality against the declared prompt constraints before presenting results.
