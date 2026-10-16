# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

## Model

Kaggle attached model path used during acceptance:

`/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1`

## Fresh-runtime evidence

A fresh TPU runtime independently reproduced 512×512 and 1024×1024 after `/kaggle/working` had been reset. The project helpers were reconstructed in the new runtime and the model was loaded directly from the attached Kaggle model.

## Required metadata

Each accepted run records prompt, negative prompt, seed, resolution, steps, CFG, execution mode, attention chunk configuration, per-step timing, latent path, and status. VAE records include output PNG path and SHA256.

## Reproducible Kaggle execution

The repository versions the runtime implementation, tests, documentation, prompts, seeds, timings, hashes, and compact JSON evidence needed to reproduce the accepted Kaggle TPU workflow, including `evidence/reference/production.json`. Run the maintained scripts from a tagged repository revision with the attached model and the documented TPU topology.

## Notebook-template reproduction path

For an end-to-end Kaggle reproduction path, use the clean `notebooks/kaggle-tpu-v5e8-production.ipynb` template included in this repository. It is intentionally stored without execution counts or outputs and is suitable for experimentation, verification, and showcase demonstrations.

## Replica×model throughput evidence

The throughput path uses one eight-device PJRT client and an explicit `(replica=2, model=4)` mesh. It was accepted at both 512×512 and 1024×1024 with two distinct seeds, CPU VAE decode, SHA256 output checks, and visual review. The 1024 result was then exercised across three style-diverse rounds: stylized 3D, 2D illustration, and photorealistic imagery.

Reference image assets and their SHA256 checksums are listed in `evidence/reference/showcase.json`. This public manifest does not claim exact-prompt reproducibility where the generation prompt was not preserved.

The six publication-facing showcase files are copied to `assets/showcase-*.png`. Their SHA256 values are recorded in `evidence/reference/showcase.json`.
