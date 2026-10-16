# Kaggle Runtime Guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](KAGGLE.vi.md)

## Attach the model

Open the [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/tongyi-mai-z-image) and use **Add Input → Models** in the Kaggle notebook to attach the **PyTorch / default / version 1** variation. The path below is the resulting read-only mount, not a directory to create or an upload destination. Confirm `model_index.json`, `transformer/`, and `text_encoder/` exist there before running inference:

`/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1`

The accepted environment used Python 3.12, PyTorch 2.8.0+cpu, PyTorch/XLA 2.8.0, Diffusers 0.37.1, Transformers 5.12.1, and a Kaggle TPU v5litepod-8 allocation exposing eight devices.

## Kaggle execution path

Use the maintained repository runtime scripts with the attached model to reproduce the accepted 512×512 smoke and 1024×1024 `(replica=4, model=2)` / global-batch-4 path. High-resolution modes are opt-in because their wall time is substantially larger.


## Clean notebook template

The repository includes `notebooks/kaggle-tpu-v5e8-production.ipynb`, a clean Kaggle notebook template for experimentation, independent verification, and showcase demonstration. It contains no saved execution counts or cell outputs. Import it into a Kaggle TPU v5e-8 session, attach the Z-Image model, and run the workflow from the top.

## TPU ownership and anti-idle behavior

The accepted Kaggle execution procedure performs a real XLA health operation early and runs a live topology probe before measured TPU inference. Do not run a second PJRT client while replica×model inference owns all eight devices.

## Pair-throughput topology

The accepted pair workload uses one PJRT process over all eight devices with mesh `(replica=2, model=4)`. Independent 4-chip PJRT clients are not an accepted topology on the tested Kaggle runtime because the tested multi-chip subsets failed during libtpu/PJRT bootstrap.

## Reproducibility

Keep prompt, negative prompt, seed, resolution, steps, CFG mode, chunk settings, timing records, PNG path, and SHA256 for every accepted run. See [Reproducibility](REPRODUCIBILITY.md) and the frozen evidence under `evidence/`.
