# Contributing

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for improving this Kaggle TPU inference project. Keep changes small, reviewable, and reproducible.

## Development rules

- Do not commit model weights, latent caches, XLA compilation caches, credentials, private Kaggle artifacts, or generated runtime scratch data.
- Keep source changes focused; avoid mixing unrelated code, documentation, and benchmark changes in one commit.
- Preserve the accepted single-image 1×8 path unless a change explicitly targets it and includes equivalent acceptance evidence.
- Do not describe independent 4-chip PJRT clients as supported; partition discovery failed before model loading on the tested runtime.

## Evidence required for inference changes

Record the exact resolution, prompt/negative prompt, seed, steps, CFG, CFG execution mode, attention chunking settings, topology, and whether timing is cold/compile or warm. Performance claims must be backed by machine-readable evidence and must name the comparison baseline.

For visual acceptance, a technically successful denoise is not enough when the prompt has explicit constraints. Preserve rejected artifacts and use a new remediation identifier instead of overwriting evidence.

## Validation

Before opening a pull request, run:

```bash
python -m pytest -q
python -m compileall -q chunked_zimage_attention.py run_with_kernel_env.py run_zimage_spmd.py scripts zimage_tpu
python scripts/repository_audit.py .
git diff --check
```

If a change touches evidence JSON, verify those files parse cleanly as well.
