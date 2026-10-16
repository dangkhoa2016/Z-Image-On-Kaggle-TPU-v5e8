# Pull Request

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](pull_request_template.vi.md)

## Summary

Describe what changed and why.

## Validation

- [ ] `python -m pytest -q` passes
- [ ] Python syntax checks pass
- [ ] Evidence JSON is valid
- [ ] `python scripts/repository_audit.py .` passes
- [ ] `git diff --check` passes
- [ ] No model weights, credentials, XLA cache, latent cache, or runtime scratch data were added
- [ ] Benchmark/performance claims include reproducible evidence and name the comparison baseline
- [ ] Visual acceptance changes preserve rejected evidence and remediation provenance

## Evidence

List the exact commands, runtime, resolution, seed, steps, CFG mode, topology, timing type, and relevant artifact/checksum references.
