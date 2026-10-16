# Pull Request

> 🌐 Ngôn ngữ / Language: [English](pull_request_template.md) | **Tiếng Việt**

## Tóm tắt

Mô tả thay đổi và lý do.

## Validation

- [ ] `python -m pytest -q` PASS
- [ ] Python syntax checks PASS
- [ ] Evidence JSON hợp lệ
- [ ] `python scripts/repository_audit.py .` PASS
- [ ] `git diff --check` PASS
- [ ] Không thêm model weights, credential, XLA cache, latent cache hoặc runtime scratch data
- [ ] Benchmark/performance claim có reproducible evidence và nêu rõ comparison baseline
- [ ] Thay đổi visual acceptance giữ rejected evidence và remediation provenance

## Evidence

Liệt kê exact command, runtime, resolution, seed, steps, CFG mode, topology, timing type và artifact/checksum reference liên quan.
