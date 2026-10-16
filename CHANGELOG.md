# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

All notable public changes are documented here.

## v1.0.0 — 2026-10-15

### Added

- Full Z-Image BF16 inference on Kaggle TPU v5e-8 with manual PyTorch/XLA SPMD sharding.
- Clean Kaggle notebook template for experimentation, verification, and showcase demonstration; no saved execution state or outputs.
- Baseline 512×512 / 1024×1024 acceptance plus high-resolution sequential-CFG and query-chunked modes.
- One-process `(replica=2, model=4)` pair-throughput path with matched 1×8 batch=2 controls.
- Machine-readable benchmark/evidence records and reviewed publication assets.
- English/Vietnamese documentation, governance, licensing notes, and repository audit automation.

### Measured pair-throughput results

- 512: `126.263757719 s` → `95.001391616 s`, `1.329×` versus the matched 1×8 batch=2 control.
- 1024: `288.645413737 s` → `127.217999720 s`, `2.268903884×` versus the matched 1×8 batch=2 control.



- **2026-10-09:** repository bootstrap, licensing boundary, Kaggle TPU inference foundation.
