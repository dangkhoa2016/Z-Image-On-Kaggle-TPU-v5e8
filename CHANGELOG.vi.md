# Changelog

> 🌐 Ngôn ngữ / Language: [English](CHANGELOG.md) | **Tiếng Việt**

Tất cả thay đổi public đáng chú ý được ghi tại đây.

## v1.0.0 — 2026-10-15

### Đã bổ sung

- Full Z-Image BF16 inference trên Kaggle TPU v5e-8 với manual PyTorch/XLA SPMD sharding.
- Clean Kaggle notebook template dành cho thử nghiệm, kiểm chứng và showcase demo; không lưu execution state hoặc output.
- Acceptance baseline 512×512 / 1024×1024 cùng high-resolution sequential-CFG và query-chunked mode.
- Đường pair-throughput one-process `(replica=2, model=4)` với matched 1×8 batch=2 control.
- Machine-readable benchmark/evidence record và reviewed publication asset.
- Tài liệu English/Vietnamese, governance, licensing notes và repository audit automation.

### Kết quả pair-throughput đã đo

- 512: `126.263757719 s` → `95.001391616 s`, nhanh hơn `1.329×` so với matched 1×8 batch=2 control.
- 1024: `288.645413737 s` → `127.217999720 s`, nhanh hơn `2.268903884×` so với matched 1×8 batch=2 control.



- **2026-10-09:** repository bootstrap, licensing boundary, nền tảng Kaggle TPU inference.
