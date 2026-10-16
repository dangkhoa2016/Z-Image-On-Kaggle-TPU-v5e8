# Giới hạn

> 🌐 Ngôn ngữ / Language: [English](LIMITATIONS.md) | **Tiếng Việt**

- Dự án xác minh inference, không bao gồm training hoặc fine-tuning.
- Very-high-resolution query chunking chậm hơn đáng kể ở 2560×2048.
- XLA compile latency có thể lớn ở các step đầu và thay đổi theo tensor shape.
- Không nên diễn giải latency giữa các shape khác nhau như một đường pixels-per-second đơn giản.
- Model sinh ảnh có thể thất bại về prompt adherence hoặc chất lượng hình ảnh dù execution technically thành công; visual review vẫn bắt buộc cho showcase acceptance.
- Các multi-chip PJRT subset 2 chip và 4 chip độc lập không bootstrap được trên Kaggle runtime đã đo. Đường pair-throughput được chấp nhận là một PJRT client dùng toàn bộ tám devices với `(replica=2, model=4)`.
- Output single-mesh two-image control và replicated-model two-image workload gần nhau nhưng không bit-identical; dự án không đưa ra claim exact equivalence.
- Các ảnh tham chiếu có mã SHA256 công khai; chất lượng ảnh và mức độ tuân thủ prompt vẫn cần được kiểm tra trực quan.
- Các số benchmark chỉ áp dụng cho Kaggle runtime, model build, prompt, seed, scheduler, execution mode và shape đã test; đây không phải claim hiệu năng TPU phổ quát.
