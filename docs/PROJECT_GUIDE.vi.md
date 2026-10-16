# Hướng dẫn dự án

> 🌐 Ngôn ngữ / Language: [English](PROJECT_GUIDE.md) | **Tiếng Việt**

Tài liệu này tách phần hướng dẫn theo từng nhóm người đọc ra khỏi README chính để README có thể tập trung vào kết quả kỹ thuật của dự án. Hãy chọn đường đọc phù hợp với mục đích của bạn.

## Nếu bạn chỉ muốn hiểu dự án thật nhanh

Bắt đầu từ [README tiếng Việt](../README.vi.md), đặc biệt là các phần **Vì sao dự án này tồn tại?**, **Repository mang lại những gì?**, **Kết quả chính trong một bảng** và **Kết quả trực quan**. Các phần này tóm tắt bài toán kỹ thuật, topology production, giới hạn đã đo và output đã accepted mà không bắt buộc phải đọc implementation.

Sau đó có thể mở:

- [Kiến trúc](ARCHITECTURE.vi.md) để hiểu thiết kế một PJRT process với mesh 4×2.
- [Production Reference Metrics](../evidence/reference/production.json) để xem các số đo TPU tham chiếu.
- [Release notes v1.0.0](releases/v1.0.0.vi.md) để xem contract của lần public đầu tiên.

## Nếu bạn muốn chạy model trên Kaggle

Đọc [Hướng dẫn Kaggle](KAGGLE.vi.md) trước, sau đó dùng clean notebook template trong `notebooks/` như một giao diện execution có hướng dẫn. Implementation có thể tái sử dụng vẫn nằm trong `zimage_tpu/`, các top-level runtime entry point và `scripts/`.

Luồng chạy ở mức cao:

1. Attach Kaggle Z-Image model mirror.
2. Xác minh pinned runtime environment.
3. Chạy BF16 PyTorch/XLA health operation thật.
4. Probe đủ tám TPU devices và mesh `(replica=4, model=2)`.
5. Chạy workload smoke 512.
6. Chạy production workload 1024 global batch 4.
7. Đối chiếu runtime evidence và output hashes.

Các thử nghiệm high-resolution được để opt-in vì tốn tài nguyên và thời gian hơn đáng kể so với production path thông thường.

## Nếu bạn review implementation kỹ thuật

Tập trung vào các thành phần sau:

- `run_zimage_spmd.py` — baseline SPMD inference entry point.
- `chunked_zimage_attention.py` — full-context query-chunked attention.
- `zimage_tpu/replica_mesh.py` — abstraction cho replica/model mesh.
- `zimage_tpu/dual_runtime.py` — hỗ trợ multi-prompt/topology dùng trong qualification.
- `scripts/run_zimage_tpu_batch4.py` — four-replica production runner.
- `scripts/probe_tpu_partition.py` — topology và device probing.
- `scripts/encode_prompt_cpu.py` và `scripts/decode_vae.py` — các stage chạy phía CPU.

Nên đọc [Kiến trúc](ARCHITECTURE.vi.md) trước khi đánh giá từng file, vì tài liệu này giải thích lý do prompt encoding và VAE decode nằm trên CPU, còn transformer được phân chia trên TPU mesh.

## Nếu bạn review các tuyên bố hiệu năng

Không nên chỉ so sánh wall-clock time. XLA graph compilation phụ thuộc tensor shape, vì vậy dự án tách cold compilation, warm transformer execution, VAE time và end-to-end acceptance.

Sử dụng các nguồn sau:

- [Phương pháp benchmark](BENCHMARK_METHODOLOGY.vi.md)
- [Production Reference Metrics](../evidence/reference/production.json)
- [Benchmark / Acceptance Table](../evidence/reference/resolution.json)
- [Throughput Comparison](../evidence/reference/throughput.json)

Topology production hiện tại là `(replica=4, model=2)`. Kết quả `(replica=2, model=4)` trước đây chỉ được giữ như matched historical comparison.

## Nếu bạn muốn tái lập hoặc audit evidence

Bắt đầu từ [Tái lập kết quả](REPRODUCIBILITY.vi.md). Publication model của dự án đi theo hướng evidence-first: prompt, seed, runtime identity, topology, timing, acceptance decision và output SHA256 đều được version để có thể kiểm tra độc lập với screenshot hay phần diễn giải bằng prose.

Các thư mục evidence chính:

```text
evidence/reference/     # public reference metrics, throughput, resolution, image checksums
assets/                 # ảnh accepted hiển thị trong README/release
```

Repository audit và CPU-safe tests kiểm tra các publication invariant mà không cần một TPU runner.

## Nếu bạn đánh giá khả năng high-resolution

Đọc [Kiến trúc](ARCHITECTURE.vi.md), [Phương pháp benchmark](BENCHMARK_METHODOLOGY.vi.md) và [Giới hạn](LIMITATIONS.vi.md).

Điểm quan trọng là query chunking chỉ chia query dimension nhưng vẫn giữ full K/V context. Đây là kỹ thuật quản lý bộ nhớ cho full attention, không phải chuyển sang local/window attention. Stretch boundary đã accepted là 2560×2048 với sequential CFG, query chunk 4096 và explicit XLA synchronization.

## Nếu bạn muốn mở rộng dự án

Đọc [Đóng góp](../CONTRIBUTING.vi.md), [Bảo mật](../SECURITY.vi.md) và [Hỗ trợ](../SUPPORT.vi.md). Mọi benchmark claim mới cần có khả năng tái lập, phải giữ ranh giới rõ giữa measured evidence và interpretation, và không được âm thầm thay thế evidence failed/rejected.

Các hướng mở rộng có thể gồm nghiên cứu topology mới, qualification thêm tensor shape, tối ưu runtime hoặc tăng mức tự động hóa cho image-quality checks. Training và fine-tuning nằm ngoài phạm vi hiện tại.

## Chỉ mục tài liệu

| Chủ đề | Tài liệu chính |
|---|---|
| Tổng quan dự án | [README tiếng Việt](../README.vi.md) |
| Runtime architecture | [Kiến trúc](ARCHITECTURE.vi.md) |
| Chạy trên Kaggle | [Hướng dẫn Kaggle](KAGGLE.vi.md) |
| Quy tắc benchmark | [Phương pháp benchmark](BENCHMARK_METHODOLOGY.vi.md) |
| Tái lập kết quả | [Reproducibility](REPRODUCIBILITY.vi.md) |
| Xử lý sự cố | [Troubleshooting](TROUBLESHOOTING.vi.md) |
| Boundary / non-goal | [Giới hạn](LIMITATIONS.vi.md) |
| Model licensing | [Model License](MODEL_LICENSE.vi.md) |
| Bản public đầu tiên | [Release notes v1.0.0](releases/v1.0.0.vi.md) |
