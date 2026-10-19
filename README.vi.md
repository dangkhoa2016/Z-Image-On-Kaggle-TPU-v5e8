# Z-Image trên Kaggle TPU v5e-8

[![CI](https://github.com/dangkhoa2016/Z-Image-On-Kaggle-TPU-v5e8/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Z-Image-On-Kaggle-TPU-v5e8/actions/workflows/ci.yml)
[![Repository Audit](https://github.com/dangkhoa2016/Z-Image-On-Kaggle-TPU-v5e8/actions/workflows/repository-audit.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Z-Image-On-Kaggle-TPU-v5e8/actions/workflows/repository-audit.yml)
![Version](https://img.shields.io/badge/version-v1.0.0-orange)
[![Kaggle](https://img.shields.io/badge/Kaggle-TPU%20v5e--8-20BEFF?logo=kaggle)](#kiến-trúc-production)
[![Precision](https://img.shields.io/badge/precision-BF16-blue)](#kiến-trúc-production)
[![Model](https://img.shields.io/badge/model-Tongyi--MAI%2FZ--Image-orange)](https://huggingface.co/Tongyi-MAI/Z-Image)
[![Code License](https://img.shields.io/github/license/dangkhoa2016/Z-Image-On-Kaggle-TPU-v5e8)](LICENSE)
[![Model License](https://img.shields.io/badge/model%20license-Apache--2.0-green)](docs/MODEL_LICENSE.vi.md)

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Chạy **Z-Image đầy đủ, không distilled, với transformer 6.15B tham số** trên Kaggle **TPU v5e-8** miễn phí và khai thác cả tám TPU devices bằng PyTorch/XLA SPMD. Repository này biến một workload sinh ảnh khó fit vào TPU thành một execution path có kiến trúc production, benchmark, high-resolution mode, tests và evidence đủ rõ để cộng đồng có thể kiểm tra lại.

Dự án **không** dùng GGUF, quantization hay distillation. Transformer inference giữ **BF16**; prompt encoding và VAE decode chạy trên CPU, còn transformer denoising chạy trên TPU.

## Vì sao dự án này tồn tại?

Với một model cỡ Z-Image, việc “chạy lên được TPU” chưa phải là kết quả kỹ thuật đủ hữu ích. Một triển khai thực tế còn phải trả lời được: transformer được shard thế nào trên toàn bộ v5e-8, làm sao tạo nhiều ảnh đồng thời, giới hạn HBM ở đâu, topology nào nên dùng cho workload thông thường, độ phân giải có thể đẩy tới mức nào và người khác sẽ kiểm chứng các tuyên bố đó bằng gì.

Repository này đưa các câu trả lời đó vào code, tests và evidence có version, thay vì chỉ để lại output của một phiên notebook đơn lẻ.

## Repository mang lại những gì?

- Execution path **PyTorch/XLA BF16** có thể tái sử dụng cho model Z-Image đầy đủ.
- Manual **SPMD parameter sharding trên cả 8 TPU devices**; runtime đã accepted shard 521 parameter tensors của transformer.
- Production mesh **`(replica=4, model=2)`**, tạo bốn ảnh 1024×1024 đồng thời trong một PJRT process.
- Ba execution mode theo nhu cầu bộ nhớ: **batched CFG**, **sequential CFG** và **full-context query-chunked attention**.
- Production policy đã đo thực tế cho 512, 768 và 1024, bao gồm giới hạn HBM đã thử nghiệm tại 1024.
- High-resolution validation tới **2560×2048 (5.24 MP)** mà không thay full attention bằng local/window attention.
- CPU-safe regression tests, repository audit gates, benchmark evidence machine-readable, seed cố định, output hash và visual acceptance artifacts.
- Bộ tài liệu kỹ thuật song ngữ Anh/Việt.

Nếu muốn biết nên bắt đầu từ đâu tùy mục đích review, chạy thử hay nghiên cứu, xem [Hướng dẫn dự án](docs/PROJECT_GUIDE.vi.md).

## Kết quả chính trong một bảng

| Hạng mục | Kết quả đã xác minh |
|---|---|
| Upstream model | `Tongyi-MAI/Z-Image`, full / undistilled |
| Kích thước transformer | 6.15B tham số |
| Accelerator | Kaggle TPU v5e-8 / 8 devices |
| Precision | BF16 transformer inference |
| Transformer tensors được shard | 521 |
| Production topology | `(replica=4, model=2)` |
| Workload production 1024 | 4 ảnh đồng thời / global batch 4 |
| Frozen benchmark 1024 | **90.264 giây / 4 ảnh** (`0.04431 img/s`) |
| Maximum safe tested batch tại 1024 | **4** |
| Độ phân giải lớn nhất đã xác minh | **2560×2048 / 5.24 MP** |

Nếu quy đổi theo throughput của bốn output đồng thời, frozen benchmark 1024 tương đương 22.566 giây/ảnh. Một fresh Kaggle TPU execution của cùng workload bốn seed hoàn thành trong **90.099 giây**.

## Kiến trúc production

Production path đã accepted sử dụng **một PJRT process duy nhất trên cả tám TPU devices**:

```text
Prompt + negative prompt
        │
        ▼
CPU text encoder
        │
        ▼
Prompt embeddings
        │
        ▼
┌────────────────────────────────────────┐
│ Kaggle TPU v5e-8                       │
│                                        │
│  Replica 0 ─ Model shard 0 + shard 1   │
│  Replica 1 ─ Model shard 0 + shard 1   │
│  Replica 2 ─ Model shard 0 + shard 1   │
│  Replica 3 ─ Model shard 0 + shard 1   │
│                                        │
│  Z-Image transformer · BF16 · SPMD     │
└────────────────────────────────────────┘
        │
        ▼
Latents → CPU VAE decode → PNG + SHA256 evidence
```

Mesh production là `(replica=4, model=2)`: bốn replica sinh bốn sample đồng thời; trong mỗi replica, transformer được model-shard qua hai TPU devices.

Production policy đã đo:

| Độ phân giải | Topology | Global batch | Trạng thái |
|---:|---|---:|---|
| 512×512 | `(replica=4, model=2)` | 8 | PASS |
| 768×768 | `(replica=4, model=2)` | 4 | PASS |
| 1024×1024 | `(replica=4, model=2)` | 4 | PASS / maximum safe tested |

Tại 1024, batch 5–8 đều đã được thử và fail XLA compilation do áp lực HBM. Vì vậy batch 4 là một **giới hạn đã đo**, không phải một con số ước lượng.

Xem [Kiến trúc](docs/ARCHITECTURE.vi.md) và [Phương pháp benchmark](docs/BENCHMARK_METHODOLOGY.vi.md) để đọc toàn bộ lập luận kỹ thuật.

## Ba inference mode

**Batched CFG** là đường chạy thông thường cho workload 512/1024. Nhánh positive và negative được xử lý cùng batch để ưu tiên throughput.

**Sequential CFG** chạy hai nhánh CFG lần lượt để giảm peak HBM. Với standard attention, đường này đã được xác minh tới **2048×1904**.

**Sequential CFG + full-context query chunking** chỉ chia query dimension; mỗi query chunk vẫn attend tới toàn bộ K/V context. Vì vậy đây không phải local/window attention. Đường này đã PASS ở **2048×2048** và **2560×2048**; run accepted tại 2560×2048 dùng query chunk 4096 cùng explicit XLA synchronization.

## Kết quả trực quan

Các ảnh accepted được lưu cùng evidence mô tả cách chúng được tạo và review, thay vì chỉ trình bày hình đẹp mà không có provenance.

### Publication acceptance — 1024×1024

![Publication acceptance 1024×1024](assets/publication-1024-preview.jpg)

### Sáu showcase 1024×1024 đã accepted

| Stylized 3D | Stylized 3D |
|---|---|
| ![Miniature mountain village](assets/showcase-3d-mountain-village.png) | ![Futuristic eco-city](assets/showcase-3d-eco-city.png) |
| **Miniature mountain village** | **Futuristic eco-city** |

| Illustration | Illustration |
|---|---|
| ![Watercolor canal town](assets/showcase-illustration-canal-town.png) | ![Storybook cliffside castle](assets/showcase-illustration-cliffside-castle.png) |
| **Watercolor canal town** | **Storybook cliffside castle** |

| Photoreal | Photoreal |
|---|---|
| ![Mechanical wristwatch with leather strap](assets/showcase-photoreal-watch-final.png) | ![Cinematic alpine lake](assets/showcase-photoreal-alpine-lake.png) |
| **Đồng hồ cơ không branding** | **Hồ alpine cinematic** |


### High-resolution boundary — 2560×2048

[![2560×2048 preview](assets/stretch-2560x2048-preview.jpg)](assets/stretch-2560x2048-full-redecoded.png)

Run accepted sử dụng 28 steps, CFG 3.0, seed 42, sequential CFG, query chunk 4096 và explicit XLA synchronization. Checksum của output accepted ban đầu và checksum của PNG full-resolution được re-decode sau đó được ghi tách biệt; repository không tuyên bố hai file này byte-identical.

## Chạy trên Kaggle

Từ [Kaggle Models](https://www.kaggle.com/models/dangkhoa2016/tongyi-mai-z-image), dùng **Add Input → Models** để attach variation **PyTorch / default / version 1**. Kiểm tra thư mục model được mount chỉ đọc:

```text
/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1
```

Publication environment đã xác minh:

```text
Python       3.12
PyTorch      2.8.0+cpu
PyTorch/XLA  2.8.0
Diffusers    0.37.1
Transformers 5.12.1
TPU          v5litepod-8 / 8 devices
```

Trình tự validation khuyến nghị:

```text
1. Attach model
2. Load tagged runtime source
3. Chạy BF16 XLA health check thật
4. Probe live mesh (replica=4, model=2)
5. Chạy workload smoke 512
6. Chạy production workload 1024 global batch 4
7. Export và đối chiếu reproducibility evidence
```

Repository có clean Kaggle notebook template tại `notebooks/kaggle-tpu-v5e8-production.ipynb` để hướng dẫn execution; các runtime scripts vẫn là implementation source of truth có thể tái sử dụng. High-resolution qualification là opt-in vì tốn tài nguyên và thời gian đáng kể hơn production workload mặc định.

Xem [Hướng dẫn Kaggle](docs/KAGGLE.vi.md) và [Hướng dẫn tái lập](docs/REPRODUCIBILITY.vi.md).

## Evidence thay vì chỉ có screenshot

Các tuyên bố về hiệu năng được gắn với JSON/Markdown evidence có version, prompt và seed cố định, runtime identity, output SHA256 và acceptance record. Cold XLA compilation và warm transformer execution được tách riêng vì graph compilation phụ thuộc mạnh vào tensor shape.

Evidence chính:

- [Production Reference Metrics](evidence/reference/production.json)
- [Benchmark / Acceptance Table](evidence/reference/resolution.json)
- [Replica×Model Throughput Comparison](evidence/reference/throughput.json)
- [Six-Image SHA256 Reference Manifest](evidence/reference/showcase.json)

Benchmark pair `(replica=2, model=4)` trước đây vẫn được giữ như matched-control evidence lịch sử; nó không thay thế topology production `(replica=4, model=2)` hiện tại. Hướng chạy các PJRT client độc lập trên subset 4 chip cũng đã được thử và fail trước model load, vì vậy các multi-replica path accepted đều dùng một PJRT process trên toàn bộ tám devices.

## Bản đồ tài liệu

| Tài liệu | Mục đích |
|---|---|
| [Hướng dẫn dự án](docs/PROJECT_GUIDE.vi.md) | Chọn đường đọc nhanh nhất cho review, sử dụng hoặc nghiên cứu |
| [Kiến trúc](docs/ARCHITECTURE.vi.md) | Runtime design, sharding, topology và execution modes |
| [Hướng dẫn Kaggle](docs/KAGGLE.vi.md) | Environment và quy trình chạy trên Kaggle |
| [Phương pháp benchmark](docs/BENCHMARK_METHODOLOGY.vi.md) | Quy tắc đo và cách diễn giải kết quả |
| [Tái lập kết quả](docs/REPRODUCIBILITY.vi.md) | Evidence model, hashes và quy trình reproduction |
| [Troubleshooting](docs/TROUBLESHOOTING.vi.md) | Các failure mode thường gặp của runtime/XLA |
| [Giới hạn](docs/LIMITATIONS.vi.md) | Boundary đã xác minh và các non-goal |
| [Release notes v1.0.0](docs/releases/v1.0.0.vi.md) | Contract của bản public đầu tiên |

## Cấu trúc repository

```text
.
├── zimage_tpu/                 # reusable runtime modules
├── scripts/                    # probes, runners, CPU encode/decode helpers
├── tests/                      # CPU-safe regression và publication checks
├── notebooks/                  # clean Kaggle execution template
├── evidence/                   # machine-readable + community-facing evidence
├── assets/                     # accepted publication/showcase images
├── docs/                       # architecture, benchmark, Kaggle, reproducibility
├── run_zimage_spmd.py          # baseline SPMD inference entry point
└── chunked_zimage_attention.py # full-context query-chunked attention
```

## Phạm vi và giới hạn

Dự án xác minh **inference**, không phải training hoặc fine-tuning. First-step XLA compilation có thể lâu và thay đổi theo tensor shape. Chế độ 2560×2048 là capability boundary, không phải cấu hình được khuyến nghị để chạy thường xuyên. Chất lượng ảnh sinh và prompt adherence vẫn cần visual review.

Hãy đọc [Giới hạn](docs/LIMITATIONS.vi.md) trước khi suy rộng kết quả ra ngoài các cấu hình đã đo.

## License và attribution

Code do repository phát triển dùng MIT License. Z-Image weights là third-party material từ upstream; model card upstream công bố Apache-2.0 và repository này không redistribute weights. Xem [Model License và Attribution](docs/MODEL_LICENSE.vi.md) và [License Notes](LICENSE-NOTES.vi.md).

Z-Image được phát triển bởi Tongyi-MAI. Dự án này tập trung vào engineering, qualification và reproducible inference trên Kaggle TPU v5e-8, không tuyên bố quyền tác giả đối với upstream model.
