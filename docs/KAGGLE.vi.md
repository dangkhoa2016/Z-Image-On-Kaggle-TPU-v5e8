# Hướng dẫn Kaggle Runtime

> 🌐 Ngôn ngữ / Language: [English](KAGGLE.md) | **Tiếng Việt**

## Attach model

Mở [Kaggle model mirror](https://www.kaggle.com/models/dangkhoa2016/tongyi-mai-z-image) và dùng **Add Input → Models** trong Kaggle notebook để attach variation **PyTorch / default / version 1**. Đường dẫn dưới đây là vị trí mount chỉ đọc sau khi attach, không phải thư mục cần tự tạo hoặc nơi upload model. Kiểm tra có `model_index.json`, `transformer/` và `text_encoder/` trước khi chạy inference:

`/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1`

Environment đã được acceptance sử dụng Python 3.12, PyTorch 2.8.0+cpu, PyTorch/XLA 2.8.0, Diffusers 0.37.1, Transformers 5.12.1 và allocation Kaggle TPU v5litepod-8 với tám devices.

## Execution path trên Kaggle

Dùng các runtime script được maintain trong repository cùng model đã attach để tái lập 512×512 smoke và production path 1024×1024 `(replica=4, model=2)` / global batch 4. Các mode high-resolution là opt-in vì wall time lớn hơn đáng kể.


## Clean notebook template

Repository có `notebooks/kaggle-tpu-v5e8-production.ipynb`, một clean Kaggle notebook template dành cho thử nghiệm, kiểm chứng độc lập và showcase demo. Notebook không lưu execution count hoặc cell output. Import notebook vào Kaggle TPU v5e-8, attach model Z-Image rồi chạy workflow từ đầu.

## TPU ownership và anti-idle

Quy trình Kaggle đã acceptance chạy XLA health operation thật từ sớm và live topology probe trước measured TPU inference. Không chạy PJRT client thứ hai khi replica×model inference đang sở hữu toàn bộ tám devices.

## Topology pair-throughput

Workload hai ảnh được chấp nhận sử dụng một PJRT process trên toàn bộ tám devices với mesh `(replica=2, model=4)`. Hai PJRT client 4-chip độc lập không phải topology được chấp nhận trên Kaggle runtime đã test vì các multi-chip subset đã thử đều fail trong libtpu/PJRT bootstrap.

## Khả năng tái lập

Giữ prompt, negative prompt, seed, resolution, steps, CFG mode, chunk settings, timing records, PNG path và SHA256 cho mỗi run được acceptance. Xem [Reproducibility](REPRODUCIBILITY.vi.md) và frozen evidence trong `evidence/`.
