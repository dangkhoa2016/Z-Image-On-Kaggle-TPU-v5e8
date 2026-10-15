# Kiến trúc

> 🌐 Ngôn ngữ / Language: [English](ARCHITECTURE.md) | **Tiếng Việt**

## Runtime

Kaggle TPU v5e-8 được sử dụng thông qua PyTorch/XLA SPMD. Transformer Z-Image được nạp ở BF16 và 521 tensor tham số được shard thủ công trên 8 TPU devices.

## Pipeline

1. Prompt và negative prompt được encode trên CPU bằng đường chính thức `ZImagePipeline.encode_prompt()`.
2. Prompt embeddings được chuyển lên TPU.
3. Diffusion transformer thực hiện denoising theo một trong ba execution mode đã freeze.
4. Final latent được chuyển trở lại CPU.
5. VAE decode chạy riêng để tránh transformer và VAE cùng gây áp lực HBM.

## Các inference mode đã freeze

- **Baseline:** batched CFG, standard attention. Đã chứng minh ở 512×512 và 1024×1024.
- **High-resolution:** sequential CFG, standard attention. Đã chứng minh tới 2048×1904.
- **Very-high-resolution:** sequential CFG + full-context query chunking. Đã chứng minh ở 2048×2048 và 2560×2048.

## Query-chunked attention

Chỉ query dimension được chia thành các chunk. Mỗi query chunk vẫn attention tới toàn bộ key/value sequence. Output được nối lại theo đúng thứ tự query ban đầu. Với đường 2560×2048, explicit XLA synchronization giữa các chunk ngăn nhiều attention-score temporary lớn cùng tồn tại trong một HBM schedule.

## Thử nghiệm topology throughput

Đường production single-image được chấp nhận vẫn là một process SPMD mesh dùng đủ 8 devices. Một thử nghiệm throughput riêng kiểm tra liệu allocation Kaggle v5e-8 có thể được tận dụng hiệu quả hơn cho workload hai ảnh hay không.

Các PJRT client 4-chip độc lập **không** được chấp nhận: independent TPU partition discovery chứng minh runtime này có thể khởi tạo subset 1 chip và toàn bộ 8 chip, nhưng các subset multi-chip 2 chip và 4 chip đã thử đều fail trong libtpu/PJRT bootstrap trước khi load model. Vì vậy thử nghiệm fail-closed thay vì tuyên bố split dual-process thành công.

Fallback thành công sử dụng một PJRT client trên cả 8 devices với mesh 2D:

```text
mesh_shape = (2, 4)
axis_names = ("replica", "model")
```

Transformer parameters được replicate theo axis `replica` và shard 4-way theo axis `model`. Batch hai mẫu được partition theo `replica`. Sequential CFG được giữ nguyên, vì vậy mỗi CFG forward xử lý đồng thời cả hai sample trên hai replica.

Topology này mang tính bổ sung: nó không thay đổi default single-image 1×8 đã được chấp nhận trong `run_zimage_spmd.py`.
