# Xử lý sự cố

> 🌐 Ngôn ngữ / Language: [English](TROUBLESHOOTING.md) | **Tiếng Việt**

## `metadata.google.internal` / lỗi TPU discovery

Child shell của Desktop Commander có thể không kế thừa TPU environment từ Kaggle Jupyter kernel. Dùng `run_with_kernel_env.py` để subprocess nhận kernel environment trước khi import/khởi tạo `torch_xla`.

## `/dev/vfio/*: Device or resource busy`

Một XLA process hoặc heartbeat trước đó vẫn đang giữ TPU computation client. Dừng process/process group cũ, chờ VFIO được release rồi mới khởi động XLA process tiếp theo. Không coi đây là lỗi của model.

## Auto-SPMD solver failure

XLA auto-sharding thất bại trên graph Z-Image. Đường được chấp nhận dùng manual parameter sharding rõ ràng trên mesh 8 devices.

## HBM OOM với CFG

Batched CFG làm tăng effective attention batch và có thể tạo FP32 attention-score temporary rất lớn. Dùng sequential CFG cho high resolution.

## HBM OOM ở very high resolution

Dùng full-context query chunking. `2048×2048` đã được chứng minh với chunk 8192. `2560×2048` đã được chứng minh với chunk 4096 cộng explicit XLA synchronization giữa các chunk.

## Không benchmark khi heartbeat đang chạy

Heartbeat chỉ dành cho các khoảng host-heavy. Dừng heartbeat trước measured TPU inference và chỉ bật lại sau khi TPU client được release.

## Multi-chip subset fail bootstrap trước khi có Python evidence

Trên Kaggle v5e-8 runtime đã đo, subset 1 chip và toàn bộ allocation 8 devices khởi tạo thành công, trong khi subset 2 chip và 4 chip đã thử đều fail trong libtpu/PJRT bootstrap. Không diễn giải lỗi này thành Z-Image model OOM và không tiếp tục load model. Fallback throughput được chấp nhận là một PJRT client với mesh `(replica=2, model=4)` trên toàn bộ 8 devices.

## Không khởi động PJRT client thứ hai khi đang chạy replica×model

Đường replica×model đã sở hữu toàn bộ 8 TPU devices. Global heartbeat, probe hoặc inference client thứ hai có thể gây VFIO/PJRT ownership conflict. Giữ heartbeat tắt trong measured inference và dùng host-only monitoring để theo dõi tiến độ.

## Output technically pass nhưng vi phạm prompt

Xác minh mã SHA256 của PNG và kiểm tra chất lượng ảnh theo các ràng buộc prompt trước khi trình bày kết quả.
