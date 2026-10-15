# Phương pháp Benchmark

> 🌐 Ngôn ngữ / Language: [English](BENCHMARK_METHODOLOGY.md) | **Tiếng Việt**

- Dtype: BF16 cho transformer weights/activations.
- Seed: 42.
- Scheduler: `FlowMatchEulerDiscreteScheduler` từ model đã attach.
- Acceptance mặc định: 28 steps, CFG 3.0.
- XLA execution được đồng bộ rõ ràng trước các mốc timing.
- Các step đầu/cold được giữ lại thay vì che đi.
- Warm forward median loại hai step đầu khi có ít nhất ba step.
- Thời gian VAE decode được đo riêng trên CPU.
- Compile/cold và warm performance không bị trộn thành một metric không nhãn.

Xem `evidence/reference/resolution.json` để biết kết quả đã freeze.

## So sánh throughput hai ảnh

Thử nghiệm topology đo **pair wall time**, không phải kernel time tách biệt trên từng device. single-mesh two-image control và replicated-model two-image workload dùng cùng resolution, 28 steps, CFG 3.0, sequential CFG, frozen prompt embeddings và hai seed cố định. Cả hai sample phải hoàn tất denoising và decode thành PNG hợp lệ trước khi topology được chấp nhận.

| Resolution | 1×8 model mesh, batch=2 | 2×4 replica×model, batch=2 | replicated-model two-image workload speedup vs single-mesh two-image control |
|---|---:|---:|---:|
| 512×512 | 126.264 s | 95.001 s | 1.329× |
| 1024×1024 | 288.645 s | 127.218 s | **2.269×** |

Ở 1024, warm transformer-forward median là 8.266 s cho control 1×8 batch=2 và 1.145 s cho replica×model, tương ứng cải thiện warm-forward 7.223×. Các cold compile step vẫn được tính trong pair denoise time.

Bộ ảnh tham chiếu gồm sáu ảnh 1024×1024 đã được kiểm tra SHA256. Các mã hash được công bố trong `evidence/reference/showcase.json`.

Xem `evidence/reference/throughput.json` và `evidence/reference/showcase.json`.
