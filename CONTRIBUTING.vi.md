# Đóng góp

> 🌐 Ngôn ngữ / Language: [English](CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã cải thiện dự án inference Kaggle TPU này. Hãy giữ thay đổi nhỏ, dễ review và có thể tái lập.

## Quy tắc phát triển

- Không commit model weights, latent cache, XLA compilation cache, credential, private Kaggle artifact hoặc runtime scratch data được sinh ra.
- Giữ source change tập trung; tránh trộn code, tài liệu và benchmark không liên quan trong một commit.
- Giữ nguyên đường single-image 1×8 đã được acceptance trừ khi thay đổi nhắm trực tiếp vào nó và có acceptance evidence tương đương.
- Không mô tả các PJRT client 4-chip độc lập là supported; partition discovery đã fail trước khi load model trên runtime đã test.

## Evidence bắt buộc cho thay đổi inference

Ghi chính xác resolution, prompt/negative prompt, seed, steps, CFG, CFG execution mode, attention chunking settings, topology và timing là cold/compile hay warm. Performance claim phải có machine-readable evidence và phải chỉ rõ baseline so sánh.

Với visual acceptance, denoise technically thành công chưa đủ nếu prompt có constraint rõ ràng. Giữ lại artifact bị reject và dùng remediation identifier mới thay vì overwrite evidence.

## Validation

Trước khi mở pull request, chạy:

```bash
python -m pytest -q
python -m compileall -q chunked_zimage_attention.py run_with_kernel_env.py run_zimage_spmd.py scripts zimage_tpu
python scripts/repository_audit.py .
git diff --check
```

Nếu thay đổi chạm evidence JSON, hãy xác minh các file đó parse sạch.
