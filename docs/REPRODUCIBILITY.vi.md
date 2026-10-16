# Khả năng tái lập

> 🌐 Ngôn ngữ / Language: [English](REPRODUCIBILITY.md) | **Tiếng Việt**

## Model

Đường dẫn Kaggle attached model đã dùng trong acceptance:

`/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1`

## Evidence trên fresh runtime

Một TPU runtime mới đã tái lập độc lập 512×512 và 1024×1024 sau khi `/kaggle/working` được reset. Các helper của dự án được dựng lại trên runtime mới và model được load trực tiếp từ Kaggle model đã attach.

## Metadata bắt buộc

Mỗi run được chấp nhận ghi lại prompt, negative prompt, seed, resolution, steps, CFG, execution mode, attention chunk configuration, timing theo step, latent path và status. Record của VAE gồm output PNG path và SHA256.

## Tái lập execution trên Kaggle

Repository version runtime implementation, tests, documentation, prompt, seed, timing, hash và compact JSON evidence cần thiết để tái lập accepted Kaggle TPU workflow, bao gồm `evidence/reference/production.json`. Chạy các script được maintain từ tagged repository revision với model đã attach và topology TPU đã document.

## Đường tái lập bằng notebook template

Để tái lập end-to-end trên Kaggle, dùng clean template `notebooks/kaggle-tpu-v5e8-production.ipynb` có trong repository. Notebook được lưu chủ động không có execution count hoặc output và phù hợp cho thử nghiệm, kiểm chứng và showcase demo.

## Evidence throughput replica×model

Đường throughput dùng một PJRT client cho đủ 8 devices và mesh rõ ràng `(replica=2, model=4)`. Topology này đã được chấp nhận ở cả 512×512 và 1024×1024 với hai seed khác nhau, CPU VAE decode, SHA256 output checks và visual review. Kết quả 1024 sau đó được chạy qua ba nhóm style khác nhau: stylized 3D, 2D illustration và photorealistic.

Danh sách ảnh mẫu và mã SHA256 nằm trong `evidence/reference/showcase.json`. Manifest công khai không khẳng định khả năng tái lập chính xác prompt nếu prompt gốc không được lưu.

Sáu showcase file public nằm ở `assets/showcase-*.png`. SHA256 của chúng được ghi trong `evidence/reference/showcase.json`.
