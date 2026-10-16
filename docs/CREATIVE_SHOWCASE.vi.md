# Creative Production Showcase

> 🌐 Ngôn ngữ / Language: [English](CREATIVE_SHOWCASE.md) | **Tiếng Việt**

Notebook production tạo tám ảnh creative độc lập ở độ phân giải 1024x1024, bên cạnh sáu ảnh kiểm chứng kỹ thuật, tổng cộng 14 PNG gốc theo mặc định. Các ảnh creative được tạo bởi model Full Z-Image trên Kaggle TPU v5e-8, chia thành hai batch TPU thực, mỗi batch bốn prompt riêng biệt. Prompt, nhóm nội dung và danh sách seed cụ thể được lưu trong scripts/showcase/creative-prompts.json (501–504, 615, 506–508).

Mỗi lượt chạy phải cung cấp timing inference, timing giải mã CPU VAE, checksum PNG và bằng chứng thuộc cùng phiên chạy. Gallery HTML song ngữ kiểm SHA256 từng ảnh; không trình bày ảnh từ phiên trước như inference mới. Mở notebooks/kaggle-tpu-v5e8-production.ipynb, attach Kaggle model, chọn TPU v5e-8 và chạy toàn bộ các cell.

Notebook trong Git chỉ là template mã nguồn chưa thực thi, không chứa output của lần chạy. Để nghiệm thu phát hành cần kiểm tra notebook thực sự đã được Kaggle thực thi và chứa output thật. Kaggle CLI có thể bỏ output trong notebook và tự xếp lịch chạy lại; không xác nhận publish thành công trước khi kiểm chứng version. Link Cloudflare gallery chỉ dùng tạm thời khi máy host vẫn hoạt động.

Run All mặc định tạo hai ảnh 512×512 độc lập, bốn ảnh kỹ thuật 1024×1024 và tám ảnh creative 1024×1024 (14 ảnh). Biến môi trường ZIMAGE_RUN_2000PX=True bật thêm ảnh 2048×1904 (15 ảnh). Encode 8 prompt với một lần tải encoder, TPU chạy 4+4 trên mesh 4×2, CPU VAE được tải một lần để giải mã 8 ảnh creative.
