# Chính sách bảo mật

> 🌐 Ngôn ngữ / Language: [English](SECURITY.md) | **Tiếng Việt**

## Secret và dữ liệu nhạy cảm

Không đưa Kaggle token, Hugging Face token, GitHub PAT, Cloudflare credential, SSH private key, private model URL hoặc secret khác vào issue, notebook, log, evidence, screenshot hay commit.

Trước khi đính kèm log, hãy loại bỏ environment variable và authorization header. Runtime evidence chỉ nên chứa lượng thông tin tối thiểu cần cho khả năng tái lập.

## Báo cáo vulnerability

Với báo cáo nhạy cảm về security, hãy liên hệ riêng với maintainer tại `i.am@dangkhoa.dev` thay vì mở public issue. Hãy kèm mô tả ngắn gọn, file/version bị ảnh hưởng, bước reproduce, impact và mitigation gợi ý nếu có.

Không công bố exploit detail trước khi maintainer có thời gian hợp lý để điều tra và chuẩn bị fix.

## Phạm vi

Policy này áp dụng cho code và configuration do repository viết. Security issue thuộc Z-Image, PyTorch/XLA, Kaggle hoặc dependency upstream khác cũng nên được báo cáo tới upstream project phù hợp khi áp dụng.
