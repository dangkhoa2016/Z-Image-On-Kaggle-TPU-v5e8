# Security Policy

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

## Secrets and sensitive data

Do not include Kaggle tokens, Hugging Face tokens, GitHub PATs, Cloudflare credentials, SSH private keys, private model URLs, or other secrets in issues, notebooks, logs, evidence, screenshots, or commits.

Before attaching logs, remove environment variables and authorization headers. Runtime evidence should contain only the minimum information required for reproducibility.

## Reporting a vulnerability

For security-sensitive reports, contact the project maintainer privately at `i.am@dangkhoa.dev` rather than opening a public issue. Include a concise description, affected files/versions, reproduction steps, impact, and any suggested mitigation.

Please do not publish exploit details before the maintainer has had a reasonable opportunity to investigate and prepare a fix.

## Scope

This policy covers repository-authored code and project configuration. Security issues in Z-Image itself, PyTorch/XLA, Kaggle, or other upstream dependencies should also be reported to the appropriate upstream project when applicable.
