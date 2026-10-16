# Production Creative Showcase

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CREATIVE_SHOWCASE.vi.md)

The production notebook runs eight distinct 1024x1024 creative prompts alongside six technical verification images, giving 14 original PNGs by default. Eight creative images are produced using the original full Z-Image model on Kaggle TPU v5e-8 in two real batches of four independent prompts each. Prompt texts, categories and explicit seeds are preserved in scripts/showcase/creative-prompts.json (501–504, 615, 506–508).

Every run must supply its own inference timing, CPU VAE decoding timing, PNG checksum, and current-run evidence. The bilingual HTML gallery checks exact SHA256 values; a previous run is never presented as fresh inference. Open notebooks/kaggle-tpu-v5e8-production.ipynb, attach the Kaggle model mirror, select TPU v5e-8, and execute all code cells.

The repository notebook is an unexecuted template: execution output is not included in the Git source notebook. Publication acceptance requires checking the actual Kaggle-executed notebook, including genuine cell output. Kaggle CLI submissions may remove embedded outputs and schedule a fresh run; do not claim publication success without verifying the resulting Kaggle version. The Cloudflare gallery URL is temporary and is available only while its host is online.

Default Run All generates two independent 512×512 images, four technical 1024×1024 images and eight creative 1024×1024 images (14 total). Set ZIMAGE_RUN_2000PX=True for an additional 2048×1904 render (15 total). Encoding uses one text encoder load, TPU inference runs 4+4 on mesh 4×2 and one CPU VAE load decodes eight creative latents.
