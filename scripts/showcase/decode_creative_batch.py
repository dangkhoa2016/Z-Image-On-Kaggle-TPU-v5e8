#!/usr/bin/env python3
"""Decode eight independently generated creative latents with one CPU VAE load."""
import argparse, hashlib, json, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    args=ap.parse_args()
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image
    base=Path(__file__).resolve().parents[2]
    entries=json.loads(Path(args.manifest).read_text())
    assert len(entries)==8 and all(Path(x["latent"]).is_file() for x in entries)
    model="/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1"
    torch.set_num_threads(24)
    vae=AutoencoderKL.from_pretrained(model,subfolder="vae",torch_dtype=torch.float32,local_files_only=True).eval()
    for row in entries:
        run_dir=Path(row["run_dir"])
        run_dir.mkdir(parents=True,exist_ok=True)
        latent=torch.load(row["latent"],map_location="cpu",weights_only=True).to(vae.dtype)
        latent=latent/vae.config.scaling_factor+vae.config.shift_factor
        start=time.perf_counter()
        with torch.no_grad(): decoded=vae.decode(latent,return_dict=False)[0]
        elapsed=time.perf_counter()-start
        arr=((decoded.clamp(-1,1)+1)/2*255).round().to(torch.uint8).permute(0,2,3,1)[0].cpu().numpy()
        png=run_dir/"output.png"
        Image.fromarray(arr).save(png)
        assert Image.open(png).size==(1024,1024)
        digest=hashlib.sha256(png.read_bytes()).hexdigest()
        evidence={"test":row["vae_slug"],"status":"PASS","width":1024,"height":1024,"vae_decode_seconds":elapsed,"output":str(png),"sha256":digest,"shape":list(decoded.shape),"pixel_mean":float(arr.mean())}
        (base/"evidence/runtime"/(row["vae_slug"]+".json")).write_text(json.dumps(evidence,indent=2)+"\n")
        print("VAE_PASS",row["id"],digest,round(elapsed,3),flush=True)
    print("ALL_EIGHT_CPU_VAE_PASS",flush=True)

if __name__=="__main__":
    main()
