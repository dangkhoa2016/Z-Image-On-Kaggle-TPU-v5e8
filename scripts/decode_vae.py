#!/usr/bin/env python3
import argparse,hashlib,json,os,time,torch
from pathlib import Path
from diffusers import AutoencoderKL
from PIL import Image
ap=argparse.ArgumentParser(); ap.add_argument('--latent',required=True); ap.add_argument('--run-dir',required=True); ap.add_argument('--slug',required=True); ap.add_argument('--width',type=int,required=True); ap.add_argument('--height',type=int,required=True); a=ap.parse_args()
MODEL='/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1'; os.makedirs(a.run_dir,exist_ok=True)
vae=AutoencoderKL.from_pretrained(MODEL,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).eval(); lat=torch.load(a.latent,map_location='cpu').to(vae.dtype); lat=(lat/vae.config.scaling_factor)+vae.config.shift_factor
t0=time.perf_counter()
with torch.no_grad(): img=vae.decode(lat,return_dict=False)[0]
dt=time.perf_counter()-t0; arr=((img.clamp(-1,1)+1)/2*255).round().to(torch.uint8).permute(0,2,3,1)[0].cpu().numpy(); out=os.path.join(a.run_dir,'output.png'); Image.fromarray(arr).save(out); sha=hashlib.sha256(open(out,'rb').read()).hexdigest(); rec={'test':a.slug,'status':'PASS','width':a.width,'height':a.height,'vae_decode_seconds':dt,'output':out,'sha256':sha,'shape':list(img.shape),'pixel_mean':float(arr.mean())}; json.dump(rec,open(str(Path(__file__).resolve().parents[1] / 'evidence/runtime' / (a.slug + '.json')),'w'),indent=2); print('VAE_PASS',json.dumps(rec),flush=True)
