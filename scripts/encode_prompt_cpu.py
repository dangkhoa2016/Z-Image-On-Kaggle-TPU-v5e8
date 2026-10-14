#!/usr/bin/env python3
import argparse, torch
from diffusers import ZImagePipeline
ap=argparse.ArgumentParser(); ap.add_argument('--prompt',required=True); ap.add_argument('--negative-prompt',default=''); ap.add_argument('--output',required=True); a=ap.parse_args()
model='/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1'
pipe=ZImagePipeline.from_pretrained(model, transformer=None, vae=None, torch_dtype=torch.bfloat16, local_files_only=True)
pos,neg=pipe.encode_prompt(a.prompt,device=torch.device('cpu'),do_classifier_free_guidance=True,negative_prompt=a.negative_prompt)
_,empty=pipe.encode_prompt(a.prompt,device=torch.device('cpu'),do_classifier_free_guidance=True,negative_prompt='')
torch.save({'prompt':a.prompt,'negative_prompt':a.negative_prompt,'positive':pos,'negative':neg,'empty_negative':empty},a.output)
print('ENCODE_PASS', [tuple(x.shape) for x in pos], [tuple(x.shape) for x in neg], flush=True)
