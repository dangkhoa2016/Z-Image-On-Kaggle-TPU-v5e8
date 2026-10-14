import torch,json,time
from pathlib import Path
from diffusers import ZImagePipeline
root=Path(__file__).resolve().parents[2]/'evidence/showcase'
items=json.loads((Path(__file__).with_name('creative-prompts.json')).read_text())
assert len(items)==8 and [x['seed'] for x in items]==[501,502,503,504,615,506,507,508]
# Reuse only if all cached tensors match exact approved prompts.
if all((root/f'prompt-{i}.pt').exists() for i in range(len(items))):
    validated=True
    for i,row in enumerate(items):
        cached=torch.load(root/f'prompt-{i}.pt',map_location='cpu',weights_only=True)
        if cached.get('prompt')!=row['prompt'] or cached.get('negative_prompt')!=row['negative_prompt']:
            validated=False
            break
    if validated:
        print('APPROVED_EMBEDDINGS_REUSED_WITH_EXACT_PROMPT_VALIDATION',flush=True)
        raise SystemExit(0)
torch.set_num_threads(24)
print('ENCODE_THREADS',torch.get_num_threads(),'EXTRA_UNUSED_EMPTY_NEGATIVE_ENCODING_DISABLED',flush=True)
print('LOAD_TEXT_ENCODER',flush=True)
st=time.monotonic()
pipe=ZImagePipeline.from_pretrained('/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1', transformer=None,vae=None,torch_dtype=torch.bfloat16,local_files_only=True)
print('ENCODER_LOADED',round(time.monotonic()-st,3),flush=True)
for i,x in enumerate(items):
 out=root/f'prompt-{i}.pt'
 print('ENCODE_START',i,x['id'],flush=True)
 st=time.monotonic()
 pos,neg=pipe.encode_prompt(x['prompt'],device=torch.device('cpu'),do_classifier_free_guidance=True,negative_prompt=x['negative_prompt'])
 torch.save({'prompt':x['prompt'],'negative_prompt':x['negative_prompt'],'positive':pos,'negative':neg},out)
 print('ENCODE_PASS',i,'SECONDS',round(time.monotonic()-st,2),'BYTES',out.stat().st_size,flush=True)
print('ALL_EIGHT_PROMPT_EMBEDS_READY',flush=True)
