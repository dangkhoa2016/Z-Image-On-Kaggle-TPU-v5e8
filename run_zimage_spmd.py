#!/usr/bin/env python3
import argparse,json,os,time,traceback
BASE='/kaggle/working/zimage-tpu'; MODEL='/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1'
def build_arg_parser():
 ap=argparse.ArgumentParser(); ap.add_argument('--width',type=int,required=True); ap.add_argument('--height',type=int,required=True); ap.add_argument('--steps',type=int,default=28); ap.add_argument('--guidance',type=float,default=3.0); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--slug',required=True); ap.add_argument('--negative-mode',choices=['explicit','empty'],default='explicit'); ap.add_argument('--cfg-execution',choices=['batched','sequential'],default='batched'); ap.add_argument('--caption-pad-multiple',type=int,default=32); ap.add_argument('--attention-query-chunk-size',type=int,default=0); ap.add_argument('--attention-chunk-sync',action='store_true'); ap.add_argument('--embeddings',required=True); ap.add_argument('--xla-cache-dir',default=f'{BASE}/xla-cache'); ap.add_argument('--instance-id',default='single'); return ap
def main():
 a=build_arg_parser().parse_args()
 import numpy as np,torch,torch_xla
 import torch_xla.runtime as xr
 import torch_xla.distributed.spmd as xs
 from torch_xla.distributed.spmd import Mesh
 from diffusers import ZImageTransformer2DModel,FlowMatchEulerDiscreteScheduler
 import diffusers.models.transformers.transformer_z_image as zimage_impl
 if a.width%16 or a.height%16: raise SystemExit('width/height must be divisible by 16')
 if a.caption_pad_multiple < 1: raise SystemExit('caption-pad-multiple must be >= 1')
 zimage_impl.SEQ_MULTI_OF=a.caption_pad_multiple
 ev=f'{BASE}/evidence/runtime/{a.slug}.json'; progress=f'{BASE}/evidence/runtime/{a.slug}-progress.json'; rec={'test':a.slug,'status':'STARTED','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'xla_cache_dir':a.xla_cache_dir,'instance_id':a.instance_id}
 try:
  xr.initialize_cache(a.xla_cache_dir,readonly=False)
  emb=torch.load(a.embeddings,map_location='cpu'); pos_cpu=emb['positive'][0]; neg_cpu=(emb['empty_negative'][0] if a.negative_mode=='empty' else emb['negative'][0])
  rec.update(prompt=emb['prompt'],negative_prompt=('' if a.negative_mode=='empty' else emb['negative_prompt']),negative_mode=a.negative_mode)
  xr.use_spmd(); n=xr.global_runtime_device_count(); mesh=Mesh(np.arange(n),(n,),('model',)); dev=torch_xla.device()
  model=ZImageTransformer2DModel.from_pretrained(MODEL,subfolder='transformer',torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).eval().to(dev)
  if a.attention_query_chunk_size > 0:
   from diffusers.models.attention_processor import Attention
   from chunked_zimage_attention import ChunkedZSingleStreamAttnProcessor
   replaced=0
   for m in model.modules():
    if isinstance(m, Attention): m.set_processor(ChunkedZSingleStreamAttnProcessor(a.attention_query_chunk_size, sync_chunks=a.attention_chunk_sync)); replaced+=1
   print(f'CHUNKED_ATTENTION enabled chunk={a.attention_query_chunk_size} sync={a.attention_chunk_sync} processors={replaced}', flush=True)
  sharded=0
  for _,p in model.named_parameters():
   if p.ndim==0: continue
   cs=[(int(s),ax) for ax,s in enumerate(p.shape) if int(s)>=n and int(s)%n==0]
   if not cs: continue
   _,axis=max(cs); spec=[None]*p.ndim; spec[axis]='model'; xs.mark_sharding(p,mesh,tuple(spec)); sharded+=1
  pos=[pos_cpu.to(dev,dtype=torch.bfloat16)]; neg=[neg_cpu.to(dev,dtype=torch.bfloat16)]
  lh,lw=a.height//8,a.width//8
  lat=torch.randn((1,16,lh,lw),generator=torch.Generator().manual_seed(a.seed),dtype=torch.float32).to(dev)
  sched=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler',local_files_only=True); sched.set_timesteps(a.steps,device=dev)
  rec.update(global_devices=n,params_sharded=sharded,guidance_scale=a.guidance,cfg_execution=a.cfg_execution,caption_pad_multiple=a.caption_pad_multiple,attention_query_chunk_size=a.attention_query_chunk_size,attention_chunk_sync=a.attention_chunk_sync,positive_tokens=int(pos_cpu.shape[0]),negative_tokens=int(neg_cpu.shape[0]),steps=a.steps,seed=a.seed,resolution=[a.width,a.height],latent_shape_initial=[1,16,lh,lw])
  times=[]; t0=time.perf_counter()
  for i,t in enumerate(sched.timesteps):
   timestep=(1000-t.expand(1))/1000; st=time.perf_counter()
   with torch.no_grad():
    if a.cfg_execution=='batched':
     inp=lat.to(torch.bfloat16).repeat(2,1,1,1).unsqueeze(2); outs=model(list(inp.unbind(0)),timestep.repeat(2),pos+neg,return_dict=False)[0]; poso,nego=outs[0].float(),outs[1].float()
    else:
     inp=lat.to(torch.bfloat16).unsqueeze(2); poso=model(list(inp.unbind(0)),timestep,pos,return_dict=False)[0][0].float(); nego=model(list(inp.unbind(0)),timestep,neg,return_dict=False)[0][0].float()
   torch_xla.sync(); fwd=time.perf_counter()-st; pred=poso+a.guidance*(poso-nego); noise=-pred.unsqueeze(0).squeeze(2); ss=time.perf_counter(); lat=sched.step(noise,t,lat,return_dict=False)[0]; torch_xla.sync(); sch=time.perf_counter()-ss
   row={'step':i+1,'forward_seconds':fwd,'scheduler_seconds':sch,'timestep':float(t.cpu())}; times.append(row); json.dump(dict(rec,status='RUNNING',completed_steps=i+1,step_times=times,last_update=time.strftime('%Y-%m-%dT%H:%M:%S%z')),open(progress,'w'),indent=2); print(f'STEP {i+1:02d}/{a.steps} fwd={fwd:.3f}s sched={sch:.3f}s',flush=True)
  total=time.perf_counter()-t0; cpu=lat.cpu(); lp=f'{BASE}/cache/latent_{a.slug}.pt'; torch.save(cpu,lp); rec.update(status='PASS',completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),total_denoise_seconds=total,step_times=times,latent_path=lp,latent_shape=list(cpu.shape),latent_mean=float(cpu.mean()),latent_std=float(cpu.std())); json.dump(rec,open(ev,'w'),indent=2); json.dump(rec,open(progress,'w'),indent=2); print('RUN_PASS',json.dumps({'total':total,'latent_path':lp,'shape':list(cpu.shape)}),flush=True)
 except Exception as e:
  rec.update(status='FAIL',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()); json.dump(rec,open(ev,'w'),indent=2); print('RUN_FAIL',traceback.format_exc(),flush=True); raise
if __name__=='__main__': main()
