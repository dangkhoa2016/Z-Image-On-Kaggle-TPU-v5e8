#!/usr/bin/env python3
import os,sys

def kernel_env(pid):
    try:
        raw=open(f'/proc/{pid}/environ','rb').read().split(b'\0')
        env={}
        for item in raw:
            if b'=' in item:
                k,v=item.split(b'=',1); env[k.decode(errors='ignore')]=v.decode(errors='ignore')
        return env
    except Exception:
        return {}

candidates=[]
for p in os.listdir('/proc'):
    if not p.isdigit():
        continue
    try:
        cmd=open(f'/proc/{p}/cmdline','rb').read()
    except Exception:
        continue
    if b'ipykernel_launcher' not in cmd:
        continue
    env=kernel_env(p)
    score=0
    if env.get('PJRT_DEVICE')=='TPU': score+=10
    if env.get('TPU_SKIP_MDS_QUERY')=='1': score+=3
    if env.get('TPU_ACCELERATOR_TYPE'): score+=2
    if env.get('XRT_TPU_CONFIG'): score+=1
    candidates.append((score,int(p),env))
if not candidates:
    raise SystemExit('No Jupyter kernel PID found')
candidates.sort(reverse=True)
score,pid,env=candidates[0]
if score <= 0:
    raise SystemExit('No TPU-enabled Jupyter kernel environment found')
for k,v in os.environ.items():
    env.setdefault(k,v)
os.execvpe(sys.argv[1],sys.argv[1:],env)
