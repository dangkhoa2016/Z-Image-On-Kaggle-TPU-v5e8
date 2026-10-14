#!/usr/bin/env python3
import argparse,time,datetime,torch,torch_xla
ap=argparse.ArgumentParser(); ap.add_argument('--interval',type=int,default=120); a=ap.parse_args()
d=torch_xla.device(); print('HEARTBEAT_START',datetime.datetime.now(datetime.timezone.utc).isoformat(),d,flush=True)
i=0
while True:
    x=torch.arange(4096,device=d,dtype=torch.bfloat16).reshape(64,64); y=(x@x.T).sum(); torch_xla.sync()
    print('HEARTBEAT',i,datetime.datetime.now(datetime.timezone.utc).isoformat(),float(y.cpu()),flush=True)
    i+=1; time.sleep(a.interval)
