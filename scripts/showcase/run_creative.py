#!/usr/bin/env python3
"""Generate eight independent 1024px creative images in two four-replica TPU batches."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from PIL import Image

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/"evidence/showcase"
items=json.loads((BASE/"scripts/showcase/creative-prompts.json").read_text())
assert len(items)==8 and [x["seed"] for x in items]==[501,502,503,504,615,506,507,508]
for i,row in enumerate(items):
    blob=ROOT/f"prompt-{i}.pt"
    assert blob.is_file(),blob
    import torch
    record=torch.load(blob,map_location="cpu",weights_only=True)
    assert record["prompt"]==row["prompt"] and record["negative_prompt"]==row["negative_prompt"]
outputs=[]
for label,indices in (("creative-first",[0,1,2,3]),("creative-second",[4,5,6,7])):
    slug="showcase-"+label
    embeddings=[ROOT/f"prompt-{i}.pt" for i in indices]
    seeds=[items[i]["seed"] for i in indices]
    cmd=[sys.executable,str(BASE/"scripts/showcase/run_zimage_creative.py"),"--run","--width","1024","--height","1024","--steps","28","--guidance","3.0","--replicas","4","--batch-size","4","--seeds",",".join(map(str,seeds)),"--slug",slug,"--embeddings",str(embeddings[0]),"--embeddings-per-replica",*[str(b) for b in embeddings]]
    log=ROOT/(label+"-tpu.log")
    with log.open("w") as out:
        result=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,cwd=BASE,env=os.environ|{"PYTHONPATH":str(BASE),"PJRT_DEVICE":"TPU"})
    if result.returncode:raise RuntimeError(f"TPU batch failed: {log} exit={result.returncode}")
    evidence=json.loads((BASE/"evidence/dual"/(slug+".json")).read_text())
    assert evidence["status"]=="PASS" and evidence["prompt_mode"]=="independent_per_replica"
    assert evidence["replicas"]==4 and len(evidence["latent_paths"])==4
    assert evidence["prompts"]==[items[i]["prompt"] for i in indices]
    for slot,i in enumerate(indices):
        outputs.append({**items[i],"index":i,"slot":slot,"batch":slug,"latent":evidence["latent_paths"][slot],"run_dir":str(BASE/f"evidence/runs/run-{slug}-{slot}"),"vae_slug":f"vae-{slug}-{slot}","denoise_batch_seconds":evidence["total_denoise_seconds"],"replicas":4,"model_shards":2})
    print("TPU_BATCH_PASS",label,"images",len(indices),"denoise",round(evidence["total_denoise_seconds"],3),flush=True)
assert len(outputs)==8
latent_manifest=ROOT/"creative-latents.json"
latent_manifest.write_text(json.dumps(outputs,indent=2)+"\n")
cmd=[sys.executable,str(BASE/"scripts/showcase/decode_creative_batch.py"),"--manifest",str(latent_manifest)]
log=ROOT/"creative-vae.log"
with log.open("w") as f:
    done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=BASE)
if done.returncode:raise RuntimeError(f"VAE decoding failed: {log} exit={done.returncode}")
rows=[]
for row in outputs:
    png=Path(row["run_dir"])/"output.png"
    with Image.open(png) as img:
        img.verify()
    with Image.open(png) as img:
        assert img.size==(1024,1024)
    digest=hashlib.sha256(png.read_bytes()).hexdigest()
    vae=json.loads((BASE/"evidence/runtime"/(row["vae_slug"]+".json")).read_text())
    assert vae["status"]=="PASS" and vae["sha256"]==digest
    rows.append({key:value for key,value in row.items() if key not in ("latent","run_dir","vae_slug")}|{"png":str(png),"sha256":digest,"bytes":png.stat().st_size,"vae_decode_seconds":vae["vae_decode_seconds"],"mesh":[4,2],"status":"PASS"})
    print("FRESH_PNG_PASS",row["id"],digest,flush=True)
assert len(set(x["sha256"] for x in rows))==8
(ROOT/"fresh-six-manifest.json").write_text(json.dumps({"status":"PASS","count":6,"all_rendered_fresh":True,"items":rows[:6]},indent=2)+"\n")
(ROOT/"creative-extra-acceptance.json").write_text(json.dumps({"status":"PASS","count":2,"all_rendered_fresh":True,"items":rows[6:]},indent=2)+"\n")
print("EIGHT_CREATIVE_IMAGES_VERIFIED",len(rows),flush=True)
