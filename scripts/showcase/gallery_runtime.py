#!/usr/bin/env python3
"""Bilingual static HTML gallery from the current machine's actual TPU logs, JSON evidence and PNGs.

Strict mode requires notebook run-context.json and refuses older files.
No latent generation, model loading, TPU activation, or image re-generation occurs here.
"""
from pathlib import Path
from PIL import Image,ImageOps
from datetime import datetime,timezone
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import argparse,base64,hashlib,html,json,os,re,shutil,socket,subprocess,sys,time,urllib.request

PROJECT=Path(__file__).resolve().parents[2]
DEFAULT_DEST=PROJECT/'evidence/showcase/gallery'
TEMPLATE=Path(__file__).with_name('gallery_template.html')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def load_json(path):
    p=Path(path)
    assert p.is_file(),f'NOT_FOUND {p}'
    d=json.loads(p.read_text())
    assert d.get('status')=='PASS',f'EVIDENCE_NOT_PASS {p}'
    return d

def build(repo,output,run_start_epoch=None,run_id=None,allow_existing=False):
    repo=Path(repo).resolve()
    output=Path(output).resolve()
    if not allow_existing:
        context=repo/'evidence/showcase/run-context.json'
        c=json.loads(context.read_text())
        assert c['state']=='STARTED' and c.get('run_id'),'MISSING_VALID_RUN_CONTEXT'
        assert c['started_epoch']>0
        run_start_epoch=float(c['started_epoch'])
        run_id=c['run_id']
    else:
        assert run_start_epoch is not None and run_id, 'Historical inspection requires explicit run start and run ID'
    assert run_start_epoch is not None and float(run_start_epoch)>0
    def fresh(path):
        p=Path(path)
        assert p.is_file(),f'MISSING_CURRENT_RUN_FILE {p}'
        assert p.stat().st_mtime>=float(run_start_epoch)-2.0, f'STALE_EVIDENCE {p}'
        return p
    def ev(rel):
        p=fresh(repo/rel)
        return load_json(p)
    single=ev('evidence/runtime/publication-512.json')
    context=json.loads((repo/'evidence/showcase/run-context.json').read_text())
    high_enabled=context.get('run_highres') is True
    high=ev('evidence/runtime/publication-2048x1904.json') if high_enabled else None
    second=ev('evidence/runtime/publication-512-alternate.json')
    extra=ev('evidence/showcase/creative-extra-acceptance.json')
    assert extra['count']==2 and extra['all_rendered_fresh']
    batch=ev('evidence/dual/publication-tpu-b4-1024.json')
    c4=ev('evidence/dual/showcase-creative-first.json')
    c2=ev('evidence/dual/showcase-creative-second.json')
    cm=ev('evidence/showcase/fresh-six-manifest.json')
    assert cm['count']==6 and cm['all_rendered_fresh']
    assert len({x['prompt'] for x in cm['items']})==6
    assert c4['prompt_mode']=='independent_per_replica' and c2['prompt_mode']=='independent_per_replica'
    assert c4['mesh_shape']==[4,2] and c2['mesh_shape']==[4,2]
    assert len(batch['seeds'])==4 and batch['mesh_shape']==[4,2]
    assert single['resolution']==[512,512] and second['resolution']==[512,512]
    if high_enabled: assert high['resolution']==[2048,1904]
    assert all(x['status']=='PASS' for x in cm['items'])
    assert len({x['seed'] for x in cm['items']})==6
    rows=[]
    creatives=cm['items']+extra['items']
    assert len(creatives)==8
    assert [x['seed'] for x in creatives]==[501,502,503,504,615,506,507,508]
    assert len({x['prompt'] for x in creatives})==8
    originals=[
      repo/'evidence/runs/run-publication-512-seed42/output.png',
      repo/'evidence/runs/run-publication-512-alternate-seed43/output.png',
      *(repo/f'evidence/runs/run-publication-tpu-b4-1024-{i}/output.png' for i in range(4)),
      *(Path(x['png']) for x in creatives)
    ]
    if high_enabled: originals.append(repo/'evidence/runs/run-publication-2048x1904-seed42/output.png')
    expected_count=15 if high_enabled else 14
    assert len(originals)==expected_count
    out_images=output/'images';out_thumbs=output/'thumbs'
    out_images.mkdir(parents=True,exist_ok=True)
    out_thumbs.mkdir(parents=True,exist_ok=True)
    for idx,path in enumerate(originals):
        # The only images served are explicit output paths that map back into this repository.
        p=fresh(path).resolve()
        assert p.is_relative_to((repo/'evidence/runs').resolve()),'OUTPUT_OUTSIDE_REPOSITORY'
        assert p.name=='output.png'
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:
            dims=list(im.size)
            small=ImageOps.exif_transpose(im).convert('RGB')
            small.thumbnail((720,720),Image.Resampling.LANCZOS)
            preview=out_thumbs/f'{idx+1:02}.jpg'
            small.save(preview,'JPEG',quality=80,optimize=True)
        digest=sha(p)
        if idx in (0,1) or (high_enabled and idx==14):
            is_high=idx==14
            b=high if is_high else (single if idx==0 else second)
            suffix='publication-2048x1904' if is_high else ('publication-512' if idx==0 else 'publication-512-alternate')
            vae=ev(f'evidence/runtime/vae-{suffix}.json')
            assert vae['sha256']==digest
            assert dims==([2048,1904] if is_high else [512,512])
            slot={'group':'technical','style':'cinematic',
                  'title':('High resolution · 2048×1904' if is_high else f'TPU smoke · 512×512 · seed {b["seed"]}'),
                  'seed':b['seed'],'mesh':'1×8','batch':b['test'],'batch_size':1,
                  'denoise_s':b['total_denoise_seconds'],'vae_s':vae['vae_decode_seconds'],
                  'prompt':b['prompt'],'negative':b.get('negative_prompt',''),
                  'cfg':b.get('guidance_scale',3.0),'steps':b['steps'],
                  'started_at':b.get('started_at'),'run_type':'single'}
        elif 2<=idx<=5:
            vae=ev(f'evidence/runtime/vae-publication-tpu-b4-1024-{idx-2}.json')
            assert vae['sha256']==digest and dims==[1024,1024]
            slot={'group':'technical','style':'cinematic','title':f'Parallel · replica {idx-2}',
                  'seed':batch['seeds'][idx-2],'mesh':'4×2','batch':batch['slug'],'batch_size':4,
                  'denoise_s':batch['total_denoise_seconds'],'vae_s':vae['vae_decode_seconds'],
                  'prompt':batch['prompt'],'negative':batch.get('negative_prompt',''),
                  'cfg':batch['guidance'],'steps':batch['steps'],
                  'started_at':batch.get('started_at'),'run_type':'concurrent_batch'}
        else:
            x=creatives[idx-6]
            b=c4 if idx-6<4 else c2
            assert x['prompt'] in b['prompts'] and x['seed'] in b['seeds']
            assert digest==x['sha256'] and dims==[1024,1024]
            vae=ev(f"evidence/runtime/vae-{x['batch']}-{x['slot']}.json")
            assert vae['sha256']==digest
            slot={'group':'creative','style':x['category'],'title':x['id'].replace('-',' ').title(),
                  'seed':x['seed'],'mesh':'×'.join(map(str,b['mesh_shape'])),'batch':b['slug'],
                  'batch_size':b['batch_size'],'denoise_s':b['total_denoise_seconds'],
                  'vae_s':vae['vae_decode_seconds'],'prompt':x['prompt'],
                  'negative':x.get('negative_prompt',''),'cfg':b['guidance'],
                  'steps':b['steps'],'started_at':b.get('started_at'),'run_type':'concurrent_batch'}
        assert slot['denoise_s']>0 and slot['vae_s']>0 and slot['prompt']
        filename=f'{idx+1:02}-'+re.sub(r'[^a-z0-9-]+','-',p.parent.name.lower())+'.png'
        target=out_images/filename
        if not target.is_file() or sha(target)!=digest:shutil.copy2(p,target)
        assert sha(target)==digest
        rows.append(dict(id=idx,provenance='FRESH_INFERENCE',name=p.parent.name,image='images/'+filename,thumb=f'thumbs/{idx+1:02}.jpg',
                         width=dims[0],height=dims[1],sha256=digest,bytes=p.stat().st_size,
                         denoise_s=round(slot.pop('denoise_s'),3),vae_s=round(slot.pop('vae_s'),3),**slot))
    assert len({x['sha256'] for x in rows})==expected_count,'DUPLICATE_OUTPUT'
    manifest=dict(status='PASS',count=expected_count,run_id=run_id,run_started_epoch=run_start_epoch,
                  generated_at=datetime.now(timezone.utc).isoformat(),source='local_notebook_evidence',
                  timing_semantics='TPU denoise shared by the whole batch; CPU VAE per image; does not include setup/load',
                  items=rows)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    def esc(s):return html.escape(str(s),quote=True)
    cards={}
    for x in rows:
        chips=''.join(f'<span class="chip">{esc(i)}</span>' for i in
                      [x['style'],f"Seed {x['seed']}",f"Mesh {x['mesh']}",f"{x['width']}×{x['height']}"])
        # User-provided prompts are escaped; never interpreted as HTML.
        cards[x['id']]=(
          f'<article class="card" data-group="{x["group"]}" data-size="{x["width"]}">'
          f'<button class="open" data-id="{x["id"]}" aria-label="{esc(x["title"])}"><img loading="lazy" src="{esc(x["thumb"])}" alt="{esc(x["title"])}"></button>'
          f'<div class="body"><h3>{esc(x["title"])}</h3><div class="chips">{chips}</div>'
          f'<div class="metrics"><div><small data-i18n="denoise">TPU denoise · batch</small><b>{x["denoise_s"]:.2f} s</b></div>'
          f'<div><small data-i18n="vae">CPU VAE · this image</small><b>{x["vae_s"]:.2f} s</b></div></div>'
          f'<p class="meta"><span data-i18n="batch">Batch</span>: {esc(x["batch"])} · {x["batch_size"]} <span data-i18n="images">images</span> · {x["steps"]} steps · CFG {x["cfg"]}</p>'
          f'<details><summary data-i18n="promptDetails">Prompt, seed & evidence</summary><h4>Prompt</h4><pre>{esc(x["prompt"])}</pre>'
          f'<h4 data-i18n="negative">Negative prompt</h4><pre>{esc(x["negative"])}</pre>'
          f'<h4>SHA256</h4><pre>{esc(x["sha256"])}</pre>'
          f'<p class="meta">{esc(x.get("started_at") or "")} · {x["bytes"]} bytes</p>'
          f'<button class="copy" data-copy="{x["id"]}" data-i18n="copy">Copy prompt</button></details>'
          f'<p class="actions"><a href="{esc(x["image"])}" target="_blank" rel="noopener" data-i18n="openOriginal">Open original PNG ↗</a>'
          f'<a href="{esc(x["image"])}" download data-i18n="download">Download PNG ↓</a></p></div></article>'
        )
    sections=''
    for w,language_key in [(512,'group512'),(1024,'group1024'),(2048,'group2048')]:
        vals=[x for x in rows if x['width']==w]
        if not vals:continue
        sections+=f'<section class="section" data-section="{w}"><h2><span data-i18n="{language_key}">{w} px</span> <span class="count">{len(vals)} <span data-i18n="images">images</span></span></h2><div class="grid">'
        sections+=''.join(cards[x['id']] for x in vals)+'</div></section>'
    template=TEMPLATE.read_text()
    assert '__DATA__' in template and '__SECTIONS__' in template,'GALLERY_TEMPLATE_INVALID'
    safe=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    substitutions=dict(__SECTIONS__=sections,__DATA__=safe,__COUNT__=str(len(rows)),
        __TECH_COUNT__=str(sum(x['group']=='technical' for x in rows)),
        __CREATIVE_COUNT__=str(sum(x['group']=='creative' for x in rows)),
        __DEVICES__=str(single['global_devices']),__MESH__='×'.join(map(str,batch['mesh_shape'])),
        __MAX_RES__=f"{high['resolution'][0]}×{high['resolution'][1]}" if high_enabled else '1024×1024')
    for key,value in substitutions.items(): template=template.replace(key,value)
    assert '__DATA__' not in template and '__SECTIONS__' not in template
    (output/'index.html').write_text(template)
    print('GALLERY_DYNAMIC_BUILD_PASS',len(rows),'RUN_ID',run_id,'SOURCE',repo,'OUT',output,flush=True)
    return manifest

def start_server(repo,output,port,args):
    # Refuse a duplicate active server; this cannot accidentally serve old users' data.
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',port))!=0,f'PORT_IN_USE_{port}'
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**k):super().__init__(*a,directory=str(output),**k)
        def do_GET(self):
            if self.path.split('?')[0] in ('/','/index.html','/manifest.json'):
                try:build(repo,output,args.run_start_epoch,args.run_id,args.allow_existing)
                except Exception as ex:
                    body=('Gallery not ready: '+str(ex)).encode()
                    self.send_response(503);self.send_header('Content-Type','text/plain; charset=utf-8')
                    self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.serve_forever()

def serve(args):
    repo=Path(args.repo).resolve()
    out=Path(args.output).resolve()
    build(repo,out,args.run_start_epoch,args.run_id,args.allow_existing)
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',args.port))!=0,f'PORT_IN_USE_{args.port}'
    with (out/'http.log').open('a') as log:
        srv=subprocess.Popen([sys.executable,__file__,'--run-server','--repo',str(repo),'--output',str(out),
                              '--port',str(args.port)]+(['--allow-existing','--run-start-epoch',str(args.run_start_epoch),'--run-id',args.run_id] if args.allow_existing else []),
                              stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (out/'server.pid').write_text(str(srv.pid))
    for _ in range(15):
        time.sleep(.2)
        if srv.poll() is not None:raise RuntimeError('Gallery HTTP server died')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/manifest.json',timeout=3) as r:
                assert json.load(r)['count']==(15 if json.loads((repo/'evidence/showcase/run-context.json').read_text()).get('run_highres') else 14)
            break
        except Exception:pass
    else:raise RuntimeError('Dynamic gallery HTTP endpoint failed')
    with (out/'tunnel.log').open('a') as log:
        p=subprocess.Popen([args.cloudflared,'tunnel','--no-autoupdate','--url',f'http://127.0.0.1:{args.port}'],
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (out/'tunnel.pid').write_text(str(p.pid))
    for _ in range(60):
        time.sleep(.5)
        match=re.search(r'https://[a-z0-9-]+\.trycloudflare\.com',(out/'tunnel.log').read_text(errors='replace'))
        if match:
            print('FULL_RES_GALLERY_URL',match.group(0),flush=True)
            print('FULL_RES_IMAGE_COUNT',len(json.loads((out/'manifest.json').read_text())['items']),flush=True)
            print('EVIDENCE_RUN_ID',json.loads((out/'manifest.json').read_text())['run_id'],flush=True)
            return
    raise RuntimeError('Cloudflare URL unavailable')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default=str(PROJECT))
    p.add_argument('--output',default=str(DEFAULT_DEST))
    p.add_argument('--serve',action='store_true')
    p.add_argument('--run-server',action='store_true')
    p.add_argument('--port',type=int,default=8768)
    p.add_argument('--cloudflared',default='/kaggle/working/cloudflared')
    p.add_argument('--allow-existing',action='store_true',help='Manual historic preview only; NOT for production notebook')
    p.add_argument('--run-start-epoch',type=float)
    p.add_argument('--run-id')
    args=p.parse_args()
    assert not args.allow_existing or (args.run_start_epoch and args.run_id)
    if args.run_server:start_server(Path(args.repo),Path(args.output),args.port,args)
    elif args.serve:serve(args)
    else:build(Path(args.repo),Path(args.output),args.run_start_epoch,args.run_id,args.allow_existing)
if __name__=='__main__':main()
