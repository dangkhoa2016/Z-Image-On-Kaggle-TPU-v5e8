#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zimage_tpu.replica_mesh import (
    choose_model_shard_axis,
    replica_batch_partition_spec,
    replica_model_mesh_shape,
)

BASE = REPO_ROOT
MODEL = Path('/kaggle/input/models/dangkhoa2016/tongyi-mai-z-image/pytorch/default/1')
DEFAULT_CACHE = BASE / 'xla-cache-tpu-batch4'


def describe_config() -> dict[str, object]:
    return {
        'mesh_shape': [4, 2],
        'replicas': 4,
        'model_shards': 2,
        'axis_names': ['replica', 'model'],
        'cfg_execution': 'sequential',
        'batch_size': 4,
        'seeds': [42, 43, 44, 45],
        'xla_cache_dir': str(DEFAULT_CACHE),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--describe-config', action='store_true')
    parser.add_argument('--describe-run-config', action='store_true')
    parser.add_argument('--mesh-probe-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--guidance', type=float, default=3.0)
    parser.add_argument('--replicas', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seeds', default='42,43,44,45')
    parser.add_argument('--slug')
    parser.add_argument('--embeddings')
    parser.add_argument('--embeddings-b')
    parser.add_argument('--xla-cache-dir', default=str(DEFAULT_CACHE))
    parser.add_argument('--output-json', default=str(BASE / 'evidence/dual/d10-mesh-probe.json'))
    return parser


def require_run_args(args: argparse.Namespace) -> None:
    missing = [name for name in ('width', 'height', 'slug', 'embeddings') if getattr(args, name) in (None, '')]
    if missing:
        raise SystemExit('missing run arguments: ' + ', '.join('--' + x.replace('_', '-') for x in missing))
    if args.width % 16 or args.height % 16:
        raise SystemExit('width/height must be divisible by 16')
    if args.steps < 1:
        raise SystemExit('steps must be >= 1')
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    batch_size = args.batch_size or len(seeds)
    if batch_size < 1:
        raise SystemExit('batch-size must be >= 1')
    if len(seeds) != batch_size:
        raise SystemExit(f'expected {batch_size} seeds, got {len(seeds)}')
    if len(set(seeds)) != len(seeds):
        raise SystemExit('seeds must be unique')
    if 8 % args.replicas:
        raise SystemExit('8 devices must be divisible by replicas')


def describe_run_config(args: argparse.Namespace) -> dict[str, object]:
    require_run_args(args)
    return {
        'width': args.width,
        'height': args.height,
        'steps': args.steps,
        'guidance': args.guidance,
        'seeds': [int(x) for x in args.seeds.split(',') if x.strip()],
        'slug': args.slug,
        'embeddings': args.embeddings,
        'embeddings_b': args.embeddings_b or args.embeddings,
        'cfg_execution': 'sequential',
        'replicas': args.replicas,
        'batch_size': args.batch_size or len([x for x in args.seeds.split(',') if x.strip()]),
        'model_shards': 8 // args.replicas,
        'mesh_shape': [args.replicas, 8 // args.replicas],
        'xla_cache_dir': args.xla_cache_dir,
    }


def run_mesh_probe(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    xr.initialize_cache(args.xla_cache_dir, readonly=False)
    xr.use_spmd()
    global_count = xr.global_runtime_device_count()
    addressable_count = xr.addressable_runtime_device_count()
    if global_count != 8 or addressable_count != 8:
        raise RuntimeError(f'TPU batch4 showcase requires 8 visible devices, got global={global_count} addressable={addressable_count}')

    mesh = Mesh(np.arange(8), (args.replicas, 8 // args.replicas), ('replica', 'model'))
    device = torch_xla.device()
    batch_size = args.batch_size or len([x for x in args.seeds.split(',') if x.strip()])
    batch = torch.arange(batch_size * 128 * 128, dtype=torch.float32).reshape(batch_size, 128, 128).to(torch.bfloat16).to(device)
    weight = torch.eye(128, dtype=torch.bfloat16).to(device)
    xs.mark_sharding(batch, mesh, replica_batch_partition_spec(batch.ndim))
    xs.mark_sharding(weight, mesh, ('model', None))
    out = batch @ weight
    torch_xla.sync()
    cpu = out.float().cpu()

    record = {
        'status': 'PASS',
        'mesh_shape': [args.replicas, 8 // args.replicas],
        'axis_names': ['replica', 'model'],
        'global_runtime_device_count': global_count,
        'addressable_runtime_device_count': addressable_count,
        'input_shape': list(batch.shape),
        'output_shape': list(cpu.shape),
        'sample_sums': [float(x.sum()) for x in cpu],
        'started_at': started,
        'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    target = Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True))
    return 0


def run_zimage(args: argparse.Namespace) -> int:
    require_run_args(args)
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh
    from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel
    import diffusers.models.transformers.transformer_z_image as zimage_impl

    evidence_path = BASE / 'evidence' / 'dual' / f'{args.slug}.json'
    progress_path = BASE / 'evidence' / 'dual' / f'{args.slug}-progress.json'
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    rec: dict[str, object] = {
        **describe_run_config(args),
        'status': 'STARTED',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }

    try:
        xr.initialize_cache(args.xla_cache_dir, readonly=False)
        xr.use_spmd()
        global_count = xr.global_runtime_device_count()
        addressable_count = xr.addressable_runtime_device_count()
        if global_count != 8 or addressable_count != 8:
            raise RuntimeError(f'TPU batch4 showcase requires 8 visible devices, got global={global_count} addressable={addressable_count}')
        mesh = Mesh(np.arange(8), (args.replicas, 8 // args.replicas), ('replica', 'model'))
        device = torch_xla.device()

        emb = torch.load(args.embeddings, map_location='cpu')
        pos_cpu = emb['positive'][0]
        neg_cpu = emb['negative'][0]
        rec.update(prompt=emb.get('prompt'), negative_prompt=emb.get('negative_prompt'))

        zimage_impl.SEQ_MULTI_OF = 32
        load_started = time.perf_counter()
        model = ZImageTransformer2DModel.from_pretrained(
            str(MODEL),
            subfolder='transformer',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval().to(device)
        model_load_seconds = time.perf_counter() - load_started

        model_shards = 8 // args.replicas
        sharded = 0
        for _, param in model.named_parameters():
            if param.ndim == 0:
                continue
            axis = choose_model_shard_axis(tuple(int(x) for x in param.shape), model_shards)
            if axis is None:
                continue
            spec: list[str | None] = [None] * param.ndim
            spec[axis] = 'model'
            xs.mark_sharding(param, mesh, tuple(spec))
            sharded += 1

        def mark_replica_tensor(value):
            if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == batch_size:
                xs.mark_sharding(value, mesh, replica_batch_partition_spec(value.ndim))
            return value

        original_prepare = model._prepare_sequence
        def sharded_prepare(*inner_args, **inner_kwargs):
            values = list(original_prepare(*inner_args, **inner_kwargs))
            for value in values:
                mark_replica_tensor(value)
            return tuple(values)
        model._prepare_sequence = sharded_prepare

        original_build = model._build_unified_sequence
        def sharded_build(*inner_args, **inner_kwargs):
            values = list(original_build(*inner_args, **inner_kwargs))
            for value in values:
                mark_replica_tensor(value)
            return tuple(values)
        model._build_unified_sequence = sharded_build

        seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
        batch_size=args.batch_size or len(seeds)
        pos = [pos_cpu.to(device, dtype=torch.bfloat16) for _ in seeds]
        neg = [neg_cpu.to(device, dtype=torch.bfloat16) for _ in seeds]
        lh, lw = args.height // 8, args.width // 8
        lat = torch.cat([torch.randn((1,16,lh,lw), generator=torch.Generator().manual_seed(seed), dtype=torch.float32) for seed in seeds], dim=0).to(device)
        xs.mark_sharding(lat, mesh, replica_batch_partition_spec(lat.ndim))

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(MODEL), subfolder='scheduler', local_files_only=True
        )
        scheduler.set_timesteps(args.steps, device=device)

        rec.update(
            global_runtime_device_count=global_count,
            addressable_runtime_device_count=addressable_count,
            params_sharded=sharded,
            model_load_seconds=model_load_seconds,
            latent_shape_initial=list(lat.shape),
        )
        times = []
        denoise_started = time.perf_counter()
        for index, timestep_raw in enumerate(scheduler.timesteps):
            timestep = (1000 - timestep_raw.expand(batch_size)) / 1000
            xs.mark_sharding(timestep, mesh, ('replica',))
            inp = lat.to(torch.bfloat16).unsqueeze(2)
            xs.mark_sharding(inp, mesh, replica_batch_partition_spec(inp.ndim))

            forward_started = time.perf_counter()
            with torch.no_grad():
                pos_list = model(list(inp.unbind(0)), timestep, pos, return_dict=False)[0]
                pos_out = torch.stack(pos_list, dim=0).float().squeeze(2)
                neg_list = model(list(inp.unbind(0)), timestep, neg, return_dict=False)[0]
                neg_out = torch.stack(neg_list, dim=0).float().squeeze(2)
            torch_xla.sync()
            forward_seconds = time.perf_counter() - forward_started

            prediction = pos_out + args.guidance * (pos_out - neg_out)
            noise = -prediction
            scheduler_started = time.perf_counter()
            lat = scheduler.step(noise, timestep_raw, lat, return_dict=False)[0]
            xs.mark_sharding(lat, mesh, replica_batch_partition_spec(lat.ndim))
            torch_xla.sync()
            scheduler_seconds = time.perf_counter() - scheduler_started
            row = {
                'step': index + 1,
                'forward_seconds': forward_seconds,
                'scheduler_seconds': scheduler_seconds,
                'timestep': float(timestep_raw.cpu()),
            }
            times.append(row)
            progress_path.write_text(json.dumps({**rec, 'status': 'RUNNING', 'completed_steps': index + 1, 'step_times': times}, indent=2) + '\n')
            print(f'STEP {index + 1:02d}/{args.steps} fwd={forward_seconds:.3f}s sched={scheduler_seconds:.3f}s', flush=True)

        denoise_seconds = time.perf_counter() - denoise_started
        cpu = lat.cpu()
        latent_paths=[]
        for i in range(cpu.shape[0]):
            p=BASE / 'cache' / f'latent_{args.slug}-{i}.pt'
            torch.save(cpu[i:i+1], p)
            latent_paths.append(str(p))
        rec.update(
            status='PASS', completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            total_denoise_seconds=denoise_seconds, step_times=times,
            latent_paths=latent_paths, latent_shape=list(cpu.shape),
            latent_means=[float(cpu[i].mean()) for i in range(cpu.shape[0])],
            latent_stds=[float(cpu[i].std()) for i in range(cpu.shape[0])],
        )
        evidence_path.write_text(json.dumps(rec, indent=2) + '\n')
        progress_path.write_text(json.dumps(rec, indent=2) + '\n')
        print('RUN_PASS', json.dumps({'total': denoise_seconds, 'latent_paths': latent_paths}), flush=True)
        return 0
    except Exception as exc:
        rec.update(
            status='FAIL',
            completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        evidence_path.write_text(json.dumps(rec, indent=2) + '\n')
        print('RUN_FAIL', traceback.format_exc(), flush=True)
        return 1


def main() -> int:
    args = build_parser().parse_args()
    if args.describe_config:
        print(json.dumps(describe_config(), sort_keys=True))
        return 0
    if args.describe_run_config:
        print(json.dumps(describe_run_config(args), sort_keys=True))
        return 0
    if args.mesh_probe_only:
        return run_mesh_probe(args)
    if args.run:
        return run_zimage(args)
    raise SystemExit('select --describe-config, --describe-run-config, --mesh-probe-only, or --run')


if __name__ == '__main__':
    raise SystemExit(main())
