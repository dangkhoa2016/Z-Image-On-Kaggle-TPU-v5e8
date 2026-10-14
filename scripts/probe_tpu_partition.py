#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

FIELDS = (
    "status",
    "instance_id",
    "visible_chips_env",
    "global_runtime_device_count",
    "addressable_runtime_device_count",
    "devices",
    "bf16_probe_sum",
    "started_at",
    "completed_at",
    "error",
)


def empty_record() -> dict[str, object]:
    return {key: None for key in FIELDS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe-schema", action="store_true")
    parser.add_argument("--instance-id")
    parser.add_argument("--expected-devices", type=int)
    parser.add_argument("--output-json")
    return parser


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_record(path: str | None, record: dict[str, object]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    if args.describe_schema:
        print(json.dumps(empty_record()))
        return 0
    if not args.instance_id or args.expected_devices is None or not args.output_json:
        raise SystemExit("--instance-id, --expected-devices, and --output-json are required")

    rec = empty_record()
    rec.update(
        status="STARTED",
        instance_id=args.instance_id,
        visible_chips_env=os.environ.get("TPU_VISIBLE_CHIPS"),
        started_at=stamp(),
    )
    try:
        import torch
        import torch_xla
        import torch_xla.runtime as xr

        global_count = xr.global_runtime_device_count()
        addressable_count = xr.addressable_runtime_device_count()
        attrs = xr.global_runtime_device_attributes()
        rec.update(
            global_runtime_device_count=global_count,
            addressable_runtime_device_count=addressable_count,
            devices=attrs,
        )
        if global_count != args.expected_devices or addressable_count != args.expected_devices:
            raise RuntimeError(
                f"expected {args.expected_devices} runtime devices, got "
                f"global={global_count} addressable={addressable_count}"
            )

        device = torch_xla.device()
        lhs = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128).to(torch.bfloat16).to(device)
        rhs = torch.eye(128, dtype=torch.bfloat16).to(device)
        out = lhs @ rhs
        torch_xla.sync()
        probe_sum = float(out.float().sum().cpu())
        rec.update(status="PASS", bf16_probe_sum=probe_sum, completed_at=stamp(), error=None)
        write_record(args.output_json, rec)
        print(json.dumps(rec, sort_keys=True))
        return 0
    except Exception as exc:
        rec.update(
            status="FAIL",
            completed_at=stamp(),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        write_record(args.output_json, rec)
        print(json.dumps(rec, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
