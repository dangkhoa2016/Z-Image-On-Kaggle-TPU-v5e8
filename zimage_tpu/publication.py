from __future__ import annotations

import hashlib
import json
from pathlib import Path


_ASSET_BY_SLOT = {
    "3d_a": "assets/showcase-3d-mountain-village.png",
    "3d_b": "assets/showcase-3d-eco-city.png",
    "illustration_a": "assets/showcase-illustration-canal-town.png",
    "illustration_b": "assets/showcase-illustration-cliffside-castle.png",
    "photoreal_a": "assets/showcase-photoreal-watch.png",
    "photoreal_b": "assets/showcase-photoreal-alpine-lake.png",
}


def _load_json(path: Path):
    return json.loads(path.read_text())


def load_showcase_rows(root: Path) -> list[dict[str, object]]:
    return list(_load_json(Path(root) / "evidence/reference/showcase.json")["images"])


def verify_showcase_assets(root: Path) -> list[dict[str, object]]:
    rows = load_showcase_rows(root)
    for row in rows:
        actual = hashlib.sha256((Path(root) / row["asset"]).read_bytes()).hexdigest()
        if actual != row["sha256"]:
            raise ValueError(f"SHA256 mismatch for {row['asset']}")
    return rows


def load_pair_throughput_rows(root: Path) -> list[dict[str, float | int]]:
    payload = _load_json(Path(root) / "evidence/reference/throughput.json")
    rows: list[dict[str, float | int]] = []
    for resolution in ("512", "1024"):
        item = payload["comparisons"][resolution]
        rows.append(
            {
                "resolution": int(resolution),
                "control_seconds": item["single_mesh_batch2_pair_seconds"],
                "replica_model_seconds": item["replica_model_batch2_pair_seconds"],
                "speedup": item["replica_model_speedup_vs_single_mesh"],
                "control_warm_median": item["single_mesh_warm_forward_median"],
                "replica_model_warm_median": item["replica_model_warm_forward_median"],
            }
        )
    return rows


def load_resolution_rows(root: Path) -> list[dict[str, object]]:
    payload = _load_json(Path(root) / "evidence/reference/resolution.json")
    return [dict(row) for row in payload]
