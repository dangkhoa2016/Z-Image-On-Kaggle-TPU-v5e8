from pathlib import Path

import pytest

from zimage_tpu.publication import (
    load_pair_throughput_rows,
    load_resolution_rows,
    load_showcase_rows,
    verify_showcase_assets,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_showcase_rows_preserves_reference_order():
    rows = load_showcase_rows(ROOT)
    assert len(rows) == 6
    assert [row["category"] for row in rows] == [
        "3d", "3d", "illustration", "illustration", "photoreal", "photoreal"
    ]
    assert [row["seed"] for row in rows] == [101, 102, 211, 202, 615, 302]
    assert [row["asset"] for row in rows] == [
        "assets/showcase-3d-mountain-village.png",
        "assets/showcase-3d-eco-city.png",
        "assets/showcase-illustration-canal-town.png",
        "assets/showcase-illustration-cliffside-castle.png",
        "assets/showcase-photoreal-watch-final.png",
        "assets/showcase-photoreal-alpine-lake.png",
    ]


def test_verify_showcase_assets_matches_frozen_sha256():
    rows = verify_showcase_assets(ROOT)
    assert [row["sha256"] for row in rows] == [
        "b58acfdafbf7f4393f257e04a42d3c6e7c458f4eebd2acaed802063b31494da4",
        "853353399df9d76e3c9988fc684a3577b4b0f41f39e6dd61ebb83fa4c7661668",
        "f366922d77e1df7d286941971d8ff847a59e886dddbdd9ea7bab5894838e8879",
        "89570629f1530c08db63864102923708806a076dc19a637b7ecbea1a0f3c1e35",
        "89023bbb74bebd325afca7d28dbfe5a46afff052f4b7f4c8b1fbc14b629100d7",
        "b8705382c50fbd1fdd869b96bb06d5b3d42c9e2125f2934232b176d56bb622b8",
    ]


def test_verify_showcase_assets_fails_closed_on_mismatch(tmp_path: Path):
    import shutil

    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    (tmp_path / "assets" / "showcase-3d-mountain-village.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_showcase_assets(tmp_path)


def test_load_pair_throughput_rows_pins_matched_controls():
    rows = load_pair_throughput_rows(ROOT)
    assert rows == [
        {
            "resolution": 512,
            "control_seconds": 126.2637577190003,
            "replica_model_seconds": 95.00139161600009,
            "speedup": 1.3290727174751724,
            "control_warm_median": 0.849760574000129,
            "replica_model_warm_median": 0.28751509699986855,
        },
        {
            "resolution": 1024,
            "control_seconds": 288.6454137370001,
            "replica_model_seconds": 127.21799972000008,
            "speedup": 2.268903884452617,
            "control_warm_median": 8.266436781000266,
            "replica_model_warm_median": 1.1445404069997949,
        },
    ]


def test_load_resolution_rows_preserves_all_accepted_stretch_modes():
    rows = load_resolution_rows(ROOT)
    assert [row["resolution"] for row in rows] == [
        "512x512", "1024x1024", "1536x1024", "2048x1024",
        "2048x1536", "2048x1904", "2048x2048", "2560x2048",
    ]
    assert all(row["status"] == "PASS" for row in rows)
    assert rows[-1]["mode"] == "seq+chunk4096+sync"
    assert rows[-1]["denoise_s"] == pytest.approx(2463.987464998001)
