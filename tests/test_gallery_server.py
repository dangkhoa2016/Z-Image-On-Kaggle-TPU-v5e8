import json
import socket
from pathlib import Path

import pytest

from scripts.showcase.serve_gallery import select_gallery_port, validate_manifest


def test_select_gallery_port_preserves_existing_service():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy = occupied.getsockname()[1]
        selected = select_gallery_port(busy)
        assert selected != busy
        assert selected > 0


def test_gallery_manifest_requires_distinct_images(tmp_path: Path):
    manifest = {"status": "PASS", "count": 14, "run_id": "current-run",
                "items": [{"sha256": str(i)} for i in range(14)]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert validate_manifest(tmp_path)["run_id"] == "current-run"
    manifest["items"][13]["sha256"] = "0"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        validate_manifest(tmp_path)
