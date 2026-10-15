#!/usr/bin/env python3
"""Serve the current verified gallery without disturbing existing local servers."""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GALLERY = REPO / "evidence/showcase/gallery-complete"
DEFAULT_CLOUDFLARED = Path("/kaggle/working/cloudflared")


def choose_available_port(preferred: int = 8788) -> int:
    """Choose the familiar port when free, otherwise an OS-allocated free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", preferred if preferred > 0 else 0))
        return sock.getsockname()[1]


def select_gallery_port(preferred: int = 8788) -> int:
    try:
        return choose_available_port(preferred)
    except OSError:
        return choose_available_port(0)


def validate_manifest(gallery: Path) -> dict:
    data = json.loads((gallery / "manifest.json").read_text())
    assert data["status"] == "PASS" and data["count"] in (14, 15)
    assert data.get("run_id"), "gallery run_id is missing"
    assert len(data["items"]) == data["count"]
    assert len({x["sha256"] for x in data["items"]}) == data["count"]
    return data


def ensure_cloudflared(path: Path) -> None:
    if not path.is_file():
        temporary = path.with_suffix(".download")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        try:
            urllib.request.urlretrieve(url, temporary)
            assert temporary.stat().st_size > 20_000_000, "CLOUDFLARED_DOWNLOAD_INCOMPLETE"
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    path.chmod(path.stat().st_mode | 0o111)
    version = subprocess.run([str(path), "--version"], capture_output=True, text=True)
    assert version.returncode == 0 and "cloudflared version" in version.stdout
    print("CLOUDFLARED_READY", version.stdout.strip(), flush=True)


def verify_local(port: int, run_id: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manifest.json", timeout=2) as response:
            data = json.load(response)
        return data.get("status") == "PASS" and data.get("run_id") == run_id
    except Exception:
        return False


def launch_http(gallery: Path, run_id: str, preferred: int, attempts: int = 8):
    log = gallery / "http.log"
    for _ in range(attempts):
        port = select_gallery_port(preferred)
        with log.open("a") as out:
            process = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                 "--directory", str(gallery)],
                stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        for _ in range(30):
            if verify_local(port, run_id):
                print("GALLERY_LOCAL_READY", port, "RUN_ID", run_id, "PID", process.pid, flush=True)
                return port, process
            if process.poll() is not None:
                break
            time.sleep(0.2)
        # Only terminate our own unsuccessful child, not any pre-existing server.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        preferred = 0
    raise RuntimeError("GALLERY_HTTP_START_FAILED")


def launch_tunnel(gallery: Path, port: int, cloudflared: Path, run_id: str):
    # Fresh log: an old trycloudflare.com address must never be reused.
    log_path = gallery / "tunnel.log"
    with log_path.open("w") as out:
        process = subprocess.Popen(
            [str(cloudflared), "tunnel", "--no-autoupdate",
             "--url", f"http://127.0.0.1:{port}"],
            stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    expression = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"GALLERY_TUNNEL_EXIT_{process.returncode}")
        match = expression.search(log_path.read_text(errors="replace"))
        if match:
            url = match.group(0)
            # Check the public URL resolves to THIS run's manifest.
            try:
                with urllib.request.urlopen(url + "/manifest.json", timeout=4) as response:
                    data = json.load(response)
                if data.get("run_id") == run_id and data.get("status") == "PASS":
                    print("VERIFIED_GALLERY_URL", url, "RUN_ID", run_id, flush=True)
                    return url, process
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError("GALLERY_TUNNEL_UNAVAILABLE_OR_WRONG_RUN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--preferred-port", type=int, default=8788)
    parser.add_argument("--cloudflared", type=Path, default=DEFAULT_CLOUDFLARED)
    parser.add_argument("--skip-tunnel", action="store_true", help="HTTP smoke test only")
    args = parser.parse_args()
    gallery = args.gallery.resolve()
    manifest = validate_manifest(gallery)
    if not args.skip_tunnel:
        ensure_cloudflared(args.cloudflared)
    port, server = launch_http(gallery, manifest["run_id"], args.preferred_port)
    if args.skip_tunnel:
        print("GALLERY_SMOKE_PASS", manifest["count"], "PORT", port, flush=True)
        return
    url, tunnel = launch_tunnel(gallery, port, args.cloudflared, manifest["run_id"])
    print("GALLERY_READY", url, "ORIGINAL_IMAGES", manifest["count"],
          "SERVER_PID", server.pid, "TUNNEL_PID", tunnel.pid, flush=True)


if __name__ == "__main__":
    main()
