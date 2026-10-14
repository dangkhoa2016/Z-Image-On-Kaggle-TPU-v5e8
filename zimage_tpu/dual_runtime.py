from __future__ import annotations

from pathlib import Path


def parse_chip_group(text: str) -> tuple[int, ...]:
    try:
        chips = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("chip group must contain integer chip ids") from exc
    if not chips:
        raise ValueError("chip group must not be empty")
    if len(chips) != len(set(chips)):
        raise ValueError("chip ids must be unique")
    return chips


def validate_disjoint_groups(
    a: tuple[int, ...], b: tuple[int, ...], total_chips: int = 8
) -> None:
    combined = a + b
    if any(chip < 0 or chip >= total_chips for chip in combined):
        raise ValueError(f"chip id outside valid range 0..{total_chips - 1}")
    overlap = set(a).intersection(b)
    if overlap:
        raise ValueError(f"chip groups overlap: {sorted(overlap)}")


def instance_paths(
    base: Path, round_id: str, instance_id: str
) -> dict[str, Path]:
    root = Path(base) / "evidence" / "dual" / round_id
    return {
        "cache": Path(base) / f"xla-cache-dual-{instance_id}",
        "evidence": root / instance_id,
        "log": root / f"{instance_id}.log",
    }


def throughput_speedup(sequential_pair_seconds: float, dual_pair_seconds: float) -> float:
    if sequential_pair_seconds <= 0 or dual_pair_seconds <= 0:
        raise ValueError("timings must be positive")
    return sequential_pair_seconds / dual_pair_seconds


def classify_failure(stderr: str, returncode: int) -> str:
    text = stderr.lower()
    if "vfio" in text or "/dev/vfio" in text:
        return "vfio"
    if "pjrt" in text:
        return "pjrt"
    if returncode in {-9, 137} or "killed" in text or "cannot allocate memory" in text:
        return "host_oom"
    if "hbm" in text and ("out of memory" in text or "resource_exhausted" in text or "oom" in text):
        return "tpu_hbm_oom"
    if "xla" in text and ("compile" in text or "compilation" in text):
        return "xla_compile"
    return "unknown_process_failure"
