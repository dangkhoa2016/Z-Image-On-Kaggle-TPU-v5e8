from pathlib import Path

import pytest

from zimage_tpu.dual_runtime import (
    classify_failure,
    instance_paths,
    parse_chip_group,
    throughput_speedup,
    validate_disjoint_groups,
)


def test_parse_chip_group_accepts_four_unique_chips():
    assert parse_chip_group("0,1,2,3") == (0, 1, 2, 3)


def test_parse_chip_group_rejects_duplicates():
    with pytest.raises(ValueError, match="unique"):
        parse_chip_group("0,1,1,3")


def test_validate_disjoint_groups_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_groups((0, 1, 2, 3), (3, 4, 5, 6))


def test_validate_disjoint_groups_rejects_out_of_range():
    with pytest.raises(ValueError, match="range"):
        validate_disjoint_groups((0, 1, 2, 8), (4, 5, 6, 7))


def test_instance_paths_are_unique_per_round_and_instance(tmp_path: Path):
    a = instance_paths(tmp_path, "round-01", "a")
    b = instance_paths(tmp_path, "round-01", "b")
    assert a["cache"] != b["cache"]
    assert a["evidence"] != b["evidence"]
    assert a["log"] != b["log"]


def test_throughput_speedup():
    assert throughput_speedup(327.52, 218.3466666667) == pytest.approx(1.5, rel=1e-6)


@pytest.mark.parametrize(
    ("stderr", "returncode", "expected"),
    [
        ("vfio: device or resource busy", 1, "vfio"),
        ("PJRT client already owns device", 1, "pjrt"),
        ("Killed\n", -9, "host_oom"),
        ("RESOURCE_EXHAUSTED: HBM out of memory", 1, "tpu_hbm_oom"),
        ("XLA compilation failed while lowering graph", 1, "xla_compile"),
        ("unexpected child failure", 3, "unknown_process_failure"),
    ],
)
def test_classify_failure(stderr, returncode, expected):
    assert classify_failure(stderr, returncode) == expected


def _required_inference_args():
    return [
        "--width", "512",
        "--height", "512",
        "--slug", "parser-test",
        "--embeddings", "/tmp/embeddings.pt",
    ]


def test_parser_preserves_single_instance_cache_defaults():
    from run_zimage_spmd import build_arg_parser

    args = build_arg_parser().parse_args(_required_inference_args())
    assert args.xla_cache_dir == str(Path(__file__).resolve().parents[1] / "xla-cache")
    assert args.instance_id == "single"


def test_parser_accepts_explicit_instance_cache_override():
    from run_zimage_spmd import build_arg_parser

    args = build_arg_parser().parse_args(
        _required_inference_args()
        + ["--xla-cache-dir", "/tmp/xla-cache-dual-a", "--instance-id", "a"]
    )
    assert args.xla_cache_dir == "/tmp/xla-cache-dual-a"
    assert args.instance_id == "a"


def test_partition_probe_schema_mode_is_cpu_only_and_complete():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/probe_tpu_partition.py", "--describe-schema"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {
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
    }


def test_replica_model_mesh_shape_for_v5e8():
    from zimage_tpu.replica_mesh import replica_model_mesh_shape

    assert replica_model_mesh_shape(total_devices=8, replicas=2) == (2, 4)


def test_replica_model_mesh_shape_rejects_non_divisible_device_count():
    from zimage_tpu.replica_mesh import replica_model_mesh_shape

    with pytest.raises(ValueError, match="divisible"):
        replica_model_mesh_shape(total_devices=8, replicas=3)


def test_choose_model_shard_axis_uses_model_axis_size_not_total_devices():
    from zimage_tpu.replica_mesh import choose_model_shard_axis

    assert choose_model_shard_axis((3840, 10240), model_shards=4) == 1
    assert choose_model_shard_axis((3, 7), model_shards=4) is None


def test_replica_batch_partition_spec_marks_only_batch_dimension():
    from zimage_tpu.replica_mesh import replica_batch_partition_spec

    assert replica_batch_partition_spec(4) == ("replica", None, None, None)


def test_replica_mesh_runner_describe_config_is_cpu_only():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/run_zimage_replica_model_mesh.py", "--describe-config"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mesh_shape"] == [2, 4]
    assert payload["axis_names"] == ["replica", "model"]
    assert payload["cfg_execution"] == "sequential"
    assert payload["batch_size"] == 2


def test_replica_mesh_runner_describe_run_config_preserves_safe_defaults():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_zimage_replica_model_mesh.py",
            "--describe-run-config",
            "--width", "512",
            "--height", "512",
            "--steps", "1",
            "--slug", "d10-smoke",
            "--embeddings", "/tmp/emb.pt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["steps"] == 1
    assert payload["cfg_execution"] == "sequential"
    assert payload["seed_a"] == 42
    assert payload["seed_b"] == 43
    assert payload["replicas"] == 2
    assert payload["model_shards"] == 4


def test_multi_prompt_control_describe_config_is_cpu_only():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/run_multi_prompt_control.py", "--describe-config"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mesh_shape"] == [8]
    assert payload["axis_names"] == ["model"]
    assert payload["cfg_execution"] == "sequential"
    assert payload["batch_size"] == 2
    assert payload["replicas"] == 1
    assert payload["model_shards"] == 8


def test_multi_prompt_control_describe_run_config_preserves_pair_defaults():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multi_prompt_control.py",
            "--describe-run-config",
            "--width", "512",
            "--height", "512",
            "--steps", "28",
            "--slug", "d9-control",
            "--embeddings", "/tmp/emb.pt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["steps"] == 28
    assert payload["seed_a"] == 42
    assert payload["seed_b"] == 43
    assert payload["replicas"] == 1
    assert payload["model_shards"] == 8


def test_replica_mesh_run_config_accepts_distinct_prompt_embeddings():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_zimage_replica_model_mesh.py",
            "--describe-run-config",
            "--width", "1024",
            "--height", "1024",
            "--steps", "28",
            "--slug", "showcase-round",
            "--embeddings", "/tmp/a.pt",
            "--embeddings-b", "/tmp/b.pt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["embeddings"] == "/tmp/a.pt"
    assert payload["embeddings_b"] == "/tmp/b.pt"


def test_replica_mesh_run_config_defaults_second_prompt_to_first():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_zimage_replica_model_mesh.py",
            "--describe-run-config",
            "--width", "512",
            "--height", "512",
            "--slug", "same-prompt",
            "--embeddings", "/tmp/a.pt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["embeddings_b"] == "/tmp/a.pt"


def test_tpu_batch4_runner_describe_config_is_cpu_only():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/run_zimage_tpu_batch4.py", "--describe-config"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mesh_shape"] == [4, 2]
    assert payload["axis_names"] == ["replica", "model"]
    assert payload["replicas"] == 4
    assert payload["model_shards"] == 2
    assert payload["batch_size"] == 4
    assert payload["seeds"] == [42, 43, 44, 45]
    assert payload["cfg_execution"] == "sequential"


def test_tpu_batch4_runner_describe_run_config_targets_1024_showcase():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_zimage_tpu_batch4.py",
            "--describe-run-config",
            "--width", "1024",
            "--height", "1024",
            "--steps", "28",
            "--guidance", "3.0",
            "--slug", "notebook-tpu-b4-1024",
            "--embeddings", "/tmp/emb.pt",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mesh_shape"] == [4, 2]
    assert payload["batch_size"] == 4
    assert payload["seeds"] == [42, 43, 44, 45]
    assert payload["width"] == 1024
    assert payload["height"] == 1024
    assert payload["steps"] == 28
