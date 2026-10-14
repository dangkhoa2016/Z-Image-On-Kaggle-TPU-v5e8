from __future__ import annotations


def replica_model_mesh_shape(total_devices: int, replicas: int) -> tuple[int, int]:
    if total_devices <= 0 or replicas <= 0:
        raise ValueError("device and replica counts must be positive")
    if total_devices % replicas:
        raise ValueError("total devices must be divisible by replicas")
    return replicas, total_devices // replicas


def choose_model_shard_axis(shape: tuple[int, ...], model_shards: int) -> int | None:
    if model_shards <= 0:
        raise ValueError("model_shards must be positive")
    candidates = [
        (int(size), axis)
        for axis, size in enumerate(shape)
        if int(size) >= model_shards and int(size) % model_shards == 0
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def replica_batch_partition_spec(ndim: int) -> tuple[str | None, ...]:
    if ndim < 1:
        raise ValueError("ndim must be >= 1")
    return ("replica",) + (None,) * (ndim - 1)
