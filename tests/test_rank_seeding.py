"""CPU tests for the multi-GPU correctness fixes T1, T3, T4, T5 (CODEBASE.md §8.1)."""
import os
import socket

import ml_collections
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from trainers.trainer import (
    _broadcast_sigma_data,
)


# ----------------------------------------------------------------------- T1
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bcast_worker(rank, world, port, q):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    cfg = ml_collections.ConfigDict({"diffusion": {"continuous": {"sigma_data": 0.5}}})
    if rank == 0:
        cfg.diffusion.continuous.sigma_data = 0.399844765663147   # what the estimator writes
    val = _broadcast_sigma_data(cfg, torch.device("cpu"), True)
    q.put((rank, val, float(cfg.diffusion.continuous.sigma_data)))
    dist.destroy_process_group()


def test_sigma_data_is_rank0s_value_on_every_rank():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world, port = 3, _free_port()
    procs = [ctx.Process(target=_bcast_worker, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    # Drain before joining: a child that has queued data does not exit until
    # it is read. Spawn start-up imports the trainer, which is slow on a
    # throttled login node.
    got = sorted(q.get(timeout=300) for _ in range(world))
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert all(v == c == 0.399844765663147 for _, v, c in got), got


def test_sigma_data_broadcast_is_a_noop_without_ddp():
    cfg = ml_collections.ConfigDict({"diffusion": {"continuous": {"sigma_data": 0.42}}})
    assert _broadcast_sigma_data(cfg, torch.device("cpu"), False) == 0.42

