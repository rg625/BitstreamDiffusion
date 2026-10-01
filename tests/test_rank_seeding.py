"""CPU tests for the multi-GPU correctness fixes T1, T3, T4, T5 (CODEBASE.md §8.1)."""
import os
import socket

import ml_collections
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from trainers.trainer import (
    Trainer, _broadcast_sigma_data, _make_distributed_sampler, _rank_seed, _seed_everything,
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



# ----------------------------------------------------------------------- T3
def test_rank_seed_is_distinct_per_rank_and_step():
    seeds = {_rank_seed(42, r, s) for r in range(8) for s in (0, 5000, 10000)}
    assert len(seeds) == 24


def test_rank0_step0_keeps_the_base_seed():
    assert _rank_seed(42, 0, 0) == 42


def test_ranks_draw_different_noise():
    draws = []
    for r in range(4):
        _seed_everything(_rank_seed(42, r))
        draws.append(torch.randn(16))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(draws[i], draws[j])


class _ResumeStub:
    _restore_rng_on_resume = Trainer._restore_rng_on_resume
    _set_rng_state = Trainer._set_rng_state
    _rng_state = Trainer._rng_state

    def __init__(self, rank, world):
        self.rank, self.world_size, self.is_master = rank, world, rank == 0
        self.cfg = ml_collections.ConfigDict({"train": {"seed": 42}})


def test_resume_no_longer_copies_rank0_rng_into_every_rank():
    torch.manual_seed(123)
    saved = _ResumeStub(0, 4)._rng_state()          # rank 0's state, as checkpointed
    after = []
    for r in range(4):
        _ResumeStub(r, 4)._restore_rng_on_resume(saved, step=20000)
        after.append(torch.randn(8))
    torch.set_rng_state(saved["torch_cpu"])
    rank0_saved_stream = torch.randn(8)
    assert all(not torch.allclose(a, rank0_saved_stream) for a in after)
    assert len({tuple(a.tolist()) for a in after}) == 4


def test_resume_is_exact_for_a_single_process():
    torch.manual_seed(7)
    saved = _ResumeStub(0, 1)._rng_state()
    expected = torch.randn(8)
    torch.manual_seed(999)
    _ResumeStub(0, 1)._restore_rng_on_resume(saved, step=20000)
    assert torch.equal(torch.randn(8), expected)


def test_resume_reseed_is_deterministic():
    a, b = _ResumeStub(2, 4), _ResumeStub(2, 4)
    a._restore_rng_on_resume(None, step=100)
    x = torch.randn(4)
    b._restore_rng_on_resume(None, step=100)
    assert torch.equal(torch.randn(4), x)



# ----------------------------------------------------------------------- T4
def test_distributed_sampler_honours_the_training_seed():
    ds = list(range(1000))
    order = lambda seed: list(_make_distributed_sampler(  # noqa: E731
        ds, world_size=4, rank=1, seed=seed, shuffle=True, drop_last=True))
    s42 = _make_distributed_sampler(ds, world_size=4, rank=1, seed=42, shuffle=True, drop_last=True)
    assert s42.seed == 42
    assert order(42) != order(43)
    assert order(42) == order(42)

