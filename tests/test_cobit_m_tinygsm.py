"""CoBit-M on TinyGSM (configs/tasks/tinygsm_bits_cobit_m.py): CPU checks.

The run is the unguided CoBit-S TinyGSM data/objective with the CoBit-M OWT
trunk and optimiser, a cosine laid out over 500k that stops at 250k, the
pre-T4 data order (shuffle seed 0), and step=*.pt only at 100k/200k/250k.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import ml_collections
import pytest
import torch
from torch.utils.data import DistributedSampler

import train
from trainers.trainer import (
    Trainer, _data_order_seed, _make_distributed_sampler, _resolve_checkpointing_cfg, _stop_step,
)
from utils.optim import WarmUpCosineDecayScheduler

ROOT = Path(__file__).resolve().parents[1]


def _load(rel, monkeypatch=None, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


def _diff(a, b, prefixes):
    fa, fb = _flat(a.to_dict()), _flat(b.to_dict())
    return {k for k in set(fa) | set(fb)
            if k.startswith(prefixes) and fa.get(k, "<absent>") != fb.get(k, "<absent>")}


M = "configs/tasks/tinygsm_bits_cobit_m.py"
S = "configs/tasks/tinygsm_bits.py"
OWT = "configs/owt/rate_bits_edm_weight_medium_24x1024.py"


# ------------------------------------------------------------------ config
def test_data_side_is_exactly_tinygsm_cobit_s():
    m, s = _load(M), _load(S)
    assert _diff(m, s, ("data.", "cond.", "diffusion.")) == set()
    for k in ("batch_size", "seed", "ema_decay", "loss_type", "loss_weighting",
              "sigma_sampling_strategy", "self_condition_prob", "amp_dtype", "epochs"):
        assert getattr(m.train, k) == getattr(s.train, k), k
    assert m.cond.p_uncond == 0.0                       # unguided
    for name in ("generation", "external_ppl", "mauve", "visualization", "vlb"):
        assert getattr(m.train, name).enabled is False  # validation as CoBit-S


def test_entropy_schedule_settings_match_both_parents():
    m, s, o = _load(M), _load(S), _load(OWT)
    ent = {k for k in _flat(m.train.to_dict()) if k.startswith("entropy")}
    assert ent
    for k in ent:
        assert _flat(m.train.to_dict())[k] == _flat(s.train.to_dict())[k] == _flat(o.train.to_dict())[k], k


def test_model_is_cobit_s_head_with_cobit_m_trunk():
    m, s, o = _load(M), _load(S), _load(OWT)
    assert _diff(m, o, ("model.",)) == set()            # whole model block = CoBit-M OWT
    assert _diff(m, s, ("model.",)) == {"model.embed_dim", "model.dim_ff",
                                        "model.n_blocks", "model.n_heads"}
    assert (m.model.n_blocks, m.model.embed_dim, m.model.n_heads, m.model.dim_ff) == (24, 1024, 16, 4096)


def test_optimiser_is_cobit_m_owt_with_a_500k_cosine():
    m, o = _load(M), _load(OWT)
    assert _diff(m, o, ("optim.",)) == {"optim.total_steps"}
    assert m.optim.total_steps == 500_000
    assert (m.optim.lr, m.optim.weight_decay, m.optim.beta2, m.optim.scheduler, m.optim.warmup) == \
        (2e-4, 0.01, 0.99, "cosine_decay", 5_000)


def test_stop_checkpoints_and_data_order_defaults():
    m = _load(M)
    assert m.experiment == "tasks/tinygsm/cobit_m_bits_s42"
    assert m.train.stop_at_step == 250_000 and _stop_step(m) == 250_000
    assert m.train.data_order_seed == 0 and _data_order_seed(m) == 0 and m.train.seed == 42
    ck = _resolve_checkpointing_cfg(m)
    assert ck["interval_at_steps"] == {100_000, 200_000, 250_000}
    assert ck["resume_interval_every_steps"] == 2_000
    assert ck["save_top_k"] == 0


def test_env_overrides(monkeypatch):
    m = _load(M, monkeypatch, COBIT_M_STOP_AT_STEP=400, COBIT_M_CKPT_STEPS="300",
              COBIT_M_TAG="smoke_n1", COBIT_M_ACCUM=2)
    assert _stop_step(m) == 400 and m.optim.total_steps == 500_000
    assert _resolve_checkpointing_cfg(m)["interval_at_steps"] == {300}
    assert m.experiment == "tasks/tinygsm/cobit_m_bits_smoke_n1"
    assert m.train.grad_accum_steps == 2


def test_parameter_count_is_cobit_m():
    from models import create_model
    m, o = _load(M), _load(OWT)
    with torch.device("meta"):
        n_m = sum(p.numel() for p in create_model(m).parameters())
        n_o = sum(p.numel() for p in create_model(o).parameters())
    assert 440e6 < n_m < 480e6, n_m
    # Same trunk and head; only per-token input/position sizes can differ.
    assert abs(n_m - n_o) / n_o < 0.01, (n_m, n_o)


# ------------------------------------------------------------- LR schedule
def _lr_after(steps, start=0, total=500_000):
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    sch = WarmUpCosineDecayScheduler(opt, warmup_steps=5_000, total_steps=total,
                                     base_lr=2e-4, global_step=start)
    for _ in range(steps):
        sch.step()
    return opt.param_groups[0]["lr"]


def test_lr_is_the_500k_cosine():
    assert _lr_after(5_000) == pytest.approx(2e-4)
    expect_250k = 1e-6 + (2e-4 - 1e-6) * 0.5 * (1 + math.cos(math.pi * 245_000 / 495_000))
    assert _lr_after(1, start=249_999) == pytest.approx(expect_250k)
    assert 0.9e-4 < expect_250k < 1.1e-4
    assert _lr_after(1, start=499_999) == pytest.approx(1e-6)


def test_stopping_at_250k_and_resuming_continues_the_same_cosine():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.0)
    full = WarmUpCosineDecayScheduler(opt, warmup_steps=5_000, total_steps=500_000, base_lr=2e-4)
    full.load_state_dict({"step": 250_000})
    full.step()
    continuous = opt.param_groups[0]["lr"]
    assert _lr_after(1, start=250_000) == continuous


def test_stop_step_takes_the_earlier_of_the_two():
    cfg = lambda total, stop: ml_collections.ConfigDict(  # noqa: E731
        {"optim": {"total_steps": total}, "train": {"stop_at_step": stop}})
    assert _stop_step(cfg(500_000, 250_000)) == 250_000
    assert _stop_step(cfg(500_000, 0)) == 500_000
    assert _stop_step(cfg(250_000, 500_000)) == 250_000
    plain = ml_collections.ConfigDict({"optim": {"total_steps": 250_000}, "train": {}})
    assert _stop_step(plain) == 250_000                 # every existing config


# -------------------------------------------------------------- data order
def test_data_order_seed_zero_reproduces_the_pre_t4_sampler():
    ds = list(range(5_000))
    old = DistributedSampler(ds, num_replicas=4, rank=2, shuffle=True, drop_last=True)  # no seed=
    new = _make_distributed_sampler(ds, world_size=4, rank=2, seed=0, shuffle=True, drop_last=True)
    for e in (0, 3):
        old.set_epoch(e)
        new.set_epoch(e)
        assert list(old) == list(new)


def test_data_order_seed_falls_back_to_the_training_seed():
    assert _data_order_seed(ml_collections.ConfigDict({"train": {"seed": 42}})) == 42
    assert _data_order_seed(ml_collections.ConfigDict({"train": {"seed": 42, "data_order_seed": 0}})) == 0


@pytest.mark.parametrize("world", [8, 2])
def test_global_batches_do_not_depend_on_the_gpu_count(world):
    n, B = 11_704, 512                                  # divisible by 2, 4 and 8, like the cache

    def global_batches(w):
        per = B // w
        ranks = []
        for r in range(w):
            s = _make_distributed_sampler(list(range(n)), world_size=w, rank=r, seed=0,
                                          shuffle=True, drop_last=True)
            s.set_epoch(1)
            idx = list(s)
            ranks.append([idx[i:i + per] for i in range(0, len(idx) - per + 1, per)])
        return [sorted(sum((ranks[r][b] for r in range(w)), [])) for b in range(len(ranks[0]))]

    assert global_batches(world) == global_batches(4)


# ------------------------------------------------------------- checkpoints
class _CkptStub(Trainer):
    def __init__(self, ck, tmp_path):                   # bypass the real __init__
        self.is_master = True
        self.ckpt_interval_enabled = ck["interval_enabled"]
        self.ckpt_interval_every_steps = ck["interval_every_steps"]
        self.ckpt_interval_keep_last = ck["interval_keep_last"]
        self.ckpt_interval_at_steps = ck["interval_at_steps"]
        self._next_interval_ckpt_step = ck["interval_every_steps"]
        self._interval_ckpt_paths = []
        self.ckpt_dir = tmp_path
        self.global_step = 0

    def _build_ckpt_state(self, epoch):
        return {"step": self.global_step}

    def _checkpoint_path(self, name):
        return self.ckpt_dir / f"{name}.pt"


def _saved_steps(cfg, tmp_path, last):
    t = _CkptStub(_resolve_checkpointing_cfg(cfg), tmp_path)
    for s in range(1, last + 1):
        t.global_step = s
        t._maybe_save_interval_ckpt(epoch=0)
    return sorted(int(p.stem.split("=")[1]) for p in tmp_path.glob("step=*.pt"))


def test_numbered_checkpoints_only_at_the_listed_steps(monkeypatch, tmp_path):
    m = _load(M, monkeypatch, COBIT_M_CKPT_STEPS="100,200,250")
    assert _saved_steps(m, tmp_path, 300) == [100, 200, 250]


def test_every_steps_behaviour_is_unchanged_without_a_list(tmp_path):
    cfg = ml_collections.ConfigDict({"train": {"checkpointing": {
        "interval": {"enabled": True, "every_steps": 25, "keep_last": 0}}}})
    assert _saved_steps(cfg, tmp_path, 100) == [25, 50, 75, 100]


# ------------------------------------------------------------------ resume
def _resume(cfg, saved):
    layout = train._snapshot_layout(cfg)
    train._update_cfg_from_dict(cfg, saved)
    return dict((k, (s, f)) for k, s, f in train._restore_layout(cfg, layout))


def test_resume_to_500k_moves_the_stop_and_checkpoints_not_the_schedule(monkeypatch):
    saved = _load(M).to_dict()                          # the 250k run's config.json
    fresh = _load(M, monkeypatch, COBIT_M_STOP_AT_STEP=500_000,
                  COBIT_M_CKPT_STEPS="300000,400000,500000")
    changed = _resume(fresh, saved)
    assert fresh.train.stop_at_step == 500_000 and _stop_step(fresh) == 500_000
    assert list(fresh.train.checkpointing.interval.at_steps) == [300_000, 400_000, 500_000]
    assert fresh.optim.total_steps == 500_000 and fresh.optim.lr == 2e-4
    assert fresh.train.data_order_seed == 0
    assert set(changed) == {"train.stop_at_step", "train.checkpointing.interval.at_steps"}


def test_plain_resume_changes_nothing():
    assert _resume(_load(M), _load(M).to_dict()) == {}
