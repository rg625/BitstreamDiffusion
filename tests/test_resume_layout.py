"""Resume: the fresh config's compute layout must survive the saved-config merge.

train.py merges runs/<exp>/config.json over the freshly built config on every
resume, which silently discarded TOK_ACCUM when the token run was moved from
one node (64/GPU x 4 x accum 2) to two (64/GPU x 8 x accum 1): the resumed job
ran 32/GPU x 8 x accum 2. Layout keys now come from the fresh config; what a
step IS (the effective batch, the objective, the model) still comes from the
saved snapshot.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from ml_collections import config_dict

import train

ROOT = Path(__file__).resolve().parents[1]


def _token_cfg(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location(
        "tok_cfg", ROOT / "configs/tasks/tinygsm_tokens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def _resume(cfg, saved):
    layout = train._snapshot_layout(cfg)
    train._update_cfg_from_dict(cfg, saved)
    return train._restore_layout(cfg, layout)


def test_fresh_accum_wins_and_effective_batch_stays_saved(monkeypatch):
    saved = _token_cfg(monkeypatch, TOK_BATCH=512, TOK_ACCUM=2, TOK_RESUME_EVERY=1000).to_dict()
    cfg = _token_cfg(monkeypatch, TOK_BATCH=512, TOK_ACCUM=1, TOK_RESUME_EVERY=2000)
    changed = dict((k, (s, f)) for k, s, f in _resume(cfg, saved))
    assert cfg.train.grad_accum_steps == 1
    assert cfg.train.checkpointing.resume_interval.every_steps == 2000
    assert cfg.train.batch_size == 512
    assert changed == {"train.grad_accum_steps": (2, 1),
                       "train.checkpointing.resume_interval.every_steps": (1000, 2000)}


def test_effective_batch_is_not_a_layout_key(monkeypatch):
    # A resume must never change what a step is: a different TOK_BATCH on the
    # command line is overwritten by the saved one.
    saved = _token_cfg(monkeypatch, TOK_BATCH=512).to_dict()
    cfg = _token_cfg(monkeypatch, TOK_BATCH=1024)
    _resume(cfg, saved)
    assert cfg.train.batch_size == 512


def test_key_absent_from_fresh_config_is_left_as_saved():
    # The bits configs never set grad_accum_steps; a resumed bits run must keep
    # whatever its snapshot says rather than gain or lose the key.
    cfg = config_dict.ConfigDict({"train": {"batch_size": 512}})
    saved = {"train": {"batch_size": 512, "grad_accum_steps": 3}}
    assert _resume(cfg, saved) == []
    assert cfg.train.grad_accum_steps == 3


def test_unchanged_layout_reports_nothing(monkeypatch):
    saved = _token_cfg(monkeypatch, TOK_ACCUM=2).to_dict()
    cfg = _token_cfg(monkeypatch, TOK_ACCUM=2)
    assert _resume(cfg, saved) == []


def test_real_saved_config_of_the_moved_run(monkeypatch):
    p = ROOT / "runs/tasks/tinygsm/tok_gh500k_token_ce_s42/config.json"
    if not p.exists():
        pytest.skip("run directory not present on this machine")
    saved = json.loads(p.read_text())
    cfg = _token_cfg(monkeypatch, TOK_BATCH=512, TOK_ACCUM=1, TOK_TAG="gh500k")
    _resume(cfg, saved)
    assert cfg.train.grad_accum_steps == 1
    assert cfg.train.batch_size == saved["train"]["batch_size"] == 512
    assert cfg.model.out_dim == 49153 and cfg.train.loss_type == "token_ce"
