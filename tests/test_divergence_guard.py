"""Divergence guard: abort a diverged run instead of burning the budget.

The first CE/SM pilot diverged and then trained with bit-identical weights for
~55k steps, wasting ~55 GPU-h. These tests pin the rule that stops that.
"""
import ml_collections
import pytest

from trainers.trainer import Trainer


class _Stub:
    _check_divergence = Trainer._check_divergence

    def __init__(self, **kw):
        self.cfg = ml_collections.ConfigDict()
        self.cfg.train = ml_collections.ConfigDict()
        if kw.pop("enabled", True):
            g = ml_collections.ConfigDict()
            g.enabled = True
            g.factor = kw.pop("factor", 20.0)
            g.patience = kw.pop("patience", 10)
            g.min_steps = kw.pop("min_steps", 0)
            g.ema_decay = kw.pop("ema_decay", 0.0)   # 0 => EMA is the raw loss
            self.cfg.train.divergence_guard = g
        self.global_step = 0
        self._div_ema = self._div_best = None
        self._div_strikes = 0
        import collections
        self._spike_ring = collections.deque(maxlen=2000)
        self._last_grad_norm = None

    def _dump_spike_ring(self, reason):   # no-op unless a test overrides it
        pass

    def feed(self, losses):
        for v in losses:
            self.global_step += 1
            self._check_divergence(v)


def test_off_by_default():
    s = _Stub(enabled=False)
    s.feed([0.01] * 50 + [1e9] * 5000)     # would trip any enabled guard


def test_steadily_decreasing_loss_never_trips():
    s = _Stub(patience=5)
    s.feed([1.0 / (i + 1) for i in range(2000)])


def test_noisy_but_healthy_loss_never_trips():
    """Real training is spiky; a guard that fires on spikes is useless."""
    s = _Stub(factor=20.0, patience=200, ema_decay=0.99)
    losses = []
    for i in range(5000):
        base = 0.05
        losses.append(base * (30.0 if i % 500 == 0 else 1.0))   # periodic spikes
    s.feed(losses)


def test_sustained_blowup_aborts():
    s = _Stub(factor=20.0, patience=10)
    with pytest.raises(SystemExit, match="divergence-guard"):
        s.feed([0.02] * 100 + [50.0] * 200)


def test_non_finite_loss_aborts_immediately():
    s = _Stub()
    with pytest.raises(SystemExit, match="non-finite"):
        s.feed([0.02] * 10 + [float("nan")])


def test_transient_spike_shorter_than_patience_is_tolerated():
    s = _Stub(factor=20.0, patience=50)
    s.feed([0.02] * 100 + [50.0] * 20 + [0.02] * 100)


def test_guard_is_not_armed_during_warmup():
    """Early loss falls fast; 'best so far' is not meaningful yet, so a large
    early swing must not abort. Once past min_steps the guard behaves normally."""
    s = _Stub(factor=2.0, patience=1, min_steps=500)
    s.feed([10.0] * 400 + [0.01] * 50)          # swings 1000x, still unarmed
    assert s._div_best is None
    s.feed([0.01] * 100)                        # arms, best settles at ~0.01
    assert s._div_best is not None
    with pytest.raises(SystemExit):
        s.feed([100.0] * 200)


def test_best_is_seeded_from_the_first_armed_step_not_from_warmup():
    """If a run is already bad when the guard arms, the guard cannot know that:
    it anchors on what it sees. Pins the semantics so it is not mistaken for a
    bug later."""
    s = _Stub(factor=20.0, patience=5, min_steps=100)
    s.feed([0.001] * 99)                        # never seen by the guard
    s.feed([50.0] * 50)                         # arms here; 50.0 becomes "best"
    assert s._div_best == pytest.approx(50.0)


def test_the_actual_pilot_trajectory_would_have_been_caught_early():
    """binary_sm: ~0.024 through 19.5k, then >1 and rising. The guard must fire
    within a few hundred steps of the real onset, not 65k steps later."""
    s = _Stub(factor=20.0, patience=200, ema_decay=0.99, min_steps=2000)
    healthy = [0.024] * 19540
    with pytest.raises(SystemExit) as e:
        s.feed(healthy + [30.0] * 1000)
    assert s.global_step < 19540 + 800, f"fired too late: step {s.global_step}"


# Measured from the four 5k smoke runs: (best EMA, sustained post-break level).
# Both SM arms break; neither CE arm does.
_SMOKE = {
    "sm_clamped":   (0.0327, 0.0327 * 15.0),   # broke at step 3,900
    "sm_noclamp":   (0.0337, 0.0337 * 4.9),    # broke at step 4,720
    "ce_clamped":   (0.1019, 0.1019 * 1.1),    # healthy
    "ce_noclamp":   (0.1025, 0.1025 * 1.1),    # healthy
}


@pytest.mark.parametrize("arm", ["sm_clamped", "sm_noclamp"])
def test_default_factor_catches_both_observed_sm_breaks(arm):
    best, level = _SMOKE[arm]
    s = _Stub(factor=4.0, patience=10, min_steps=0)
    with pytest.raises(SystemExit):
        s.feed([best] * 50 + [level] * 300)


@pytest.mark.parametrize("arm", ["ce_clamped", "ce_noclamp"])
def test_default_factor_never_fires_on_a_healthy_ce_arm(arm):
    best, level = _SMOKE[arm]
    s = _Stub(factor=4.0, patience=10, min_steps=0)
    s.feed([best] * 50 + [level] * 1000)
    # even the transient 1.9x peak must not trip it
    s.feed([best * 1.9] * 50)


def test_factor_5_would_miss_the_unclamped_sm_break():
    """Pins why the default is 4 and not higher: the unclamped SM arm settles at
    only ~4.9x its best, so 5 and 10 both miss a real divergence."""
    best, level = _SMOKE["sm_noclamp"]
    for f in (5.0, 10.0, 20.0):
        s = _Stub(factor=f, patience=10, min_steps=0)
        s.feed([best] * 50 + [level] * 500)      # no abort


# --- Calibration against the known-good production run ------------------------
# Measured from the full 500k production history (all event files merged):
#   max EMA/best = 6.29, stable plateau ~5.4x for the last 450k steps.
# The rise starts at entropy_warmup_steps=40000 over entropy_transition_steps
# =10000 -- the sigma-schedule handover changes the loss scale. It is not a
# divergence, and a guard that fires on it is useless.
PRODUCTION_MAX_RATIO = 6.29
# Smallest ratio among the six measured 20k divergences (ce seed 44).
SMALLEST_REAL_DIVERGENCE = 17.2


def test_guard_would_not_have_aborted_the_production_run():
    """The previous default of 4 would have killed a run that went on to train
    successfully to 500k. This is the regression that matters most."""
    s = _Stub(factor=4.0, patience=10, min_steps=0)
    with pytest.raises(SystemExit):
        s.feed([0.023] * 50 + [0.023 * PRODUCTION_MAX_RATIO] * 500)

    s = _Stub(factor=10.0, patience=10, min_steps=0)
    s.feed([0.023] * 50 + [0.023 * PRODUCTION_MAX_RATIO] * 5000)   # must survive


def test_guard_still_catches_the_smallest_measured_divergence():
    s = _Stub(factor=10.0, patience=10, min_steps=0)
    with pytest.raises(SystemExit):
        s.feed([0.065] * 50 + [0.065 * SMALLEST_REAL_DIVERGENCE] * 300)


def test_threshold_sits_between_the_two_populations():
    assert PRODUCTION_MAX_RATIO < 10.0 < SMALLEST_REAL_DIVERGENCE


# --- per-step ring buffer -----------------------------------------------------
# The 100-step scalar logs cannot see a spike: the production-era break went
# 0.033 -> 13.7 inside one 20-step interval and the spike step itself was never
# sampled. The ring keeps every step at full resolution.

def _ring_stub(tmp_path, ring_steps=50, **kw):
    import collections
    s = _Stub(**kw)
    s.cfg.train.divergence_guard.ring_steps = ring_steps
    s.cfg.evaluation = ml_collections.ConfigDict()
    s.cfg.evaluation.checkpoint_path = str(tmp_path / "run" / "checkpoints" / "last.pt")
    s._spike_ring = collections.deque(maxlen=ring_steps)
    s.is_master = True
    s._last_grad_norm = 0.5
    from trainers.trainer import Trainer
    s._dump_spike_ring = Trainer._dump_spike_ring.__get__(s)
    return s


def test_ring_records_every_step_not_every_hundredth(tmp_path):
    s = _ring_stub(tmp_path, ring_steps=500, factor=10.0, patience=10, min_steps=0)
    s.feed([0.03] * 200)
    steps = [r[0] for r in s._spike_ring]
    assert steps == sorted(steps) and len(steps) == 200
    assert steps[1] - steps[0] == 1, "ring must be per-step"


def test_ring_is_bounded_and_keeps_the_most_recent(tmp_path):
    s = _ring_stub(tmp_path, ring_steps=50, factor=10.0, patience=10, min_steps=0)
    s.feed([0.03] * 400)
    assert len(s._spike_ring) == 50
    assert s._spike_ring[-1][0] == 400


def test_ring_is_dumped_on_abort_and_contains_the_spike(tmp_path):
    import json
    s = _ring_stub(tmp_path, ring_steps=500, factor=10.0, patience=10, min_steps=0)
    with pytest.raises(SystemExit):
        s.feed([0.03] * 100 + [50.0] * 60)
    out = tmp_path / "run" / "spike_ring.json"
    assert out.exists(), "ring buffer was not written on abort"
    d = json.loads(out.read_text())
    assert d["reason"] == "divergence"
    losses = [r[1] for r in d["rows"]]
    assert max(losses) >= 50.0, "the spike itself must be in the dump"
    assert d["columns"] == ["step", "loss", "grad_norm_preclip"]


def test_ring_is_dumped_on_a_non_finite_loss(tmp_path):
    import json
    s = _ring_stub(tmp_path, ring_steps=500, factor=10.0, patience=10, min_steps=0)
    with pytest.raises(SystemExit, match="non-finite"):
        s.feed([0.03] * 20 + [float("nan")])
    d = json.loads((tmp_path / "run" / "spike_ring.json").read_text())
    assert d["reason"] == "non_finite"


def test_dump_failure_never_masks_the_abort(tmp_path):
    s = _ring_stub(tmp_path, ring_steps=50, factor=10.0, patience=10, min_steps=0)
    s.cfg.evaluation.checkpoint_path = "/proc/definitely/not/writable/x/y.pt"
    with pytest.raises(SystemExit, match="divergence-guard"):
        s.feed([0.03] * 100 + [50.0] * 60)
