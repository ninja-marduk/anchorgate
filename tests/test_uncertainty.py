import numpy as np
import pytest

from anchorgate import Benchmark, Gate, Verdict, anchor, anomaly_skill
from anchorgate.uncertainty import (
    block_indices,
    correlation_sums,
    interval,
    median_correlation,
    skill_from_sums,
    skill_horizon,
    verdict_from_interval,
)
from conftest import make_forecast


def test_block_indices_shape_and_range():
    idx = block_indices(12, 3, 50, seed=1)
    assert idx.shape == (50, 12)
    assert idx.min() >= 0 and idx.max() < 12
    # blocks are consecutive modulo the record length
    first = idx[0]
    assert all((first[i + 1] - first[i]) % 12 == 1 for i in range(2))
    assert block_indices(5, 30, 4).shape == (4, 5)
    with pytest.raises(ValueError):
        block_indices(0, 3, 4)


def test_skill_from_sums_matches_the_point_statistic():
    rng = np.random.default_rng(0)
    num = rng.uniform(1, 2, size=(10, 4))
    den = rng.uniform(2, 3, size=(10, 4))
    idx = np.tile(np.arange(10), (3, 1))
    boot = skill_from_sums(num, den, idx)
    assert boot.shape == (3, 4)
    assert boot[0] == pytest.approx(1 - num.sum(axis=0) / den.sum(axis=0))


def test_median_correlation_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3, 7))
    y = 0.5 * x + rng.normal(size=x.shape)
    idx = np.arange(20)[None, :]
    got = median_correlation(correlation_sums(x, y), idx)[0]
    direct = np.median(
        [np.corrcoef(x[:, :, c].ravel(), y[:, :, c].ravel())[0, 1] for c in range(7)]
    )
    assert got == pytest.approx(direct)


def test_interval_and_verdicts():
    s = np.arange(101, dtype=float)
    lo, hi = interval(s, 0.9)
    assert (lo, hi) == pytest.approx((5.0, 95.0))
    assert interval(np.array([np.nan]), 0.9) == (pytest.approx(np.nan, nan_ok=True),) * 2
    assert verdict_from_interval(0.1, 0.3, 0.0) is Verdict.PASS
    assert verdict_from_interval(-0.3, -0.1, 0.0) is Verdict.FAIL
    assert verdict_from_interval(-0.3, 0.0, 0.0) is Verdict.FAIL
    assert verdict_from_interval(-0.1, 0.1, 0.0) is Verdict.INSUFFICIENT
    assert verdict_from_interval(float("nan"), 0.1, 0.0) is Verdict.FAIL


def test_skill_horizon_counts_the_unbroken_run():
    leads = np.array([1, 2, 3, 4, 5, 6])
    assert skill_horizon(np.array([0.1, 0.05, 0.01, -0.1, 0.2, 0.3]), leads, 0.0) == 3
    assert skill_horizon(np.array([-0.1, 0.5, 0.5, 0.5, 0.5, 0.5]), leads, 0.0) == 0
    assert skill_horizon(np.array([np.nan, 0.5, 0.5, 0.5, 0.5, 0.5]), leads, 0.0) == 0
    assert skill_horizon(np.full(6, 0.2), leads, 0.0) == 6


def test_gate_is_insufficient_when_the_interval_straddles_the_threshold(obs):
    """A forecast that is perfect on even origins and twice as wrong as the climatology on
    odd ones has a skill score near zero whose block-bootstrap interval spans both signs;
    the gate must not call it either way."""
    runs = []
    for s in (1, 2):
        clim = make_forecast(obs, "climatology", seed=s)
        pred = clim.pred.copy()
        odd = np.arange(pred.shape[0]) % 2 == 1
        pred[~odd] = clim.target[~odd]
        pred[odd] = clim.target[odd] + np.sqrt(2.0) * (clim.pred[odd] - clim.target[odd])
        runs.append(
            clim.__class__(
                pred=pred,
                target=clim.target,
                origin_time=clim.origin_time,
                lead=clim.lead,
                model="alternating",
                features="BASIC",
                seed=s,
                space_shape=clim.space_shape,
                lat=clim.lat,
                lon=clim.lon,
                mask=clim.mask,
            )
        )
    res = anchor(Benchmark(runs, obs), gate=Gate(min_seeds=2, block=1, n_boot=400), validate=False)
    cfg = res.tables["by_configuration"].iloc[0]
    assert abs(cfg["msess_mean"]) < 0.25
    assert cfg["msess_lower"] < 0.0 < cfg["msess_upper"]
    assert cfg["verdict"] == "INSUFFICIENT" and res.verdict is Verdict.INSUFFICIENT
    lt = res.tables["by_lead_configuration"]
    assert set(lt.columns) >= {"lead", "msess_mean", "msess_lower", "msess_upper"}
    assert (lt["msess_lower"] <= lt["msess_upper"]).all()


def test_anchor_intervals_bracket_the_point_statistic(benchmark):
    res = anchor(benchmark, gate=Gate(level=0.9, block=2, n_boot=300))
    cfg = res.tables["by_configuration"].set_index("model")
    for model in ("perfect", "half", "climatology"):
        lo, hi = cfg.loc[model, "msess_lower"], cfg.loc[model, "msess_upper"]
        assert (
            lo <= cfg.loc[model, "msess_mean"] + 1e-9 and hi >= cfg.loc[model, "msess_mean"] - 1e-9
        )
    assert cfg.loc["perfect", "skill_horizon"] == int(benchmark.lead.max())
    assert cfg.loc["climatology", "skill_horizon"] == 0
    assert "interval" in res.rule and "skill horizon" in res.notes[1]


def test_anomaly_verdict_reads_the_correlation_interval(benchmark):
    res = anomaly_skill(benchmark, gate=Gate(n_boot=300))
    cfg = res.tables["by_configuration"].set_index("model")
    assert cfg.loc["perfect", "acc_cell_median_lower"] == pytest.approx(1.0)
    assert np.isnan(cfg.loc["climatology", "acc_cell_median_lower"])
    assert cfg.loc["climatology", "verdict"] == "FAIL"
    assert "MSESS" in res.rule and "both" not in res.rule
