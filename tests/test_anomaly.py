"""Component 4 on synthetic forecasts with known anomaly structure."""

from __future__ import annotations

import numpy as np
import pytest

from anchorgate import (
    Benchmark,
    anomaly_scores,
    anomaly_skill,
    conformal_by_origin,
    pattern_acc,
    temporal_acc,
)
from anchorgate.references import climatology_table
from conftest import make_forecast


def _clim_for(obs, forecast):
    return climatology_table(obs)[forecast.target_period % 12]


def test_temporal_and_pattern_acc_on_known_cases(obs):
    half = make_forecast(obs, "half", seed=1)
    clim = _clim_for(obs, half)
    t_anom = half.target - clim
    assert temporal_acc(half.pred - clim, t_anom) == pytest.approx(1.0)
    assert pattern_acc(half.pred - clim, t_anom) == pytest.approx(1.0)
    scores = anomaly_scores(half.pred - clim, t_anom)
    assert scores["anomaly_r2"] == pytest.approx(0.75, abs=0.01)  # anomaly mean is not exactly 0
    assert scores["anomaly_sd_ratio"] == pytest.approx(0.5)
    assert scores["cells_positive"] == 1.0
    # a climatology forecast has no anomaly: every correlation is undefined, not zero
    c = make_forecast(obs, "climatology", seed=1)
    s = anomaly_scores(c.pred - clim, t_anom)
    assert np.isnan(s["acc_cell_mean"]) and np.isnan(s["acc_pooled"])


def test_pooled_acc_is_inflated_by_a_static_pattern(obs):
    """A forecast whose anomaly is a time-invariant per-cell offset has no temporal
    skill, yet the pooled correlation is positive."""
    perfect = make_forecast(obs, "perfect", seed=1)
    clim = _clim_for(obs, perfect)
    t_anom = perfect.target - clim
    static = np.broadcast_to(t_anom.mean(axis=(0, 1)), t_anom.shape)
    s = anomaly_scores(static, t_anom)
    assert np.isnan(s["acc_cell_mean"])  # zero temporal variance in every cell
    assert s["acc_pooled"] > 0.1  # inflated: the pattern alone correlates


def test_anomaly_skill_verdicts(benchmark):
    res = anomaly_skill(benchmark)
    cfg = res.tables["by_configuration"].set_index("model")
    assert cfg.loc["perfect", "verdict"] == "PASS"
    assert cfg.loc["half", "verdict"] == "PASS"
    assert cfg.loc["climatology", "verdict"] == "FAIL"
    assert cfg.loc["single", "verdict"] == "INSUFFICIENT"
    assert cfg.loc["half", "msess_mean"] == pytest.approx(0.75)
    assert res.verdict.value == "PASS"
    v = res.values.set_index("label")
    assert v.loc["climatology", "msess"] == 0.0
    assert "by_lead" in res.tables and res.tables["by_lead"]["lead"].nunique() == 6


def test_anomaly_skill_fails_without_temporal_skill(obs):
    fcs = [make_forecast(obs, "climatology", seed=s) for s in (1, 2)]
    res = anomaly_skill(Benchmark(fcs, obs))
    assert res.verdict.value == "FAIL"


def test_conformal_by_origin(obs):
    perfect = make_forecast(obs, "perfect", seed=1)
    c = conformal_by_origin(perfect.pred, perfect.target, n_calibration=10)
    assert (c["half_width"] == 0).all() and (c["coverage"] == 1).all()
    noisy = make_forecast(obs, "noisy", seed=3)
    n = noisy.n_origin
    strata = {"first_half": np.arange(noisy.n_cell) < noisy.n_cell // 2}
    c = conformal_by_origin(
        noisy.pred, noisy.target, n_calibration=n // 2, alpha=0.1, strata=strata
    )
    assert c["coverage"].mean() > 0.8
    assert "coverage_first_half" in c
    assert (c["width"] > 0).all() and (c["lower_below_zero"] >= 0).all()
    assert c["level_guaranteed"].iloc[0] <= 1.0
    with pytest.raises(ValueError):
        conformal_by_origin(noisy.pred, noisy.target, n_calibration=n)


def test_anomaly_skill_with_conformal_tables(benchmark):
    res = anomaly_skill(benchmark, conformal=True, n_calibration=8)
    assert "conformal" in res.tables and "conformal_summary" in res.tables
    summ = res.tables["conformal_summary"].set_index("label")
    assert summ.loc["climatology", "coverage"] >= 0.9
    assert any("conformal intervals" in n for n in res.notes)
