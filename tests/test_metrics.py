import numpy as np
import pytest

from anchorgate import metrics as m


def test_pooled_r2_perfect_and_mean():
    obs = np.random.default_rng(0).normal(size=(5, 3, 7))
    assert m.pooled_r2(obs, obs) == pytest.approx(1.0)
    assert m.pooled_r2(np.full_like(obs, obs.mean()), obs) == pytest.approx(0.0)


def test_per_cell_nse_climatology_scores_zero_against_cell_mean():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(20, 2, 4)) + np.arange(4)[None, None, :] * 10
    cell_mean = np.broadcast_to(obs.reshape(-1, 4).mean(0), obs.shape)
    assert m.per_cell_nse(cell_mean, obs) == pytest.approx(0.0)
    assert m.per_cell_nse(obs, obs) == pytest.approx(1.0)
    assert m.pooled_r2(cell_mean, obs) > 0.9  # pooled R2 rewards between-cell variance


def test_msess_known_values():
    obs = np.random.default_rng(2).normal(size=(10, 4))
    ref = np.zeros_like(obs)
    assert m.msess(obs, obs, ref) == pytest.approx(1.0)
    assert m.msess(ref, obs, ref) == pytest.approx(0.0)
    half = 0.5 * obs
    assert m.msess(half, obs, ref) == pytest.approx(0.75)
    worse = 2 * obs
    assert m.msess(worse, obs, ref) == pytest.approx(0.0)
    assert m.msess(3 * obs, obs, ref) < 0


def test_nan_pairs_are_dropped_consistently():
    obs = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
    pred = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    assert m.mae(pred, obs) == 0.0
    assert m.rmse(pred, obs) == 0.0
    assert m.bias(pred, obs) == 0.0


def test_categorical_scores_perfect_random_constant():
    rng = np.random.default_rng(3)
    o = (rng.uniform(size=5000) < 1 / 3).astype(float)
    perfect = m.categorical_scores(o, o)
    assert perfect["hss"] == pytest.approx(1.0)
    assert perfect["pss"] == pytest.approx(1.0)
    assert perfect["ets"] == pytest.approx(1.0)
    random = m.categorical_scores((rng.uniform(size=5000) < 1 / 3).astype(float), o)
    for k in ("hss", "pss", "ets"):
        assert abs(random[k]) < 0.05
    constant = m.categorical_scores(np.zeros_like(o), o)
    assert constant["pod"] == 0.0
    assert constant["hss"] == pytest.approx(0.0)


def test_tercile_events_respect_missing():
    lo, hi = np.array([1.0, 1.0]), np.array([2.0, 2.0])
    ev = m.tercile_events(np.array([0.5, np.nan]), lo, hi)
    assert ev["dry"][0] == 1.0 and ev["wet"][0] == 0.0
    assert np.isnan(ev["dry"][1])
