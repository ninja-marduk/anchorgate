import numpy as np
import pytest

from anchorgate.references import (
    build_references,
    climatology,
    climatology_table,
    persistence,
    seasonal_naive,
    training_terciles,
)
from conftest import LEADS, origins_after_cutoff


def test_climatology_uses_training_only(obs):
    table = climatology_table(obs)
    cal = obs.calendar[: obs.cutoff]
    jan = obs.training[cal == 1].mean(0)
    assert np.allclose(table[0], jan)
    # the held-out January values do not enter the table
    polluted = obs.series.copy()
    polluted[obs.cutoff :] += 1000
    from anchorgate import Observations

    o2 = Observations(
        series=polluted.reshape(obs.n_time, *obs.space_shape), time=obs.time, cutoff=obs.cutoff
    )
    assert np.allclose(climatology_table(o2), table)


def test_references_on_the_forecast_grid(obs):
    oi = origins_after_cutoff(obs)
    op = obs.period[oi]
    clim = climatology(obs, op, LEADS)
    pers = persistence(obs, op, LEADS)
    seas = seasonal_naive(obs, op, LEADS)
    assert clim.shape == pers.shape == seas.shape == (len(oi), len(LEADS), obs.n_cell)
    assert np.allclose(pers[:, 0], obs.series[oi])
    assert np.allclose(pers[:, 0], pers[:, -1])
    assert np.allclose(seas[:, 0], obs.series[oi + 1 - 12])
    table = climatology_table(obs)
    assert np.allclose(clim[0, 0], table[(op[0] + 1) % 12])


def test_build_references_rejects_unknown(obs):
    with pytest.raises(ValueError, match="unknown reference"):
        build_references(obs, obs.period[:2], LEADS, ("climatology", "oracle"))


def test_training_terciles_bracket_the_training_values(obs):
    lo, hi = training_terciles(obs)
    cal = obs.calendar[: obs.cutoff]
    jan = obs.training[cal == 1]
    frac_dry = (jan < lo[0]).mean(0)
    frac_wet = (jan > hi[0]).mean(0)
    assert np.all(np.abs(frac_dry - 1 / 3) < 0.1)
    assert np.all(np.abs(frac_wet - 1 / 3) < 0.1)


def test_reference_forecast_packages_a_reference(obs):
    from anchorgate import Benchmark, anchor, reference_forecast
    from anchorgate.synthetic import origins_after_cutoff

    leads = np.arange(1, 7)
    origins = origins_after_cutoff(obs, leads)
    pers = reference_forecast(obs, origins, leads, "persistence")
    seas = reference_forecast(obs, origins, leads, "seasonal_naive")
    assert pers.model == "persistence" and pers.seed is None
    assert pers.pred.shape == (origins.size, 6, obs.n_cell)
    bench = Benchmark([pers, seas], obs)
    bench.validate()
    res = anchor(bench, validate=False)
    v = res.values.set_index("label")
    # the packaged references score exactly as the built-in ones
    assert v.loc["persistence (single run)", "msess"] == pytest.approx(
        v.loc["persistence", "msess"]
    )
    assert v.loc["seasonal_naive (single run)", "msess"] == pytest.approx(
        v.loc["seasonal_naive", "msess"]
    )


def test_synthetic_autocorrelation_gives_lagged_information():
    from anchorgate.synthetic import synthetic_observations

    white = synthetic_observations(seed=3, ar=0.0)
    red = synthetic_observations(seed=3, ar=0.8)
    from anchorgate.references import climatology_table

    def lag1_corr(o):
        a = o.series - climatology_table(o)[o.calendar - 1]
        a = a[: o.cutoff]
        return np.corrcoef(a[1:].ravel(), a[:-1].ravel())[0, 1]

    assert lag1_corr(red) > 0.5 > lag1_corr(white) + 0.3
