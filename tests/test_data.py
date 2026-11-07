import numpy as np
import pandas as pd
import pytest

from anchorgate import (
    Benchmark,
    Forecast,
    Observations,
    forecasts_from_directory,
    infer_origin_time,
)
from anchorgate._time import period_of_year, period_to_datetime, to_period
from conftest import LEADS, make_forecast, make_observations, origins_after_cutoff, targets_for


def test_period_roundtrip():
    t = np.array(["1982-01", "1999-12", "2025-02"], dtype="datetime64[M]")
    p = to_period(t)
    assert np.array_equal(period_to_datetime(p), t)
    assert np.array_equal(period_of_year(p), [1, 12, 2])
    assert p[1] - p[0] == 17 * 12 + 11


def test_observations_contract(obs):
    assert obs.n_cell == 12 and obs.space_shape == (3, 4)
    assert obs.cutoff_period == obs.period[obs.cutoff]
    assert np.allclose(obs.at(obs.period[:3]), obs.series[:3])
    assert np.isnan(obs.at(np.array([obs.period[0] - 1]))).all()
    with pytest.raises(ValueError):
        Observations(series=obs.series, time=obs.time, cutoff=0)


def test_forecast_target_periods(obs):
    f = make_forecast(obs, "perfect")
    assert f.n_lead == len(LEADS) and f.horizon == 6
    assert np.array_equal(f.target_period[0], f.origin_period[0] + LEADS)
    assert f.label.endswith("seed 1")
    assert make_forecast(obs, "perfect", seed=None).label.endswith("(single run)")


def test_benchmark_validates_identical_targets(obs):
    a = make_forecast(obs, "perfect", seed=1)
    b = make_forecast(obs, "half", seed=1)
    Benchmark([a, b], obs).validate()
    bad = make_forecast(obs, "half", seed=2)
    bad.target[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="targets differ"):
        Benchmark([a, bad], obs).validate()


def test_benchmark_rejects_targets_in_training_period(obs):
    oi = np.arange(10, 20)
    f = Forecast(
        pred=targets_for(obs, oi), target=targets_for(obs, oi), origin_time=obs.time[oi], lead=LEADS
    )
    with pytest.raises(ValueError, match="training period"):
        Benchmark([f], obs).validate()


def test_benchmark_detects_wrong_origin_times(obs):
    f = make_forecast(obs, "perfect")
    shifted = Forecast(pred=f.pred, target=f.target, origin_time=f.origin_time + 1, lead=f.lead)
    with pytest.raises(ValueError, match="do not match the observations"):
        Benchmark([shifted], obs).validate()


def test_benchmark_notes_duplicates_and_mixed_seeds(obs):
    a = make_forecast(obs, "perfect", seed=1)
    b = make_forecast(obs, "perfect", seed=2)
    c = make_forecast(obs, "perfect", seed=None)
    notes = Benchmark([a, b, c], obs).validate()
    assert any("duplicate" in n for n in notes)
    assert any("mixes labelled seeds" in n for n in notes)


def test_long_table_roundtrip(obs):
    f = make_forecast(obs, "half", seed=7)
    rows = []
    for o in range(f.n_origin):
        for h in range(f.n_lead):
            for c in range(f.n_cell):
                rows.append(
                    {
                        "origin_time": str(f.origin_time[o]),
                        "lead": int(f.lead[h]),
                        "cell": c,
                        "pred": f.pred[o, h, c],
                        "target": f.target[o, h, c],
                        "model": "half",
                        "features": "BASIC",
                        "seed": 7,
                    }
                )
    g = Forecast.from_long_table(pd.DataFrame(rows))
    assert np.allclose(g.pred, f.pred) and np.allclose(g.target, f.target)
    assert g.seed == 7 and g.model == "half"
    Benchmark([g], obs).validate()


def test_netcdf_roundtrip(obs, tmp_path):
    f = make_forecast(obs, "noisy", seed=3)
    f.to_netcdf(tmp_path / "f.nc")
    obs.to_netcdf(tmp_path / "o.nc")
    g = Forecast.from_netcdf(tmp_path / "f.nc")
    o2 = Observations.from_netcdf(tmp_path / "o.nc")
    assert np.allclose(g.pred, f.pred) and g.seed == 3 and g.space_shape == f.space_shape
    assert o2.cutoff == obs.cutoff and np.array_equal(o2.mask, obs.mask)
    Benchmark([g], o2).validate()


def test_infer_origin_time_and_directory_loader(obs, tmp_path):
    oi = origins_after_cutoff(obs)[5:25]
    f = make_forecast(obs, "half", origin_idx=oi)
    assert np.array_equal(infer_origin_time(f.target, obs, LEADS), f.origin_time)
    for seed in (42, 123):
        d = tmp_path / f"SEED{seed}" / "map_exports"
        d.mkdir(parents=True)
        np.save(d / "predictions.npy", f.pred.reshape(-1, 6, 3, 4, 1).astype(np.float32))
        np.save(d / "targets.npy", f.target.reshape(-1, 6, 3, 4, 1).astype(np.float32))
    fcs = forecasts_from_directory(
        tmp_path / "SEED*" / "map_exports" / "predictions.npy", obs, model="half", features="BASIC"
    )
    assert [x.seed for x in fcs] == [123, 42]
    assert fcs[0].space_shape == (3, 4)
    Benchmark(fcs, obs).validate(atol=1e-3)


def test_from_field_builds_grid_coordinates():
    import xarray as xr

    o = make_observations()
    da = xr.DataArray(
        o.series.reshape(o.n_time, 3, 4),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": o.time.astype("datetime64[ns]"),
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [10.0, 11.0, 12.0, 13.0],
        },
        name="precip",
    )
    o2 = Observations.from_field(da, cutoff_time=o.time[o.cutoff])
    assert o2.cutoff == o.cutoff and o2.lat is not None and o2.lat[4] == 2.0 and o2.lon[4] == 10.0
