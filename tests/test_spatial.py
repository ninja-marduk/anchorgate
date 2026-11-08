"""Component 5 on synthetic fields with known winners and known collapse."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from anchorgate import (
    Benchmark,
    Declaration,
    collapse_diagnostic,
    correlation_leakage,
    graph_audit,
    skill_by_strata,
    spatial_diagnostics,
    spatial_from_declaration,
    strata_bands,
    strata_from_declaration,
    winner_map,
)
from conftest import make_forecast


def _cell_mean_forecast(obs, seed=1):
    cm = np.nanmean(obs.series[: obs.cutoff], axis=0)
    return make_forecast(
        obs,
        "custom",
        seed=seed,
        model="cellmean",
        transform=lambda t, c: np.broadcast_to(cm, t.shape).copy(),
    )


def _grid_edges(shape):
    """Four-neighbour edge list on a (ny, nx) grid, both directions."""
    ny, nx = shape
    idx = np.arange(ny * nx).reshape(shape)
    pairs = []
    for y in range(ny):
        for x in range(nx):
            if x + 1 < nx:
                pairs += [(idx[y, x], idx[y, x + 1]), (idx[y, x + 1], idx[y, x])]
            if y + 1 < ny:
                pairs += [(idx[y, x], idx[y + 1, x]), (idx[y + 1, x], idx[y, x])]
    return np.asarray(pairs).T


def test_strata_bands():
    v = np.array([100.0, 1600.0, 3000.0, np.nan])
    lab = strata_bands(v, [1500, 2800], ["low", "mid", "high"])
    assert lab.tolist() == ["low", "mid", "high", ""]
    auto = strata_bands(v, [1500])
    assert auto[0] == "-inf..1500" and auto[1] == "1500..inf"
    with pytest.raises(ValueError):
        strata_bands(v, [1500], ["only"])


def test_winner_map_and_strata(obs):
    fcs = [make_forecast(obs, "half", seed=s) for s in (1, 2)]
    bench = Benchmark(fcs, obs)
    share, win, names = winner_map(bench)
    assert share.iloc[0]["arm"] == "half[BASIC]"
    assert share.iloc[0]["share"] == pytest.approx(1.0)
    assert (win == names.index("half[BASIC]")).all()
    labels = np.array(["a"] * 6 + ["b"] * 6, dtype=object)
    t = skill_by_strata(bench, labels, stratum_name="half_of_grid")
    assert set(t["half_of_grid"]) == {"a", "b"}
    model = t[t["arm"] == "half[BASIC]"]
    assert model["msess"].to_numpy() == pytest.approx(0.75)
    assert model["n_seeds"].tolist() == [2, 2]
    assert t[t["arm"] == "climatology"]["n_cells"].sum() == 12
    with pytest.raises(ValueError):
        skill_by_strata(bench, labels[:5])


def test_collapse_diagnostic_tells_a_cell_mean_from_a_forecast(obs):
    bench = Benchmark([make_forecast(obs, "perfect", seed=1), _cell_mean_forecast(obs)], obs)
    c = collapse_diagnostic(bench).set_index("label")
    cm = c.loc["cellmean[BASIC] seed 1"]
    assert cm["sd_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert cm["r2_vs_cell_mean"] == pytest.approx(1.0)
    perfect = c.loc["perfect[BASIC] seed 1"]
    assert perfect["sd_ratio"] == pytest.approx(1.0)
    assert perfect["r2_vs_cell_mean"] < 0.9
    assert c.loc["climatology", "sd_ratio"] < 1.0
    assert c.loc["cell_mean", "sd_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_graph_audit_on_a_grid(obs):
    edges = _grid_edges(obs.space_shape)
    g = graph_audit(
        edges,
        obs.lat if obs.lat is not None else np.repeat(np.arange(3), 4) * 0.05,
        obs.lon if obs.lon is not None else np.tile(np.arange(4), 3) * 0.05,
    )
    s = g["summary"].iloc[0]
    assert s["n_edges"] == edges.shape[1]
    assert s["isolated_nodes"] == 0
    assert s["share_under_50km"] == 1.0
    assert "length" in g
    dense = np.zeros((12, 12))
    dense[edges[0], edges[1]] = 1.0
    g2 = graph_audit(
        dense,
        np.repeat(np.arange(3), 4) * 0.05,
        np.tile(np.arange(4), 3) * 0.05,
        elevation=np.arange(12) * 100.0,
    )
    assert "elevation_contrast" in g2 and "weight_share_under_25km" in g2["summary"]
    with pytest.raises(ValueError):
        graph_audit(np.array([[0, 99]]).T, np.zeros(12), np.zeros(12))


def test_correlation_leakage_runs_on_synthetic_observations(obs):
    t = correlation_leakage(obs, n_cells=12).iloc[0]
    assert t["n_cells_sampled"] == 12
    assert 0 <= t["disagreement_share"] <= 1
    assert t["n_steps_training"] == obs.cutoff


def test_spatial_diagnostics_verdicts(obs):
    good = Benchmark([make_forecast(obs, "half", seed=s) for s in (1, 2)], obs)
    res = spatial_diagnostics(good, strata={"band": np.array(["a"] * 6 + ["b"] * 6, dtype=object)})
    assert res.verdict.value == "PASS"
    assert "strata_band" in res.tables and "collapse" in res.tables
    assert len(res.tables["winner_cells"]) == 12
    bad = Benchmark(
        [make_forecast(obs, "noisy", seed=s) for s in (1, 2)] + [_cell_mean_forecast(obs, 3)], obs
    )
    res2 = spatial_diagnostics(
        bad,
        adjacency=_grid_edges(obs.space_shape),
        lat=np.repeat(np.arange(3), 4) * 0.05,
        lon=np.tile(np.arange(4), 3) * 0.05,
        leakage=True,
    )
    assert res2.verdict.value == "FAIL"
    assert any("collapsed" in n for n in res2.notes)
    assert "graph_summary" in res2.tables and "leakage" in res2.tables


def test_strata_from_declaration(obs, tmp_path):
    obs.to_netcdf(tmp_path / "obs.nc")
    elev = xr.DataArray(
        np.arange(12, dtype=float).reshape(obs.space_shape) * 300, dims=("y", "x"), name="elevation"
    )
    elev.to_dataset().to_netcdf(tmp_path / "static.nc")
    np.save(tmp_path / "labels.npy", np.array(["n"] * 6 + ["s"] * 6))
    decl = Declaration.from_dict(
        {
            "name": "strata",
            "data": {"observations": {"format": "netcdf", "path": "obs.nc"}},
            "strata": {
                "elevation": {
                    "path": "static.nc",
                    "variable": "elevation",
                    "edges": [1500, 2800],
                    "labels": ["low", "mid", "high"],
                },
                "region": {"path": "labels.npy"},
            },
        }
    )
    bench = Benchmark([make_forecast(obs, "half", seed=s) for s in (1, 2)], obs)
    strata = strata_from_declaration(decl, bench, tmp_path)
    assert set(strata) == {"elevation", "region"}
    assert sorted(set(strata["elevation"])) == ["high", "low", "mid"]
    res = spatial_from_declaration(bench, decl, tmp_path)
    assert "strata_elevation" in res.tables and "strata_region" in res.tables
