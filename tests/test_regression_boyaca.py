"""Regression against the published Boyaca numbers.

Runs only when ``ANCHORGATE_THESIS_ROOT`` points at a checkout of the case-study
repository with its ``models/output`` arrays and the engineered dataset in place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from anchorgate import (
    Benchmark,
    Observations,
    anchor,
    audit,
    decompose_fusion,
    factorial,
    forecasts_from_directory,
    fusion_ladder,
    group_contrasts,
    holm,
    ladder,
)

ROOT = os.environ.get("ANCHORGATE_THESIS_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="ANCHORGATE_THESIS_ROOT not set")

DATASET = (
    "data/output/complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
CUTOFF_INDEX = 414


def _observations() -> Observations:
    import xarray as xr

    root = Path(ROOT or ".")
    with xr.open_dataset(root / DATASET) as ds:
        da = ds["total_precipitation"].load()
    return Observations.from_field(da, cutoff_time=da["time"].values[CUTOFF_INDEX])


@pytest.fixture(scope="module")
def boyaca() -> Benchmark:
    obs = _observations()
    out = Path(ROOT or ".") / "models" / "output"
    fcs = []
    fcs += forecasts_from_directory(
        out
        / "V2_Enhanced_Models/SEED*/map_exports/H12/BASIC/ConvLSTM_Bidirectional/predictions.npy",
        obs,
        model="ConvLSTM-Bidirectional",
        features="BASIC",
    )
    fcs += forecasts_from_directory(
        out / "V4_GNN_TAT_Models/SEED*/map_exports/H12/BASIC/GNN_TAT_GAT/predictions.npy",
        obs,
        model="GNN-TAT-GAT",
        features="BASIC",
    )
    fcs += forecasts_from_directory(
        out / "V10_Late_Fusion/SEED*/predictions.npy",
        obs,
        model="Late Fusion (released)",
        features="BASIC",
    )
    bench = Benchmark(fcs, obs)
    bench.validate(atol=1e-3)
    return bench


@pytest.fixture(scope="module")
def p30_factorial() -> Benchmark:
    """The retrained GNN-TAT factorial: 3 operators x 2 bundles x 3 seeds."""
    obs = _observations()
    out = Path(ROOT or ".") / "models" / "output" / "V4_GNN_TAT_Models_p30"
    fcs = []
    for bundle in ("BASIC", "PAFC"):
        for operator in ("GAT", "GCN", "SAGE"):
            fcs += forecasts_from_directory(
                out / f"SEED*/map_exports/H12/{bundle}/GNN_TAT_{operator}/predictions.npy",
                obs,
                model=operator,
                features=bundle,
            )
    bench = Benchmark(fcs, obs)
    bench.validate(atol=1e-3)
    return bench


def test_grid_and_origins(boyaca):
    f = boyaca.reference
    assert (f.n_origin, f.n_lead, f.n_cell) == (33, 12, 61 * 65)
    assert str(f.target_time[0, 0]) == "2021-07"
    assert boyaca.observations.cutoff == CUTOFF_INDEX


def test_climatology_anchor_numbers(boyaca):
    res = anchor(boyaca, validate=False)
    v = res.values.set_index("label")
    assert v.loc["climatology", "pooled_r2"] == pytest.approx(0.730, abs=0.002)
    assert v.loc["Late Fusion (released)[BASIC] seed 42", "pooled_r2"] == pytest.approx(
        0.672, abs=0.002
    )
    res_purged = anchor(boyaca, score_origins=range(21, 33), validate=False)
    vp = res_purged.values.set_index("label")
    assert vp.loc["climatology", "pooled_r2"] == pytest.approx(0.763, abs=0.002)
    assert res_purged.verdict.value == "FAIL"


def test_overlap_ladder(boyaca):
    res = audit(boyaca, ladder(boyaca, score=list(range(21, 33))), out_of_sample=["purged"])
    t = res.values.set_index("assignment")["fraction"]
    assert t["shuffled"] == pytest.approx(394 / 396)
    assert t["contiguous"] == pytest.approx(347 / 396)
    assert t["purged"] == 0.0
    assert t["released"] > 0.999
    assert res.tables["effective_units"]["n_disjoint_windows"].iloc[0] == 3


def test_seed_spread_on_the_purged_block(boyaca):
    res = anchor(boyaca, score_origins=range(21, 33), validate=False)
    cfg = res.tables["by_configuration"].set_index("model")
    assert cfg.loc["ConvLSTM-Bidirectional", "n_seeds"] == 3
    assert np.isfinite(cfg.loc["ConvLSTM-Bidirectional", "pooled_r2_sd"])
    assert (cfg["msess_mean"] < 0).all()


def test_factorial_reproduces_the_blocked_analysis(p30_factorial):
    res = factorial(p30_factorial, n_perm=20000, validate=False)
    assert len(res.values) == 18
    anova = res.tables["anova"].set_index("source")
    assert anova.loc["features", "F"] == pytest.approx(6.40, abs=0.01)
    assert anova.loc["features", "p"] == pytest.approx(0.0299, abs=0.0005)
    assert anova.loc["model", "p"] == pytest.approx(0.756, abs=0.001)
    assert anova.loc["residual", "df"] == 10
    perm = res.tables["permutation"].set_index("factor")
    assert perm.loc["features", "p"] == pytest.approx(0.0079, abs=0.003)
    assert perm.loc["model", "p"] == pytest.approx(0.835, abs=0.03)
    c = res.tables["contrasts"]
    top = c[(c["a"] == "GAT[PAFC]") & (c["b"] == "GCN[BASIC]")].iloc[0]
    assert top["mean_d"] == pytest.approx(0.094, abs=0.001)
    assert top["sd_d"] == pytest.approx(0.028, abs=0.001)
    assert top["p_holm"] == pytest.approx(0.419, abs=0.002)
    assert int(c["separates"].sum()) == 0 and int((c["p"] < 0.05).sum()) == 1
    assert res.verdict.value == "FAIL"
    assert res.tables["mde"]["median"].iloc[0] == pytest.approx(0.228, abs=0.001)
    cfg = res.tables["configurations"]
    assert cfg["inflation_peak"].median() == pytest.approx(0.045, abs=0.001)
    units = res.tables["effective_units"].iloc[0]
    assert (units["n_periods"], units["n_disjoint_windows"]) == (44, 3)
    fr = res.tables["friedman"].iloc[0]
    assert fr["cd_effective"] > fr["cd"]


def test_group_contrasts_reproduce_the_holm_family():
    """The five architecture contrasts of the published benchmark, configurations as units."""
    import pandas as pd

    root = Path(ROOT or ".")
    unified = pd.read_csv(root / "docs/models/comparative/data/unified_metrics_h12.csv")
    v4 = pd.read_csv(root / "models/output/V4_GNN_TAT_Models/metrics_spatial_v4_gnn_tat_h12.csv")
    rows = []
    for family, d in (("ConvLSTM", unified[unified.Version == "V2"]), ("GNN-TAT", v4)):
        for (_m, e), g in d.groupby(["Model", "Experiment"]):
            rows.append(
                {
                    "family": family,
                    "features": e,
                    "rmse_lead12": float(g[g.H == 12].RMSE.iloc[0]),
                    "rmse_mean": float(g.RMSE.mean()),
                }
            )
    t = pd.DataFrame(rows)
    pair = [("ConvLSTM", "GNN-TAT")]
    fam = [group_contrasts(t, "family", "rmse_lead12", pairs=pair)]
    fam += [
        group_contrasts(t[t.features == e], "family", "rmse_lead12", pairs=pair)
        for e in ("BASIC", "KCE", "PAFC")
    ]
    fam.append(group_contrasts(t, "family", "rmse_mean", pairs=[("GNN-TAT", "ConvLSTM")]))
    f = pd.concat(fam, ignore_index=True)
    assert f["U_a"].tolist() == pytest.approx([187.0, 16.0, 23.0, 23.0, 57.0])
    assert f["p"].iloc[0] == pytest.approx(0.0322, abs=0.0005)
    assert f["p"].iloc[-1] == pytest.approx(0.0153, abs=0.0005)
    adjusted = holm(f["p"].to_numpy())
    assert adjusted.min() == pytest.approx(0.077, abs=0.001)
    assert f["p_floor"].iloc[2] == pytest.approx(2 / 220)


def test_fusion_decomposition_on_the_purged_block(boyaca):
    base = [("ConvLSTM-Bidirectional", "BASIC"), ("GNN-TAT-GAT", "BASIC")]
    ens = ("Late Fusion (released)", "BASIC")
    res = decompose_fusion(
        boyaca,
        base=base,
        ensemble=ens,
        assignment="purged",
        score_origins=range(21, 33),
        validate=False,
    )
    v = res.values
    assert v["fusion"].mean() == pytest.approx(0.598, abs=0.001)
    assert v["fusion"].std(ddof=1) == pytest.approx(0.022, abs=0.001)
    assert v["best_raw"].mean() == pytest.approx(0.639, abs=0.001)
    assert v["best_recal"].mean() == pytest.approx(0.577, abs=0.001)
    assert v["total_gain"].mean() == pytest.approx(-0.041, abs=0.001)
    assert v["total_gain"].std(ddof=1) == pytest.approx(0.041, abs=0.001)
    assert v["in_sample_gap"].mean() == pytest.approx(0.084, abs=0.001)
    assert v["supplied_ensemble"].mean() == pytest.approx(0.669, abs=0.001)
    picks = dict(zip(v["seed"], v["picked_raw"], strict=True))
    assert picks[42].startswith("ConvLSTM") and picks[456].startswith("ConvLSTM")
    assert picks[123].startswith("GNN-TAT")
    assert res.verdict.value == "PASS"  # combination +0.021 over the recalibrated learner
    table, _ = fusion_ladder(boyaca, base=base, ensemble=ens, score=range(21, 33))
    rungs = table.set_index("rung")
    assert rungs.loc["purged", "overlap_fraction"] == 0.0
    assert rungs.loc["shuffled", "fusion"] == pytest.approx(0.665, abs=0.01)
    assert rungs.loc["released", "fusion"] == pytest.approx(0.668, abs=0.02)
    assert (rungs.drop("purged")["verdict"] == "INSUFFICIENT").all()
