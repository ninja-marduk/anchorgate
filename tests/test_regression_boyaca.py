"""Regression against the published Boyaca numbers.

Runs only when ``ANCHORGATE_THESIS_ROOT`` points at a checkout of the case-study
repository with its ``models/output`` arrays and the engineered dataset in place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from anchorgate import Benchmark, Observations, anchor, audit, forecasts_from_directory, ladder

ROOT = os.environ.get("ANCHORGATE_THESIS_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="ANCHORGATE_THESIS_ROOT not set")

DATASET = (
    "data/output/complete_dataset_with_features_with_clusters_elevation_windows_"
    "imfs_with_onehot_elevation_clean.nc"
)
CUTOFF_INDEX = 414


@pytest.fixture(scope="module")
def boyaca() -> Benchmark:
    import xarray as xr

    root = Path(ROOT or ".")
    with xr.open_dataset(root / DATASET) as ds:
        da = ds["total_precipitation"].load()
    obs = Observations.from_field(da, cutoff_time=da["time"].values[CUTOFF_INDEX])
    out = root / "models" / "output"
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
