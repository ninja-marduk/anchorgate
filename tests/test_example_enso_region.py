"""The regional ENSO example, offline: the origin grid, the scored block, the region
parser and the fitted runs on the synthetic field with a synthetic index."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anchorgate import Benchmark, Verdict, overlap, purged_holdout
from conftest import make_observations

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def example():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    return importlib.import_module("enso_ridge_region")


def test_region_parser(example):
    name, box, regime = example.parse_region("mantaro")
    assert name.startswith("Mantaro") and box == (-13.5, -10.5, -76.5, -73.5)
    assert regime == "unimodal tropical"
    name, box, _ = example.parse_region("-8,-6,106,109")
    assert box == (-8.0, -6.0, 106.0, 109.0) and "box" in name
    with pytest.raises(ValueError):
        example.parse_region("andes")  # Boyaca and Mantaro both match
    with pytest.raises(ValueError):
        example.parse_region("5,4,0,1")


def test_origin_grid_and_block(example):
    obs = make_observations(n_years=40, cutoff=32 * 12)
    origins = example.origin_grid(obs)
    leads = example.LEADS
    # every origin after the cutoff whose first target is not training and whose
    # twelfth target is inside the record, one per month
    assert origins[0] == obs.cutoff - 1 and origins[-1] == obs.n_time - 13
    assert np.all(np.diff(origins) == 1)
    assert np.array_equal(leads, np.arange(1, 13))
    block = example.scored_block(origins.size)
    assert block == list(range(origins.size - 12, origins.size))
    pu = purged_holdout(origins.size, leads, block)
    fit = pu.splits[0].fit
    # embargo of horizon - 1 origins before the block, nothing after it
    assert fit.max() == block[0] - 12 and "embargo 11" in pu.description
    assert int(overlap(pu, obs.period[origins], leads)["n_overlapping"].sum()) == 0
    with pytest.raises(ValueError):
        example.scored_block(11)


def test_declaration_design_hash_is_region_independent(example):
    obs = make_observations(n_years=40, cutoff=32 * 12)
    n = example.origin_grid(obs).size
    a = example.declaration("A", (0.0, 1.0, 0.0, 1.0), "x", obs, n)
    b = example.declaration("B", (5.0, 6.0, 5.0, 6.0), "y", obs, n)
    assert a.design_hash == b.design_hash and a.sha256 != b.sha256
    assert a.score_origins(n, 12).tolist() == example.scored_block(n)
    assert a.gate.n_boot == 1000 and a.gate.level == 0.9 and a.gate.block == 3
    assert a.contrasts == [["ENSO_LOCAL", "ENSO_ONLY"]]


def test_protocol_runs_offline_on_synthetic_field(example):
    obs = make_observations(n_years=40, shape=(3, 4), cutoff=32 * 12)
    rng = np.random.default_rng(0)
    index = pd.DatetimeIndex(obs.time.astype("datetime64[ns]"))
    nino = pd.Series(rng.normal(0, 1, obs.n_time).cumsum() * 0.1, index=index)
    origins = example.origin_grid(obs)
    runs = example.fit_runs(obs, nino, origins)
    assert len(runs) == 9 and {f.model for f in runs} == {"ridge", "damped_persistence"}
    bench = Benchmark(runs, obs)
    bench.validate()
    decl = example.declaration("synthetic", (0.0, 1.0, 0.0, 1.0), "none", obs, origins.size)
    results = example.run_protocol(bench, decl)
    assert [r.component[:2] for r in results] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    c0 = results[0].values.set_index("assignment")
    assert results[0].verdict == Verdict.PASS and c0.loc["purged", "fraction"] == 0.0
    assert c0.loc["shuffled", "fraction"] > 0.5
    c1 = results[1].tables["by_configuration"]
    assert set(c1["features"]) == {"ENSO_LOCAL", "ENSO_ONLY", "LOCAL"}
    assert (c1["n_seeds"] == 3).all()
    assert all(r.deviations == [] for r in results[1:])
    assert all(r.declaration_hash == decl.sha256 for r in results)
