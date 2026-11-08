"""Component 3 on synthetic ensembles with known decompositions."""

from __future__ import annotations

import numpy as np
import pytest

from anchorgate import (
    Benchmark,
    FoldAssignment,
    decompose_fusion,
    fusion_ladder,
    purged_holdout,
    resolve_assignment,
)
from anchorgate.folds import Split
from conftest import make_forecast, make_observations


def _bench(seeds=(1, 2), second="scaled"):
    obs = make_observations()
    a = [make_forecast(obs, "half", seed=s, model="half") for s in seeds]
    if second == "scaled":
        b = [
            make_forecast(obs, "custom", seed=s, model="scaled", transform=lambda t, c: 2 * t + 10)
            for s in seeds
        ]
    else:
        b = [make_forecast(obs, "noisy", seed=s, model="noisy") for s in seeds]
    return Benchmark(a + b, obs), obs


def test_pure_calibration_decomposes_to_zero_combination():
    bench, _ = _bench()
    res = decompose_fusion(bench, assignment="purged")
    v = res.values
    assert (v["raw: scaled[BASIC]"] < 0).all()  # unusable before recalibration
    assert v["best_recal"].to_numpy() == pytest.approx(1.0, abs=1e-9)
    assert v["combination_gain"].to_numpy() == pytest.approx(0.0, abs=1e-9)
    assert (v["calibration_gain"] > 0).all()
    assert res.verdict.value == "FAIL"  # nothing left for combination to add
    assert res.tables["assignment"]["overlap_fraction"].iloc[0] == 0.0


def test_combination_adds_when_errors_are_independent():
    obs = make_observations()
    rng = np.random.default_rng(5)
    fcs = []
    for s in (1, 2, 3):
        for name in ("u", "v"):
            noise_seed = s * 10 + (0 if name == "u" else 1)
            fcs.append(
                make_forecast(
                    obs,
                    "custom",
                    seed=s,
                    model=name,
                    transform=lambda t, c, k=noise_seed: (
                        t + np.random.default_rng(k).normal(0, 30, t.shape)
                    ),
                )
            )
    bench = Benchmark(fcs, obs)
    res = decompose_fusion(bench, assignment="purged")
    assert res.verdict.value == "PASS"
    assert (res.values["combination_gain"] > 0).all()
    assert res.values["n_splits"].iloc[0] == 1
    del rng


def test_overlapping_assignment_cannot_pass():
    bench, _ = _bench()
    res = decompose_fusion(bench, assignment="shuffled")
    assert res.verdict.value == "INSUFFICIENT"
    assert res.tables["assignment"]["overlap_fraction"].iloc[0] > 0
    assert any("not out of sample" in n for n in res.notes)


def test_single_run_is_insufficient_unless_allowed():
    bench, _ = _bench(seeds=(1,))
    assert decompose_fusion(bench, assignment="purged").verdict.value == "INSUFFICIENT"
    assert decompose_fusion(bench, assignment="purged", min_seeds=1).verdict.value in {
        "PASS",
        "FAIL",
    }


def test_supplied_ensemble_is_scored_for_reference():
    bench, obs = _bench()
    ens = [make_forecast(obs, "perfect", seed=s, model="ens") for s in (1, 2)]
    full = Benchmark(bench.forecasts + ens, obs)
    res = decompose_fusion(
        full,
        base=[("half", "BASIC"), ("scaled", "BASIC")],
        ensemble=("ens", "BASIC"),
        assignment="purged",
    )
    assert res.values["supplied_ensemble"].to_numpy() == pytest.approx(1.0)
    assert any("supplied ensemble" in n for n in res.notes)


def test_custom_assignment_and_scalar_rung():
    bench, _ = _bench()
    n = bench.n_origin
    fa = FoldAssignment(
        "half_split", n, [Split(fit=np.arange(0, n // 2), score=np.arange(n // 2 + 6, n))]
    )
    res = decompose_fusion(bench, assignment=fa, score_origins=range(n // 2 + 6, n))
    assert res.tables["assignment"]["name"].iloc[0] == "half_split"
    scalar = decompose_fusion(bench, assignment="released", score_origins=range(n - 6, n))
    assert scalar.values["n_splits"].iloc[0] == 5
    assert scalar.verdict.value == "INSUFFICIENT"
    with pytest.raises(ValueError):
        resolve_assignment(bench, "stratified")
    with pytest.raises(ValueError):
        decompose_fusion(bench, base=[("half", "BASIC")])
    with pytest.raises(ValueError):
        decompose_fusion(bench, base=[("half", "BASIC"), ("nope", "BASIC")])


def test_ladder_scores_every_rung_on_the_same_origins():
    bench, _ = _bench()
    block = list(range(bench.n_origin - 6, bench.n_origin))
    table, results = fusion_ladder(bench, score=block)
    assert table["rung"].tolist() == ["released", "shuffled", "contiguous", "purged"]
    assert table["overlap_fraction"].iloc[-1] == 0.0
    assert (table["overlap_fraction"].iloc[:-1] > 0).all()
    assert len(results) == 4
    assert all(r.component.startswith("C3") for r in results)
    purged = purged_holdout(bench.n_origin, bench.lead, block)
    assert results[-1].tables["assignment"]["name"].iloc[0] == purged.name
