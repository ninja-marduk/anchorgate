"""Component 2 on synthetic designs with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anchorgate import (
    Benchmark,
    cliffs_delta,
    factorial,
    friedman_nemenyi,
    group_contrasts,
    holm,
    inflation,
    minimum_detectable_effect,
    nemenyi_cd,
    paired_contrasts,
    permutation_test,
    rcbd_anova,
    responses,
)
from conftest import make_forecast, make_observations


def _crossed_benchmark(seeds=(1, 2, 3), feature_gain=0.3, model_gain=0.0):
    """2 models x 2 features x seeds; ``features`` carries a real effect, ``model`` not."""
    obs = make_observations()
    fcs = []
    for s in seeds:
        for feat, w in (("A", 0.5), ("B", 0.5 + feature_gain)):
            for i, model in enumerate(("m1", "m2")):

                def tr(t, c, w=w + i * model_gain, s=s, tag=feat + model):
                    rng = np.random.default_rng(s * 1000 + sum(map(ord, tag)))
                    return c + w * (t - c) + rng.normal(0, 5, t.shape)

                fcs.append(
                    make_forecast(obs, "custom", seed=s, model=model, features=feat, transform=tr)
                )
    fcs.append(make_forecast(obs, "noisy", seed=None, model="single", features="A"))
    return Benchmark(fcs, obs)


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.2]) == pytest.approx([0.2])
    assert (holm([0.5, 0.9]) <= 1).all()


def test_cliffs_delta_extremes():
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_nemenyi_cd_matches_the_published_table():
    # q_0.05 / sqrt(2) for k = 2..8 (Demsar 2006): 1.960 2.343 2.569 2.728 2.850 2.949 3.031
    for k, q in ((2, 1.960), (3, 2.343), (5, 2.728), (8, 3.031)):
        assert nemenyi_cd(k, 1) == pytest.approx(q * np.sqrt(k * (k + 1) / 6), abs=2e-3)
    assert nemenyi_cd(8, 33) == pytest.approx(3.031 * np.sqrt(72 / (6 * 33)), abs=2e-3)
    assert nemenyi_cd(8, 3) > nemenyi_cd(8, 33)


def test_minimum_detectable_effect_formula():
    out = minimum_detectable_effect([0.1, 0.2, 0.4], n=3)
    factor = 4.302652729911275 / np.sqrt(3)
    assert out["factor"] == pytest.approx(factor)
    assert out["median"] == pytest.approx(0.2 * factor)
    assert np.isnan(minimum_detectable_effect([0.1], n=1)["median"])


def test_rcbd_anova_recovers_a_planted_effect():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(4):
        block = rng.normal(0, 0.05)
        for f, f_eff in (("A", 0.0), ("B", 0.3)):
            for m in ("x", "y"):
                rows.append(
                    {
                        "seed": seed,
                        "features": f,
                        "model": m,
                        "y": block + f_eff + rng.normal(0, 0.01),
                    }
                )
    t = pd.DataFrame(rows)
    a = rcbd_anova(t, ("model", "features"), "seed").set_index("source")
    assert a.loc["features", "p"] < 1e-6
    assert a.loc["model", "p"] > 0.05
    assert a.loc["block (seed)", "ss"] > a.loc["residual", "ss"]  # the block absorbs the seed shift
    assert a.loc["residual", "df"] == 4 * 4 - 1 - 3 - 1 - 1 - 1
    assert a["ss"].drop(["total"]).sum() == pytest.approx(a.loc["total", "ss"])


def test_rcbd_anova_rejects_incomplete_designs():
    t = pd.DataFrame({"seed": [1, 1, 2], "model": ["a", "b", "a"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        rcbd_anova(t, ("model",), "seed")


def test_permutation_test_sees_a_real_effect_and_misses_a_null_one():
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(3):
        for f, f_eff in (("A", 0.0), ("B", 0.3)):
            for m in ("x", "y"):
                rows.append(
                    {"seed": seed, "features": f, "model": m, "y": f_eff + rng.normal(0, 0.02)}
                )
    t = pd.DataFrame(rows)
    assert permutation_test(t, "features", n_perm=2000, seed=0)["p"] < 0.05
    assert permutation_test(t, "model", n_perm=2000, seed=0)["p"] > 0.2


def test_paired_contrasts_use_the_paired_dispersion():
    rows = []
    for seed, shift in zip((1, 2, 3), (0.0, 0.2, 0.4), strict=True):  # a large common seed shift
        rows.append({"seed": seed, "model": "a", "features": "F", "y": shift + 0.10})
        rows.append({"seed": seed, "model": "b", "features": "F", "y": shift + 0.00})
    c = paired_contrasts(pd.DataFrame(rows), ("model", "features"), "seed").iloc[0]
    assert c["mean_d"] == pytest.approx(0.10)
    assert c["sd_d"] == pytest.approx(0.0)
    assert c["marginal_sd"] > 0.2  # what a naive comparison would use
    assert c["wins"] == 3
    assert c["cliffs_delta"] == pytest.approx(1 / 3)  # the arms overlap across seeds


def test_paired_contrasts_declared_family_and_single_seed():
    rows = [
        {"seed": 1, "model": "a", "features": "F", "y": 0.5},
        {"seed": 2, "model": "a", "features": "F", "y": 0.6},
        {"seed": 1, "model": "b", "features": "F", "y": 0.4},
        {"seed": 2, "model": "b", "features": "F", "y": 0.5},
        {"seed": 1, "model": "c", "features": "F", "y": 0.9},
    ]
    t = pd.DataFrame(rows)
    c = paired_contrasts(
        t, ("model", "features"), "seed", family=[["a[F]", "b[F]"], ["a[F]", "c[F]"]]
    )
    assert len(c) == 2
    single = c[c["b"] == "c[F]"].iloc[0]
    assert single["n"] == 1 and np.isnan(single["p"])
    with pytest.raises(ValueError):
        paired_contrasts(t, ("model", "features"), "seed", family=[["a[F]", "zz"]])


def test_inflation_is_best_minus_mean():
    t = pd.DataFrame(
        {"seed": [1, 2, 3], "model": ["m"] * 3, "features": ["F"] * 3, "y": [0.5, 0.6, 0.7]}
    )
    r = inflation(t).iloc[0]
    assert r["inflation"] == pytest.approx(0.1)
    assert r["best"] == pytest.approx(0.7)
    low = inflation(t, higher_is_better=False).iloc[0]
    assert low["best"] == pytest.approx(0.5) and low["inflation"] == pytest.approx(0.1)


def test_group_contrasts_mann_whitney():
    t = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 4, "y": [1, 2, 3, 4, 5, 10, 11, 12, 13]})
    g = group_contrasts(t, "group", "y").iloc[0]
    assert g["cliffs_delta"] == -1.0
    assert g["p"] < 0.05
    assert g["p_floor"] == pytest.approx(2 / 126)  # C(9, 5) = 126
    assert g["U_a"] + g["U_b"] == 20


def test_friedman_nemenyi_ranks_a_clear_winner():
    rng = np.random.default_rng(2)
    base = rng.normal(100, 10, (30, 1))
    m = np.hstack([base - 5, base, base + 5]) + rng.normal(0, 0.5, (30, 3))
    out = friedman_nemenyi(m, ["good", "mid", "bad"], n_effective=2)
    assert out["ranks"]["treatment"].tolist() == ["good", "mid", "bad"]
    assert out["test"]["p"].iloc[0] < 1e-6
    pairs = out["pairs"].set_index(["a", "b"])
    assert bool(pairs.loc[("good", "bad"), "beyond_cd"])
    assert not bool(pairs.loc[("good", "bad"), "beyond_cd_effective"])  # two units cannot


def test_responses_aggregates():
    b = _crossed_benchmark(seeds=(1, 2))
    mean = responses(b, "pooled_r2", aggregate="lead_mean")
    peak = responses(b, "pooled_r2", aggregate="lead_peak")
    pooled = responses(b, "pooled_r2", aggregate="pooled")
    assert len(mean) == len(peak) == len(pooled) == len(b.forecasts)
    assert (peak["y"] >= mean["y"] - 1e-12).all()
    low = responses(b, "rmse", aggregate="lead_peak")
    assert (low["y"] <= responses(b, "rmse", aggregate="lead_mean")["y"] + 1e-12).all()
    with pytest.raises(ValueError):
        responses(b, "pooled_r2", aggregate="best")


def test_factorial_component_on_a_crossed_design():
    b = _crossed_benchmark()
    res = factorial(b, n_perm=1000)
    assert res.verdict.value == "PASS"
    anova = res.tables["anova"].set_index("source")
    assert anova.loc["features", "p"] < 0.01
    assert anova.loc["model", "p"] > 0.05
    perm = res.tables["permutation"].set_index("factor")
    assert perm.loc["features", "p"] < 0.05
    cfg = res.tables["configurations"].set_index("configuration")
    assert cfg.loc["single[A]", "n_seeds"] == 1
    assert any("single[A]" in n for n in res.notes)
    c = res.tables["contrasts"]
    assert len(c) == 6  # four multi-seed configurations
    assert c[(c["a"] == "m1[A]") & (c["b"] == "m1[B]")]["separates"].iloc[0]
    assert "mde" in res.tables and "friedman" in res.tables
    assert res.tables["effective_units"]["n_disjoint_windows"].iloc[0] >= 1


def test_factorial_unresolved_design_fails_and_reports_mde():
    b = _crossed_benchmark(seeds=(1, 2), feature_gain=0.0)  # a null design
    res = factorial(b, n_perm=500, friedman=False)
    assert res.verdict.value == "FAIL"
    assert res.tables["contrasts"]["p_holm"].min() > 0.05
    assert "mde" in res.tables
    assert any("minimum detectable effect" in n for n in res.notes)


def test_factorial_declared_family():
    from anchorgate import Declaration

    b = _crossed_benchmark()
    decl = Declaration.from_dict(
        {"name": "t", "contrasts": [["m1[A]", "m1[B]"]], "factorial": {"n_perm": 300}}
    )
    res = factorial(b, decl)
    assert len(res.tables["contrasts"]) == 1
    assert res.declaration_hash == decl.sha256
