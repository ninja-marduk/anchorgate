import numpy as np

from anchorgate import (
    Benchmark,
    Verdict,
    audit,
    contiguous_kfold,
    effective_units,
    ladder,
    overlap,
    purged_holdout,
    released_scalar_shuffle,
    scalar_shuffle_overlap,
    shuffled_kfold,
)
from conftest import make_forecast, make_observations, origins_after_cutoff

LEAD12 = np.arange(1, 13)


def sliding_origins(n: int = 33) -> np.ndarray:
    return np.arange(1000, 1000 + n)


def test_thesis_ladder_numbers_are_reproduced():
    """33 origins sliding by one month, 12 leads, 5 folds, seed 42: the published audit."""
    op = sliding_origins()
    sh = overlap(shuffled_kfold(33, k=5, seed=42), op, LEAD12)
    assert int(sh["n_overlapping"].sum()) == 394 and int(sh["n_pairs"].sum()) == 396
    co = overlap(contiguous_kfold(33, k=5), op, LEAD12)
    assert int(co["n_overlapping"].sum()) == 347
    pu = overlap(purged_holdout(33, LEAD12, range(21, 33)), op, LEAD12)
    assert int(pu["n_overlapping"].sum()) == 0
    assert np.array_equal(purged_holdout(33, LEAD12, range(21, 33)).splits[0].fit, np.arange(0, 10))


def test_purged_holdout_two_sided_embargo():
    a = purged_holdout(60, LEAD12, range(24, 36))
    fit = a.splits[0].fit
    # targets of the block: 25..47; fit origins need o+12 < 25 (o < 13) or o+1 > 47 (o > 46)
    assert fit.min() == 0 and 12 in fit and 13 not in fit
    assert 46 not in fit and 47 in fit and fit.max() == 59
    assert int(overlap(a, sliding_origins(60), LEAD12)["n_overlapping"].sum()) == 0


def test_scalar_shuffle_is_fully_in_sample():
    frac = scalar_shuffle_overlap(sliding_origins(), LEAD12, n_cell=50, k=5, seed=42)
    assert frac > 0.999


def test_effective_units():
    u = effective_units(sliding_origins(), LEAD12)
    assert u["n_pairs"] == 396 and u["n_periods"] == 44
    assert u["multiplicity_min"] == 1 and u["multiplicity_max"] == 12
    assert u["n_disjoint_windows"] == 3


def test_audit_verdicts(obs):
    bench = Benchmark([make_forecast(obs, "perfect")], obs)
    n = bench.n_origin
    rungs = ladder(bench, score=list(range(n - 12, n)))
    res = audit(bench, rungs, out_of_sample=["purged"])
    assert res.verdict == Verdict.PASS
    table = res.values.set_index("assignment")
    assert table.loc["purged", "fraction"] == 0.0
    assert table.loc["shuffled", "fraction"] > 0.5
    assert table.loc["released", "fraction"] > 0.99
    res_bad = audit(bench, [shuffled_kfold(n, name="shuffled")], out_of_sample=["shuffled"])
    assert res_bad.verdict == Verdict.FAIL
    res_na = audit(bench, [released_scalar_shuffle(n)])
    assert res_na.verdict == Verdict.NOT_APPLICABLE


def test_audit_on_sparse_origins_has_no_overlap():
    o = make_observations()
    oi = origins_after_cutoff(o)[::12][:4]  # one origin per year, leads 1..6
    bench = Benchmark([make_forecast(o, "perfect", origin_idx=oi)], o)
    res = audit(bench, [shuffled_kfold(4, k=2, name="shuffled")], out_of_sample=["shuffled"])
    assert res.verdict == Verdict.PASS
