import numpy as np
import pytest

from anchorgate import Benchmark, Declaration, Gate, Verdict, anchor, gate_table, quick_anchor
from conftest import make_forecast


def test_known_answers(benchmark):
    res = anchor(benchmark, by_lead=True, categorical=True)
    v = res.values.set_index("label")
    assert v.loc["climatology", "msess"] == pytest.approx(0.0)
    # per-cell NSE removes between-cell variance, not the seasonal cycle the climatology carries
    assert 0.0 < v.loc["climatology", "per_cell_nse"] < v.loc["climatology", "pooled_r2"]
    assert v.loc["perfect[BASIC] seed 1", "msess"] == pytest.approx(1.0)
    assert v.loc["perfect[BASIC] seed 1", "pooled_r2"] == pytest.approx(1.0)
    assert v.loc["half[BASIC] seed 1", "msess"] == pytest.approx(0.75)
    assert v.loc["climatology[BASIC] seed 1", "msess"] == pytest.approx(0.0)
    cfg = gate_table(res).set_index("model")
    assert cfg.loc["perfect", "verdict"] == "PASS"
    assert cfg.loc["half", "verdict"] == "PASS"
    assert cfg.loc["climatology", "verdict"] == "FAIL"
    assert cfg.loc["single", "verdict"] == "INSUFFICIENT"
    assert res.verdict == Verdict.PASS
    assert set(res.tables) == {
        "by_configuration",
        "by_lead",
        "by_lead_configuration",
        "categorical",
    }
    cat = res.tables["categorical"].set_index(["label", "event"])
    assert cat.loc[("perfect[BASIC] seed 1", "dry"), "hss"] == pytest.approx(1.0)
    assert abs(cat.loc[("climatology", "wet"), "hss"]) < 0.2


def test_gate_fails_when_nothing_beats_the_reference(obs):
    bench = Benchmark([make_forecast(obs, "climatology", seed=s) for s in (1, 2, 3)], obs)
    res = anchor(bench)
    assert res.verdict == Verdict.FAIL
    assert res.declaration_hash is None


def test_single_runs_cannot_pass(obs):
    bench = Benchmark([make_forecast(obs, "perfect", seed=None)], obs)
    assert anchor(bench).verdict == Verdict.INSUFFICIENT
    assert anchor(bench, gate=Gate(min_seeds=1)).verdict == Verdict.PASS


def test_scoring_block_mask_and_declaration(obs):
    bench = Benchmark([make_forecast(obs, "half", seed=s) for s in (1, 2)], obs)
    n = bench.n_origin
    decl = Declaration.from_dict(
        {
            "name": "t",
            "scoring": {"purged": {"start": n - 12, "stop": n}},
            "gate": {"metric": "msess", "threshold": 0.5, "min_seeds": 2},
        }
    )
    res = anchor(bench, decl, mask="domain")
    assert res.declaration_hash == decl.sha256
    assert res.deviations == []
    over = anchor(bench, decl, score_origins=range(0, 5), validate=False)
    assert any("score_origins" in d for d in over.deviations)
    assert over.parameters["score_origins"] == [0, 1, 2, 3, 4]
    assert int(res.values["n_origins"].iloc[0]) == 12
    assert int(res.values["n_cells"].iloc[0]) == obs.n_cell // 2
    assert res.verdict == Verdict.PASS
    strict = anchor(bench, decl, gate=Gate(threshold=0.9))
    assert strict.verdict == Verdict.FAIL
    with pytest.raises(ValueError):
        anchor(bench, score_origins=[n + 5])


def test_quick_anchor_single_model(obs):
    res = quick_anchor(make_forecast(obs, "half", seed=1), obs, gate=Gate(min_seeds=1))
    assert res.verdict == Verdict.PASS
    assert np.isfinite(res.values["pooled_r2"]).all()


def test_by_lead_table_is_consistent(benchmark):
    res = anchor(benchmark)
    bl = res.tables["by_lead"]
    perfect = bl[bl["label"] == "perfect[BASIC] seed 1"]
    assert len(perfect) == benchmark.reference.n_lead
    assert np.allclose(perfect["msess"], 1.0)
