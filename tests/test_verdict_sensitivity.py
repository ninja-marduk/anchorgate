"""The gate-constant grid of ``examples/verdict_sensitivity.py`` on the synthetic
benchmark: known answers do not depend on the constants, the bounds bracket the seed
mean, component 4 and the seed pairs are covered, and the outputs are written."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example() -> ModuleType:
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        "verdict_sensitivity", EXAMPLES / "verdict_sensitivity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_settings_grid_shape():
    vs = load_example()
    assert len(vs.settings()) == 11 + 8 + 1
    assert len(vs.settings([1, 2])) == 11 + 8 + 1
    assert len(vs.settings([1, 2, 3])) == 11 + 8 + 1 + 3 * 2
    kinds = {s.kind for s in vs.settings([1, 2, 3])}
    assert kinds == {"block", "level", "threshold", "min_seeds", "seeds"}


def test_grid_known_answers_and_outputs(benchmark, tmp_path):
    vs = load_example()
    seeds = sorted({f.seed for f in benchmark.forecasts if f.seed is not None})
    grid = vs.settings(seeds)
    df = vs.run_grid([benchmark], score_origins=None, n_boot=200)
    assert list(df.columns) == vs.COLUMNS
    n_configs = df["model"].nunique()
    assert n_configs == 4
    # the constants grid of component 1 with every seed; min_seeds is checked apart
    c1 = df[
        (df["component"] == "C1")
        & (df["seeds"] == vs.ALL_SEEDS)
        & (df["setting_kind"] != "min_seeds")
    ]
    c1_points = [
        s for s in grid if s.component == "C1" and s.seeds is None and s.kind != "min_seeds"
    ]
    assert len(c1) == len(c1_points) * n_configs
    assert set(df["component"]) == {"C1", "C4"}
    assert (c1.loc[c1["model"] == "perfect[BASIC]", "verdict"] == "PASS").all()
    assert (c1.loc[c1["model"] == "climatology[BASIC]", "verdict"] == "FAIL").all()
    assert (c1.loc[c1["model"] == "single[BASIC]", "verdict"] == "INSUFFICIENT").all()
    assert np.isfinite(c1["lower"]).all() and np.isfinite(c1["upper"]).all()
    assert (c1["lower"] <= c1["mean"] + 1e-9).all()
    assert (c1["mean"] <= c1["upper"] + 1e-9).all()
    assert (c1.loc[c1["model"] == "perfect[BASIC]", "skill_horizon"] == 6).all()
    # min_seeds 3 with three seeds leaves every component-1 verdict where it was
    base = c1[(c1["setting_kind"] == "block") & (c1["block"] == vs.BASE_BLOCK)].set_index("model")
    min3 = df[df["setting_kind"] == "min_seeds"].set_index("model")
    if len(seeds) >= 3:
        assert (min3["verdict"] == base.loc[min3.index, "verdict"]).all()
    c4 = df[df["component"] == "C4"]
    assert (c4["metric"] == "acc_cell_median").all()
    assert (c4["skill_horizon"] == -1).all()
    pairs = df[df["setting_kind"] == "seeds"]
    if len(seeds) >= 3:
        assert pairs["seeds"].nunique() == len(seeds) * (len(seeds) - 1) // 2
        assert set(pairs["component"]) == {"C1", "C4"}
    else:
        assert pairs.empty
    csv, tex = vs.write_outputs(df, tmp_path / "report")
    assert csv.exists() and tex.exists()
    body = tex.read_text(encoding="utf-8")
    assert body.startswith("% Generated")
    assert "\\begin{tabular*}" in body and "\\end{tabular*}" in body
    assert body.count("\\\\") >= n_configs + 2
    assert "PASS" in body and "FAIL" in body and "INSUFFICIENT" in body
    c4_tex = tmp_path / "report" / "verdict_sensitivity_c4_table.tex"
    assert c4_tex.exists() and "ACC" in c4_tex.read_text(encoding="utf-8")
    text = vs.summary(df)
    assert "never move" in text and "C4" in text
