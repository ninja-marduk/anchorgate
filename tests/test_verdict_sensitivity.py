"""The gate-constant grid of ``examples/verdict_sensitivity.py`` on the synthetic
benchmark: known answers do not depend on the constants, the bounds bracket the seed
mean, and the outputs are written."""

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


def test_grid_known_answers_and_outputs(benchmark, tmp_path):
    vs = load_example()
    grid = vs.settings()
    assert len(grid) == 11
    df = vs.run_grid([benchmark], score_origins=None, n_boot=200)
    assert list(df.columns) == vs.COLUMNS
    n_configs = df["model"].nunique()
    assert n_configs == 4
    assert len(df) == len(grid) * n_configs
    assert set(df["setting_kind"]) == {"block", "level", "threshold"}
    assert (df.loc[df["model"] == "perfect[BASIC]", "verdict"] == "PASS").all()
    assert (df.loc[df["model"] == "climatology[BASIC]", "verdict"] == "FAIL").all()
    assert (df.loc[df["model"] == "single[BASIC]", "verdict"] == "INSUFFICIENT").all()
    assert np.isfinite(df["lower"]).all() and np.isfinite(df["upper"]).all()
    assert (df["lower"] <= df["msess_mean"] + 1e-9).all()
    assert (df["msess_mean"] <= df["upper"] + 1e-9).all()
    assert (df.loc[df["model"] == "perfect[BASIC]", "skill_horizon"] == 6).all()
    csv, tex = vs.write_outputs(df, tmp_path / "report")
    assert csv.exists() and tex.exists()
    body = tex.read_text(encoding="utf-8")
    assert body.startswith("% Generated")
    assert "\\begin{tabular*}" in body and "\\end{tabular*}" in body
    assert body.count("\\\\") >= n_configs + 2
    assert "PASS" in body and "FAIL" in body and "INSUFFICIENT" in body
    text = vs.summary(df)
    assert "never move" in text
