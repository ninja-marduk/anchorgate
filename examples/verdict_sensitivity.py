"""How much of a component-1 verdict is owed to the constants of the gate?

The gate reads PASS, FAIL or INSUFFICIENT from a circular moving-block bootstrap over
the scored origins. On the purged block of the Boyaca case study there are twelve
origins, so blocks of three leave four blocks behind every interval, and a referee may
ask whether a different block length, interval level or threshold would move a verdict.
This example answers by recomputing the verdict of six configurations on that block,
origins 21 to 32, under a grid of gate settings:

* block length 1, 2, 3, 4 and 6 origins at level 0.90 and threshold 0;
* level 0.80, 0.90 and 0.95 at block 3 and threshold 0;
* threshold 0.00, 0.02 and 0.05 at block 3 and level 0.90;

with 1000 resamples everywhere and the same bootstrap seed, so two settings differ only
by the constant under study. The configurations are the three deposited ones
(ConvLSTM-Bidirectional, GNN-TAT-GAT and the released late fusion, read through the
declaration of ``boyaca_case_study``) and the three statistical ones of ``enso_ridge.py``
(ridge on Nino 3.4 lags, ridge on Nino 3.4 and local lags, damped persistence), refitted
here with the functions of that example so the forecasts are the ones it produces. The
Nino 3.4 table is downloaded on first use into ``<out>/data``.

    python examples/verdict_sensitivity.py --thesis-root /path/to/deposit --out report_sensitivity/

The root is also read from ``ANCHORGATE_THESIS_ROOT``. Outputs: ``verdict_sensitivity.csv``
(one row per setting and configuration), ``verdict_sensitivity_table.tex`` (a compact
``tabular*`` body for the manuscript) and a printed summary of which verdicts move.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anchorgate import Benchmark, Declaration, Gate, anchor
from anchorgate.data import benchmark_from_declaration

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import enso_ridge  # noqa: E402

TEMPLATE = HERE / "boyaca_case_study" / "declaration.yaml"
N_BOOT = 1000
BLOCKS = (1, 2, 3, 4, 6)
LEVELS = (0.80, 0.90, 0.95)
THRESHOLDS = (0.00, 0.02, 0.05)
BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD = 3, 0.90, 0.0
TABLE_BLOCKS = (1, 3, 6)
TABLE_LEVELS = (0.80, 0.95)
TABLE_THRESHOLDS = (0.02, 0.05)
COLUMNS = [
    "setting_kind",
    "block",
    "level",
    "threshold",
    "model",
    "msess_mean",
    "lower",
    "upper",
    "verdict",
    "skill_horizon",
]
# configuration key (model[features]) -> name printed in the table, in table order
DISPLAY = {
    "ConvLSTM-Bidirectional[BASIC]": "ConvLSTM-Bidirectional",
    "GNN-TAT-GAT[BASIC]": "GNN-TAT-GAT",
    "Late-Fusion-released[BASIC]": "Late fusion (released)",
    "ridge[ENSO_ONLY]": "Ridge, ENSO only",
    "ridge[ENSO_LOCAL]": "Ridge, ENSO and local",
    "damped_persistence[LOCAL]": "Damped persistence",
}


@dataclass(frozen=True)
class Setting:
    """One point of the grid: which constant is varied, and the three gate constants."""

    kind: str
    block: int
    level: float
    threshold: float

    def gate(self, template: Gate, n_boot: int) -> Gate:
        return Gate(
            metric=template.metric,
            threshold=self.threshold,
            min_seeds=template.min_seeds,
            reference=template.reference,
            level=self.level,
            block=self.block,
            n_boot=n_boot,
        )

    @property
    def label(self) -> str:
        return f"{self.kind} b={self.block} L={self.level:.2f} t={self.threshold:.2f}"


def settings() -> list[Setting]:
    """The eleven grid points; the base setting appears once under every kind."""
    out = [Setting("block", b, BASE_LEVEL, BASE_THRESHOLD) for b in BLOCKS]
    out += [Setting("level", BASE_BLOCK, lv, BASE_THRESHOLD) for lv in LEVELS]
    out += [Setting("threshold", BASE_BLOCK, BASE_LEVEL, th) for th in THRESHOLDS]
    return out


# ------------------------------------------------------------------------------ grid


def run_grid(
    benchmarks: Sequence[Benchmark],
    *,
    score_origins: Sequence[int] | None,
    n_boot: int = N_BOOT,
    gate: Gate | None = None,
    grid: Sequence[Setting] | None = None,
) -> pd.DataFrame:
    """Component-1 verdicts of every configuration of ``benchmarks`` under every setting.

    The benchmarks are scored separately (they need not share targets) and their
    per-configuration tables are stacked. ``gate`` supplies the metric, the reference
    and ``min_seeds``; the grid supplies the rest. The bootstrap seed is the one
    ``anchor`` always uses, so two settings differ only by the constant under study.
    """
    template = gate or Gate()
    rows: list[dict[str, object]] = []
    for s in grid if grid is not None else settings():
        g = s.gate(template, n_boot)
        for bench in benchmarks:
            res = anchor(bench, score_origins=score_origins, gate=g, by_lead=False, validate=False)
            for _, r in res.tables["by_configuration"].iterrows():
                rows.append(
                    {
                        "setting_kind": s.kind,
                        "block": s.block,
                        "level": s.level,
                        "threshold": s.threshold,
                        "model": f"{r['model']}[{r['features']}]",
                        "msess_mean": float(r[f"{g.metric}_mean"]),
                        "lower": float(r[f"{g.metric}_lower"]),
                        "upper": float(r[f"{g.metric}_upper"]),
                        "verdict": str(r["verdict"]),
                        "skill_horizon": int(r["skill_horizon"]),
                    }
                )
    return pd.DataFrame(rows, columns=COLUMNS)


def _pick(df: pd.DataFrame, kind: str, block: int, level: float, threshold: float) -> pd.DataFrame:
    m = (
        (df["setting_kind"] == kind)
        & (df["block"] == block)
        & np.isclose(df["level"], level)
        & np.isclose(df["threshold"], threshold)
    )
    return df[m].set_index("model")


def _num(v: float) -> str:
    if not np.isfinite(v):
        return "--"
    return f"{v:.3f}".replace("-", "$-$")


def _tex(s: str) -> str:
    return s.replace("_", r"\_")


def configuration_order(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [k for k in DISPLAY if k in present] + sorted(present - set(DISPLAY))


def latex_table(df: pd.DataFrame) -> str:
    """A ``tabular*`` body: point estimate, then lower bound and verdict at blocks 1/3/6
    (level 0.90), at levels 0.80/0.95 (block 3), and the verdict at thresholds 0.02 and
    0.05 (block 3, level 0.90). Minus signs are ``$-$``; verdicts are capitals."""
    base = _pick(df, "block", BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD)
    by_block = {b: _pick(df, "block", b, BASE_LEVEL, BASE_THRESHOLD) for b in TABLE_BLOCKS}
    by_level = {lv: _pick(df, "level", BASE_BLOCK, lv, BASE_THRESHOLD) for lv in TABLE_LEVELS}
    by_thr = {th: _pick(df, "threshold", BASE_BLOCK, BASE_LEVEL, th) for th in TABLE_THRESHOLDS}
    n_pairs = len(TABLE_BLOCKS) + len(TABLE_LEVELS)
    spec = "@{\\extracolsep{\\fill}}l r" + " rl" * n_pairs + " l" * len(TABLE_THRESHOLDS) + "@{}"
    c0 = 3
    c1 = c0 + 2 * len(TABLE_BLOCKS) - 1
    c2 = c1 + 1
    c3 = c2 + 2 * len(TABLE_LEVELS) - 1
    c4 = c3 + 1
    c5 = c4 + len(TABLE_THRESHOLDS) - 1
    lines = [
        "% Generated by examples/verdict_sensitivity.py. Each paired column holds the lower",
        "% bound of the bootstrap interval of the seed-mean MSESS and the verdict it yields.",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{spec}}}",
        "\\toprule",
        " & & "
        f"\\multicolumn{{{2 * len(TABLE_BLOCKS)}}}{{c}}{{Block length (level {BASE_LEVEL:.2f})}} & "
        f"\\multicolumn{{{2 * len(TABLE_LEVELS)}}}{{c}}{{Level (block {BASE_BLOCK})}} & "
        f"\\multicolumn{{{len(TABLE_THRESHOLDS)}}}{{c}}"
        f"{{Threshold (block {BASE_BLOCK}, level {BASE_LEVEL:.2f})}} \\\\",
        f"\\cmidrule(lr){{{c0}-{c1}}} \\cmidrule(lr){{{c2}-{c3}}} \\cmidrule(lr){{{c4}-{c5}}}",
        "Configuration & MSESS & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{b}}}" for b in TABLE_BLOCKS)
        + " & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lv:.2f}}}" for lv in TABLE_LEVELS)
        + " & "
        + " & ".join(f"{th:.2f}" for th in TABLE_THRESHOLDS)
        + " \\\\",
        "\\midrule",
    ]
    for key in configuration_order(df):
        cells = [_tex(DISPLAY.get(key, key)), _num(float(base.loc[key, "msess_mean"]))]
        for b in TABLE_BLOCKS:
            t = by_block[b]
            cells += [_num(float(t.loc[key, "lower"])), str(t.loc[key, "verdict"])]
        for lv in TABLE_LEVELS:
            t = by_level[lv]
            cells += [_num(float(t.loc[key, "lower"])), str(t.loc[key, "verdict"])]
        for th in TABLE_THRESHOLDS:
            cells.append(str(by_thr[th].loc[key, "verdict"]))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", ""]
    return "\n".join(lines)


def write_outputs(df: pd.DataFrame, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "verdict_sensitivity.csv"
    tex = out / "verdict_sensitivity_table.tex"
    df.to_csv(csv, index=False, float_format="%.6f", lineterminator="\n")
    tex.write_text(latex_table(df), encoding="utf-8", newline="\n")
    return csv, tex


def summary(df: pd.DataFrame) -> str:
    """Plain-text grid plus the lists of verdicts that move and that never move."""
    wide = df.copy()
    wide["setting"] = [
        Setting(k, int(b), float(lv), float(th)).label
        for k, b, lv, th in zip(
            wide["setting_kind"], wide["block"], wide["level"], wide["threshold"], strict=True
        )
    ]
    is_base = (
        (wide["setting_kind"] == "block")
        & (wide["block"] == BASE_BLOCK)
        & np.isclose(wide["level"], BASE_LEVEL)
        & np.isclose(wide["threshold"], BASE_THRESHOLD)
    )
    order = configuration_order(df)
    lines = ["Verdict under every setting (interval of the seed-mean MSESS, verdict, horizon):"]
    for key in order:
        sub = wide[wide["model"] == key]
        point = float(sub["msess_mean"].iloc[0])
        lines.append(f"  {DISPLAY.get(key, key)}  (seed-mean MSESS {point:+.3f})")
        for _, r in sub.iterrows():
            lines.append(
                f"    {r['setting']:<32s} [{r['lower']:+.3f}, {r['upper']:+.3f}]"
                f"  {r['verdict']:<12s} horizon {int(r['skill_horizon'])}"
            )
    moving: list[str] = []
    fixed: list[str] = []
    for key in order:
        sub = wide[wide["model"] == key]
        verdicts = sub["verdict"].unique().tolist()
        name = DISPLAY.get(key, key)
        if len(verdicts) == 1:
            fixed.append(f"{name}: {verdicts[0]} under every setting")
            continue
        base_verdict = sub.loc[is_base[sub.index], "verdict"].iloc[0]
        others = sub[sub["verdict"] != base_verdict]
        where = "; ".join(f"{r['setting']} -> {r['verdict']}" for _, r in others.iterrows())
        moving.append(f"{name}: {base_verdict} at the base setting; {where}")
    lines.append("")
    lines.append("Verdicts that move across the grid:")
    lines += [f"  {m}" for m in moving] or ["  none"]
    lines.append("Verdicts that never move:")
    lines += [f"  {f}" for f in fixed] or ["  none"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ data


def deposit_declaration(root: Path) -> Declaration:
    text = TEMPLATE.read_text(encoding="utf-8").replace("{thesis_root}", root.as_posix())
    return Declaration.from_dict(yaml.safe_load(text))


def deposit_benchmark(decl: Declaration, root: Path) -> Benchmark:
    """The three deposited configurations, read as the case study reads them."""
    return benchmark_from_declaration(decl.data, root=root)


def statistical_benchmark(root: Path, nino_path: Path) -> tuple[Benchmark, np.ndarray]:
    """The three configurations of ``enso_ridge.py``, fitted as that example fits them,
    with the origin indices the forecasts were issued on."""
    nino = enso_ridge.read_nino34(nino_path)
    obs = enso_ridge.load_observations(root)
    origins, source = enso_ridge.deposit_origins(root, obs)
    if source != "deposit":
        raise RuntimeError("the deposit target arrays are needed to recover the origin grid")
    fitter = enso_ridge.Fitter(obs, nino)
    runs = [
        fitter.forecast(feats, s, origins) for feats in enso_ridge.CONFIGS for s in enso_ridge.SEEDS
    ]
    return Benchmark(runs, obs), origins


# ------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--thesis-root", default=os.environ.get("ANCHORGATE_THESIS_ROOT"))
    ap.add_argument("--nino", default=None, help="local copy of the PSL nina34.anom.data table")
    ap.add_argument("--out", default=str(HERE / "report_sensitivity"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    if not args.thesis_root:
        ap.error("--thesis-root or ANCHORGATE_THESIS_ROOT is required")
    root = Path(args.thesis_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    decl = deposit_declaration(root)
    deposited = deposit_benchmark(decl, root)
    for note in deposited.validate():
        print(f"note: {note}")
    purged = decl.score_origins(deposited.n_origin, deposited.horizon)
    if purged.tolist() != enso_ridge.PURGED:
        raise RuntimeError("the declared block and the block of enso_ridge.py differ")
    print(
        f"deposit: {len(deposited.forecasts)} runs on {deposited.n_origin} origins;"
        f" scored block {purged[0]}..{purged[-1]} ({time.perf_counter() - t0:.0f} s)"
    )

    nino_path = Path(args.nino) if args.nino else enso_ridge.fetch_nino34(out / "data")
    statistical, origins = statistical_benchmark(root, nino_path)
    for note in statistical.validate():
        print(f"note: {note}")
    print(
        f"statistical: {len(statistical.forecasts)} runs issued on"
        f" {origins.size} origins ({time.perf_counter() - t0:.0f} s)"
    )

    grid = run_grid(
        [deposited, statistical],
        score_origins=purged.tolist(),
        n_boot=args.n_boot,
        gate=decl.gate,
    )
    csv, tex = write_outputs(grid, out)
    print()
    print(summary(grid))
    print()
    print(latex_table(grid))
    print(f"written: {csv} and {tex} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
