"""How much of a verdict is owed to the constants of the gate, and to which seeds were run?

The gate reads PASS, FAIL or INSUFFICIENT from a circular moving-block bootstrap over
the scored origins. On the purged block of the Boyaca case study there are twelve
origins, so blocks of three leave four blocks behind every interval, and a referee may
ask whether a different block length, interval level or threshold would move a verdict,
whether the verdict of component 4 (anomaly skill) moves with the same constants, and
whether any verdict depends on which seeds happened to be run. This example answers by
recomputing the verdicts of six configurations on that block, origins 21 to 32, under:

* component 1, block length 1, 2, 3, 4 and 6 origins at level 0.90 and threshold 0;
  level 0.80, 0.90 and 0.95 at block 3 and threshold 0; threshold 0.00, 0.02 and 0.05 at
  block 3 and level 0.90;
* component 4, the same block lengths and levels (its gate has no threshold to vary);
* components 1 and 4 at the default constants on every pair of the three seeds, and
  component 1 with the minimum number of seeds raised from two to three;

with 1000 resamples everywhere and the same bootstrap seed, so two settings differ only
by the constant under study. The configurations are the three deposited ones
(ConvLSTM-Bidirectional, GNN-TAT-GAT and the released late fusion, read through the
declaration of ``boyaca_case_study``) and the three statistical ones of ``enso_ridge.py``
(ridge on Nino 3.4 lags, ridge on Nino 3.4 and local lags, damped persistence), refitted
here with the functions of that example so the forecasts are the ones it produces. The
Nino 3.4 table is downloaded on first use into ``<out>/data``.

    python examples/verdict_sensitivity.py --thesis-root /path/to/deposit --out report_sensitivity/

The root is also read from ``ANCHORGATE_THESIS_ROOT``. Outputs: ``verdict_sensitivity.csv``
(one row per setting and configuration), ``verdict_sensitivity_table.tex`` (the component-1
grid as a compact ``tabular*`` body), ``verdict_sensitivity_c4_table.tex`` (the component-4
grid), ``verdict_sensitivity_seeds_table.tex`` (the seed pairs) and a printed summary of
which verdicts move.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anchorgate import Benchmark, Declaration, Gate, anchor, anomaly_skill
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
ALL_SEEDS = "all"
METRIC_OF = {"C1": "msess", "C4": "acc_cell_median"}
COLUMNS = [
    "component",
    "setting_kind",
    "block",
    "level",
    "threshold",
    "min_seeds",
    "seeds",
    "model",
    "metric",
    "mean",
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
    """One point of the grid: the component, which constant is varied, the three gate
    constants, the minimum number of seeds and the seeds kept (``None`` for all)."""

    kind: str
    block: int
    level: float
    threshold: float
    component: str = "C1"
    min_seeds: int | None = None
    seeds: tuple[int, ...] | None = None

    def gate(self, template: Gate, n_boot: int) -> Gate:
        return Gate(
            metric=template.metric,
            threshold=self.threshold,
            min_seeds=template.min_seeds if self.min_seeds is None else self.min_seeds,
            reference=template.reference,
            level=self.level,
            block=self.block,
            n_boot=n_boot,
        )

    @property
    def seeds_label(self) -> str:
        return ALL_SEEDS if self.seeds is None else ",".join(str(s) for s in self.seeds)

    @property
    def label(self) -> str:
        extra = ""
        if self.seeds is not None:
            extra = f" seeds={self.seeds_label}"
        if self.min_seeds is not None:
            extra += f" min_seeds={self.min_seeds}"
        core = f"b={self.block} L={self.level:.2f} t={self.threshold:.2f}"
        return f"{self.component} {self.kind} {core}{extra}"


def settings(seeds: Sequence[int] = ()) -> list[Setting]:
    """The grid: eleven component-1 points, eight component-4 points, the seed pairs for
    both components (when three or more seeds are given) and ``min_seeds`` 3 for component
    1. The base setting appears once under every kind."""
    out = [Setting("block", b, BASE_LEVEL, BASE_THRESHOLD) for b in BLOCKS]
    out += [Setting("level", BASE_BLOCK, lv, BASE_THRESHOLD) for lv in LEVELS]
    out += [Setting("threshold", BASE_BLOCK, BASE_LEVEL, th) for th in THRESHOLDS]
    out += [Setting("block", b, BASE_LEVEL, BASE_THRESHOLD, component="C4") for b in BLOCKS]
    out += [Setting("level", BASE_BLOCK, lv, BASE_THRESHOLD, component="C4") for lv in LEVELS]
    out.append(Setting("min_seeds", BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD, min_seeds=3))
    if len(seeds) >= 3:
        for pair in itertools.combinations(sorted(seeds), 2):
            for comp in ("C1", "C4"):
                out.append(
                    Setting("seeds", BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD, comp, seeds=pair)
                )
    return out


# ------------------------------------------------------------------------------ grid


def _subset(bench: Benchmark, seeds: tuple[int, ...] | None) -> Benchmark | None:
    if seeds is None:
        return bench
    kept = [f for f in bench.forecasts if f.seed in seeds]
    if not kept:
        return None
    return Benchmark(kept, bench.observations)


def _rows_of(res_table: pd.DataFrame, s: Setting, g: Gate, metric: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, r in res_table.iterrows():
        rows.append(
            {
                "component": s.component,
                "setting_kind": s.kind,
                "block": s.block,
                "level": s.level,
                "threshold": s.threshold,
                "min_seeds": g.min_seeds,
                "seeds": s.seeds_label,
                "model": f"{r['model']}[{r['features']}]",
                "metric": metric,
                "mean": float(r[f"{metric}_mean"]),
                "lower": float(r[f"{metric}_lower"]),
                "upper": float(r[f"{metric}_upper"]),
                "verdict": str(r["verdict"]),
                "skill_horizon": int(r["skill_horizon"]) if "skill_horizon" in r else -1,
            }
        )
    return rows


def run_grid(
    benchmarks: Sequence[Benchmark],
    *,
    score_origins: Sequence[int] | None,
    n_boot: int = N_BOOT,
    gate: Gate | None = None,
    grid: Sequence[Setting] | None = None,
) -> pd.DataFrame:
    """Component-1 and component-4 verdicts of every configuration of ``benchmarks`` under
    every setting.

    The benchmarks are scored separately (they need not share targets) and their
    per-configuration tables are stacked. ``gate`` supplies the metric, the reference
    and ``min_seeds``; the grid supplies the rest. The bootstrap seed is the one the
    components always use, so two settings differ only by the constant under study.
    """
    template = gate or Gate()

    def _run(s: Setting, bench: Benchmark) -> list[dict[str, object]]:
        g = s.gate(template, n_boot)
        sub = _subset(bench, s.seeds)
        if sub is None:
            return []
        if s.component == "C1":
            res = anchor(sub, score_origins=score_origins, gate=g, by_lead=False, validate=False)
        else:
            res = anomaly_skill(
                sub, score_origins=score_origins, gate=g, by_lead=False, validate=False
            )
        return _rows_of(res.tables["by_configuration"], s, g, METRIC_OF[s.component])

    rows: list[dict[str, object]] = []
    if grid is not None:
        for s in grid:
            for bench in benchmarks:
                rows.extend(_run(s, bench))
        return pd.DataFrame(rows, columns=COLUMNS)
    # the constants grid on every benchmark; the seed pairs are formed per benchmark,
    # because two benchmarks need not share seed numbers
    for s in settings():
        for bench in benchmarks:
            rows.extend(_run(s, bench))
    for bench in benchmarks:
        seeds = sorted({int(f.seed) for f in bench.forecasts if f.seed is not None})
        for s in settings(seeds):
            if s.kind == "seeds":
                rows.extend(_run(s, bench))
    return pd.DataFrame(rows, columns=COLUMNS)


def _pick(
    df: pd.DataFrame,
    kind: str,
    block: int,
    level: float,
    threshold: float,
    component: str = "C1",
    seeds: str = ALL_SEEDS,
) -> pd.DataFrame:
    m = (
        (df["component"] == component)
        & (df["setting_kind"] == kind)
        & (df["block"] == block)
        & np.isclose(df["level"], level)
        & np.isclose(df["threshold"], threshold)
        & (df["seeds"] == seeds)
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


def _grid_table(df: pd.DataFrame, component: str, metric_label: str, thresholds: bool) -> str:
    base = _pick(df, "block", BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD, component)
    by_block = {
        b: _pick(df, "block", b, BASE_LEVEL, BASE_THRESHOLD, component) for b in TABLE_BLOCKS
    }
    by_level = {
        lv: _pick(df, "level", BASE_BLOCK, lv, BASE_THRESHOLD, component) for lv in TABLE_LEVELS
    }
    by_thr = (
        {
            th: _pick(df, "threshold", BASE_BLOCK, BASE_LEVEL, th, component)
            for th in TABLE_THRESHOLDS
        }
        if thresholds
        else {}
    )
    n_pairs = len(TABLE_BLOCKS) + len(TABLE_LEVELS)
    spec = "@{\\extracolsep{\\fill}}l r" + " rl" * n_pairs + " l" * len(by_thr) + "@{}"
    c0 = 3
    c1 = c0 + 2 * len(TABLE_BLOCKS) - 1
    c2 = c1 + 1
    c3 = c2 + 2 * len(TABLE_LEVELS) - 1
    c4 = c3 + 1
    c5 = c4 + len(by_thr) - 1
    head = (
        " & & "
        f"\\multicolumn{{{2 * len(TABLE_BLOCKS)}}}{{c}}{{Block length (level {BASE_LEVEL:.2f})}} & "
        f"\\multicolumn{{{2 * len(TABLE_LEVELS)}}}{{c}}{{Level (block {BASE_BLOCK})}}"
    )
    rule = f"\\cmidrule(lr){{{c0}-{c1}}} \\cmidrule(lr){{{c2}-{c3}}}"
    if by_thr:
        head += (
            f" & \\multicolumn{{{len(by_thr)}}}{{c}}"
            f"{{Threshold (block {BASE_BLOCK}, level {BASE_LEVEL:.2f})}}"
        )
        rule += f" \\cmidrule(lr){{{c4}-{c5}}}"
    sub = (
        f"Configuration & {metric_label} & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{b}}}" for b in TABLE_BLOCKS)
        + " & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lv:.2f}}}" for lv in TABLE_LEVELS)
    )
    if by_thr:
        sub += " & " + " & ".join(f"{th:.2f}" for th in by_thr)
    lines = [
        f"% Generated by examples/verdict_sensitivity.py ({component}). Each paired column",
        f"% holds the lower bound of the bootstrap interval of the seed-mean {metric_label}",
        "% and the verdict it yields.",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{spec}}}",
        "\\toprule",
        head + " \\\\",
        rule,
        sub + " \\\\",
        "\\midrule",
    ]
    for key in configuration_order(df):
        if key not in base.index:
            continue
        cells = [_tex(DISPLAY.get(key, key)), _num(float(base.loc[key, "mean"]))]
        for b in TABLE_BLOCKS:
            t = by_block[b]
            cells += [_num(float(t.loc[key, "lower"])), str(t.loc[key, "verdict"])]
        for lv in TABLE_LEVELS:
            t = by_level[lv]
            cells += [_num(float(t.loc[key, "lower"])), str(t.loc[key, "verdict"])]
        for th in by_thr:
            cells.append(str(by_thr[th].loc[key, "verdict"]))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", ""]
    return "\n".join(lines)


def latex_table(df: pd.DataFrame) -> str:
    """The component-1 grid as a ``tabular*`` body: point estimate, then lower bound and
    verdict at blocks 1/3/6 (level 0.90), at levels 0.80/0.95 (block 3), and the verdict at
    thresholds 0.02 and 0.05 (block 3, level 0.90). Minus signs are ``$-$``."""
    return _grid_table(df, "C1", "MSESS", thresholds=True)


def latex_table_c4(df: pd.DataFrame) -> str:
    """The component-4 grid: the median per-cell anomaly correlation and its lower bound
    and verdict at the same block lengths and levels."""
    return _grid_table(df, "C4", "ACC", thresholds=False)


def latex_table_seeds(df: pd.DataFrame) -> str:
    """Verdicts of components 1 and 4 at the default constants on every pair of seeds, and
    of component 1 with ``min_seeds`` raised to 3."""
    seeds_rows = df[df["setting_kind"] == "seeds"]
    base1 = _pick(df, "block", BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD, "C1")
    base4 = _pick(df, "block", BASE_BLOCK, BASE_LEVEL, BASE_THRESHOLD, "C4")
    min3 = df[(df["setting_kind"] == "min_seeds") & (df["component"] == "C1")].set_index("model")

    def pair_verdicts(key: str, comp: str) -> list[str]:
        """The verdicts of one configuration over its seed pairs, in seed order; the pairs
        are positional because two benchmarks need not share seed numbers."""
        sub = seeds_rows[(seeds_rows["model"] == key) & (seeds_rows["component"] == comp)]
        return [
            f"{r['verdict']} ({_tex(str(r['seeds']))})"
            for _, r in sub.sort_values("seeds").iterrows()
        ]

    n_pairs = max((len(pair_verdicts(k, "C1")) for k in configuration_order(df)), default=0)
    spec = "@{\\extracolsep{\\fill}}l" + " l" * (2 + 2 * n_pairs + 1) + "@{}"
    lines = [
        "% Generated by examples/verdict_sensitivity.py (seeds). Verdicts at the default",
        "% constants with every seed, with each pair of seeds (the pair in brackets), and",
        "% with min_seeds raised to 3.",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{spec}}}",
        "\\toprule",
        " & \\multicolumn{" + str(1 + n_pairs) + "}{c}{Component 1} & "
        "\\multicolumn{" + str(1 + n_pairs) + "}{c}{Component 4} & C1 \\\\",
        f"\\cmidrule(lr){{2-{2 + n_pairs}}} \\cmidrule(lr){{{3 + n_pairs}-{3 + 2 * n_pairs}}}",
        "Configuration & all seeds & "
        + " & ".join(f"pair {i + 1}" for i in range(n_pairs))
        + " & all seeds & "
        + " & ".join(f"pair {i + 1}" for i in range(n_pairs))
        + " & min 3 \\\\",
        "\\midrule",
    ]
    for key in configuration_order(df):
        if key not in base1.index:
            continue
        cells = [_tex(DISPLAY.get(key, key)), str(base1.loc[key, "verdict"])]
        v1 = pair_verdicts(key, "C1")
        cells += v1 + ["--"] * (n_pairs - len(v1))
        cells.append(str(base4.loc[key, "verdict"]) if key in base4.index else "--")
        v4 = pair_verdicts(key, "C4")
        cells += v4 + ["--"] * (n_pairs - len(v4))
        cells.append(str(min3.loc[key, "verdict"]) if key in min3.index else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", ""]
    return "\n".join(lines)


def write_outputs(df: pd.DataFrame, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "verdict_sensitivity.csv"
    tex = out / "verdict_sensitivity_table.tex"
    df.to_csv(csv, index=False, float_format="%.6f", lineterminator="\n")
    tex.write_text(latex_table(df), encoding="utf-8", newline="\n")
    if (df["component"] == "C4").any():
        (out / "verdict_sensitivity_c4_table.tex").write_text(
            latex_table_c4(df), encoding="utf-8", newline="\n"
        )
    if (df["setting_kind"] == "seeds").any():
        (out / "verdict_sensitivity_seeds_table.tex").write_text(
            latex_table_seeds(df), encoding="utf-8", newline="\n"
        )
    return csv, tex


def summary(df: pd.DataFrame) -> str:
    """Plain-text grid plus the lists of verdicts that move and that never move, per
    component."""
    wide = df.copy()
    wide["setting"] = [
        Setting(
            k,
            int(b),
            float(lv),
            float(th),
            comp,
            None if sd == ALL_SEEDS and kind_ != "min_seeds" else int(ms),
            None if sd == ALL_SEEDS else tuple(int(x) for x in sd.split(",")),
        ).label
        for k, b, lv, th, comp, sd, ms, kind_ in zip(
            wide["setting_kind"],
            wide["block"],
            wide["level"],
            wide["threshold"],
            wide["component"],
            wide["seeds"],
            wide["min_seeds"],
            wide["setting_kind"],
            strict=True,
        )
    ]
    is_base = (
        (wide["setting_kind"] == "block")
        & (wide["block"] == BASE_BLOCK)
        & np.isclose(wide["level"], BASE_LEVEL)
        & np.isclose(wide["threshold"], BASE_THRESHOLD)
        & (wide["seeds"] == ALL_SEEDS)
    )
    order = configuration_order(df)
    lines: list[str] = []
    for comp in ("C1", "C4"):
        part = wide[wide["component"] == comp]
        if part.empty:
            continue
        metric = METRIC_OF[comp]
        lines.append(
            f"{comp}: verdict under every setting (interval of the seed-mean {metric}, verdict):"
        )
        moving: list[str] = []
        fixed: list[str] = []
        for key in order:
            sub = part[part["model"] == key]
            if sub.empty:
                continue
            base_rows = sub[is_base[sub.index]]
            point = float(base_rows["mean"].iloc[0]) if len(base_rows) else float("nan")
            lines.append(f"  {DISPLAY.get(key, key)}  (seed-mean {metric} {point:+.3f})")
            for _, r in sub.iterrows():
                hz = f" horizon {int(r['skill_horizon'])}" if r["skill_horizon"] >= 0 else ""
                lines.append(
                    f"    {r['setting']:<52s} [{r['lower']:+.3f}, {r['upper']:+.3f}]"
                    f"  {r['verdict']:<12s}{hz}"
                )
            verdicts = sub["verdict"].unique().tolist()
            name = DISPLAY.get(key, key)
            if len(verdicts) == 1:
                fixed.append(f"{name}: {verdicts[0]} under every setting")
                continue
            base_verdict = base_rows["verdict"].iloc[0] if len(base_rows) else verdicts[0]
            others = sub[sub["verdict"] != base_verdict]
            where = "; ".join(f"{r['setting']} -> {r['verdict']}" for _, r in others.iterrows())
            moving.append(f"{name}: {base_verdict} at the base setting; {where}")
        lines.append("")
        lines.append(f"{comp}: verdicts that move across the grid:")
        lines += [f"  {m}" for m in moving] or ["  none"]
        lines.append(f"{comp}: verdicts that never move:")
        lines += [f"  {f}" for f in fixed] or ["  none"]
        lines.append("")
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
    print(latex_table(grid))
    print(latex_table_c4(grid))
    print(latex_table_seeds(grid))
    print(f"written: {csv} and {tex} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
