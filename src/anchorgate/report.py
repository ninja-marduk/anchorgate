"""Render results as Markdown and JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .data import Benchmark
from .declaration import Declaration
from .result import Result

INTEGER_COLUMNS = {
    "seed",
    "n_seeds",
    "n_runs",
    "n_origins",
    "n_cells",
    "n_fit",
    "n_score",
    "n_pairs",
    "n_overlapping",
    "n_splits",
    "split",
    "lead",
    "n",
    "df",
    "n_perm",
    "wins",
    "n_a",
    "n_b",
    "n_units",
    "k",
    "n_effective",
    "cell",
    "n_nodes",
    "n_edges",
    "isolated_nodes",
    "n_calibration",
    "n_test",
    "n_cells_sampled",
    "n_steps_full",
    "n_steps_training",
    "edges_full",
    "edges_training",
    "edges_only_with_held_out",
    "edges_lost_without_held_out",
}


def _table(frame: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int = 60) -> str:
    if frame.empty:
        return "_(empty)_\n"
    df = frame.head(max_rows).copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            finite = df[c].dropna()
            integral = len(finite) > 0 and bool(np.all(np.mod(finite.to_numpy(), 1) == 0))
            fmt = "{:.0f}" if integral and str(c) in INTEGER_COLUMNS else floatfmt

            def _cell(v: float, fmt: str = fmt) -> str:
                return "" if pd.isna(v) else fmt.format(v)

            df[c] = df[c].map(_cell)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_{len(frame) - max_rows} more rows in the JSON report_")
    return "\n".join(lines) + "\n"


def render_markdown(
    results: Sequence[Result],
    benchmark: Benchmark | None = None,
    declaration: Declaration | None = None,
) -> str:
    """Verdict table first, then each component's numbers."""
    out = ["# anchorgate report", ""]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out.append(f"Generated {stamp} with anchorgate {__version__}.")
    if declaration is not None:
        out.append(
            f"Declared design: `{declaration.name}`, design hash `{declaration.design_hash}`,"
            f" file hash `{declaration.sha256}` (from {declaration.source})."
        )
        if declaration.conclusions:
            out.append(
                "Declared conclusions: "
                + "; ".join(f"on {k.upper()}, {v}" for k, v in declaration.conclusions.items())
            )
    if benchmark is not None:
        f = benchmark.reference
        o = benchmark.observations
        out.append(
            f"Benchmark: {len(benchmark.forecasts)} forecasts, {f.n_origin} origins,"
            f" leads {f.lead.min()}..{f.lead.max()}, {f.n_cell} cells;"
            f" observations {o.n_time} steps, training cutoff index {o.cutoff}."
        )
        if benchmark.notes:
            out.append("")
            out.append("Validation notes:")
            out.extend(f"- {n}" for n in benchmark.notes)
    out.extend(["", "## Verdicts", ""])
    out.append(
        _table(
            pd.DataFrame(
                [
                    {"component": r.component, "verdict": r.verdict.value, "rule": r.rule}
                    for r in results
                ]
            )
        )
    )
    for r in results:
        out.extend(
            [
                "",
                f"## {r.component}: {r.title}",
                "",
                f"Verdict: **{r.verdict.value}**. Rule: {r.rule}",
                "",
            ]
        )
        if r.notes:
            out.extend(f"- {n}" for n in r.notes)
            out.append("")
        if r.parameters:
            shown = {k: v for k, v in r.parameters.items() if k != "score_origins"}
            so = r.parameters.get("score_origins")
            if isinstance(so, list) and so:
                shown["score_origins"] = f"{so[0]}..{so[-1]} ({len(so)})"
            out.append("Parameters used: " + ", ".join(f"{k}={v}" for k, v in shown.items()))
        if r.deviations:
            out.append("Deviations from the declaration: " + "; ".join(r.deviations))
        if r.parameters or r.deviations:
            out.append("")
        out.append(_table(r.values))
        for name, t in r.tables.items():
            out.extend(["", f"### {name}", "", _table(t)])
        if r.inputs:
            out.extend(["", "Inputs (SHA-256):", ""])
            out.extend(f"- {k}: `{v[:16]}`" for k, v in r.inputs.items())
    return "\n".join(out) + "\n"


def to_json(results: Sequence[Result], declaration: Declaration | None = None) -> str:
    doc = {
        "anchorgate": __version__,
        "declaration": None
        if declaration is None
        else {
            "name": declaration.name,
            "sha256": declaration.sha256,
            "design_hash": declaration.design_hash,
            "source": declaration.source,
            "conclusions": dict(declaration.conclusions),
        },
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(doc, indent=2, default=str)


def write_report(
    directory: str | Path,
    results: Sequence[Result],
    benchmark: Benchmark | None = None,
    declaration: Declaration | None = None,
) -> tuple[Path, Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    md = d / "report.md"
    js = d / "report.json"
    md.write_text(render_markdown(results, benchmark, declaration), encoding="utf-8")
    js.write_text(to_json(results, declaration), encoding="utf-8")
    return md, js
