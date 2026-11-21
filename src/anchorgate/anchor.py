"""Component 1: anchor every forecast to untrained references, and gate.

Every forecast and every reference is scored on identical targets. The gate passes a
configuration only when the mean over its seeds of the skill score against the
reference is positive; a configuration with fewer seeds than the declaration asks for
is reported but cannot pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._time import period_of_year
from .data import Benchmark, Forecast
from .declaration import Declaration, Gate, deviations
from .metrics import categorical_scores, score_table, tercile_events
from .references import build_references, training_terciles
from .result import Result, Verdict, array_hash
from .uncertainty import (
    block_indices,
    interval,
    skill_from_sums,
    skill_horizon,
    squared_error_sums,
    verdict_from_interval,
)

SCORE_COLUMNS = ["pooled_r2", "per_cell_nse", "msess", "rmse", "mae", "bias"]


def _select(arr: np.ndarray, origins: np.ndarray, cells: np.ndarray | None) -> np.ndarray:
    a = arr[origins]
    return a if cells is None else a[..., cells]


def anchor(
    benchmark: Benchmark,
    declaration: Declaration | None = None,
    *,
    score_origins: Sequence[int] | np.ndarray | None = None,
    references: Sequence[str] | None = None,
    gate: Gate | None = None,
    mask: np.ndarray | str | None = None,
    by_lead: bool = True,
    categorical: bool = False,
    validate: bool = True,
) -> Result:
    """Score forecasts and references on the same targets and apply the gate.

    Explicit keyword arguments override the declaration. ``score_origins`` defaults to
    the declaration's scoring block, or to every origin.
    """
    if validate:
        benchmark.validate()
    decl = declaration
    n, horizon = benchmark.n_origin, benchmark.horizon
    if score_origins is None:
        origins = decl.score_origins(n, horizon) if decl else np.arange(n)
    else:
        origins = np.asarray(sorted(int(o) for o in score_origins), dtype=int)
    if origins.size == 0 or np.min(origins) < 0 or np.max(origins) >= n:
        raise ValueError(f"scored origins must lie in 0..{n - 1}")
    ref_names = (
        tuple(references)
        if references is not None
        else (decl.references if decl else ("climatology", "persistence", "seasonal_naive"))
    )
    g = gate or (decl.gate if decl else Gate())
    if g.reference not in ref_names:
        ref_names = (*ref_names, g.reference)
    cells = benchmark.cell_mask(mask)
    obs = benchmark.observations
    refs = build_references(obs, benchmark.origin_period, benchmark.lead, ref_names)
    target = _select(benchmark.target, origins, cells)
    gate_ref = _select(refs[g.reference], origins, cells)
    boot_idx = block_indices(int(origins.size), g.block, g.n_boot, seed=0)
    se_ref = squared_error_sums(gate_ref, target, per_lead=False)
    se_ref_lead = squared_error_sums(gate_ref, target, per_lead=True)
    boot: dict[str, np.ndarray] = {}
    boot_lead: dict[str, np.ndarray] = {}

    rows: list[dict[str, object]] = []
    lead_rows: list[dict[str, object]] = []
    cat_rows: list[dict[str, object]] = []
    inputs: dict[str, str] = {"observations": array_hash(obs.series)}
    lo = np.empty(0)
    hi = np.empty(0)
    obs_ev: dict[str, np.ndarray] = {}
    if categorical:
        lo_t, hi_t = training_terciles(obs)
        cal = period_of_year(benchmark.reference.target_period[origins]) - 1
        lo, hi = lo_t[cal], hi_t[cal]
        if cells is not None:
            lo, hi = lo[..., cells], hi[..., cells]
        obs_ev = tercile_events(target, lo, hi)

    def score_one(
        name: str, kind: str, pred: np.ndarray, model: str, features: str, seed: int | None
    ) -> None:
        row: dict[str, object] = {
            "kind": kind,
            "model": model,
            "features": features,
            "seed": seed,
            "label": name,
            "n_origins": int(origins.size),
            "n_cells": int(target.shape[-1]),
        }
        row.update(score_table(pred, target, gate_ref))
        rows.append(row)
        if kind == "forecast" and g.metric == "msess":
            boot[name] = skill_from_sums(
                squared_error_sums(pred, target, per_lead=False), se_ref, boot_idx
            )
            boot_lead[name] = skill_from_sums(
                squared_error_sums(pred, target, per_lead=True), se_ref_lead, boot_idx
            )
        if by_lead:
            for j, lead in enumerate(benchmark.lead.tolist()):
                lr: dict[str, object] = {
                    "label": name,
                    "kind": kind,
                    "model": model,
                    "features": features,
                    "seed": seed,
                    "lead": int(lead),
                }
                lr.update(score_table(pred[:, j], target[:, j], gate_ref[:, j]))
                lead_rows.append(lr)
        if categorical:
            ev = tercile_events(pred, lo, hi)
            for event in ("dry", "wet"):
                cr: dict[str, object] = {
                    "label": name,
                    "kind": kind,
                    "model": model,
                    "features": features,
                    "seed": seed,
                    "event": event,
                }
                cr.update(categorical_scores(ev[event], obs_ev[event]))
                cat_rows.append(cr)

    for name in ref_names:
        score_one(name, "reference", _select(refs[name], origins, cells), name, "", None)
    for f in benchmark.forecasts:
        inputs[f.label] = f.hash()
        score_one(f.label, "forecast", _select(f.pred, origins, cells), f.model, f.features, f.seed)

    values = pd.DataFrame(rows)
    lead_df = pd.DataFrame(lead_rows)
    per_config, lead_cfg = _aggregate(
        values[values["kind"] == "forecast"], lead_df, g, boot, boot_lead, benchmark.lead
    )
    verdicts = per_config["verdict"].tolist() if len(per_config) else []
    if any(v == Verdict.PASS.value for v in verdicts):
        verdict = Verdict.PASS
    elif any(v == Verdict.FAIL.value for v in verdicts):
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.INSUFFICIENT
    notes = []
    ref_row = values[(values["kind"] == "reference") & (values["label"] == g.reference)].iloc[0]
    notes.append(
        f"{g.reference}: pooled R2 {ref_row['pooled_r2']:.3f},"
        f" per-cell NSE {ref_row['per_cell_nse']:.3f} on {int(origins.size)} origins"
    )
    for _, r in per_config.iterrows():
        notes.append(
            f"{r['model']}[{r['features']}]: {r['verdict']} (seed mean {g.metric}"
            f" {r[g.metric + '_mean']:+.3f}, {g.level:.0%} interval"
            f" [{r[g.metric + '_lower']:+.3f}, {r[g.metric + '_upper']:+.3f}],"
            f" n_seeds {int(r['n_seeds'])}, skill horizon {int(r['skill_horizon'])})"
        )
    tables = {"by_configuration": per_config}
    if by_lead:
        tables["by_lead"] = lead_df
        tables["by_lead_configuration"] = lead_cfg
    if categorical:
        tables["categorical"] = pd.DataFrame(cat_rows)
    if decl is not None:
        if len(per_config):
            for _, r in per_config[per_config["n_seeds"] < decl.seeds_expected].iterrows():
                notes.append(
                    f"{r['model']}[{r['features']}] has {int(r['n_seeds'])} seed(s); the"
                    f" declaration expected {decl.seeds_expected}"
                )
        said = decl.conclusion_for(verdict.value)
        if said:
            notes.append(f"declared conclusion on {verdict.value}: {said}")
    params: dict[str, object] = {
        "score_origins": origins.tolist(),
        "references": list(ref_names),
        "gate": vars(g),
        "mask": "domain" if isinstance(mask, str) else ("array" if mask is not None else None),
        "by_lead": by_lead,
        "categorical": categorical,
    }
    devs = deviations(
        decl,
        {
            "score_origins": score_origins is not None,
            "references": references is not None,
            "gate": gate is not None,
        },
    )
    return Result(
        component="C1 anchoring",
        title=f"Anchoring to {', '.join(ref_names)} on {int(origins.size)} origins",
        values=values,
        verdict=verdict,
        rule=(
            f"A configuration passes when the lower bound of the {g.level:.0%} block-bootstrap"
            f" interval (blocks of {g.block} origins) of the mean over at least {g.min_seeds}"
            f" seeds of {g.metric} against {g.reference} exceeds {g.threshold:g}; it fails when"
            " the upper bound does not; an interval that straddles the threshold is INSUFFICIENT."
        ),
        inputs=inputs,
        notes=notes,
        tables=tables,
        declaration_hash=decl.sha256 if decl else None,
        parameters=params,
        deviations=devs,
    )


def _aggregate(
    forecast_rows: pd.DataFrame,
    lead_rows: pd.DataFrame,
    g: Gate,
    boot: dict[str, np.ndarray],
    boot_lead: dict[str, np.ndarray],
    leads: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed mean and spread per configuration, the bootstrap interval of the seed mean of
    the gate metric, the skill horizon and the verdict; and the same per lead."""
    cols = ["model", "features"]
    lead_vals = np.asarray(leads, dtype=int)
    cfg_cols = [
        *cols,
        "n_seeds",
        "n_runs",
        *(f"{c}_mean" for c in SCORE_COLUMNS),
        *(f"{c}_sd" for c in SCORE_COLUMNS),
        f"{g.metric}_lower",
        f"{g.metric}_upper",
        "skill_horizon",
        "verdict",
    ]
    lead_cols = [
        *cols,
        "lead",
        "n_runs",
        f"{g.metric}_mean",
        f"{g.metric}_lower",
        f"{g.metric}_upper",
    ]
    if forecast_rows.empty:
        return pd.DataFrame(columns=cfg_cols), pd.DataFrame(columns=lead_cols)
    out = []
    lead_out = []
    for (model, features), grp in forecast_rows.groupby(cols, sort=False):
        labelled = grp[grp["seed"].notna()]
        # distinct labelled seeds, plus one per unlabelled run (which cannot be told apart)
        n_seeds = int(labelled["seed"].nunique()) + int(grp["seed"].isna().sum())
        rec: dict[str, object] = {
            "model": model,
            "features": features,
            "n_seeds": n_seeds,
            "n_runs": len(grp),
        }
        means: dict[str, float] = {}
        for c in SCORE_COLUMNS:
            vals = grp[c].to_numpy(dtype=float)
            means[c] = float(np.nanmean(vals)) if vals.size else float("nan")
            rec[f"{c}_mean"] = means[c]
            rec[f"{c}_sd"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan
        stat = means.get(g.metric, float("nan"))
        labels = [lab for lab in grp["label"].tolist() if lab in boot]
        lower = upper = float("nan")
        horizon = 0
        if labels:
            draws = np.mean([boot[lab] for lab in labels], axis=0)
            lower, upper = interval(draws, g.level)
            draws_lead = np.mean([boot_lead[lab] for lab in labels], axis=0)
            lo_hi = [interval(draws_lead[:, j], g.level) for j in range(draws_lead.shape[1])]
            lo_lead = np.array([lh[0] for lh in lo_hi])
            horizon = skill_horizon(lo_lead, lead_vals, g.threshold)
            if len(lead_rows):
                lr = lead_rows[(lead_rows["model"] == model) & (lead_rows["features"] == features)]
                for j, lead in enumerate(lead_vals.tolist()):
                    pts = lr.loc[lr["lead"] == lead, g.metric].to_numpy(dtype=float)
                    lead_out.append(
                        {
                            "model": model,
                            "features": features,
                            "lead": int(lead),
                            "n_runs": int(pts.size),
                            f"{g.metric}_mean": float(np.nanmean(pts)) if pts.size else np.nan,
                            f"{g.metric}_lower": lo_hi[j][0],
                            f"{g.metric}_upper": lo_hi[j][1],
                        }
                    )
        rec[f"{g.metric}_lower"] = lower
        rec[f"{g.metric}_upper"] = upper
        rec["skill_horizon"] = horizon
        if n_seeds < g.min_seeds:
            rec["verdict"] = Verdict.INSUFFICIENT.value
        elif labels:
            rec["verdict"] = verdict_from_interval(lower, upper, g.threshold).value
        elif np.isfinite(stat) and stat > g.threshold:
            rec["verdict"] = Verdict.PASS.value
        else:
            rec["verdict"] = Verdict.FAIL.value
        out.append(rec)
    return pd.DataFrame(out), pd.DataFrame(lead_out, columns=lead_cols)


def gate_table(result: Result) -> pd.DataFrame:
    """The per-configuration verdict table of an anchoring result."""
    return result.tables["by_configuration"]


def quick_anchor(forecast: Forecast, observations: object, **kw: object) -> Result:
    """Anchor a single forecast (one line for a reader with one model)."""
    from .data import Observations

    if not isinstance(observations, Observations):
        raise TypeError("observations must be an Observations instance")
    return anchor(Benchmark([forecast], observations), **kw)  # type: ignore[arg-type]
