"""Component 4: skill on deseasonalised anomalies.

A per-cell period climatology carries the seasonal cycle and the spatial pattern of the
field, which is why it scores high on pooled R2 and above zero on per-cell NSE. Removing
that climatology from forecast and target leaves the anomalies, on which the
climatology scores exactly zero. Skill on anomalies is therefore the only evidence that
a model knows something the climatology does not.

The anomaly correlation is computed per cell over time (the standard definition) and
per origin over space (the pattern correlation), and the two are reported separately.
Pooling both axes into one correlation conflates a time-invariant spatial pattern with
forecast skill; that pooled number is printed next to the others so a reader can see
how much it inflates, but it never enters a verdict.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import Benchmark
from .declaration import Declaration, Gate, deviations
from .factorial import _origins, _select
from .metrics import msess, pooled_r2
from .references import build_references
from .result import Result, Verdict

# ------------------------------------------------------------------ correlations


def _corr_along(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Pearson correlation of ``a`` and ``b`` along ``axis``; NaN where undefined."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a = np.where(m, a, np.nan)
    b = np.where(m, b, np.nan)
    n = m.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = np.nanmean(a, axis=axis, keepdims=True)
        mean_b = np.nanmean(b, axis=axis, keepdims=True)
        ac = a - mean_a
        bc = b - mean_b
        # a series that is constant up to rounding has no correlation, not a spurious one
        flat_a = np.nanstd(a, axis=axis) <= 1e-9 * (np.abs(np.squeeze(mean_a, axis)) + 1e-12)
        flat_b = np.nanstd(b, axis=axis) <= 1e-9 * (np.abs(np.squeeze(mean_b, axis)) + 1e-12)
        num = np.nansum(ac * bc, axis=axis)
        den = np.sqrt(np.nansum(ac**2, axis=axis) * np.nansum(bc**2, axis=axis))
        ok = (den > 0) & (n >= 3) & ~flat_a & ~flat_b
        out = np.where(ok, num / np.where(den > 0, den, 1), np.nan)
    return np.asarray(out, dtype=np.float64)


def temporal_acc(pred_anom: np.ndarray, target_anom: np.ndarray) -> np.ndarray:
    """Per-cell anomaly correlation over the leading (time) axes: ``(..., cells)`` ->
    ``(cells,)``. This is the standard ACC of forecast verification."""
    p = np.asarray(pred_anom, dtype=np.float64).reshape(-1, np.shape(pred_anom)[-1])
    t = np.asarray(target_anom, dtype=np.float64).reshape(-1, np.shape(target_anom)[-1])
    return _corr_along(p, t, axis=0)


def pattern_acc(pred_anom: np.ndarray, target_anom: np.ndarray) -> np.ndarray:
    """Per-time-step spatial anomaly correlation: ``(..., cells)`` -> ``(...,)``."""
    return _corr_along(pred_anom, target_anom, axis=-1)


def pooled_acc(pred_anom: np.ndarray, target_anom: np.ndarray) -> float:
    """One correlation over every (time, cell) pair. Inflated by any static spatial
    pattern; reported for comparison only."""
    p = np.asarray(pred_anom, dtype=np.float64).ravel()
    t = np.asarray(target_anom, dtype=np.float64).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 3 or np.std(p[m]) == 0 or np.std(t[m]) == 0:
        return float("nan")
    return float(np.corrcoef(p[m], t[m])[0, 1])


def anomaly_scores(pred_anom: np.ndarray, target_anom: np.ndarray) -> dict[str, float]:
    """The row of anomaly scores for one forecast on one set of (origin, lead) pairs."""
    cell = temporal_acc(pred_anom, target_anom)
    pat = pattern_acc(pred_anom, target_anom)
    good = np.isfinite(cell)
    return {
        "acc_cell_mean": float(np.nanmean(cell)) if good.any() else float("nan"),
        "acc_cell_median": float(np.nanmedian(cell)) if good.any() else float("nan"),
        "cells_positive": float(np.mean(cell[good] > 0)) if good.any() else float("nan"),
        "acc_pattern": float(np.nanmean(pat)) if np.isfinite(pat).any() else float("nan"),
        "acc_pooled": pooled_acc(pred_anom, target_anom),
        "anomaly_r2": pooled_r2(pred_anom, target_anom),
        "anomaly_sd_ratio": _sd_ratio(pred_anom, target_anom),
    }


def _sd_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """Median over cells of sd(forecast) / sd(target) over the leading axes."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1, np.shape(pred)[-1])
    t = np.asarray(target, dtype=np.float64).reshape(-1, np.shape(target)[-1])
    sp, st = np.nanstd(p, axis=0), np.nanstd(t, axis=0)
    ok = st > 0
    return float(np.nanmedian(sp[ok] / st[ok])) if ok.any() else float("nan")


# -------------------------------------------------------------------- conformal


def conformal_by_origin(
    pred: np.ndarray,
    target: np.ndarray,
    n_calibration: int,
    alpha: float = 0.10,
    score_quantile: float = 0.90,
    strata: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Split-conformal intervals whose exchangeable unit is the forecast origin.

    ``pred`` and ``target`` are ``(origin, lead, cell)``. The first ``n_calibration``
    origins calibrate; for each lead the per-origin ``score_quantile`` of the absolute
    residual over cells is the conformity score, and the ``ceil((n + 1)(1 - alpha))``-th
    smallest score is the half-width. Intervals are truncated at zero (a non-negative
    variable), and coverage is reported marginally and inside each stratum.

    One row per lead; columns ``coverage``, ``width`` (mean truncated width), the
    fraction of untruncated lower bounds below zero, the conformal level the
    calibration size can guarantee, and ``coverage_<stratum>`` when strata are given.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n_origin, n_lead = p.shape[0], p.shape[1]
    if not 1 <= n_calibration < n_origin:
        raise ValueError("n_calibration must leave at least one test origin")
    k = int(np.ceil((n_calibration + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n_calibration - 1)
    level = min(np.ceil((n_calibration + 1) * (1 - alpha)) / n_calibration, 1.0)
    rows = []
    for j in range(n_lead):
        res_cal = np.abs(t[:n_calibration, j] - p[:n_calibration, j])
        per_origin = np.nanpercentile(res_cal, 100 * score_quantile, axis=1)
        q = float(np.sort(per_origin)[k])
        err = np.abs(t[n_calibration:, j] - p[n_calibration:, j])
        lo = p[n_calibration:, j] - q
        hi = p[n_calibration:, j] + q
        row: dict[str, object] = {
            "lead": j + 1,
            "half_width": q,
            "coverage": float(np.nanmean(err <= q)),
            "width": float(np.nanmean(hi - np.maximum(lo, 0))),
            "lower_below_zero": float(np.nanmean(lo < 0)),
            "level_guaranteed": float(level),
            "n_calibration": n_calibration,
            "n_test": n_origin - n_calibration,
        }
        if strata:
            for name, cells in strata.items():
                row[f"coverage_{name}"] = float(np.nanmean(err[:, cells] <= q))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- component


def anomaly_skill(
    benchmark: Benchmark,
    declaration: Declaration | None = None,
    *,
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    by_lead: bool = True,
    conformal: bool = False,
    n_calibration: int | None = None,
    alpha: float = 0.10,
    strata: dict[str, np.ndarray] | None = None,
    gate: Gate | None = None,
    validate: bool = True,
) -> Result:
    """Score every forecast on anomalies against the training climatology.

    ``values`` has one row per forecast with the pooled-over-leads anomaly scores and
    the MSESS against climatology (identical on anomalies and on raw values, since the
    same climatology is subtracted from both sides). Tables: ``by_configuration`` (seed
    means and the verdict per configuration), ``by_lead`` and, when ``conformal`` is set,
    ``conformal`` for every forecast and for the climatology on the same origins.
    """
    if validate:
        benchmark.validate()
    decl = declaration
    g = gate or (decl.gate if decl else Gate())
    if score_origins is None:
        origins = (
            decl.score_origins(benchmark.n_origin, benchmark.horizon)
            if decl
            else np.arange(benchmark.n_origin)
        )
    else:
        origins = _origins(benchmark, score_origins)
    cells = benchmark.cell_mask(mask)
    obs = benchmark.observations
    clim_full = build_references(obs, benchmark.origin_period, benchmark.lead, ("climatology",))[
        "climatology"
    ]
    clim = _select(clim_full, origins, cells)
    target = _select(benchmark.target, origins, cells)
    target_anom = target - clim

    rows: list[dict[str, object]] = []
    lead_rows: list[dict[str, object]] = []
    conf_frames: list[pd.DataFrame] = []
    inputs: dict[str, str] = {}
    n_cal = n_calibration if n_calibration is not None else max(1, origins.size // 2)

    def score_one(
        label: str, kind: str, pred: np.ndarray, model: str, features: str, seed: int | None
    ) -> None:
        anom = pred - clim
        row: dict[str, object] = {
            "kind": kind,
            "model": model,
            "features": features,
            "seed": seed,
            "label": label,
            "n_origins": int(origins.size),
            "n_cells": int(target.shape[-1]),
            "msess": msess(pred, target, clim),
        }
        row.update(anomaly_scores(anom, target_anom))
        rows.append(row)
        if by_lead:
            for j, lead in enumerate(benchmark.lead.tolist()):
                lr: dict[str, object] = {
                    "label": label,
                    "kind": kind,
                    "model": model,
                    "features": features,
                    "seed": seed,
                    "lead": int(lead),
                    "msess": msess(pred[:, j], target[:, j], clim[:, j]),
                }
                lr.update(anomaly_scores(anom[:, j], target_anom[:, j]))
                lead_rows.append(lr)
        if conformal and origins.size >= 2:
            c = conformal_by_origin(pred, target, n_cal, alpha=alpha, strata=strata)
            c.insert(0, "label", label)
            conf_frames.append(c)

    score_one("climatology", "reference", clim, "climatology", "", None)
    for f in benchmark.forecasts:
        inputs[f.label] = f.hash()
        score_one(f.label, "forecast", _select(f.pred, origins, cells), f.model, f.features, f.seed)
    values = pd.DataFrame(rows)

    # per configuration: seed means and the verdict
    fc = values[values["kind"] == "forecast"]
    cfg_rows = []
    for (model, features), grp in fc.groupby(["model", "features"], sort=False):
        labelled = grp[grp["seed"].notna()]
        n_seeds = int(labelled["seed"].nunique()) + int(grp["seed"].isna().sum())
        rec: dict[str, object] = {"model": model, "features": features, "n_seeds": n_seeds}
        for c in (
            "acc_cell_mean",
            "acc_cell_median",
            "cells_positive",
            "acc_pattern",
            "acc_pooled",
            "msess",
        ):
            v = grp[c].to_numpy(dtype=float)
            rec[f"{c}_mean"] = float(np.nanmean(v))
            rec[f"{c}_sd"] = float(np.nanstd(v, ddof=1)) if v.size > 1 else np.nan
        stat = float(rec["acc_cell_median_mean"])  # type: ignore[arg-type]
        if n_seeds < g.min_seeds:
            rec["verdict"] = Verdict.INSUFFICIENT.value
        elif np.isfinite(stat) and stat > 0 and float(rec["msess_mean"]) > 0:  # type: ignore[arg-type]
            rec["verdict"] = Verdict.PASS.value
        else:
            rec["verdict"] = Verdict.FAIL.value
        cfg_rows.append(rec)
    per_config = pd.DataFrame(cfg_rows)
    verdicts = per_config["verdict"].tolist() if len(per_config) else []
    if any(v == Verdict.PASS.value for v in verdicts):
        verdict = Verdict.PASS
    elif any(v == Verdict.FAIL.value for v in verdicts):
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.INSUFFICIENT

    notes = [
        f"anomalies against the training climatology on {int(origins.size)} origins;"
        " the climatology scores zero on every anomaly correlation by construction"
    ]
    for _, r in per_config.iterrows():
        notes.append(
            f"{r['model']}[{r['features']}]: {r['verdict']} (per-cell temporal ACC, median over"
            f" cells, seed mean {r['acc_cell_median_mean']:+.3f}; pattern ACC"
            f" {r['acc_pattern_mean']:+.3f}; pooled ACC {r['acc_pooled_mean']:+.3f}, inflated;"
            f" MSESS {r['msess_mean']:+.3f}; n_seeds {int(r['n_seeds'])})"
        )
    tables: dict[str, pd.DataFrame] = {"by_configuration": per_config}
    if by_lead:
        tables["by_lead"] = pd.DataFrame(lead_rows)
    if conf_frames:
        conf = pd.concat(conf_frames, ignore_index=True)
        tables["conformal"] = conf
        summary = conf.groupby("label", sort=False)[["coverage", "width"]].mean().reset_index()
        clim_row = summary[summary["label"] == "climatology"]
        if len(clim_row):
            notes.append(
                f"conformal intervals (alpha {alpha:g}, {n_cal} calibration origins, guaranteed"
                f" level {conf['level_guaranteed'].iloc[0]:.3f}): climatology coverage"
                f" {clim_row['coverage'].iloc[0]:.3f}, mean truncated width"
                f" {clim_row['width'].iloc[0]:.1f}; a model's interval counts only if it is"
                " narrower at the same coverage"
            )
        tables["conformal_summary"] = summary
    return Result(
        component="C4 anomaly skill",
        title=f"Deseasonalised anomaly skill on {int(origins.size)} origins",
        values=values,
        verdict=verdict,
        rule=(
            "A configuration shows skill beyond climatology only when the seed mean of its"
            " per-cell temporal anomaly correlation (median over cells) and of its MSESS against"
            f" climatology are both positive, over at least {g.min_seeds} seeds; the pooled"
            " anomaly correlation is reported but never used."
        ),
        inputs={"observations": benchmark.inputs().get("observations", ""), **inputs},
        notes=notes,
        tables=tables,
        declaration_hash=decl.sha256 if decl else None,
        parameters={
            "score_origins": origins.tolist(),
            "gate": vars(g),
            "conformal": conformal,
            "n_calibration": n_cal if conformal else None,
            "alpha": alpha,
            "by_lead": by_lead,
        },
        deviations=deviations(
            decl, {"score_origins": score_origins is not None, "gate": gate is not None}
        ),
    )
