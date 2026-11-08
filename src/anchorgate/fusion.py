"""Component 3: calibration versus combination.

An ensemble's gain over its base learners has two parts: what an affine recalibration
of the best single learner already delivers, and what combining several learners adds
on top. The two are separated out of fold, on a fold assignment whose overlap the audit
(component 0) has measured, and the best base learner is chosen on the fit origins,
never on the scored ones; the advantage of choosing it on the scored origins instead is
reported as its own number, because it is the defect the protocol exists to expose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Benchmark, Forecast
from .declaration import Declaration
from .factorial import HIGHER_IS_BETTER, METRICS, _origins, _select, configuration_label
from .folds import (
    FoldAssignment,
    contiguous_kfold,
    overlap,
    purged_holdout,
    released_scalar_shuffle,
    scalar_shuffle_overlap,
    shuffled_kfold,
)
from .metrics import msess
from .references import build_references
from .result import Result, Verdict

Configuration = tuple[str, str]


# ------------------------------------------------------------------ affine algebra


def _affine_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``y ~ x @ w + c`` over finite rows; ``(p + 1,)``."""
    a = np.column_stack([x, np.ones(len(y))])
    ok = np.isfinite(y) & np.isfinite(a).all(1)
    if ok.sum() <= a.shape[1]:
        raise ValueError("too few finite rows to fit the combiner")
    coef, *_ = np.linalg.lstsq(a[ok], y[ok], rcond=None)
    return coef


def _affine_apply(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(len(x))]) @ coef


# ------------------------------------------------------------------- run pairing


@dataclass
class _Run:
    seed: int | None
    base: list[Forecast]
    ensemble: Forecast | None


def _paired_runs(
    benchmark: Benchmark, base: Sequence[Configuration], ensemble: Configuration | None
) -> list[_Run]:
    """One run per seed common to every base configuration (and the ensemble, if given).
    Unlabelled single runs are paired by position when every configuration has one."""
    configs = benchmark.configurations()

    def key(c: Configuration) -> Configuration:
        return (str(c[0]), str(c[1]))

    for c in [*base, *([ensemble] if ensemble else [])]:
        if key(c) not in configs:
            known = ", ".join(configuration_label(k) for k in configs)
            raise ValueError(
                f"configuration {configuration_label(c)} not in the benchmark ({known})"
            )
    groups = [configs[key(c)] for c in base]
    ens_group = configs[key(ensemble)] if ensemble else None
    all_groups = groups + ([ens_group] if ens_group else [])
    if all(len(g) == 1 for g in all_groups):
        return [
            _Run(
                seed=groups[0][0].seed,
                base=[g[0] for g in groups],
                ensemble=ens_group[0] if ens_group else None,
            )
        ]
    seed_sets = [{f.seed for f in g if f.seed is not None} for g in all_groups]
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not common:
        raise ValueError(
            "no seed is shared by every configuration; label the runs or supply one run each"
        )
    runs = []
    for s in common:
        picked = [next(f for f in g if f.seed == s) for g in groups]
        ens = next(f for f in ens_group if f.seed == s) if ens_group else None
        runs.append(_Run(seed=s, base=picked, ensemble=ens))
    return runs


# ------------------------------------------------------------------- the splits


def resolve_assignment(
    benchmark: Benchmark,
    assignment: FoldAssignment | str,
    score: Sequence[int] | np.ndarray | None = None,
    k: int = 5,
    seed: int = 42,
) -> FoldAssignment:
    """A named rung (``purged``, ``shuffled``, ``contiguous``, ``released``) or a
    ready-made :class:`FoldAssignment`."""
    if isinstance(assignment, FoldAssignment):
        return assignment
    n = benchmark.n_origin
    if assignment == "purged":
        block = list(range(max(0, n - benchmark.horizon), n)) if score is None else list(score)
        return purged_holdout(n, benchmark.lead, block)
    if assignment == "shuffled":
        return shuffled_kfold(n, k=k, seed=seed)
    if assignment == "contiguous":
        return contiguous_kfold(n, k=k)
    if assignment in ("released", "scalar"):
        return released_scalar_shuffle(n)
    raise ValueError(f"unknown assignment {assignment!r}")


def _rows_of(origins: np.ndarray, n_lead: int, n_cell: int) -> np.ndarray:
    """Flat row indices of every (origin, lead, cell) scalar of the given origins."""
    per = n_lead * n_cell
    return (np.asarray(origins, dtype=int)[:, None] * per + np.arange(per)[None, :]).ravel()


def _scalar_splits(n_rows: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.random.default_rng(seed).permutation(n_rows)
    out = []
    for held in np.array_split(idx, k):
        mask = np.ones(n_rows, bool)
        mask[held] = False
        out.append((np.flatnonzero(mask), np.sort(held)))
    return out


# --------------------------------------------------------------------- the component


def decompose_fusion(
    benchmark: Benchmark,
    declaration: Declaration | None = None,
    *,
    base: Sequence[Configuration] | None = None,
    ensemble: Configuration | None = None,
    assignment: FoldAssignment | str = "purged",
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    metric: str = "pooled_r2",
    min_seeds: int = 2,
    k: int = 5,
    seed: int = 42,
    validate: bool = True,
) -> Result:
    """Split an ensemble's gain into calibration and combination, out of fold.

    For every run (seed) and every split of ``assignment``: the best raw base learner is
    chosen on the fit origins; each base learner is recalibrated by an affine fit on the
    fit origins and the best recalibrated one is chosen there too; an affine combiner
    over all base learners is fitted on the fit origins. All are scored on the held-out
    origins, restricted to ``score_origins`` when given. The oracle picks (chosen on the
    scored origins) measure the selection advantage a comparator gets when it is picked
    on the evaluation set. A supplied ``ensemble`` is scored on the same origins for
    reference; its combiner may have seen them, and the report says so.

    Explicit keyword arguments override the declaration's ``fusion`` section.
    """
    if validate:
        benchmark.validate()
    decl = declaration
    spec = dict(decl.fusion) if decl is not None else {}
    if base is None and "base" in spec:
        base = [(str(b["model"]), str(b.get("features", ""))) for b in spec["base"]]
    if ensemble is None and spec.get("ensemble"):
        e = spec["ensemble"]
        ensemble = (str(e["model"]), str(e.get("features", "")))
    if isinstance(assignment, str) and "assignment" in spec and assignment == "purged":
        assignment = str(spec["assignment"])
    metric = str(spec.get("metric", metric))
    if base is None:
        base = [
            c for c in benchmark.configurations() if ensemble is None or tuple(c) != tuple(ensemble)
        ]
    if len(base) < 2:
        raise ValueError("decomposition needs at least two base configurations")
    if (
        score_origins is None
        and decl is not None
        and ("purged" in decl.scoring or decl.scoring.get("origins", "all") != "all")
    ):
        score_origins = decl.score_origins(benchmark.n_origin, benchmark.horizon)
    fa = resolve_assignment(benchmark, assignment, score=score_origins, k=k, seed=seed)
    higher = HIGHER_IS_BETTER.get(metric, True)
    sign = 1.0 if higher else -1.0

    cells = benchmark.cell_mask(mask)
    n_lead = benchmark.reference.n_lead
    target_full = _select(benchmark.target, np.arange(benchmark.n_origin), cells)
    n_cell = target_full.shape[-1]
    ref_flat: np.ndarray | None = None
    if metric == "msess":
        refs = build_references(
            benchmark.observations, benchmark.origin_period, benchmark.lead, ("climatology",)
        )
        ref_flat = _select(refs["climatology"], np.arange(benchmark.n_origin), cells).ravel()
    y_flat = target_full.ravel()

    def score(pred: np.ndarray, rows: np.ndarray) -> float:
        if ref_flat is not None:
            return msess(pred[rows], y_flat[rows], ref_flat[rows])
        return METRICS[metric](pred[rows], y_flat[rows])

    # the rows every number is finally scored on
    if fa.kind == "scalar":
        splits = _scalar_splits(y_flat.size, k, seed)
        covered = (
            np.arange(y_flat.size)
            if score_origins is None
            else _rows_of(_origins(benchmark, score_origins), n_lead, n_cell)
        )
    else:
        if not fa.splits:
            raise ValueError("the assignment has no splits")
        splits = [
            (_rows_of(s.fit, n_lead, n_cell), _rows_of(s.score, n_lead, n_cell)) for s in fa.splits
        ]
        covered = np.unique(np.concatenate([s.score for s in fa.splits]))
        if score_origins is not None:
            wanted = _origins(benchmark, score_origins)
            covered = np.intersect1d(covered, wanted)
            if covered.size == 0:
                raise ValueError("no scored origin is held out by the assignment")
        covered = _rows_of(covered, n_lead, n_cell)

    runs = _paired_runs(benchmark, base, ensemble)
    labels = [configuration_label(c) for c in base]
    rows_out: list[dict[str, object]] = []
    inputs: dict[str, str] = {}
    for run in runs:
        x_full = np.column_stack(
            [_select(f.pred, np.arange(benchmark.n_origin), cells).ravel() for f in run.base]
        )
        for f in run.base:
            inputs[f.label] = f.hash()
        if run.ensemble is not None:
            inputs[run.ensemble.label] = run.ensemble.hash()
        oof = {
            name: np.full(y_flat.size, np.nan)
            for name in ("fusion", "best_raw", "best_recal", "oracle_raw", "oracle_recal")
        }
        in_sample: list[float] = []
        picks_raw: list[int] = []
        picks_recal: list[int] = []
        for fit_rows, held_rows in splits:
            x_fit, y_fit = x_full[fit_rows], y_flat[fit_rows]
            raw_fit = [sign * score(x_full[:, i], fit_rows) for i in range(len(base))]
            pick = int(np.argmax(raw_fit))
            cal = [_affine_fit(x_fit[:, i : i + 1], y_fit) for i in range(len(base))]
            recal_full = [_affine_apply(x_full[:, i : i + 1], cal[i]) for i in range(len(base))]
            recal_fit = [sign * score(recal_full[i], fit_rows) for i in range(len(base))]
            pick_r = int(np.argmax(recal_fit))
            coef = _affine_fit(x_fit, y_fit)
            fusion_full = _affine_apply(x_full, coef)
            in_sample.append(score(fusion_full, fit_rows))
            picks_raw.append(pick)
            picks_recal.append(pick_r)
            oof["fusion"][held_rows] = fusion_full[held_rows]
            oof["best_raw"][held_rows] = x_full[held_rows, pick]
            oof["best_recal"][held_rows] = recal_full[pick_r][held_rows]
            raw_held = [sign * score(x_full[:, i], held_rows) for i in range(len(base))]
            recal_held = [sign * score(recal_full[i], held_rows) for i in range(len(base))]
            oof["oracle_raw"][held_rows] = x_full[held_rows, int(np.argmax(raw_held))]
            oof["oracle_recal"][held_rows] = recal_full[int(np.argmax(recal_held))][held_rows]
        rec: dict[str, object] = {
            "seed": run.seed,
            "n_splits": len(splits),
            "fusion": score(oof["fusion"], covered),
            "fusion_in_sample": float(np.mean(in_sample)),
            "best_raw": score(oof["best_raw"], covered),
            "best_recal": score(oof["best_recal"], covered),
            "oracle_raw": score(oof["oracle_raw"], covered),
            "oracle_recal": score(oof["oracle_recal"], covered),
            "picked_raw": labels[picks_raw[0]] if len(set(picks_raw)) == 1 else "varies by split",
            "picked_recal": labels[picks_recal[0]]
            if len(set(picks_recal)) == 1
            else "varies by split",
        }
        for i, lab in enumerate(labels):
            rec[f"raw: {lab}"] = score(x_full[:, i], covered)
        if run.ensemble is not None:
            rec["supplied_ensemble"] = score(
                _select(run.ensemble.pred, np.arange(benchmark.n_origin), cells).ravel(), covered
            )
        f_, br, bc = float(rec["fusion"]), float(rec["best_raw"]), float(rec["best_recal"])  # type: ignore[arg-type]
        rec["calibration_gain"] = sign * (bc - br)
        rec["combination_gain"] = sign * (f_ - bc)
        rec["total_gain"] = sign * (f_ - br)
        rec["selection_advantage_raw"] = sign * (float(rec["oracle_raw"]) - br)  # type: ignore[arg-type]
        rec["selection_advantage_recal"] = sign * (float(rec["oracle_recal"]) - bc)  # type: ignore[arg-type]
        rec["in_sample_gap"] = sign * (float(rec["fusion_in_sample"]) - f_)  # type: ignore[arg-type]
        rows_out.append(rec)
    values = pd.DataFrame(rows_out)

    # overlap of the assignment, so the verdict knows whether the numbers are out of sample
    if fa.kind == "scalar":
        frac = scalar_shuffle_overlap(
            benchmark.origin_period, benchmark.lead, benchmark.reference.n_cell, k=k, seed=seed
        )
    else:
        ov = overlap(fa, benchmark.origin_period, benchmark.lead)
        frac = (
            float(ov["n_overlapping"].sum() / ov["n_pairs"].sum())
            if ov["n_pairs"].sum()
            else np.nan
        )
    out_of_sample = bool(np.isfinite(frac) and frac == 0.0)

    numeric = [
        c for c in values.columns if c not in ("seed", "picked_raw", "picked_recal", "n_splits")
    ]
    summary_rows = []
    for c in numeric:
        v = values[c].to_numpy(dtype=float)
        summary_rows.append(
            {
                "quantity": c,
                "mean": float(np.nanmean(v)),
                "sd": float(np.nanstd(v, ddof=1)) if v.size > 1 else np.nan,
                "n_runs": int(np.isfinite(v).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    n_runs = len(values)
    comb_gain = float(values["combination_gain"].mean())
    cal_gain = float(values["calibration_gain"].mean())
    tot_gain = float(values["total_gain"].mean())
    fusion_sd = float(values["fusion"].std(ddof=1)) if n_runs > 1 else float("nan")
    notes = [
        f"assignment '{fa.name}': {fa.description}; target-period overlap {100 * frac:.1f} %",
        f"{n_runs} paired run(s); base learners: {', '.join(labels)}",
        f"fusion {values['fusion'].mean():.3f} +/- {fusion_sd:.3f};"
        f" best raw base learner (chosen on fit) {values['best_raw'].mean():.3f};"
        f" recalibrated {values['best_recal'].mean():.3f} ({metric})",
        f"gain over the raw best {tot_gain:+.3f} = calibration {cal_gain:+.3f}"
        f" + combination {comb_gain:+.3f}",
        f"selection advantage of picking the base learner on the scored origins: raw"
        f" {values['selection_advantage_raw'].mean():+.3f}, recalibrated"
        f" {values['selection_advantage_recal'].mean():+.3f}",
        f"in-sample minus out-of-sample fusion: {values['in_sample_gap'].mean():+.3f}",
    ]
    picked = ", ".join(f"seed {r['seed']}: {r['picked_raw']}" for _, r in values.iterrows())
    notes.append(f"base learner picked on the fit origins: {picked}")
    if "supplied_ensemble" in values:
        notes.append(
            f"supplied ensemble {configuration_label(ensemble or ('', ''))} scores"
            f" {values['supplied_ensemble'].mean():.3f} on the same origins (its combiner may have"
            " seen them; compare with 'fusion', not with the gate)"
        )
    if n_runs < min_seeds:
        verdict = Verdict.INSUFFICIENT
        notes.append(f"fewer than {min_seeds} paired runs: the gain cannot be claimed")
    elif not out_of_sample:
        verdict = Verdict.INSUFFICIENT
        notes.append("the assignment is not out of sample: the gain cannot be claimed on it")
    elif comb_gain > 0:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL
    return Result(
        component="C3 fusion decomposition",
        title=f"Calibration versus combination on '{fa.name}' ({metric})",
        values=values,
        verdict=verdict,
        rule=(
            "An ensemble's gain is claimed only over its recalibrated best base learner, chosen"
            " on the fit origins, on an assignment with no target-period overlap and at least"
            f" {min_seeds} paired runs; the mean combination gain over runs must be positive."
        ),
        inputs=inputs,
        notes=notes,
        tables={
            "summary": summary,
            "assignment": pd.DataFrame(
                [
                    {
                        "name": fa.name,
                        "kind": fa.kind,
                        "n_splits": len(splits),
                        "overlap_fraction": frac,
                    }
                ]
            ),
        },
        declaration_hash=decl.sha256 if decl else None,
    )


def fusion_ladder(
    benchmark: Benchmark,
    *,
    base: Sequence[Configuration] | None = None,
    ensemble: Configuration | None = None,
    score: Sequence[int] | np.ndarray | None = None,
    rungs: Sequence[FoldAssignment | str] = ("released", "shuffled", "contiguous", "purged"),
    mask: np.ndarray | str | None = None,
    metric: str = "pooled_r2",
    k: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[Result]]:
    """The same decomposition on every rung, all scored on the same origins.

    Returns a one-row-per-rung table (fusion, best raw, best recalibrated, gains,
    overlap, verdict) and the full results.
    """
    n = benchmark.n_origin
    block = list(range(max(0, n - benchmark.horizon), n)) if score is None else list(score)
    results = []
    rows = []
    for r in rungs:
        fa = resolve_assignment(benchmark, r, score=block, k=k, seed=seed)
        res = decompose_fusion(
            benchmark,
            base=base,
            ensemble=ensemble,
            assignment=fa,
            score_origins=block,
            mask=mask,
            metric=metric,
            k=k,
            seed=seed,
            validate=False,
        )
        results.append(res)
        v = res.values

        def sd(column: str, frame: pd.DataFrame = v) -> float:
            return float(frame[column].std(ddof=1)) if len(frame) > 1 else float("nan")

        rows.append(
            {
                "rung": fa.name,
                "overlap_fraction": float(res.tables["assignment"]["overlap_fraction"].iloc[0]),
                "fusion": float(v["fusion"].mean()),
                "fusion_sd": sd("fusion"),
                "best_raw": float(v["best_raw"].mean()),
                "best_recal": float(v["best_recal"].mean()),
                "calibration_gain": float(v["calibration_gain"].mean()),
                "combination_gain": float(v["combination_gain"].mean()),
                "total_gain": float(v["total_gain"].mean()),
                "total_gain_sd": sd("total_gain"),
                "verdict": res.verdict.value,
            }
        )
    return pd.DataFrame(rows), results


__all__ = ["decompose_fusion", "fusion_ladder", "resolve_assignment"]
