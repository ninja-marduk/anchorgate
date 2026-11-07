"""Component 0: is a fold assignment out of sample, given the forecast horizon?

Consecutive forecast origins share target periods: origin ``o`` at lead ``h + 1``
verifies against the same period as origin ``o + 1`` at lead ``h``. Assigning whole
origins to folds, even at random, therefore leaves most held-out targets inside the
training fold. This module measures that fraction for any assignment, counts the
independent units the record holds, and builds a purged holdout whose fraction is zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Benchmark, Leads
from .result import Result, Verdict


@dataclass
class Split:
    fit: np.ndarray
    score: np.ndarray


@dataclass
class FoldAssignment:
    """Named list of (fit origins, scored origins) splits over ``n_origin`` origins."""

    name: str
    n_origin: int
    splits: list[Split] = field(default_factory=list)
    kind: str = "origin"
    description: str = ""

    @property
    def n_splits(self) -> int:
        return len(self.splits)


# -------------------------------------------------------------------- constructors


def shuffled_kfold(
    n_origin: int, k: int = 5, seed: int = 42, name: str = "shuffled"
) -> FoldAssignment:
    """Whole origins assigned to ``k`` folds at random (permutation, then equal chunks)."""
    order = np.random.default_rng(seed).permutation(np.arange(n_origin))
    splits = [
        Split(fit=np.setdiff1d(np.arange(n_origin), held), score=np.sort(held))
        for held in np.array_split(order, k)
    ]
    return FoldAssignment(
        name,
        n_origin,
        splits,
        description=f"{k} folds of whole origins, random assignment, seed {seed}",
    )


def contiguous_kfold(n_origin: int, k: int = 5, name: str = "contiguous") -> FoldAssignment:
    """Whole origins in ``k`` contiguous blocks, no embargo."""
    splits = [
        Split(fit=np.setdiff1d(np.arange(n_origin), held), score=held)
        for held in np.array_split(np.arange(n_origin), k)
    ]
    return FoldAssignment(
        name, n_origin, splits, description=f"{k} contiguous blocks of origins, no embargo"
    )


def purged_holdout(
    n_origin: int,
    lead: Leads,
    score: Sequence[int] | np.ndarray,
    name: str = "purged",
) -> FoldAssignment:
    """One split: a contiguous scored block and the fit origins whose targets never
    reach into the block's target periods (an embargo of ``horizon - 1`` origins on
    each side when leads run ``1..horizon``)."""
    lead_arr = np.asarray(lead, dtype=int)
    score_arr = np.asarray(sorted(int(s) for s in score), dtype=int)
    if score_arr.size == 0:
        raise ValueError("the scored block is empty")
    lo = int(score_arr.min() + lead_arr.min())
    hi = int(score_arr.max() + lead_arr.max())
    fit = np.array(
        [
            o
            for o in range(n_origin)
            if (o + lead_arr.max() < lo or o + lead_arr.min() > hi)
            and o not in set(score_arr.tolist())
        ],
        dtype=int,
    )
    embargo = int(lead_arr.max()) - int(lead_arr.min())
    return FoldAssignment(
        name,
        n_origin,
        [Split(fit=fit, score=score_arr)],
        description=(
            f"scored origins {score_arr.min()}..{score_arr.max()},"
            f" embargo {embargo} origins each side"
        ),
    )


def released_scalar_shuffle(n_origin: int, name: str = "released") -> FoldAssignment:
    """Marker for folds shuffled over flattened scalars; see :func:`scalar_shuffle_overlap`."""
    return FoldAssignment(
        name,
        n_origin,
        [],
        kind="scalar",
        description="folds shuffled over flattened (origin, lead, cell) scalars",
    )


# ------------------------------------------------------------------------- overlap


def _target_periods(origin_period: np.ndarray, lead: np.ndarray) -> np.ndarray:
    return np.asarray(origin_period)[:, None] + np.asarray(lead)[None, :]


def overlap(
    assignment: FoldAssignment, origin_period: np.ndarray, lead: np.ndarray
) -> pd.DataFrame:
    """Per split: held-out (origin, lead) pairs whose target period is also a target of
    some fit pair, and their fraction."""
    tp = _target_periods(origin_period, lead)
    rows = []
    for i, s in enumerate(assignment.splits):
        fit_periods = set(tp[s.fit].ravel().tolist())
        held = tp[s.score].ravel()
        n_over = int(sum(p in fit_periods for p in held.tolist()))
        rows.append(
            {
                "assignment": assignment.name,
                "split": i,
                "n_fit": len(s.fit),
                "n_score": len(s.score),
                "n_pairs": int(held.size),
                "n_overlapping": n_over,
                "fraction": n_over / held.size if held.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def scalar_shuffle_overlap(
    origin_period: np.ndarray, lead: np.ndarray, n_cell: int, k: int = 5, seed: int = 42
) -> float:
    """Fraction of held-out scalars whose target period is present among training
    scalars when folds are drawn over flattened (origin, lead, cell) values."""
    tp = _target_periods(origin_period, lead)
    n_pair = tp.size
    n = n_pair * n_cell
    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=np.int64)
    perm = rng.permutation(n)
    for f, chunk in enumerate(np.array_split(perm, k)):
        fold[chunk] = f
    pair_of_scalar = np.repeat(np.arange(n_pair), n_cell)
    period_of_scalar = tp.ravel()[pair_of_scalar]
    periods, inv = np.unique(period_of_scalar, return_inverse=True)
    counts = np.zeros((k, periods.size), dtype=np.int64)
    np.add.at(counts, (fold, inv), 1)
    total = counts.sum(0)
    other = total[inv] - counts[fold, inv]
    return float(np.mean(other > 0))


def effective_units(origin_period: np.ndarray, lead: np.ndarray) -> dict[str, float]:
    """How many independent evaluation units the origins hold.

    ``n_pairs`` scored (origin, lead) pairs; ``n_periods`` distinct target periods;
    ``multiplicity`` how many times each period is scored (min, max);
    ``n_disjoint_windows`` how many origins can be chosen so no two share a target;
    ``n_years`` the span of the targets in years.
    """
    tp = _target_periods(origin_period, lead)
    periods, counts = np.unique(tp, return_counts=True)
    span = int(lead.max() - lead.min() + 1)
    chosen: list[int] = []
    last_end = -np.inf
    for o in range(tp.shape[0]):
        if tp[o].min() > last_end:
            chosen.append(o)
            last_end = tp[o].max()
    return {
        "n_origins": float(tp.shape[0]),
        "n_pairs": float(tp.size),
        "n_periods": float(periods.size),
        "multiplicity_min": float(counts.min()),
        "multiplicity_max": float(counts.max()),
        "window_span": float(span),
        "n_disjoint_windows": float(len(chosen)),
        "n_years": float(periods.size / 12.0),
    }


# --------------------------------------------------------------------------- audit


def audit(
    benchmark: Benchmark,
    assignments: Sequence[FoldAssignment],
    out_of_sample: Sequence[str] = (),
    k: int = 5,
    seed: int = 42,
    declaration_hash: str | None = None,
) -> Result:
    """Score every assignment's overlap; PASS when each one declared out of sample
    reports zero overlapping pairs."""
    op, lead = benchmark.origin_period, benchmark.lead
    frames = []
    for a in assignments:
        if a.kind == "scalar":
            frac = scalar_shuffle_overlap(op, lead, benchmark.reference.n_cell, k=k, seed=seed)
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "assignment": a.name,
                            "split": 0,
                            "n_fit": np.nan,
                            "n_score": np.nan,
                            "n_pairs": np.nan,
                            "n_overlapping": np.nan,
                            "fraction": frac,
                        }
                    ]
                )
            )
        else:
            frames.append(overlap(a, op, lead))
    table = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "assignment",
                "split",
                "n_fit",
                "n_score",
                "n_pairs",
                "n_overlapping",
                "fraction",
            ]
        )
    )
    summary = (
        table.groupby("assignment", sort=False)
        .agg(
            n_splits=("split", "count"),
            n_pairs=("n_pairs", "sum"),
            n_overlapping=("n_overlapping", "sum"),
            fraction=("fraction", "mean"),
        )
        .reset_index()
    )
    weighted = []
    for _name, g in table.groupby("assignment", sort=False):
        if g["n_pairs"].notna().all() and g["n_pairs"].sum() > 0:
            weighted.append(float(g["n_overlapping"].sum() / g["n_pairs"].sum()))
        else:
            weighted.append(float(g["fraction"].mean()))
    summary["fraction"] = weighted
    units = effective_units(op, lead)
    verdict = Verdict.NOT_APPLICABLE
    notes = [f"{k}: {v:g}" for k, v in units.items()]
    if out_of_sample:
        bad = summary[summary["assignment"].isin(list(out_of_sample)) & (summary["fraction"] > 0)]
        missing = set(out_of_sample) - set(summary["assignment"])
        if missing:
            raise ValueError(f"declared out-of-sample assignments not audited: {sorted(missing)}")
        verdict = Verdict.FAIL if len(bad) else Verdict.PASS
        for _, r in bad.iterrows():
            notes.append(
                f"{r['assignment']}: {100 * r['fraction']:.1f} % of held-out pairs"
                " have their target period in the fit set"
            )
    return Result(
        component="C0 fold audit",
        title="Target-period overlap of fold assignments",
        values=summary,
        verdict=verdict,
        rule=(
            "A score reported as out of sample must come from an assignment with no"
            " held-out target period in its fit set."
        ),
        inputs={
            "origin_period": str(int(op[0])) + ".." + str(int(op[-1])),
            "lead": ",".join(map(str, lead.tolist())),
        },
        notes=notes,
        tables={"splits": table, "effective_units": pd.DataFrame([units])},
        declaration_hash=declaration_hash,
    )


def ladder(
    benchmark: Benchmark, score: Sequence[int] | None = None, k: int = 5, seed: int = 42
) -> list[FoldAssignment]:
    """The standard rungs: released scalar shuffle, shuffled origins, contiguous blocks,
    purged holdout on ``score`` (default: the last ``horizon`` origins)."""
    n = benchmark.n_origin
    if score is None:
        score = list(range(max(0, n - benchmark.horizon), n))
    return [
        released_scalar_shuffle(n),
        shuffled_kfold(n, k=k, seed=seed),
        contiguous_kfold(n, k=k),
        purged_holdout(n, benchmark.lead, score),
    ]
