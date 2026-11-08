"""Component 2: the seed-resolved factorial.

Runs crossed by user-named factors (model x features by default) with the seed as a
block. Every difference is judged against the dispersion of its paired, within-seed
difference, never against the marginal spread of either arm; a configuration with a
single seed is listed and left out of the family; and the minimum detectable effect at
the number of seeds is printed next to every "not significant", so that an unresolved
ranking is not read as equivalence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from .data import Benchmark
from .declaration import Declaration, deviations
from .folds import effective_units
from .metrics import bias, mae, msess, per_cell_nse, pooled_r2, rmse
from .references import build_references
from .result import Result, Verdict

METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pooled_r2": pooled_r2,
    "per_cell_nse": per_cell_nse,
    "rmse": rmse,
    "mae": mae,
    "bias": bias,
}
HIGHER_IS_BETTER = {
    "pooled_r2": True,
    "per_cell_nse": True,
    "msess": True,
    "rmse": False,
    "mae": False,
    "bias": False,
}
AGGREGATES = ("lead_mean", "lead_peak", "pooled")


def configuration_label(levels: Sequence[object]) -> str:
    """``model[features]`` for two factors, otherwise the levels joined by ``/``."""
    parts = [str(x) for x in levels]
    if len(parts) == 2:
        return f"{parts[0]}[{parts[1]}]"
    return "/".join(parts)


def _origins(benchmark: Benchmark, score_origins: Sequence[int] | np.ndarray | None) -> np.ndarray:
    n = benchmark.n_origin
    if score_origins is None:
        return np.arange(n)
    o = np.asarray(sorted(int(x) for x in score_origins), dtype=int)
    if o.size == 0 or o.min() < 0 or o.max() >= n:
        raise ValueError(f"scored origins must lie in 0..{n - 1}")
    return o


def _select(arr: np.ndarray, origins: np.ndarray, cells: np.ndarray | None) -> np.ndarray:
    a = arr[origins]
    return a if cells is None else a[..., cells]


def _scorer(
    benchmark: Benchmark,
    metric: str,
    origins: np.ndarray,
    cells: np.ndarray | None,
    reference: str,
) -> tuple[Callable[[np.ndarray, np.ndarray], float], np.ndarray | None]:
    if metric == "msess":
        refs = build_references(
            benchmark.observations, benchmark.origin_period, benchmark.lead, (reference,)
        )
        return msess, _select(refs[reference], origins, cells)  # type: ignore[return-value]
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {[*sorted(METRICS), 'msess']}")
    return METRICS[metric], None


# ------------------------------------------------------------------- responses


def lead_scores(
    benchmark: Benchmark,
    metric: str = "pooled_r2",
    *,
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    reference: str = "climatology",
) -> pd.DataFrame:
    """One row per run and lead: ``label, model, features, seed, lead, y``."""
    origins = _origins(benchmark, score_origins)
    cells = benchmark.cell_mask(mask)
    target = _select(benchmark.target, origins, cells)
    fn, ref = _scorer(benchmark, metric, origins, cells, reference)
    rows = []
    for f in benchmark.forecasts:
        pred = _select(f.pred, origins, cells)
        for j, lead in enumerate(benchmark.lead.tolist()):
            y = (
                msess(pred[:, j], target[:, j], ref[:, j])
                if ref is not None
                else fn(pred[:, j], target[:, j])
            )
            rows.append(
                {
                    "label": f.label,
                    "model": f.model,
                    "features": f.features,
                    "seed": f.seed,
                    "lead": int(lead),
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def responses(
    benchmark: Benchmark,
    metric: str = "pooled_r2",
    *,
    aggregate: str = "lead_mean",
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    reference: str = "climatology",
) -> pd.DataFrame:
    """One response per run: ``label, model, features, seed, y``.

    ``aggregate`` is ``lead_mean`` (mean over leads of the per-lead score, the response
    of a blocked analysis), ``lead_peak`` (the best lead, which is what a
    maximum-selecting report quotes and what inflates with the number of seeds and
    leads searched) or ``pooled`` (one score over every scored pair).
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}")
    if aggregate == "pooled":
        origins = _origins(benchmark, score_origins)
        cells = benchmark.cell_mask(mask)
        target = _select(benchmark.target, origins, cells)
        fn, ref = _scorer(benchmark, metric, origins, cells, reference)
        rows = []
        for f in benchmark.forecasts:
            pred = _select(f.pred, origins, cells)
            y = msess(pred, target, ref) if ref is not None else fn(pred, target)
            rows.append(
                {
                    "label": f.label,
                    "model": f.model,
                    "features": f.features,
                    "seed": f.seed,
                    "y": y,
                }
            )
        return pd.DataFrame(rows)
    ls = lead_scores(benchmark, metric, score_origins=score_origins, mask=mask, reference=reference)
    higher = HIGHER_IS_BETTER.get(metric, True)
    how = "mean" if aggregate == "lead_mean" else ("max" if higher else "min")
    g = ls.groupby(["label", "model", "features"], sort=False, dropna=False)
    out = g.agg(seed=("seed", "first"), y=("y", how)).reset_index()
    return out[["label", "model", "features", "seed", "y"]]


# ------------------------------------------------------------- small statistics


def holm(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, order preserved."""
    arr = np.asarray(p, dtype=float)
    m = arr.size
    adj = np.empty(m)
    running = 0.0
    for rank, k in enumerate(np.argsort(arr)):
        running = max(running, (m - rank) * arr[k])
        adj[k] = min(running, 1.0)
    return adj


def cliffs_delta(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """``P(a > b) - P(a < b)`` over all pairs; 1 when every ``a`` exceeds every ``b``."""
    x = np.asarray(a, dtype=float)[:, None]
    y = np.asarray(b, dtype=float)[None, :]
    if x.size == 0 or y.size == 0:
        return float("nan")
    more = int(np.count_nonzero(x > y))
    less = int(np.count_nonzero(x < y))
    return float((more - less) / (x.size * y.size))


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference of mean ranks for ``k`` treatments over ``n`` units."""
    if k < 2 or n < 1:
        return float("nan")
    q = float(stats.studentized_range.ppf(1 - alpha, k, 1e6)) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def minimum_detectable_effect(
    paired_sd: Sequence[float] | np.ndarray, n: int, alpha: float = 0.05
) -> dict[str, float]:
    """Smallest paired mean difference a two-sided paired t-test at ``n`` blocks can
    separate, ``t_crit * s_d / sqrt(n)``, over the given paired standard deviations."""
    sd = np.asarray(paired_sd, dtype=float)
    sd = sd[np.isfinite(sd)]
    if n < 2 or sd.size == 0:
        return {
            "n": float(n),
            "factor": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    factor = float(stats.t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n))
    return {
        "n": float(n),
        "factor": factor,
        "min": float(factor * sd.min()),
        "median": float(factor * np.median(sd)),
        "max": float(factor * sd.max()),
    }


# ------------------------------------------------------------- blocked analysis


def _ss_between(y: np.ndarray, labels: np.ndarray) -> float:
    gm = y.mean()
    s = 0.0
    for lv in np.unique(labels):
        m = labels == lv
        s += m.sum() * (y[m].mean() - gm) ** 2
    return float(s)


def rcbd_anova(
    table: pd.DataFrame,
    factors: Sequence[str] = ("model", "features"),
    block: str = "seed",
    response: str = "y",
) -> pd.DataFrame:
    """Randomised complete block ANOVA with one or two factors and the seed as block.

    Requires a complete design: every factor-level combination once in every block.
    """
    factors = list(factors)
    if len(factors) not in (1, 2):
        raise ValueError("rcbd_anova supports one or two factors; combine levels otherwise")
    df = table[[block, *factors, response]].dropna()
    counts = df.groupby([block, *factors], observed=True).size()
    n_blocks = df[block].nunique()
    n_cells = int(np.prod([df[f].nunique() for f in factors]))
    if (counts != 1).any() or len(counts) != n_blocks * n_cells:
        raise ValueError("the design is not a complete block design (one run per cell per block)")
    y = df[response].to_numpy(dtype=float)
    n = y.size
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_blk = _ss_between(y, df[block].to_numpy())
    rows: list[tuple[str, float, int]] = []
    df_used = n_blocks - 1
    ss_used = ss_blk
    rows.append((f"block ({block})", ss_blk, n_blocks - 1))
    main_ss = []
    main_df = []
    for f in factors:
        s = _ss_between(y, df[f].to_numpy())
        d = df[f].nunique() - 1
        main_ss.append(s)
        main_df.append(d)
        rows.append((f, s, d))
        ss_used += s
        df_used += d
    if len(factors) == 2:
        combined = df[factors[0]].astype(str) + "\x1f" + df[factors[1]].astype(str)
        ss_cells = _ss_between(y, combined.to_numpy())
        ss_int = ss_cells - sum(main_ss)
        df_int = int(np.prod(main_df))
        rows.append((f"{factors[0]} x {factors[1]}", ss_int, df_int))
        ss_used += ss_int
        df_used += df_int
    ss_err = ss_tot - ss_used
    df_err = n - 1 - df_used
    ms_err = ss_err / df_err if df_err > 0 else float("nan")
    out = []
    for source, ss_r, df_r in rows:
        ms = ss_r / df_r if df_r > 0 else float("nan")
        F = ms / ms_err if (df_r > 0 and np.isfinite(ms_err) and ms_err > 0) else float("nan")
        p = float(stats.f.sf(F, df_r, df_err)) if np.isfinite(F) else float("nan")
        out.append({"source": source, "ss": ss_r, "df": df_r, "ms": ms, "F": F, "p": p})
    out.append(
        {"source": "residual", "ss": ss_err, "df": df_err, "ms": ms_err, "F": np.nan, "p": np.nan}
    )
    out.append(
        {"source": "total", "ss": ss_tot, "df": n - 1, "ms": np.nan, "F": np.nan, "p": np.nan}
    )
    return pd.DataFrame(out)


def permutation_test(
    table: pd.DataFrame,
    factor: str,
    block: str = "seed",
    response: str = "y",
    n_perm: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test of a factor's between-level sum of squares, permuting the
    factor's labels only inside each block. No distributional assumption."""
    df = table[[block, factor, response]].dropna()
    y = df[response].to_numpy(dtype=float)
    blocks = df[block].to_numpy()
    lab, _ = pd.factorize(df[factor])
    k = int(lab.max()) + 1
    gm = y.mean()

    def statistic(labels: np.ndarray) -> np.ndarray:
        s = np.zeros(labels.shape[0])
        for j in range(k):
            m = labels == j
            n_j = m.sum(1)
            sums = (m * y[None, :]).sum(1)
            mean_j = np.divide(sums, n_j, out=np.full_like(sums, gm, dtype=float), where=n_j > 0)
            s += n_j * (mean_j - gm) ** 2
        return s

    observed = float(statistic(lab[None, :])[0])
    rng = np.random.default_rng(seed)
    perm = np.tile(lab, (n_perm, 1))
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        perm[:, idx] = rng.permuted(perm[:, idx], axis=1)
    hits = int(np.sum(statistic(perm) >= observed - 1e-12))
    return {"statistic": observed, "p": (hits + 1) / (n_perm + 1), "n_perm": float(n_perm)}


def paired_contrasts(
    table: pd.DataFrame,
    factors: Sequence[str] = ("model", "features"),
    block: str = "seed",
    response: str = "y",
    family: Sequence[Sequence[str]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Every pairwise configuration contrast as a paired difference over blocks.

    Columns: the two configurations, ``n`` paired blocks, ``mean_d``, ``sd_d`` (the
    dispersion the crossed design supplies), ``marginal_sd`` (the dispersion a naive
    comparison uses), ``t``, ``p``, ``p_holm`` over the family, ``wins`` (blocks in
    which ``a`` exceeds ``b``) and Cliff's delta between the two arms' block values.
    Pairs with fewer than two common blocks are returned with ``p`` NaN and are not in
    the Holm family.
    """
    factors = list(factors)
    df = table[[block, *factors, response]].dropna(subset=[block, response]).copy()
    df["configuration"] = [configuration_label(r) for r in df[factors].itertuples(index=False)]
    wide = df.pivot_table(index=block, columns="configuration", values=response, aggfunc="mean")
    cfgs = list(wide.columns)
    if family is None:
        pairs = list(combinations(cfgs, 2))
    else:
        pairs = []
        for a, b in family:
            if a not in cfgs or b not in cfgs:
                raise ValueError(f"contrast {a!r} vs {b!r} names a configuration not in the table")
            pairs.append((a, b))
    rows = []
    for a, b in pairs:
        both = wide[[a, b]].dropna()
        d = (both[a] - both[b]).to_numpy(dtype=float)
        n = d.size
        rec: dict[str, object] = {"a": a, "b": b, "n": n}
        if n >= 2:
            sd = float(d.std(ddof=1))
            mean = float(d.mean())
            marg = float(np.sqrt(both[a].var(ddof=1) + both[b].var(ddof=1)))
            t = mean / (sd / np.sqrt(n)) if sd > 0 else (np.inf if mean != 0 else np.nan)
            p = (
                float(2 * stats.t.sf(abs(t), n - 1))
                if np.isfinite(t)
                else (0.0 if mean != 0 else np.nan)
            )
            rec.update(
                mean_d=mean,
                sd_d=sd,
                marginal_sd=marg,
                t=float(t),
                p=p,
                wins=int(np.sum(d > 0)),
                cliffs_delta=cliffs_delta(both[a].to_numpy(), both[b].to_numpy()),
            )
        else:
            rec.update(
                mean_d=float(d.mean()) if n else np.nan,
                sd_d=np.nan,
                marginal_sd=np.nan,
                t=np.nan,
                p=np.nan,
                wins=int(np.sum(d > 0)) if n else 0,
                cliffs_delta=np.nan,
            )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    ok = out["p"].notna()
    if ok.any():
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p"].to_numpy())
    out["separates"] = out["p_holm"] < alpha
    return out.sort_values(["p", "a", "b"], na_position="last").reset_index(drop=True)


def inflation(
    table: pd.DataFrame,
    factors: Sequence[str] = ("model", "features"),
    block: str = "seed",
    response: str = "y",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Per configuration: seeds, seed mean and spread, the best seed, and the inflation
    ``best - mean`` that quoting the best seed adds (sign set so that it is positive)."""
    factors = list(factors)
    df = table[[block, *factors, response]].dropna(subset=[response]).copy()
    df["configuration"] = [configuration_label(r) for r in df[factors].itertuples(index=False)]
    rows = []
    for cfg, g in df.groupby("configuration", sort=False):
        v = g[response].to_numpy(dtype=float)
        n = int(g[block].notna().sum()) if g[block].notna().any() else len(v)
        best = float(v.max()) if higher_is_better else float(v.min())
        mean = float(v.mean())
        rows.append(
            {
                "configuration": cfg,
                "n_seeds": n,
                "mean": mean,
                "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                "best": best,
                "inflation": (best - mean) if higher_is_better else (mean - best),
            }
        )
    return pd.DataFrame(rows)


def group_contrasts(
    table: pd.DataFrame,
    group: str,
    response: str = "y",
    pairs: Sequence[Sequence[str]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mann-Whitney contrasts between groups of rows (the rows are the sampling units),
    with one Holm family over the requested pairs and Cliff's delta as effect size.

    Use this when the units are configurations, stations or regions rather than seeds;
    the function does not know whether the rows are independent and says so in the
    report, which is the reader's cue to check.
    """
    df = table[[group, response]].dropna()
    groups = list(pd.unique(df[group]))
    if pairs is None:
        pairs = list(combinations(groups, 2))
    rows = []
    for a, b in pairs:
        x = df.loc[df[group] == a, response].to_numpy(dtype=float)
        y = df.loc[df[group] == b, response].to_numpy(dtype=float)
        if x.size == 0 or y.size == 0:
            raise ValueError(f"group {a!r} or {b!r} has no rows")
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append(
            {
                "a": a,
                "b": b,
                "n_a": int(x.size),
                "n_b": int(y.size),
                "mean_a": float(x.mean()),
                "mean_b": float(y.mean()),
                "U_a": float(u),
                "U_b": float(x.size * y.size - u),
                "p": float(p),
                "cliffs_delta": cliffs_delta(x, y),
                # the smallest two-sided p the test can return at these sizes
                "p_floor": 2.0 / comb(x.size + y.size, x.size),
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy()) if len(out) else np.nan
    out["separates"] = out["p_holm"] < alpha
    return out


def friedman_nemenyi(
    matrix: np.ndarray,
    names: Sequence[str],
    alpha: float = 0.05,
    n_effective: int | None = None,
    higher_is_better: bool = False,
) -> dict[str, pd.DataFrame]:
    """Friedman test over ``matrix`` of shape ``(units, treatments)`` and the Nemenyi
    critical difference at the number of units and, if given, at ``n_effective``.

    Rank 1 is the best treatment within each unit. Returns ``test``, ``ranks`` and
    ``pairs`` tables; a pair is "beyond_cd" when its mean-rank difference exceeds the
    critical difference at the nominal ``n`` and "beyond_cd_effective" at ``n_effective``.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError("matrix must be (units, treatments) with one name per treatment")
    x = x[np.isfinite(x).all(1)]
    n, k = x.shape
    if n < 2 or k < 3:
        raise ValueError(
            "need at least two units and three treatments; two treatments are a paired contrast"
        )
    signed = -x if higher_is_better else x
    chi2, p = stats.friedmanchisquare(*signed.T)
    ranks = stats.rankdata(signed, axis=1).mean(0)
    cd = nemenyi_cd(k, n, alpha)
    cd_eff = nemenyi_cd(k, int(n_effective), alpha) if n_effective else np.nan
    test = pd.DataFrame(
        [
            {
                "n_units": n,
                "k": k,
                "statistic": float(chi2),
                "p": float(p),
                "cd": cd,
                "n_effective": n_effective if n_effective else np.nan,
                "cd_effective": cd_eff,
            }
        ]
    )
    rank_table = pd.DataFrame(
        {"treatment": list(names), "mean_rank": ranks, "mean": x.mean(0)}
    ).sort_values("mean_rank")
    pairs = []
    for i, j in combinations(range(k), 2):
        d = abs(ranks[i] - ranks[j])
        pairs.append(
            {
                "a": names[i],
                "b": names[j],
                "rank_diff": float(d),
                "beyond_cd": bool(d > cd),
                "beyond_cd_effective": bool(d > cd_eff) if np.isfinite(cd_eff) else None,
            }
        )
    return {"test": test, "ranks": rank_table.reset_index(drop=True), "pairs": pd.DataFrame(pairs)}


# ---------------------------------------------------------------------- the component


def factorial(
    benchmark: Benchmark,
    declaration: Declaration | None = None,
    *,
    factors: Sequence[str] = ("model", "features"),
    metric: str = "pooled_r2",
    aggregate: str = "lead_mean",
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    block: str = "seed",
    n_perm: int = 20000,
    seed: int = 42,
    alpha: float = 0.05,
    family: Sequence[Sequence[str]] | None = None,
    friedman: bool = True,
    validate: bool = True,
) -> Result:
    """Resolve seed variance before any claim about factors.

    ``values`` holds one response per run. Tables: ``configurations`` (seed mean,
    spread, inflation from quoting the best seed), ``anova`` (randomised complete block
    ANOVA when the design is complete), ``permutation`` (within-block permutation
    p-value per factor), ``contrasts`` (paired differences with Holm over the family),
    ``mde`` (minimum detectable effect at the number of seeds), ``effective_units`` and,
    when ``friedman`` is set, ``friedman``, ``ranks`` and ``rank_pairs`` over origins.
    Explicit keyword arguments override the declaration's ``factorial`` section.
    """
    if validate:
        benchmark.validate()
    decl = declaration
    spec = dict(decl.factorial) if decl is not None else {}
    factors = list(spec.get("factors", factors)) if "factors" in spec else list(factors)
    metric = str(spec.get("metric", metric))
    aggregate = str(spec.get("aggregate", aggregate))
    n_perm = int(spec.get("n_perm", n_perm))
    if score_origins is None:
        origins = decl.score_origins(benchmark.n_origin, benchmark.horizon) if decl else None
    else:
        origins = _origins(benchmark, score_origins)
    family_given = family is not None
    if family is None and decl is not None and decl.contrasts:
        family = [list(c) for c in decl.contrasts]
    higher = HIGHER_IS_BETTER.get(metric, True)

    table = responses(benchmark, metric, aggregate=aggregate, score_origins=origins, mask=mask)
    for f in factors:
        if f not in table.columns:
            raise ValueError(f"factor {f!r} is not a column of the response table")
    table["configuration"] = [
        configuration_label(r) for r in table[factors].itertuples(index=False)
    ]
    peak = responses(benchmark, metric, aggregate="lead_peak", score_origins=origins, mask=mask)
    infl = inflation(peak, factors, block, "y", higher_is_better=higher)
    cfg = inflation(table, factors, block, "y", higher_is_better=higher).rename(
        columns={"inflation": f"inflation_{aggregate}"}
    )
    cfg = cfg.merge(
        infl[["configuration", "best", "inflation"]].rename(
            columns={"best": "best_seed_peak", "inflation": "inflation_peak"}
        ),
        on="configuration",
        how="left",
    )
    multi = cfg[cfg["n_seeds"] >= 2]["configuration"].tolist()
    single = cfg[cfg["n_seeds"] < 2]["configuration"].tolist()
    kept = table[table["configuration"].isin(multi) & table[block].notna()]

    notes: list[str] = []
    tables: dict[str, pd.DataFrame] = {"configurations": cfg}
    if single:
        notes.append(f"excluded from the family (one seed): {', '.join(single)}")
    inputs = benchmark.inputs()

    # blocked ANOVA and permutation tests, when the design is complete
    active = [f for f in factors if kept[f].nunique() > 1]
    for f in factors:
        if f not in active and len(kept):
            notes.append(f"factor {f!r} has the single level {kept[f].iloc[0]!r} and is not tested")
    try:
        if not active:
            raise ValueError("no factor has two or more levels among the multi-seed runs")
        anova = rcbd_anova(kept, active, block, "y")
        tables["anova"] = anova
        blk = float(anova.loc[anova["source"].str.startswith("block"), "ss"].iloc[0])
        tot = float(anova.loc[anova["source"] == "total", "ss"].iloc[0])
        if tot > 0:
            notes.append(
                f"the seed block absorbs {100 * blk / tot:.1f} % of the total sum of squares"
            )
        perm_rows = []
        for f in active:
            r = permutation_test(kept, f, block, "y", n_perm=n_perm, seed=seed)
            perm_rows.append({"factor": f, **r})
            a_p = float(anova.loc[anova["source"] == f, "p"].iloc[0])
            notes.append(f"{f}: blocked ANOVA p {a_p:.4f}, within-seed permutation p {r['p']:.4f}")
        tables["permutation"] = pd.DataFrame(perm_rows)
    except ValueError as exc:
        notes.append(f"blocked ANOVA not computed: {exc}")

    contrasts = paired_contrasts(kept, factors, block, "y", family=family, alpha=alpha)
    tables["contrasts"] = contrasts
    tested = contrasts[contrasts["p"].notna()] if len(contrasts) else contrasts
    n_raw = int((tested["p"] < alpha).sum()) if len(tested) else 0
    n_holm = int(tested["separates"].sum()) if len(tested) else 0
    if len(tested):
        tighter = int((tested["sd_d"] < tested["marginal_sd"]).sum())
        notes.append(
            f"{len(tested)} contrasts in the family: {n_raw} separate at alpha {alpha:g} before"
            f" correction, {n_holm} after Holm; the paired dispersion is below the marginal"
            f" one in {tighter} of them"
        )
        n_blocks = int(tested["n"].min())
        mde = minimum_detectable_effect(tested["sd_d"].to_numpy(), n_blocks, alpha)
        span = float(
            cfg.loc[cfg["n_seeds"] >= 2, "mean"].max() - cfg.loc[cfg["n_seeds"] >= 2, "mean"].min()
        )
        mde["design_span"] = span
        mde["median_over_span"] = mde["median"] / span if span > 0 else np.nan
        tables["mde"] = pd.DataFrame([mde])
        notes.append(
            f"minimum detectable effect at {n_blocks} seeds: {mde['min']:.3f} to {mde['max']:.3f}"
            f" ({metric}, median {mde['median']:.3f}); the configuration means span {span:.3f}"
        )

    # units behind the origins, and Friedman over them
    origin_idx = _origins(benchmark, origins)
    units = effective_units(benchmark.origin_period[origin_idx], benchmark.lead)
    tables["effective_units"] = pd.DataFrame([units])
    n_eff = int(units["n_disjoint_windows"])
    notes.append(
        f"{int(units['n_origins'])} scored origins hold {int(units['n_periods'])} distinct target"
        f" periods and {n_eff} disjoint windows"
    )
    if friedman and len(multi) >= 3 and origin_idx.size >= 2:
        mat, names = _per_origin_matrix(benchmark, kept, metric, origin_idx, mask)
        if mat.shape[1] >= 3:
            fr = friedman_nemenyi(mat, names, alpha, n_effective=n_eff, higher_is_better=higher)
            tables["friedman"] = fr["test"]
            tables["ranks"] = fr["ranks"]
            tables["rank_pairs"] = fr["pairs"]
            t = fr["test"].iloc[0]
            notes.append(
                f"Friedman over {int(t['n_units'])} origins: p {t['p']:.2e}; Nemenyi CD"
                f" {t['cd']:.2f} at that N and {t['cd_effective']:.2f} at {n_eff} disjoint windows"
            )

    if not len(tested):
        verdict = Verdict.INSUFFICIENT
    elif n_holm > 0:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL
    params: dict[str, object] = {
        "factors": list(factors),
        "block": block,
        "metric": metric,
        "aggregate": aggregate,
        "n_perm": n_perm,
        "alpha": alpha,
        "family": [list(c) for c in family] if family is not None else "all pairs",
        "score_origins": origin_idx.tolist(),
    }
    devs = deviations(
        decl,
        {
            "score_origins": score_origins is not None,
            "family": family_given,
        },
    )
    return Result(
        component="C2 seed-resolved factorial",
        title=f"{' x '.join(factors)} with {block} as block, response {metric} ({aggregate})",
        values=table,
        verdict=verdict,
        rule=(
            "A difference between configurations is claimed only when its paired, within-seed"
            " difference survives Holm correction over the declared family; otherwise the"
            " ranking is unresolved at this number of seeds and the minimum detectable effect"
            " is reported. Single-seed configurations are listed, not compared."
        ),
        inputs=inputs,
        notes=notes,
        tables=tables,
        declaration_hash=decl.sha256 if decl else None,
        parameters=params,
        deviations=devs,
    )


def _per_origin_matrix(
    benchmark: Benchmark,
    kept: pd.DataFrame,
    metric: str,
    origins: np.ndarray,
    mask: np.ndarray | str | None,
) -> tuple[np.ndarray, list[str]]:
    """Seed-mean score per origin for every multi-seed configuration: ``(origins, k)``."""
    cells = benchmark.cell_mask(mask)
    target = _select(benchmark.target, origins, cells)
    fn, ref = _scorer(benchmark, metric, origins, cells, "climatology")
    by_label = {f.label: f for f in benchmark.forecasts}
    names: list[str] = []
    cols: list[np.ndarray] = []
    for cfg_name, g in kept.groupby("configuration", sort=False):
        per_seed = []
        for label in g["label"]:
            f = by_label[label]
            pred = _select(f.pred, origins, cells)
            per_seed.append(
                [
                    msess(pred[i], target[i], ref[i]) if ref is not None else fn(pred[i], target[i])
                    for i in range(origins.size)
                ]
            )
        names.append(str(cfg_name))
        cols.append(np.mean(np.asarray(per_seed, dtype=float), axis=0))
    return np.column_stack(cols) if cols else np.empty((origins.size, 0)), names
