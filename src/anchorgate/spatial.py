"""Component 5: where, on the map, does a model win?

A claim that a spatial prior "captures" orographic or regional structure is a claim
about cells. This module names them: the per-cell winner among configurations and
untrained references, skill inside user-supplied strata (elevation bands, masks,
regimes), a collapse diagnostic that tells a forecast of each cell's mean from a
forecast that varies in time, and an optional audit of a graph model's adjacency
(what its edges connect, and whether the correlation term saw the evaluation period).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import Benchmark, Observations
from .declaration import Declaration, deviations
from .factorial import _origins, _select, configuration_label
from .metrics import msess, per_cell_nse, pooled_r2
from .references import build_references
from .result import Result, Verdict

# ------------------------------------------------------------------- per-cell


def per_cell_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean squared error per cell over the leading axes: ``(..., cells)`` -> ``(cells,)``."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1, np.shape(pred)[-1])
    t = np.asarray(target, dtype=np.float64).reshape(-1, np.shape(target)[-1])
    return np.asarray(np.nanmean((t - p) ** 2, axis=0), dtype=np.float64)


def _config_mean_cell_mse(
    benchmark: Benchmark, target: np.ndarray, origins: np.ndarray, cells: np.ndarray | None
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cfg, runs in benchmark.configurations().items():
        out[configuration_label(cfg)] = np.mean(
            [per_cell_mse(_select(f.pred, origins, cells), target) for f in runs], axis=0
        )
    return out


def winner_map(
    benchmark: Benchmark,
    *,
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    references: Sequence[str] = ("climatology",),
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Per cell, the arm with the lowest seed-mean MSE among configurations and references.

    Returns the share table (arm, n_cells, share), the winner index per cell (``-1``
    where no arm has a finite error) and the arm names in index order.
    """
    origins = _origins(benchmark, score_origins)
    cells = benchmark.cell_mask(mask)
    target = _select(benchmark.target, origins, cells)
    errors = _config_mean_cell_mse(benchmark, target, origins, cells)
    refs = build_references(
        benchmark.observations, benchmark.origin_period, benchmark.lead, tuple(references)
    )
    for name in references:
        errors[name] = per_cell_mse(_select(refs[name], origins, cells), target)
    names = list(errors)
    stack = np.vstack([errors[n] for n in names])
    finite = np.isfinite(stack).any(0)
    win = np.where(finite, np.nanargmin(np.where(np.isfinite(stack), stack, np.inf), axis=0), -1)
    counts = np.bincount(win[win >= 0], minlength=len(names))
    table = (
        pd.DataFrame({"arm": names, "n_cells": counts, "share": counts / max(int(finite.sum()), 1)})
        .sort_values("n_cells", ascending=False)
        .reset_index(drop=True)
    )
    return table, win, names


# ---------------------------------------------------------------------- strata


def strata_bands(
    values: np.ndarray, edges: Sequence[float], labels: Sequence[str] | None = None
) -> np.ndarray:
    """Label each cell by the band of ``values`` it falls in (``edges`` are the interior
    cut points, so ``n + 1`` bands); NaN values get the empty label."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    e = [float(x) for x in edges]
    if labels is None:
        bounds = ["-inf", *[f"{x:g}" for x in e], "inf"]
        labels = [f"{bounds[i]}..{bounds[i + 1]}" for i in range(len(e) + 1)]
    if len(labels) != len(e) + 1:
        raise ValueError("labels must number one more than the edges")
    idx = np.digitize(v, e)
    out = np.array([labels[i] for i in idx], dtype=object)
    out[~np.isfinite(v)] = ""
    return out


def skill_by_strata(
    benchmark: Benchmark,
    strata: np.ndarray,
    *,
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    stratum_name: str = "stratum",
) -> pd.DataFrame:
    """Seed-mean pooled R2, per-cell NSE and MSESS against climatology inside each
    stratum, for every configuration and for the climatology itself."""
    origins = _origins(benchmark, score_origins)
    cells = benchmark.cell_mask(mask)
    labels_all = np.asarray(strata, dtype=object).reshape(-1)
    if labels_all.size != benchmark.reference.n_cell:
        raise ValueError("strata must give one label per cell")
    labels = labels_all if cells is None else labels_all[cells]
    target = _select(benchmark.target, origins, cells)
    clim = _select(
        build_references(
            benchmark.observations, benchmark.origin_period, benchmark.lead, ("climatology",)
        )["climatology"],
        origins,
        cells,
    )
    rows = []
    for stratum in [s for s in pd.unique(labels) if s != ""]:
        sel = labels == stratum
        t_s, c_s = target[..., sel], clim[..., sel]
        rows.append(
            {
                stratum_name: stratum,
                "arm": "climatology",
                "n_seeds": 0,
                "n_cells": int(sel.sum()),
                "pooled_r2": pooled_r2(c_s, t_s),
                "per_cell_nse": per_cell_nse(c_s, t_s),
                "msess": 0.0,
            }
        )
        for cfg, runs in benchmark.configurations().items():
            vals = np.array(
                [
                    [
                        pooled_r2(_select(f.pred, origins, cells)[..., sel], t_s),
                        per_cell_nse(_select(f.pred, origins, cells)[..., sel], t_s),
                        msess(_select(f.pred, origins, cells)[..., sel], t_s, c_s),
                    ]
                    for f in runs
                ]
            )
            m = vals.mean(0)
            sd = vals.std(0, ddof=1) if len(runs) > 1 else np.full(3, np.nan)
            rows.append(
                {
                    stratum_name: stratum,
                    "arm": configuration_label(cfg),
                    "n_seeds": len(runs),
                    "n_cells": int(sel.sum()),
                    "pooled_r2": float(m[0]),
                    "per_cell_nse": float(m[1]),
                    "msess": float(m[2]),
                    "msess_sd": float(sd[2]),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- collapse


def collapse_statistics(
    pred: np.ndarray, target: np.ndarray, cell_mean: np.ndarray
) -> dict[str, float]:
    """Does a forecast vary in time, or only reproduce the spatial pattern of the field?

    ``mean_field_corr``: correlation over cells between the forecast's and the target's
    time means; ``sd_ratio``: median over cells of sd(forecast)/sd(target) over time;
    ``var_ratio`` its square; ``r2_vs_cell_mean``: how much of the forecast's variance
    the per-cell training mean explains (one when the forecast is that mean).
    """
    p = np.asarray(pred, dtype=np.float64).reshape(-1, np.shape(pred)[-1])
    t = np.asarray(target, dtype=np.float64).reshape(-1, np.shape(target)[-1])
    mp, mt = np.nanmean(p, 0), np.nanmean(t, 0)
    ok = np.isfinite(mp) & np.isfinite(mt)
    corr = float(np.corrcoef(mp[ok], mt[ok])[0, 1]) if ok.sum() >= 3 else float("nan")
    sp, st = np.nanstd(p, 0), np.nanstd(t, 0)
    good = st > 0
    ratio = float(np.nanmedian(sp[good] / st[good])) if good.any() else float("nan")
    cm = np.broadcast_to(np.asarray(cell_mean, dtype=np.float64), p.shape)
    fin = np.isfinite(p) & np.isfinite(cm)
    ss_tot = float(np.nansum((p[fin] - np.nanmean(p[fin])) ** 2))
    r2cm = float(1 - np.nansum((p[fin] - cm[fin]) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "pooled_r2": pooled_r2(p, t),
        "mean_field_corr": corr,
        "sd_ratio": ratio,
        "var_ratio": ratio**2 if np.isfinite(ratio) else float("nan"),
        "r2_vs_cell_mean": r2cm,
    }


def collapse_diagnostic(
    benchmark: Benchmark,
    *,
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
) -> pd.DataFrame:
    """One row per forecast plus the per-cell training mean (no seasonality) and the
    per-cell climatology, with :func:`collapse_statistics`."""
    origins = _origins(benchmark, score_origins)
    cells = benchmark.cell_mask(mask)
    obs = benchmark.observations
    target = _select(benchmark.target, origins, cells)
    cell_mean = np.nanmean(obs.series[: obs.cutoff], axis=0)
    cell_mean = cell_mean if cells is None else cell_mean[cells]
    clim = _select(
        build_references(obs, benchmark.origin_period, benchmark.lead, ("climatology",))[
            "climatology"
        ],
        origins,
        cells,
    )
    rows = []
    for f in benchmark.forecasts:
        rows.append(
            {
                "label": f.label,
                "kind": "forecast",
                "model": f.model,
                "features": f.features,
                "seed": f.seed,
            }
            | collapse_statistics(_select(f.pred, origins, cells), target, cell_mean)
        )
    rows.append(
        {
            "label": "cell_mean",
            "kind": "reference",
            "model": "cell_mean",
            "features": "",
            "seed": None,
        }
        | collapse_statistics(np.broadcast_to(cell_mean, target.shape), target, cell_mean)
    )
    rows.append(
        {
            "label": "climatology",
            "kind": "reference",
            "model": "climatology",
            "features": "",
            "seed": None,
        }
        | collapse_statistics(clim, target, cell_mean)
    )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- graph


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    la1, lo1, la2, lo2 = (
        np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2)
    )
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _edges_of(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    a = np.asarray(adjacency)
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        return a[0].astype(int), a[1].astype(int), None
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 0].astype(int), a[:, 1].astype(int), None
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        r, c = np.nonzero(a)
        return r, c, a[r, c].astype(np.float64)
    raise ValueError("adjacency must be a dense (n, n) matrix or an edge list (2, E) / (E, 2)")


def graph_audit(
    adjacency: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    n_sample_pairs: int = 200_000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """What a graph model's retained edges connect.

    Returns ``summary`` (edges, nodes, degree percentiles, isolated nodes, share of edges
    shorter than 10/25/50 km, share of weight on edges under 25 km), ``length`` and
    ``elevation_contrast`` percentiles of the retained edges against a random sample of
    all node pairs. An edge list may be ``(2, E)`` or ``(E, 2)``; a dense matrix gives the
    weights.
    """
    r, c, w_dense = _edges_of(adjacency)
    w = np.asarray(weights, dtype=np.float64) if weights is not None else w_dense
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    n = lat.size
    if r.max() >= n or c.max() >= n:
        raise ValueError("edge indices exceed the number of nodes")
    length = _haversine_km(lat[r], lon[r], lat[c], lon[c])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_sample_pairs)
    j = rng.integers(0, n, n_sample_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    all_len = _haversine_km(lat[i], lon[i], lat[j], lon[j])
    q = (5, 25, 50, 75, 95, 100)

    def pct(x: np.ndarray) -> dict[str, float]:
        return {f"p{k:02d}": float(np.percentile(x, k)) for k in q}

    length_tab = pd.DataFrame(
        [{"set": "retained edges", **pct(length)}, {"set": "all pairs", **pct(all_len)}]
    )
    deg = np.bincount(np.concatenate([r, c]), minlength=n) / (
        1 if (r > c).any() and (c > r).any() else 2
    )
    summary: dict[str, object] = {
        "n_nodes": n,
        "n_edges": int(r.size),
        "degree_median": float(np.median(deg)),
        "degree_p95": float(np.percentile(deg, 95)),
        "degree_max": float(deg.max()),
        "isolated_nodes": int((deg == 0).sum()),
        "share_under_10km": float(np.mean(length < 10)),
        "share_under_25km": float(np.mean(length < 25)),
        "share_under_50km": float(np.mean(length < 50)),
        "median_length_km": float(np.median(length)),
        "median_pair_distance_km": float(np.median(all_len)),
    }
    if w is not None and w.size == r.size:
        summary["weight_share_under_25km"] = (
            float(w[length < 25].sum() / w.sum()) if w.sum() else np.nan
        )
    out = {"summary": pd.DataFrame([summary]), "length": length_tab}
    if elevation is not None:
        e = np.asarray(elevation, dtype=np.float64).reshape(-1)
        de = np.abs(e[r] - e[c])
        all_de = np.abs(e[i] - e[j])
        out["elevation_contrast"] = pd.DataFrame(
            [{"set": "retained edges", **pct(de)}, {"set": "all pairs", **pct(all_de)}]
        )
        far = length > 50
        out["summary"]["far_edges_share"] = float(far.mean())
        out["summary"]["far_edges_median_elevation_contrast"] = (
            float(np.median(de[far])) if far.any() else np.nan
        )
    return out


def correlation_leakage(
    observations: Observations,
    threshold: float = 0.3,
    weight: float = 0.5,
    min_weight: float = 0.01,
    n_cells: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """How much a correlation-derived edge set changes when the correlation is
    estimated on the training period only instead of the full record.

    The edge weight recipe is ``max(0, rho - threshold) * weight``; an edge exists when
    the weight exceeds ``min_weight``. One row with the edge counts under both estimates
    and the disagreement between them.
    """
    flat = np.asarray(observations.series, dtype=np.float64)
    valid = np.isfinite(flat).all(0)
    cells = np.flatnonzero(valid)
    if cells.size > n_cells:
        cells = np.random.default_rng(seed).choice(cells, n_cells, replace=False)
    x_full = flat[:, cells]
    x_train = flat[: observations.cutoff, cells]

    def corr(x: np.ndarray) -> np.ndarray:
        xs = (x - x.mean(0)) / (x.std(0) + 1e-9)
        return (xs.T @ xs) / x.shape[0]

    iu, ju = np.triu_indices(cells.size, 1)
    rf, rt = corr(x_full)[iu, ju], corr(x_train)[iu, ju]
    wf = np.maximum(0.0, rf - threshold) * weight
    wt = np.maximum(0.0, rt - threshold) * weight
    on_f, on_t = wf > min_weight, wt > min_weight
    appear = int(((~on_t) & on_f).sum())
    vanish = int((on_t & (~on_f)).sum())
    return pd.DataFrame(
        [
            {
                "n_cells_sampled": int(cells.size),
                "n_steps_full": int(flat.shape[0]),
                "n_steps_training": int(observations.cutoff),
                "mean_rho_full": float(rf.mean()),
                "mean_rho_training": float(rt.mean()),
                "mean_abs_delta_rho": float(np.abs(rf - rt).mean()),
                "edges_full": int(on_f.sum()),
                "edges_training": int(on_t.sum()),
                "edges_only_with_held_out": appear,
                "edges_lost_without_held_out": vanish,
                "disagreement_share": (appear + vanish) / max(int(on_f.sum()), 1),
            }
        ]
    )


# ---------------------------------------------------------------- declaration


def strata_from_declaration(
    decl: Declaration, benchmark: Benchmark, root: str | Path = "."
) -> dict[str, np.ndarray]:
    """Resolve the declaration's ``strata`` section to one label array per stratum set.

    Each entry is ``{variable: elevation, edges: [1500, 2800], labels: [low, mid, high],
    path: optional.nc}`` (the path defaults to the observations file) or
    ``{path: labels.npy}`` with one label per cell.
    """
    out: dict[str, np.ndarray] = {}
    n_cell = benchmark.reference.n_cell
    obs_path = (decl.data.get("observations") or {}).get("path")
    for name, spec in decl.strata.items():
        spec = dict(spec)
        path = Path(root) / str(spec.get("path", obs_path))
        if "variable" in spec:
            import xarray as xr

            with xr.open_dataset(path) as ds:
                da = ds[str(spec["variable"])]
                values = da.values
            if values.ndim > 2 or (values.ndim == 2 and values.size != n_cell):
                values = values[0] if values.shape[0] != n_cell else values
            labels = strata_bands(values.reshape(-1), spec["edges"], spec.get("labels"))
        else:
            labels = np.load(path, allow_pickle=False).reshape(-1).astype(str)
        if labels.size != n_cell:
            raise ValueError(f"stratum {name!r}: {labels.size} labels for {n_cell} cells")
        out[name] = labels
    return out


# ----------------------------------------------------------------------- component


def spatial_diagnostics(
    benchmark: Benchmark,
    declaration: Declaration | None = None,
    *,
    strata: dict[str, np.ndarray] | None = None,
    score_origins: Sequence[int] | np.ndarray | None = None,
    mask: np.ndarray | str | None = None,
    references: Sequence[str] | None = None,
    adjacency: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    leakage: bool = False,
    collapse_threshold: float = 0.5,
    validate: bool = True,
) -> Result:
    """Winner map, skill by strata, collapse diagnostic and the optional graph audit.

    ``values`` is the winner share table. Tables: ``winner_cells`` (one row per cell
    with the winning arm), ``collapse``, one ``strata_<name>`` table per stratum set,
    and ``graph_*`` / ``leakage`` when an adjacency or ``leakage`` is requested.
    """
    if validate:
        benchmark.validate()
    decl = declaration
    if score_origins is None:
        origins = (
            decl.score_origins(benchmark.n_origin, benchmark.horizon)
            if decl
            else np.arange(benchmark.n_origin)
        )
    else:
        origins = _origins(benchmark, score_origins)
    references_given = references is not None
    refs: tuple[str, ...] = (
        tuple(references)
        if references is not None
        else (decl.references if decl is not None else ("climatology",))
    )
    share, win, names = winner_map(benchmark, score_origins=origins, mask=mask, references=refs)
    cells = benchmark.cell_mask(mask)
    cell_index = np.arange(benchmark.reference.n_cell) if cells is None else np.flatnonzero(cells)
    winner_cells = pd.DataFrame(
        {
            "cell": cell_index,
            "winner": [names[i] if i >= 0 else "" for i in win],
        }
    )
    obs = benchmark.observations
    if obs.lat is not None and obs.lon is not None:
        winner_cells["lat"] = obs.lat[cell_index]
        winner_cells["lon"] = obs.lon[cell_index]
    tables: dict[str, pd.DataFrame] = {"winner_cells": winner_cells}
    notes: list[str] = []

    collapse = collapse_diagnostic(benchmark, score_origins=origins, mask=mask)
    tables["collapse"] = collapse
    collapsed = collapse[
        (collapse["kind"] == "forecast") & (collapse["sd_ratio"] < collapse_threshold)
    ]
    if len(collapsed):
        notes.append(
            f"collapsed towards the cell mean (sd ratio below {collapse_threshold:g}):"
            f" {', '.join(collapsed['label'].tolist())}"
        )
    for name, labels in (strata or {}).items():
        tables[f"strata_{name}"] = skill_by_strata(
            benchmark, labels, score_origins=origins, mask=mask, stratum_name=name
        )
    if adjacency is not None:
        lat = obs.lat if lat is None else np.asarray(lat)
        lon = obs.lon if lon is None else np.asarray(lon)
        if lat is None or lon is None:
            raise ValueError("the graph audit needs latitude and longitude")
        g = graph_audit(adjacency, lat, lon, elevation=elevation)
        for k, v in g.items():
            tables[f"graph_{k}"] = v
        s = g["summary"].iloc[0]
        notes.append(
            f"graph: {int(s['n_edges'])} edges, median length {s['median_length_km']:.1f} km"
            f" against {s['median_pair_distance_km']:.1f} km for all pairs;"
            f" {100 * s['share_under_25km']:.1f} % of edges under 25 km"
        )
    if leakage:
        tables["leakage"] = correlation_leakage(obs)
        lk = tables["leakage"].iloc[0]
        notes.append(
            f"correlation term estimated on the full record disagrees with the training-only"
            f" estimate on {100 * lk['disagreement_share']:.2f} % of edges"
        )

    ref_share = share[share["arm"].isin(list(refs))]
    best_ref = float(ref_share["share"].max()) if len(ref_share) else 0.0
    model_rows = share[~share["arm"].isin(list(refs))]
    best_model = model_rows.iloc[0] if len(model_rows) else None
    for _, r in share.iterrows():
        notes.append(
            f"{r['arm']}: lowest error in {int(r['n_cells'])} cells ({100 * r['share']:.1f} %)"
        )
    if best_model is None:
        verdict = Verdict.NOT_APPLICABLE
    elif float(best_model["share"]) > best_ref and float(best_model["share"]) > 0.5:
        verdict = Verdict.PASS
    else:
        verdict = Verdict.FAIL
    return Result(
        component="C5 spatial diagnostics",
        title=f"Per-cell winners, strata and collapse on {int(origins.size)} origins",
        values=share,
        verdict=verdict,
        rule=(
            "A spatial prior is credited only in the cells where the configuration has the lowest"
            " seed-mean error; the component passes when one configuration wins a majority of"
            " cells and more cells than every untrained reference."
        ),
        inputs=benchmark.inputs(),
        notes=notes,
        tables=tables,
        declaration_hash=decl.sha256 if decl else None,
        parameters={
            "score_origins": origins.tolist(),
            "references": list(refs),
            "strata": list((strata or {}).keys()),
            "collapse_threshold": collapse_threshold,
            "graph_audit": adjacency is not None,
            "leakage": leakage,
        },
        deviations=deviations(
            decl, {"score_origins": score_origins is not None, "references": references_given}
        ),
    )


def spatial_from_declaration(
    benchmark: Benchmark, decl: Declaration, root: str | Path = ".", **kw: Any
) -> Result:
    """Run :func:`spatial_diagnostics` with the strata the declaration names."""
    strata = strata_from_declaration(decl, benchmark, root) if decl.strata else None
    return spatial_diagnostics(benchmark, decl, strata=strata, **kw)


__all__ = [
    "collapse_diagnostic",
    "collapse_statistics",
    "correlation_leakage",
    "graph_audit",
    "per_cell_mse",
    "skill_by_strata",
    "spatial_diagnostics",
    "spatial_from_declaration",
    "strata_bands",
    "strata_from_declaration",
    "winner_map",
]
