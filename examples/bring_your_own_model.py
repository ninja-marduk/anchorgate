"""Bring your own model: every component on a model you can train in seconds.

The model is a per-cell ridge regression on lagged anomalies of the field (the anomaly
at the origin, one month earlier, and one year before the target), added back to the
training climatology; everything it learns, it learns on the training period only.
Three "seeds" come from bootstrap resampling of the training origins, and two feature
sets (all three lags, or the origin anomaly alone) give the seed-resolved factorial two
configurations to contrast and the fusion decomposition two base learners to combine.

By default the field is synthetic (``anchorgate.synthetic``), so the script runs
anywhere in well under a minute. Point it at a NetCDF field to run the same protocol on
real data::

    python examples/bring_your_own_model.py
    python examples/bring_your_own_model.py --observations chirps_region.nc \\
        --variable precip --cutoff 2016-07 --out report/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anchorgate import (
    Benchmark,
    Forecast,
    Gate,
    Observations,
    anchor,
    anomaly_skill,
    audit,
    decompose_fusion,
    factorial,
    ladder,
    spatial_diagnostics,
    strata_bands,
    write_report,
)
from anchorgate.references import climatology_table
from anchorgate.synthetic import origins_after_cutoff, synthetic_observations, targets_for

LEADS = np.arange(1, 7)
LAG_SETS = {"LAGS": (0, 1, "seasonal"), "LAG0": (0,)}
SEEDS = (1, 2, 3)
RIDGE = 10.0


def anomalies(obs: Observations) -> np.ndarray:
    """The field minus its per-cell training climatology, ``(time, cell)``."""
    return obs.series - climatology_table(obs)[obs.calendar - 1]


def design(anom: np.ndarray, origins: np.ndarray, lead: int, lags: tuple) -> np.ndarray:
    """Per-cell anomaly predictors at each origin for one lead: ``(origin, cell, p + 1)``."""
    cols = []
    for lag in lags:
        if lag == "seasonal":
            idx = origins + lead - 12
        else:
            idx = origins - int(lag)
        cols.append(anom[idx])
    x = np.stack(cols, axis=-1)
    return np.concatenate([x, np.ones((*x.shape[:-1], 1))], axis=-1)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Batched ridge over cells: ``x`` (n, cell, p), ``y`` (n, cell) -> ``(cell, p)``."""
    xt = np.transpose(x, (1, 2, 0))  # (cell, p, n)
    gram = xt @ np.transpose(xt, (0, 2, 1))  # (cell, p, p)
    p = gram.shape[-1]
    penalty = lam * np.eye(p)
    penalty[-1, -1] = 0.0  # do not shrink the intercept
    rhs = np.einsum("cpn,nc->cp", xt, y)
    return np.linalg.solve(gram + penalty, rhs[..., None])[..., 0]


def ridge_forecast(obs: Observations, features: str, seed: int) -> Forecast:
    lags = LAG_SETS[features]
    origins = origins_after_cutoff(obs, LEADS)
    train_origins = np.arange(12, obs.cutoff - int(LEADS.max()))
    rng = np.random.default_rng(seed)
    boot = rng.choice(train_origins, size=train_origins.size, replace=True)
    anom = anomalies(obs)
    clim = climatology_table(obs)
    pred = np.empty((origins.size, LEADS.size, obs.n_cell))
    for j, lead in enumerate(LEADS.tolist()):
        x_tr = design(anom, boot, lead, lags)
        y_tr = anom[boot + lead]
        coef = ridge_fit(x_tr, y_tr, RIDGE)
        x_te = design(anom, origins, lead, lags)
        target_month = obs.calendar[origins + lead] - 1
        pred[:, j] = np.clip(clim[target_month] + np.einsum("ocp,cp->oc", x_te, coef), 0, None)
    return Forecast(
        pred=pred,
        target=targets_for(obs, origins, LEADS),
        origin_time=obs.time[origins],
        lead=LEADS,
        model="ridge",
        features=features,
        seed=seed,
        space_shape=obs.space_shape,
        lat=obs.lat,
        lon=obs.lon,
        mask=obs.mask,
    )


def load_observations(args: argparse.Namespace) -> Observations:
    if args.observations is None:
        return synthetic_observations(seed=0, n_years=40, shape=(6, 8), ar=args.ar)
    import xarray as xr

    with xr.open_dataset(args.observations) as ds:
        da = ds[args.variable].load()
    return Observations.from_field(da, cutoff_time=np.datetime64(args.cutoff, "M"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--observations", default=None, help="NetCDF field (time, y, x); synthetic when omitted"
    )
    ap.add_argument("--variable", default="precip")
    ap.add_argument(
        "--ar",
        type=float,
        default=0.6,
        help="lag-one autocorrelation of the synthetic anomaly (0 = white noise, nothing to learn)",
    )
    ap.add_argument("--cutoff", default="2016-07", help="first month that is not training, YYYY-MM")
    ap.add_argument("--out", default=None, help="directory for report.md and report.json")
    args = ap.parse_args()

    obs = load_observations(args)
    runs = [ridge_forecast(obs, feats, s) for feats in LAG_SETS for s in SEEDS]
    bench = Benchmark(runs, obs)
    for note in bench.validate():
        print(f"note: {note}")
    n = bench.n_origin
    block = list(range(max(0, n - 2 * bench.horizon), n))  # the last two horizons of origins
    cell_mean = np.nanmean(obs.series[: obs.cutoff], axis=0)
    wet_dry = strata_bands(cell_mean, [float(np.nanmedian(cell_mean))], ["dry", "wet"])

    results = [
        audit(bench, ladder(bench, score=block), out_of_sample=["purged"]),
        anchor(bench, score_origins=block, gate=Gate(min_seeds=2), validate=False),
        factorial(bench, score_origins=block, n_perm=5000, validate=False),
        decompose_fusion(bench, assignment="purged", score_origins=block, validate=False),
        anomaly_skill(bench, score_origins=block, conformal=True, validate=False),
        spatial_diagnostics(
            bench, strata={"wet_dry": wet_dry}, score_origins=block, validate=False
        ),
    ]
    for r in results:
        print(r.summary())
        for note in r.notes[:4]:
            print(f"  {note}")
    if args.out:
        md, js = write_report(Path(args.out), results, bench)
        print(f"report: {md} and {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
