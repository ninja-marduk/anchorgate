"""An ENSO-informed ridge regression against the climatology on the Boyaca deposit.

The deep models of the case study never received an oceanic predictor, so their failure
at the gate says nothing about what is predictable. This example gives three cheap
statistical configurations the one predictor the region is known to respond to and runs
them through every component of the protocol, under a declared design, on the same
purged block as the case study:

* ``ridge[ENSO_LOCAL]``: per-cell, per-lead ridge on the Nino 3.4 anomaly at the origin
  and at lags 1, 2 and 3 months, plus the local anomaly at the origin and one month before;
* ``ridge[ENSO_ONLY]``: the same ridge on the Nino 3.4 lags alone;
* ``damped_persistence[LOCAL]``: climatology plus phi times the last observed anomaly,
  phi fitted per cell and per lead on the training origins.

Every forecast is the per-cell training climatology plus the predicted anomaly, clipped
at zero. Three seeds come from bootstrap resampling of the training origins. Forecasts
are issued on the deposit's own origin grid, recovered from one of its target arrays, so
origins 21 to 32 are the block the case study scores; without the deposit arrays the
origins after the cutoff are used instead. The Nino 3.4 index is read from the NOAA PSL
table (``nina34.anom.data``), downloaded on first use.

The components run are C0 (target-period overlap of five shuffled folds against the
purged block), C1 (anchoring to the per-cell climatology, persistence and seasonal naive
with the default gate), C2 (the feature set, ``ENSO_ONLY`` against ``ENSO_LOCAL``, as the
one factor with the seed as block), C3 (the two ridge feature sets as base learners of an
affine fusion, out of fold), C4 (skill on anomalies against the training climatology) and
C5 (per-cell winner map, wet and dry strata, collapse diagnostic). C1 is also reported on
every deposit origin. When the deposited deep models are readable, C5 is run a second
time with the two trained models added (the released fusion is left out: its combiner was
fitted on folds shuffled over the whole record, so it is in sample on the block).

    python examples/enso_ridge.py --thesis-root /path/to/deposit --out report/

The root is also read from ``ANCHORGATE_THESIS_ROOT``. Nothing is tuned on the scored
block: the ridge penalty and the lags are fixed before scoring. Besides the report, the
directory receives ``winner_cells.csv`` (the C5 winner per cell, with coordinates),
``winner_cells_with_deposit.csv`` when the deposit was read, and
``cell_skill_enso_only.csv`` (the seed-mean per-cell skill score of the index ridge
against the climatology over the block), from which a map can be drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from anchorgate import (
    Benchmark,
    Declaration,
    Forecast,
    Gate,
    Observations,
    anchor,
    anomaly_skill,
    audit,
    decompose_fusion,
    factorial,
    purged_holdout,
    shuffled_kfold,
    spatial_diagnostics,
    strata_bands,
    write_report,
)
from anchorgate._time import to_period
from anchorgate.data import _locate, benchmark_from_declaration, infer_origin_time
from anchorgate.references import build_references, climatology_table
from anchorgate.result import Result
from anchorgate.spatial import per_cell_mse
from anchorgate.synthetic import origins_after_cutoff, targets_for

HERE = Path(__file__).resolve().parent
CASE_STUDY_DECLARATION = HERE / "boyaca_case_study" / "declaration.yaml"
OBS_RELATIVE = (
    "data/output/complete_dataset_with_features_with_clusters_elevation_windows_imfs"
    "_with_onehot_elevation_clean.nc"
)
OBS_VARIABLE = "total_precipitation"
CUTOFF = "2016-07"
TARGETS_RELATIVE = (
    "models/output/V4_GNN_TAT_Models/SEED42/map_exports/H12/BASIC/GNN_TAT_GAT/targets.npy"
)
NINO_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
PURGED = list(range(21, 33))
LEADS = np.arange(1, 13)
SEEDS = (1, 2, 3)
RIDGE_PENALTY = 10.0
NINO_LAGS = (0, 1, 2, 3)
LOCAL_LAGS = (0, 1)
MISSING = -99.99
# the design shared with ``enso_ridge_region.py``: one hash, several data sets
DESIGN_NAME = "enso-ridge-chirps-region"
BLOCK_ORIGINS = 12
SHUFFLED_FOLDS = 5
FOLD_SEED = 42
FACTORIAL_PERMUTATIONS = 5000
REFERENCES = ("climatology", "persistence", "seasonal_naive")

CONFIGS: dict[str, dict[str, object]] = {
    "ENSO_LOCAL": {"model": "ridge", "nino": NINO_LAGS, "local": LOCAL_LAGS},
    "ENSO_ONLY": {"model": "ridge", "nino": NINO_LAGS, "local": ()},
    "LOCAL": {"model": "damped_persistence", "nino": (), "local": (0,)},
}


# ------------------------------------------------------------------------------ data


def fetch_nino34(cache: Path) -> Path:
    """The PSL Nino 3.4 table, downloaded once into ``cache``."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "nina34.anom.data"
    if not path.exists():
        print(f"downloading {NINO_URL}")
        with urllib.request.urlopen(NINO_URL, timeout=120) as resp:
            path.write_bytes(resp.read())
    return path


def read_nino34(path: Path) -> pd.Series:
    """Parse the PSL fixed-width table: one row per year with twelve monthly values; the
    missing marker becomes NaN. The file is treated as text only."""
    values: dict[np.datetime64, float] = {}
    row = re.compile(r"^\s*(\d{4})((?:\s+-?\d+\.\d+){12})\s*$")
    for raw in path.read_text(encoding="ascii", errors="replace").splitlines():
        m = row.match(raw)
        if m is None:
            continue
        year = int(m.group(1))
        for month, tok in enumerate(m.group(2).split(), start=1):
            v = float(tok)
            values[np.datetime64(f"{year:04d}-{month:02d}", "M")] = (
                np.nan if abs(v - MISSING) < 1e-6 else v
            )
    return pd.Series(values).sort_index()


def load_observations(root: Path) -> Observations:
    with xr.open_dataset(_locate(root / OBS_RELATIVE)) as ds:
        da = ds[OBS_VARIABLE].load()
    return Observations.from_field(da, cutoff_time=np.datetime64(CUTOFF, "M"))


def deposit_origins(root: Path, obs: Observations) -> tuple[np.ndarray, str]:
    """Origin indices of the deposit forecasts, or the generic post-cutoff grid."""
    targets = root / TARGETS_RELATIVE
    if targets.exists():
        ot = infer_origin_time(np.load(targets), obs, LEADS)
        idx = obs.index_of(to_period(ot))
        if np.all(idx >= 0):
            return idx, "deposit"
    return origins_after_cutoff(obs, LEADS), "origins_after_cutoff"


def deposit_benchmark(root: Path) -> Benchmark:
    """The three deposited configurations, read as the case study reads them."""
    text = CASE_STUDY_DECLARATION.read_text(encoding="utf-8").replace(
        "{thesis_root}", root.as_posix()
    )
    decl = Declaration.from_dict(yaml.safe_load(text))
    return benchmark_from_declaration(decl.data, root=root)


# ----------------------------------------------------------------------------- model


def design(
    anom: np.ndarray,
    nino: np.ndarray,
    origins: np.ndarray,
    nino_lags: tuple[int, ...],
    local_lags: tuple[int, ...],
) -> np.ndarray:
    """Raw predictors at each origin, ``(origin, cell, p)`` without intercept."""
    n_cell = anom.shape[1]
    cols = []
    for lag in nino_lags:
        v = nino[origins - lag]
        cols.append(np.broadcast_to(v[:, None], (origins.size, n_cell)))
    for lag in local_lags:
        cols.append(anom[origins - lag])
    return np.stack(cols, axis=-1)


def standardise(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Scale predictors with training statistics and append the intercept column."""
    z = (x - mean) / sd
    return np.concatenate([z, np.ones((*z.shape[:-1], 1))], axis=-1)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Batched ridge over cells: ``x`` (n, cell, p), ``y`` (n, cell) -> ``(cell, p)``; the
    intercept is not penalised."""
    xt = np.transpose(x, (1, 2, 0))
    gram = xt @ np.transpose(xt, (0, 2, 1))
    penalty = lam * np.eye(gram.shape[-1])
    penalty[-1, -1] = 0.0
    rhs = np.einsum("cpn,nc->cp", xt, y)
    return np.linalg.solve(gram + penalty, rhs[..., None])[..., 0]


def damped_fit(x0: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-cell phi minimising ``sum (y - phi x0)^2`` over the training origins."""
    num = np.einsum("nc,nc->c", x0, y)
    den = np.einsum("nc,nc->c", x0, x0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


class Fitter:
    """Holds the field, the index and the training origins; issues forecasts."""

    def __init__(self, obs: Observations, nino: pd.Series) -> None:
        self.obs = obs
        self.clim = climatology_table(obs)
        self.anom = obs.series - self.clim[obs.calendar - 1]
        index = pd.DatetimeIndex(obs.time.astype("datetime64[ns]"))
        self.nino = nino.reindex(index).to_numpy(dtype=float)
        self.train_origins = np.arange(12, obs.cutoff - int(LEADS.max()))

    def forecast(self, features: str, seed: int, origins: np.ndarray) -> Forecast:
        cfg = CONFIGS[features]
        nino_lags = tuple(cfg["nino"])  # type: ignore[arg-type]
        local_lags = tuple(cfg["local"])  # type: ignore[arg-type]
        rng = np.random.default_rng(seed)
        boot = rng.choice(self.train_origins, size=self.train_origins.size, replace=True)
        x_train = design(self.anom, self.nino, self.train_origins, nino_lags, local_lags)
        x_boot = design(self.anom, self.nino, boot, nino_lags, local_lags)
        x_test = design(self.anom, self.nino, origins, nino_lags, local_lags)
        if not (np.all(np.isfinite(x_boot)) and np.all(np.isfinite(x_test))):
            raise ValueError(f"{features}: a predictor is missing at some origin")
        mean = x_train.mean(axis=0)
        sd = x_train.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        pred = np.empty((origins.size, LEADS.size, self.obs.n_cell))
        for j, lead in enumerate(LEADS.tolist()):
            y_train = self.anom[boot + lead]
            target_month = self.obs.calendar[origins + lead] - 1
            if cfg["model"] == "ridge":
                coef = ridge_fit(standardise(x_boot, mean, sd), y_train, RIDGE_PENALTY)
                dev = np.einsum("ocp,cp->oc", standardise(x_test, mean, sd), coef)
            else:
                phi = damped_fit(x_boot[..., 0], y_train)
                dev = x_test[..., 0] * phi[None, :]
            pred[:, j] = np.clip(self.clim[target_month] + dev, 0, None)
        return Forecast(
            pred=pred,
            target=targets_for(self.obs, origins, LEADS),
            origin_time=self.obs.time[origins],
            lead=LEADS,
            model=str(cfg["model"]),
            features=features,
            seed=seed,
            space_shape=self.obs.space_shape,
            lat=self.obs.lat,
            lon=self.obs.lon,
            mask=self.obs.mask,
        )


# ------------------------------------------------------------------------- the design


def ridge_declaration(data: dict[str, Any]) -> Declaration:
    """The declared design of the ridge examples: every section but ``data`` is fixed and
    the scored block is declared relative to the record (its last twelve origins), so the
    Boyaca deposit and every CHIRPS region carry the same design hash."""
    doc = {
        "name": DESIGN_NAME,
        "data": data,
        "scoring": {"purged": {"last": BLOCK_ORIGINS}},
        "references": list(REFERENCES),
        "gate": {
            "metric": "msess",
            "reference": "climatology",
            "threshold": 0.0,
            "min_seeds": 2,
            "level": 0.9,
            "block": 3,
            "n_boot": 1000,
        },
        "seeds_expected": len(SEEDS),
        "folds": [
            {"kind": "shuffled", "name": "shuffled", "k": SHUFFLED_FOLDS, "seed": FOLD_SEED},
            {"kind": "purged", "name": "purged", "out_of_sample": True},
        ],
        "contrasts": [["ENSO_LOCAL", "ENSO_ONLY"]],
        "factorial": {
            "factors": ["features"],
            "metric": "msess",
            "aggregate": "lead_mean",
            "n_perm": FACTORIAL_PERMUTATIONS,
        },
        "fusion": {
            "base": [
                {"model": "ridge", "features": "ENSO_ONLY"},
                {"model": "ridge", "features": "ENSO_LOCAL"},
            ],
            "assignment": "purged",
        },
        "conclusions": {
            "pass": (
                "at least one ENSO-informed configuration carries information beyond the"
                " per-cell climatology on the purged block of this region"
            ),
            "fail": (
                "no configuration exceeds the per-cell climatology on the purged block of"
                " this region; the Nino 3.4 index alone does not predict its monthly anomalies"
            ),
        },
    }
    return Declaration.from_dict(doc)


def models_section() -> dict[str, Any]:
    """The ``models`` and ``ridge_penalty`` entries of the ``data`` section."""
    return {
        "models": {
            k: {
                "model": v["model"],
                "nino_lags": list(v["nino"]),  # type: ignore[call-overload]
                "local_lags": list(v["local"]),  # type: ignore[call-overload]
            }
            for k, v in CONFIGS.items()
        },
        "ridge_penalty": RIDGE_PENALTY,
    }


def nino_provenance(path: Path) -> dict[str, str]:
    """SHA-256 and last month of the Nino 3.4 table actually used, for the data section
    of the declaration and for a note in the report."""
    series = read_nino34(path).dropna()
    return {
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "last_month": str(series.index[-1].date())[:7],
    }


def nino_note(path: Path) -> str:
    prov = nino_provenance(path)
    return (
        f"Nino 3.4 table {path.name}: sha256 {prov['file_sha256'][:16]}, last month"
        f" {prov['last_month']}; the index is revised with the sea-surface record, so the"
        " ridge scores depend on this version"
    )


def boyaca_declaration(
    root: Path, obs: Observations, source: str, n_origin: int, nino_path: Path
) -> Declaration:
    data: dict[str, Any] = {
        "observations": {
            "source": "CHIRPS v2.0 monthly, Boyaca deposit",
            "path": OBS_RELATIVE,
            "variable": OBS_VARIABLE,
            "record": f"{obs.time[0]}..{obs.time[-1]}",
            "n_cells": obs.n_cell,
            "cutoff": CUTOFF,
            "origin_grid": source,
            "n_origins": n_origin,
        },
        "predictor": {
            "source": "NOAA PSL Nino 3.4 anomaly",
            "url": NINO_URL,
            **nino_provenance(nino_path),
        },
        **models_section(),
    }
    return ridge_declaration(data)


def run_protocol(
    bench: Benchmark,
    decl: Declaration,
    *,
    shuffled_folds: int = SHUFFLED_FOLDS,
    fold_seed: int = FOLD_SEED,
) -> list[Result]:
    """Components 0 to 5 on a validated benchmark, every choice read from the design."""
    n = bench.n_origin
    block = decl.score_origins(n, bench.horizon).tolist()
    assignments = [
        shuffled_kfold(n, k=shuffled_folds, seed=fold_seed, name="shuffled"),
        purged_holdout(n, bench.lead, block, name="purged"),
    ]
    obs = bench.observations
    cell_mean = np.nanmean(obs.series[: obs.cutoff], axis=0)
    wet_dry = strata_bands(cell_mean, [float(np.nanmedian(cell_mean))], ["dry", "wet"])
    ridge_only = Benchmark([f for f in bench.forecasts if f.model == "ridge"], obs)
    return [
        audit(bench, assignments, out_of_sample=["purged"], declaration_hash=decl.sha256),
        anchor(bench, decl, validate=False),
        factorial(ridge_only, decl, validate=False),
        decompose_fusion(bench, decl, assignment=assignments[1], validate=False),
        anomaly_skill(bench, decl, validate=False),
        spatial_diagnostics(bench, decl, strata={"wet_dry": wet_dry}, validate=False),
    ]


def cell_skill_table(bench: Benchmark, block: list[int], features: str) -> pd.DataFrame:
    """Seed-mean per-cell MSE of one configuration and of the climatology on the block, and
    the per-cell skill score between them; one row per cell with its coordinates."""
    runs = [f for f in bench.forecasts if f.model == "ridge" and f.features == features]
    if not runs:
        raise ValueError(f"no ridge run with features {features}")
    origins = np.asarray(block)
    target = bench.target[origins]
    mse_model = np.mean([per_cell_mse(f.pred[origins], target) for f in runs], axis=0)
    refs = build_references(bench.observations, bench.origin_period, bench.lead, ("climatology",))
    mse_clim = per_cell_mse(refs["climatology"][origins], target)
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = 1.0 - mse_model / mse_clim
    obs = bench.observations
    table = pd.DataFrame(
        {
            "cell": np.arange(obs.n_cell),
            "mse_model": mse_model,
            "mse_climatology": mse_clim,
            "msess_cell": skill,
        }
    )
    if obs.lat is not None and obs.lon is not None:
        table["lat"] = obs.lat
        table["lon"] = obs.lon
    return table


# ------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--thesis-root", default=os.environ.get("ANCHORGATE_THESIS_ROOT"))
    ap.add_argument("--nino", default=None, help="local copy of the PSL nina34.anom.data table")
    ap.add_argument("--out", default="report_enso", help="directory for the reports")
    ap.add_argument(
        "--no-deposit",
        action="store_true",
        help="skip the second winner map that adds the deposited deep models",
    )
    args = ap.parse_args()
    if not args.thesis_root:
        ap.error("--thesis-root or ANCHORGATE_THESIS_ROOT is required")
    root = Path(args.thesis_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    nino_path = Path(args.nino) if args.nino else fetch_nino34(out / "data")
    nino = read_nino34(nino_path)
    obs = load_observations(root)
    origins, source = deposit_origins(root, obs)
    fitter = Fitter(obs, nino)
    print(
        f"observations {obs.n_time} steps {obs.time[0]}..{obs.time[-1]}, cutoff"
        f" {obs.time[obs.cutoff]}, {obs.n_cell} cells;"
        f" nino 3.4 {nino.index[0].date()}..{nino.index[-1].date()}"
    )
    first_target = obs.time[origins[PURGED[0]] + 1]
    last_target = obs.time[origins[PURGED[-1]] + int(LEADS.max())]
    print(
        f"origin grid from {source}: {origins.size} origins; purged block targets"
        f" {first_target}..{last_target}"
    )
    print(f"training origins: {fitter.train_origins.size}; ridge penalty {RIDGE_PENALTY:g}")

    runs = [fitter.forecast(feats, s, origins) for feats in CONFIGS for s in SEEDS]
    bench = Benchmark(runs, obs)
    for note in bench.validate():
        print(f"note: {note}")
    decl = boyaca_declaration(root, obs, source, origins.size, nino_path)
    print(f"declared design {decl.sha256[:12]}; scored block {PURGED[0]}..{PURGED[-1]}")

    results = run_protocol(bench, decl)
    results[1].notes.append(nino_note(nino_path))
    gate = Gate(metric="msess", reference="climatology", threshold=0.0, min_seeds=2)
    every = anchor(bench, references=REFERENCES, gate=gate, validate=False)
    every.title += " (every deposit origin)"
    results.append(every)

    if not args.no_deposit:
        try:
            deposited = deposit_benchmark(root)
            # the released fusion was fitted on folds shuffled over the whole record, so
            # it is in sample on the block; the two trained models are out of sample
            out_of_sample = [
                f for f in deposited.forecasts if not f.model.startswith("Late-Fusion")
            ]
            combined = Benchmark(runs + out_of_sample, obs)
            for note in combined.validate():
                print(f"note: {note}")
            with_deposit = spatial_diagnostics(combined, decl, validate=False)
            with_deposit.title += (
                " (ridge configurations and the two out-of-sample deposited models together)"
            )
            results.append(with_deposit)
            with_deposit.tables["winner_cells"].to_csv(
                out / "winner_cells_with_deposit.csv", index=False, lineterminator="\n"
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"note: deposited models not added to the winner map ({exc})")

    for r in results:
        print(r.summary())
        for note in r.notes:
            print(f"  {note}")
    lead_table = results[1].tables["by_lead_configuration"]
    print("seed-mean MSESS by lead on the purged block, with the interval bounds:")
    print(lead_table.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    results[5].tables["winner_cells"].to_csv(
        out / "winner_cells.csv", index=False, lineterminator="\n"
    )
    cell_skill_table(bench, PURGED, "ENSO_ONLY").to_csv(
        out / "cell_skill_enso_only.csv", index=False, float_format="%.6f", lineterminator="\n"
    )
    md, js = write_report(out, results, bench, decl)
    print(f"report: {md} and {js} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
