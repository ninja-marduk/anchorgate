"""The whole protocol, end to end, on a CHIRPS region that is not the Boyaca deposit.

The ENSO ridge of ``enso_ridge.py`` needs only a monthly precipitation field and the Nino
3.4 index, so it can be trained and gated on any region of the global CHIRPS v2.0 monthly
product. This example reads one region remotely (lazy, by HTTP range requests; needs
``pip install anchorgate[remote]``), caches the regional cut locally, fits the same three
configurations with the same seeds, and runs components 0 to 5 through a declared design:

* C0: target-period overlap of five shuffled folds of origins against the purged block;
* C1: anchoring to the per-cell climatology, persistence and seasonal naive, with the
  default gate (seed-mean MSESS, threshold 0, two seeds, 90 % block-bootstrap interval);
* C2: the feature set (``ENSO_ONLY`` against ``ENSO_LOCAL``) as the one factor, with
  the seed as block, on the ridge runs;
* C3: the two ridge feature sets as base learners of an affine fusion, out of fold;
* C4: skill on anomalies against the training climatology;
* C5: per-cell winner map, wet and dry strata from the training mean, collapse diagnostic.

The training cutoff is 80 % of the record, as in ``eight_regimes.py``; origins run every
month after the cutoff with leads 1 to 12, and the scored block is the last twelve
admissible origins with the embargo of ``horizon - 1`` origins that
:func:`anchorgate.purged_holdout` defines. Nothing is tuned on the scored block.

    python examples/enso_ridge_region.py                              # Mantaro, Peru
    python examples/enso_ridge_region.py --region java --out report_java/
    python examples/enso_ridge_region.py --region=-8,-6,106,109        # an explicit box
    python examples/enso_ridge_region.py --local chirps-v2.0.monthly.nc

``--region`` accepts any of the eight named regions of ``eight_regimes.py`` (a
case-insensitive fragment of the name is enough) or ``lat_min,lat_max,lon_min,lon_max``.
``--source chc`` (default) reads the CHC NetCDF file by range requests, as
``eight_regimes.py`` does; the CHC server limits clients that issue many of them, and
``--source iri`` then reads the same product from the IRI Data Library's OPeNDAP mirror,
which cuts the region on the server. The regional cut and the Nino 3.4 table are cached
under ``--cache`` (default ``data/local``, which is ignored by git); CHIRPS data are read,
never redistributed.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import eight_regimes
import enso_ridge
import numpy as np
import pandas as pd
import xarray as xr

from anchorgate import (
    Benchmark,
    Declaration,
    Forecast,
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
from anchorgate.result import Result
from anchorgate.synthetic import origins_after_cutoff

Box = tuple[float, float, float, float]

LEADS = enso_ridge.LEADS
SEEDS = enso_ridge.SEEDS
CONFIGS = enso_ridge.CONFIGS
TRAIN_FRACTION = eight_regimes.TRAIN_FRACTION
BLOCK_ORIGINS = 12
SHUFFLED_FOLDS = 5
FOLD_SEED = 42
DEFAULT_REGION = "mantaro"
DEFAULT_CACHE = Path("data") / "local"
IRI_URL = (
    "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global"
    "/.precipitation/dods"
)
SOURCES = {"chc": eight_regimes.CHIRPS_URL, "iri": IRI_URL}
FACTORIAL_PERMUTATIONS = 5000
DESIGN_NAME = "enso-ridge-chirps-region"


# ---------------------------------------------------------------------------- region


def parse_region(text: str) -> tuple[str, Box, str]:
    """``(name, box, regime)`` for a named region of ``eight_regimes`` or for an
    explicit ``lat_min,lat_max,lon_min,lon_max`` box."""
    parts = text.split(",")
    if len(parts) == 4:
        la0, la1, lo0, lo1 = (float(p) for p in parts)
        if not (la0 < la1 and lo0 < lo1):
            raise ValueError("the box must be lat_min,lat_max,lon_min,lon_max with min < max")
        return f"box {la0:g}..{la1:g}N {lo0:g}..{lo1:g}E", (la0, la1, lo0, lo1), "unspecified"
    key = text.strip().lower()
    hits = [r for r in eight_regimes.REGIONS if key in str(r[0]).lower()]
    if len(hits) != 1:
        names = ", ".join(str(r[0]) for r in eight_regimes.REGIONS)
        raise ValueError(f"{text!r} matches {len(hits)} regions; choose one of: {names}")
    name, la0, la1, lo0, lo1, regime = hits[0]
    return str(name), (float(la0), float(la1), float(lo0), float(lo1)), str(regime)


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def open_iri() -> xr.Dataset:
    """The IRI Data Library mirror of CHIRPS v2.0 monthly over OPeNDAP, with its axes
    named like the CHC file. Its time axis counts months since 1960-01 at mid-month on a
    360-day calendar, which xarray does not decode; the month index is exact."""
    ds = xr.open_dataset(IRI_URL, engine="netcdf4", decode_times=False)
    months = np.floor(ds["T"].values).astype(np.int64)
    time = np.datetime64("1960-01", "M") + months
    ds = ds.rename({"T": "time", "Y": "latitude", "X": "longitude"})
    return ds.assign_coords(time=("time", time.astype("datetime64[ns]")))


def open_source(source: str, local: str | None) -> tuple[xr.Dataset, str]:
    """The lazily opened product and the URL (or path) it came from."""
    if local:
        return eight_regimes.open_chirps(local), str(local)
    if source == "iri":
        return open_iri(), IRI_URL
    if source == "chc":
        return eight_regimes.open_chirps(None), eight_regimes.CHIRPS_URL
    raise ValueError(f"unknown source {source!r}; choose one of {sorted(SOURCES)}")


def cut_region(ds: xr.Dataset, box: Box) -> xr.DataArray:
    """The CHIRPS field inside ``box`` as ``(time, latitude, longitude)`` with latitude
    ascending, loaded, with the missing sentinel replaced by NaN."""
    var = "precip" if "precip" in ds.data_vars else next(iter(ds.data_vars))
    lat = next(c for c in ds.coords if "lat" in c.lower())
    lon = next(c for c in ds.coords if "lon" in c.lower())
    la0, la1, lo0, lo1 = box
    lat_slice = slice(la0, la1) if ds[lat].values[0] <= ds[lat].values[-1] else slice(la1, la0)
    sub = ds[var].sel({lat: lat_slice, lon: slice(lo0, lo1)}).load()
    sub = sub.where(sub > -100)
    renames = {k: v for k, v in ((lat, "latitude"), (lon, "longitude")) if k != v}
    sub = sub.rename(renames) if renames else sub
    sub = sub.sortby("latitude").transpose("time", "latitude", "longitude")
    return sub.rename("precip")


def load_region_field(
    box: Box, cache: Path, source: str = "chc", local: str | None = None
) -> tuple[xr.DataArray, str]:
    """The regional cut and where it came from: the cache when present, else the
    product named by ``source`` (or the local file)."""
    cache.mkdir(parents=True, exist_ok=True)
    tag = slug(f"{box[0]}_{box[1]}_{box[2]}_{box[3]}")
    path = cache / f"chirps_v2_monthly_{'local' if local else source}_{tag}.nc"
    if path.exists():
        with xr.open_dataarray(path) as da:
            da = da.load()
        return da, str(da.attrs.get("source_url", path))
    ds, url = open_source(source, local)
    print(f"reading CHIRPS from {url}; the cut goes to {path}")
    try:
        sub = cut_region(ds, box)
    finally:
        ds.close()
    sub.attrs = {"source_url": url}
    sub.to_netcdf(path)
    return sub, url


def observations_from_field(da: xr.DataArray) -> Observations:
    """Observations with the training cutoff at 80 % of the record. Cells that are
    missing at any step (sea, outside the product) are dropped, and the field is then
    kept as a list of cells with their coordinates rather than as a grid."""
    n_time = int(da.sizes["time"])
    cutoff_time = da["time"].values[int(TRAIN_FRACTION * n_time)]
    valid = np.isfinite(da.values).all(axis=0)
    if valid.all():
        return Observations.from_field(da, cutoff_time=cutoff_time)
    la, lo = np.meshgrid(da["latitude"].values, da["longitude"].values, indexing="ij")
    keep = valid.ravel()
    series = da.values.reshape(n_time, -1)[:, keep]
    return Observations.from_cutoff_time(
        series,
        da["time"].values,
        cutoff_time,
        lat=la.ravel()[keep],
        lon=lo.ravel()[keep],
        name=f"{da.name or 'precip'} ({int(keep.sum())} of {keep.size} cells)",
    )


# ------------------------------------------------------------------------- the design


def origin_grid(obs: Observations) -> np.ndarray:
    """Every origin after the cutoff whose twelve targets lie inside the record."""
    return origins_after_cutoff(obs, LEADS)


def scored_block(n_origin: int) -> list[int]:
    """The last twelve admissible origins."""
    if n_origin < BLOCK_ORIGINS:
        raise ValueError(f"{n_origin} origins cannot hold a block of {BLOCK_ORIGINS}")
    return list(range(n_origin - BLOCK_ORIGINS, n_origin))


def declaration(
    name: str,
    box: Box,
    regime: str,
    obs: Observations,
    n_origin: int,
    url: str = SOURCES["chc"],
) -> Declaration:
    """The design, in memory: the same design hash for every region, the region and the
    record in the ``data`` section."""
    block = scored_block(n_origin)
    doc = {
        "name": DESIGN_NAME,
        "data": {
            "observations": {
                "source": "CHIRPS v2.0 global monthly",
                "url": url,
                "region": name,
                "regime": regime,
                "box": {"lat": [box[0], box[1]], "lon": [box[2], box[3]]},
                "record": f"{obs.time[0]}..{obs.time[-1]}",
                "n_cells": obs.n_cell,
                "cutoff": str(obs.time[obs.cutoff]),
                "train_fraction": TRAIN_FRACTION,
            },
            "predictor": {"source": "NOAA PSL Nino 3.4 anomaly", "url": enso_ridge.NINO_URL},
            "models": {
                k: {
                    "model": v["model"],
                    "nino_lags": list(v["nino"]),
                    "local_lags": list(v["local"]),
                }
                for k, v in CONFIGS.items()
            },
            "ridge_penalty": enso_ridge.RIDGE_PENALTY,
        },
        "scoring": {"purged": {"start": block[0], "stop": block[-1] + 1}},
        "references": ["climatology", "persistence", "seasonal_naive"],
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


def fit_runs(obs: Observations, nino: pd.Series, origins: np.ndarray) -> list[Forecast]:
    fitter = enso_ridge.Fitter(obs, nino)
    print(
        f"training origins: {fitter.train_origins.size}; ridge penalty {enso_ridge.RIDGE_PENALTY:g}"
    )
    return [fitter.forecast(feats, s, origins) for feats in CONFIGS for s in SEEDS]


def run_protocol(bench: Benchmark, decl: Declaration) -> list[Result]:
    """Components 0 to 5 on a validated benchmark, every choice read from the design."""
    n = bench.n_origin
    block = decl.score_origins(n, bench.horizon).tolist()
    assignments = [
        shuffled_kfold(n, k=SHUFFLED_FOLDS, seed=FOLD_SEED, name="shuffled"),
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


# ------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="a named region of eight_regimes.py or lat_min,lat_max,lon_min,lon_max",
    )
    ap.add_argument(
        "--source",
        default="chc",
        choices=sorted(SOURCES),
        help="chc: the CHC NetCDF by range requests; iri: the IRI Data Library OPeNDAP mirror",
    )
    ap.add_argument(
        "--local", default=None, help="a local CHIRPS monthly NetCDF instead of the remote file"
    )
    ap.add_argument("--nino", default=None, help="local copy of the PSL nina34.anom.data table")
    ap.add_argument(
        "--cache", default=str(DEFAULT_CACHE), help="directory for the regional cut and the index"
    )
    ap.add_argument("--out", default="report_enso_region", help="directory for the reports")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()

    name, box, regime = parse_region(args.region)
    cache = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nino_path = Path(args.nino) if args.nino else enso_ridge.fetch_nino34(cache)
    nino = enso_ridge.read_nino34(nino_path)
    try:
        field, url = load_region_field(box, cache, args.source, args.local)
    except Exception as exc:
        if "429" in str(exc) or isinstance(exc, FileNotFoundError):
            print(
                f"{type(exc).__name__}: {str(exc)[:120]}; the CHC server is limiting range"
                " requests; retry later, pass --source iri, or download the file once and"
                " pass --local"
            )
            return 1
        raise
    obs = observations_from_field(field)
    origins = origin_grid(obs)
    decl = declaration(name, box, regime, obs, origins.size, url)
    print(decl.describe())
    print(
        f"{name} ({regime}): {obs.n_cell} cells, {obs.n_time} steps {obs.time[0]}..{obs.time[-1]},"
        f" cutoff {obs.time[obs.cutoff]} ({TRAIN_FRACTION:.0%});"
        f" nino 3.4 {nino.index[0].date()}..{nino.index[-1].date()}"
    )
    block = scored_block(origins.size)
    last_target = obs.time[origins[block[-1]] + int(LEADS.max())]
    print(
        f"{origins.size} origins {obs.time[origins[0]]}..{obs.time[origins[-1]]}; scored block"
        f" origins {block[0]}..{block[-1]} = {obs.time[origins[block[0]]]}.."
        f"{obs.time[origins[block[-1]]]}, targets up to {last_target}"
    )

    runs = fit_runs(obs, nino, origins)
    bench = Benchmark(runs, obs)
    for note in bench.validate():
        print(f"note: {note}")
    results = run_protocol(bench, decl)
    for r in results:
        print(r.summary())
        for note in r.notes:
            print(f"  {note}")
    lead_table = results[1].tables["by_lead_configuration"]
    print("seed-mean MSESS by lead on the purged block, with the interval bounds:")
    print(lead_table.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    md, js = write_report(out, results, bench, decl)
    print(f"report: {md} and {js} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
