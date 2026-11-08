"""Eight precipitation regimes, references only: how strong is the untrained reference
a model would have to beat, and how much do ordinary folds leak, outside one case study?

The global CHIRPS v2.0 monthly product is read remotely (lazy, region by region, no
full download; needs ``pip install anchorgate[remote]``). For each region the
persistence and seasonal-naive references are packaged as forecasts and scored against
the per-cell monthly climatology on the last twelve origins of the record, exactly as a
trained model would be; the fold audit measures the target-period overlap of shuffled and
contiguous folds at a twelve-month horizon.

    python examples/eight_regimes.py                       # remote CHIRPS
    python examples/eight_regimes.py --local chirps.nc     # a local copy
    python examples/eight_regimes.py --out regimes.csv
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from anchorgate import Benchmark, Gate, Observations, anchor, audit, ladder, reference_forecast
from anchorgate.synthetic import origins_after_cutoff

CHIRPS_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
)
TRAIN_FRACTION = 0.8
LEADS = np.arange(1, 13)

# name, lat_min, lat_max, lon_min, lon_max, regime
REGIONS = [
    ("Boyaca (Colombian Andes)", 4.375, 7.375, -74.925, -71.725, "bimodal tropical"),
    ("Mantaro (Peruvian Andes)", -13.50, -10.50, -76.50, -73.50, "unimodal tropical"),
    ("Ethiopian Highlands", 8.00, 11.00, 37.00, 40.00, "monsoonal (Kiremt)"),
    ("Nepal Himalaya", 27.00, 29.00, 83.00, 86.00, "South Asian monsoon"),
    ("Southeast Brazil", -22.00, -19.00, -47.00, -44.00, "subtropical"),
    ("Central Sahel", 12.00, 15.00, -2.00, 1.00, "semi-arid, single wet season"),
    ("Java (Indonesia)", -8.00, -6.00, 106.00, 109.00, "equatorial maritime"),
    ("High Atlas (Morocco)", 30.50, 33.00, -8.00, -5.00, "Mediterranean winter"),
]


def open_chirps(local: str | None):
    import xarray as xr

    if local:
        return xr.open_dataset(local, chunks={})
    import fsspec

    handle = fsspec.open(CHIRPS_URL, mode="rb", block_size=2**22).open()
    return xr.open_dataset(handle, engine="h5netcdf", chunks={})


def region_observations(ds, box) -> Observations:
    var = "precip" if "precip" in ds.data_vars else next(iter(ds.data_vars))
    lat = next(c for c in ds.coords if "lat" in c.lower())
    lon = next(c for c in ds.coords if "lon" in c.lower())
    _, la0, la1, lo0, lo1, _ = box
    sub = ds[var].sel({lat: slice(la0, la1), lon: slice(lo0, lo1)}).load()
    sub = sub.where(sub > -100)  # CHIRPS missing sentinel
    sub = sub.rename({lat: "latitude", lon: "longitude"}) if lat != "latitude" else sub
    n_time = sub.sizes["time"]
    cutoff_time = sub["time"].values[int(TRAIN_FRACTION * n_time)]
    return Observations.from_field(sub, cutoff_time=cutoff_time)


def evaluate(obs: Observations, name: str, regime: str) -> dict[str, object]:
    origins = origins_after_cutoff(obs, LEADS)
    fcs = [reference_forecast(obs, origins, LEADS, r) for r in ("persistence", "seasonal_naive")]
    bench = Benchmark(fcs, obs)
    bench.validate()
    n = bench.n_origin
    block = list(range(n - 12, n))
    fold = audit(bench, ladder(bench, score=block), out_of_sample=["purged"])
    gate = anchor(bench, score_origins=block, gate=Gate(min_seeds=1), validate=False)
    frac = fold.values.set_index("assignment")["fraction"]
    v = gate.values.set_index("label")
    units = fold.tables["effective_units"].iloc[0]
    return {
        "region": name,
        "regime": regime,
        "n_cells": int(obs.n_cell),
        "n_origins": n,
        "clim_pooled_r2": float(v.loc["climatology", "pooled_r2"]),
        "clim_per_cell_nse": float(v.loc["climatology", "per_cell_nse"]),
        "persistence_msess": float(v.loc["persistence", "msess"]),
        "seasonal_naive_msess": float(v.loc["seasonal_naive", "msess"]),
        "overlap_shuffled": float(frac["shuffled"]),
        "overlap_contiguous": float(frac["contiguous"]),
        "overlap_purged": float(frac["purged"]),
        "disjoint_windows": int(units["n_disjoint_windows"]),
        "verdict": gate.verdict.value,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--local", default=None, help="a local CHIRPS monthly NetCDF instead of the remote file"
    )
    ap.add_argument("--out", default=None, help="CSV output")
    ap.add_argument("--regions", type=int, default=None, help="only the first N regions")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    ds = open_chirps(args.local)
    rows = []
    for box in REGIONS[: args.regions]:
        try:
            obs = region_observations(ds, box)
            rows.append(evaluate(obs, box[0], box[5]))
            r = rows[-1]
            print(
                f"{r['region']:<28}{r['regime']:<30} cells {r['n_cells']:>5}  clim R2"
                f" {r['clim_pooled_r2']:.3f}  NSE {r['clim_per_cell_nse']:.3f}  persistence"
                f" {r['persistence_msess']:+.3f}  seasonal {r['seasonal_naive_msess']:+.3f}"
                f"  overlap shuffled {100 * r['overlap_shuffled']:.1f} %"
            )
        except Exception as exc:  # one region must not stop the others
            print(f"{box[0]:<28} ERROR {type(exc).__name__}: {str(exc)[:80]}")
    table = pd.DataFrame(rows)
    if len(table):
        print(
            f"\nclimatology pooled R2 across regions: min {table['clim_pooled_r2'].min():.3f},"
            f" median {table['clim_pooled_r2'].median():.3f},"
            f" max {table['clim_pooled_r2'].max():.3f};"
            f" no untrained reference passes the gate in any region"
            if (table["verdict"] != "PASS").all()
            else "\nsome reference passes the gate; inspect the table"
        )
    if args.out and len(table):
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
