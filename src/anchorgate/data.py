"""The data contract: forecasts, observations and the benchmark that binds them.

Internally every field is flattened to ``(..., n_cell)``; the original spatial shape
is kept so maps can be rebuilt. Times are reduced to integer monthly periods (see
:mod:`anchorgate._time`).
"""

from __future__ import annotations

import glob
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ._time import as_datetime_month, period_of_year, period_to_datetime, to_period
from .result import array_hash

# --------------------------------------------------------------------------- helpers

Leads = Sequence[int] | np.ndarray


def _as_int_or_none(value: object) -> int | None:
    """``None`` for missing labels (None, NaN, empty string), else ``int``."""
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        return None if np.isnan(value) else int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    s = str(value).strip()
    return None if s == "" or s.lower() == "nan" else int(float(s))


def _shape_attr(shape: tuple[int, ...]) -> str:
    """Spatial shape as a NetCDF-safe attribute string, e.g. ``"61,65"``."""
    return ",".join(str(int(s)) for s in shape)


def _parse_shape(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(int(s) for s in value.split(",") if s.strip())
    return tuple(int(s) for s in np.atleast_1d(np.asarray(value)).tolist())


def _flatten(arr: np.ndarray, leading: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape ``arr`` to ``(*leading dims, n_cell)``; drop a trailing singleton channel."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim > leading + 1 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim < leading + 1:
        raise ValueError(f"array needs at least {leading + 1} dims, got shape {a.shape}")
    space = tuple(a.shape[leading:])
    return a.reshape((*a.shape[:leading], int(np.prod(space)))), space


def _as_cells(values: np.ndarray | None, n_cell: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    v = np.asarray(values)
    if v.size != n_cell:
        raise ValueError(f"{name} has {v.size} values for {n_cell} cells")
    return v.reshape(n_cell)


# ------------------------------------------------------------------------ Observations


@dataclass
class Observations:
    """The observed field, its time axis and the training cutoff.

    Everything a reference learns (climatology, terciles, anomaly means) is learned on
    ``time[:cutoff]``. ``cutoff`` is the index of the first step that is not training.
    """

    series: np.ndarray
    time: np.ndarray
    cutoff: int
    space_shape: tuple[int, ...] = ()
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    mask: np.ndarray | None = None
    name: str = "observations"

    def __post_init__(self) -> None:
        self.series, space = _flatten(self.series, 1)
        if not self.space_shape:
            self.space_shape = space
        self.time = as_datetime_month(self.time)
        if self.time.shape != (self.series.shape[0],):
            raise ValueError("time must have one value per observed step")
        if not np.all(np.diff(self.period) == 1):
            raise ValueError("observations must be consecutive periods without gaps")
        if not 0 < int(self.cutoff) <= self.n_time:
            raise ValueError(f"cutoff {self.cutoff} outside 1..{self.n_time}")
        self.cutoff = int(self.cutoff)
        self.lat = _as_cells(self.lat, self.n_cell, "lat")
        self.lon = _as_cells(self.lon, self.n_cell, "lon")
        m = _as_cells(self.mask, self.n_cell, "mask")
        self.mask = None if m is None else m.astype(bool)

    @property
    def n_time(self) -> int:
        return int(self.series.shape[0])

    @property
    def n_cell(self) -> int:
        return int(self.series.shape[1])

    @property
    def period(self) -> np.ndarray:
        return to_period(self.time)

    @property
    def calendar(self) -> np.ndarray:
        """Calendar month (1..12) of each step."""
        return period_of_year(self.period)

    @property
    def cutoff_period(self) -> int:
        return (
            int(self.period[self.cutoff]) if self.cutoff < self.n_time else int(self.period[-1]) + 1
        )

    @property
    def training(self) -> np.ndarray:
        return self.series[: self.cutoff]

    def index_of(self, periods: np.ndarray) -> np.ndarray:
        """Time index of each period, or -1 when the period is outside the record."""
        p = np.asarray(periods, dtype=np.int64)
        idx = p - int(self.period[0])
        return np.where((idx >= 0) & (idx < self.n_time), idx, -1)

    def at(self, periods: np.ndarray) -> np.ndarray:
        """Observed field at the given periods, NaN where outside the record."""
        idx = self.index_of(periods)
        out = np.full((*idx.shape, self.n_cell), np.nan)
        ok = idx >= 0
        out[ok] = self.series[idx[ok]]
        return out

    @classmethod
    def from_cutoff_time(
        cls, series: np.ndarray, time: np.ndarray, cutoff_time: object, **kw: Any
    ) -> Observations:
        """Build with the cutoff given as the first non-training month."""
        t = as_datetime_month(time)
        c = as_datetime_month(cutoff_time)
        idx = int(np.searchsorted(t, c))
        return cls(series=series, time=t, cutoff=idx, **kw)

    def to_xarray(self) -> xr.Dataset:
        ds = xr.Dataset(
            {"obs": (("time", "cell"), self.series)},
            coords={"time": self.time.astype("datetime64[ns]"), "cell": np.arange(self.n_cell)},
            attrs={
                "cutoff": self.cutoff,
                "space_shape": _shape_attr(self.space_shape),
                "name": self.name,
            },
        )
        if self.lat is not None:
            ds["lat"] = ("cell", self.lat)
        if self.lon is not None:
            ds["lon"] = ("cell", self.lon)
        if self.mask is not None:
            ds["mask"] = ("cell", self.mask.astype(np.int8))
        return ds

    def to_netcdf(self, path: str | Path) -> None:
        self.to_xarray().to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> Observations:
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        return cls(
            series=ds["obs"].values,
            time=ds["time"].values,
            cutoff=int(ds.attrs["cutoff"]),
            space_shape=_parse_shape(ds.attrs.get("space_shape", "")),
            lat=ds["lat"].values if "lat" in ds else None,
            lon=ds["lon"].values if "lon" in ds else None,
            mask=ds["mask"].values.astype(bool) if "mask" in ds else None,
            name=str(ds.attrs.get("name", "observations")),
        )

    @classmethod
    def from_field(
        cls,
        data: xr.DataArray,
        cutoff_time: object,
        time_dim: str = "time",
        mask: np.ndarray | None = None,
    ) -> Observations:
        """Build from a gridded xarray field ``(time, y, x)`` or ``(time, cell)``."""
        da = data.transpose(time_dim, ...)
        lat = lon = None
        if "latitude" in da.coords and "longitude" in da.coords and da.ndim == 3:
            la, lo = np.meshgrid(da["latitude"].values, da["longitude"].values, indexing="ij")
            lat, lon = la.ravel(), lo.ravel()
        elif "lat" in da.coords and "lon" in da.coords and da.ndim == 3:
            la, lo = np.meshgrid(da["lat"].values, da["lon"].values, indexing="ij")
            lat, lon = la.ravel(), lo.ravel()
        return cls.from_cutoff_time(
            da.values,
            da[time_dim].values,
            cutoff_time,
            lat=lat,
            lon=lon,
            mask=mask,
            name=str(da.name or "observations"),
        )


# ---------------------------------------------------------------------------- Forecast


@dataclass
class Forecast:
    """One run of one model: predictions and the targets they verify against.

    ``pred`` and ``target`` are ``(n_origin, n_lead, n_cell)`` after flattening;
    ``origin_time`` is the issue time of each origin; ``lead`` counts periods ahead.
    """

    pred: np.ndarray
    target: np.ndarray
    origin_time: np.ndarray
    lead: np.ndarray
    model: str = "model"
    features: str = ""
    seed: int | None = None
    space_shape: tuple[int, ...] = ()
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    mask: np.ndarray | None = None
    source: str = ""

    def __post_init__(self) -> None:
        self.pred, space = _flatten(self.pred, 2)
        self.target, _ = _flatten(self.target, 2)
        if self.pred.shape != self.target.shape:
            raise ValueError(f"pred {self.pred.shape} and target {self.target.shape} differ")
        if not self.space_shape:
            self.space_shape = space
        self.origin_time = as_datetime_month(self.origin_time)
        self.lead = np.asarray(self.lead, dtype=np.int64).reshape(-1)
        if self.origin_time.shape != (self.n_origin,):
            raise ValueError("origin_time must have one value per origin")
        if self.lead.shape != (self.n_lead,):
            raise ValueError("lead must have one value per lead axis entry")
        if self.lead.min() < 1 or np.any(np.diff(self.lead) <= 0):
            raise ValueError("lead must be strictly increasing integers >= 1")
        op = self.origin_period
        if np.any(np.diff(op) <= 0):
            raise ValueError("origin_time must be strictly increasing")
        self.lat = _as_cells(self.lat, self.n_cell, "lat")
        self.lon = _as_cells(self.lon, self.n_cell, "lon")
        m = _as_cells(self.mask, self.n_cell, "mask")
        self.mask = None if m is None else m.astype(bool)
        if self.seed is not None:
            self.seed = int(self.seed)

    # ----------------------------------------------------------------- shapes
    @property
    def n_origin(self) -> int:
        return int(self.pred.shape[0])

    @property
    def n_lead(self) -> int:
        return int(self.pred.shape[1])

    @property
    def n_cell(self) -> int:
        return int(self.pred.shape[2])

    @property
    def horizon(self) -> int:
        return int(self.lead.max())

    @property
    def origin_period(self) -> np.ndarray:
        return to_period(self.origin_time)

    @property
    def target_period(self) -> np.ndarray:
        """``(n_origin, n_lead)`` period index of every verifying target."""
        return self.origin_period[:, None] + self.lead[None, :]

    @property
    def target_time(self) -> np.ndarray:
        return period_to_datetime(self.target_period)

    @property
    def configuration(self) -> tuple[str, str]:
        return (self.model, self.features)

    @property
    def label(self) -> str:
        feat = f"[{self.features}]" if self.features else ""
        seed = f" seed {self.seed}" if self.seed is not None else " (single run)"
        return f"{self.model}{feat}{seed}"

    def hash(self) -> str:
        return array_hash(self.pred)

    def with_labels(
        self, model: str | None = None, features: str | None = None, seed: int | None = None
    ) -> Forecast:
        f = Forecast(
            pred=self.pred,
            target=self.target,
            origin_time=self.origin_time,
            lead=self.lead,
            model=self.model if model is None else model,
            features=self.features if features is None else features,
            seed=self.seed if seed is None else seed,
            space_shape=self.space_shape,
            lat=self.lat,
            lon=self.lon,
            mask=self.mask,
            source=self.source,
        )
        return f

    def unflatten(self, arr: np.ndarray) -> np.ndarray:
        """Give a ``(..., n_cell)`` array back its spatial shape."""
        return np.asarray(arr).reshape(arr.shape[:-1] + self.space_shape)

    # --------------------------------------------------------------------- io
    def to_xarray(self) -> xr.Dataset:
        ds = xr.Dataset(
            {
                "pred": (("origin", "lead", "cell"), self.pred),
                "target": (("origin", "lead", "cell"), self.target),
            },
            coords={
                "origin": self.origin_time.astype("datetime64[ns]"),
                "lead": self.lead,
                "cell": np.arange(self.n_cell),
            },
            attrs={
                "model": self.model,
                "features": self.features,
                "seed": -1 if self.seed is None else int(self.seed),
                "space_shape": _shape_attr(self.space_shape),
                "source": self.source,
            },
        )
        if self.lat is not None:
            ds["lat"] = ("cell", self.lat)
        if self.lon is not None:
            ds["lon"] = ("cell", self.lon)
        if self.mask is not None:
            ds["mask"] = ("cell", self.mask.astype(np.int8))
        return ds

    def to_netcdf(self, path: str | Path) -> None:
        self.to_xarray().to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path, **labels: Any) -> Forecast:
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        seed = int(ds.attrs.get("seed", -1))
        kw: dict[str, Any] = dict(
            model=str(ds.attrs.get("model", "model")),
            features=str(ds.attrs.get("features", "")),
            seed=None if seed < 0 else seed,
            space_shape=_parse_shape(ds.attrs.get("space_shape", "")),
            lat=ds["lat"].values if "lat" in ds else None,
            lon=ds["lon"].values if "lon" in ds else None,
            mask=ds["mask"].values.astype(bool) if "mask" in ds else None,
            source=str(ds.attrs.get("source", str(path))),
        )
        kw.update(labels)
        return cls(
            pred=ds["pred"].values,
            target=ds["target"].values,
            origin_time=ds["origin"].values,
            lead=ds["lead"].values,
            **kw,
        )

    @classmethod
    def from_long_table(
        cls,
        table: pd.DataFrame | str | Path,
        model: str | None = None,
        features: str | None = None,
        seed: int | None = None,
    ) -> Forecast:
        """Build from rows ``origin_time, lead, cell, pred, target[, model, features, seed]``.

        Cells are whatever identifiers the ``cell`` column holds, ordered by first
        appearance; missing combinations become NaN.
        """
        df = pd.read_csv(table) if not isinstance(table, pd.DataFrame) else table
        need = {"origin_time", "lead", "cell", "pred", "target"}
        if not need <= set(df.columns):
            raise ValueError(f"long table needs columns {sorted(need)}")
        origins = np.array(sorted(pd.to_datetime(df["origin_time"]).unique())).astype(
            "datetime64[M]"
        )
        leads = np.array(sorted(df["lead"].unique()), dtype=np.int64)
        cells = list(pd.unique(df["cell"]))
        oi = {o: i for i, o in enumerate(origins)}
        li = {int(lv): i for i, lv in enumerate(leads)}
        ci = {c: i for i, c in enumerate(cells)}
        shape = (len(origins), len(leads), len(cells))
        pred = np.full(shape, np.nan)
        target = np.full(shape, np.nan)
        o_idx = np.asarray(pd.to_datetime(df["origin_time"]).to_numpy()).astype("datetime64[M]")
        oo = np.array([oi[o] for o in o_idx])
        ll = np.array([li[int(lv)] for lv in df["lead"]])
        cc = np.array([ci[c] for c in df["cell"]])
        pred[oo, ll, cc] = df["pred"].to_numpy(dtype=np.float64)
        target[oo, ll, cc] = df["target"].to_numpy(dtype=np.float64)

        def pick(col: str, given: object) -> object:
            if given is not None:
                return given
            if col in df.columns and df[col].nunique() == 1:
                return df[col].iloc[0]
            return None

        m = pick("model", model)
        fe = pick("features", features)
        return cls(
            pred=pred,
            target=target,
            origin_time=origins,
            lead=leads,
            model="model" if m is None else str(m),
            features="" if fe is None else str(fe),
            seed=_as_int_or_none(pick("seed", seed)),
            space_shape=(len(cells),),
        )


def infer_origin_time(
    target: np.ndarray, obs: Observations, lead: Leads, atol: float = 1e-4
) -> np.ndarray:
    """Recover the origin times of a target array by matching it to the observations.

    Used for layouts that save arrays without a time axis. The first origin's first-lead
    target is matched against every observed step; origins are assumed consecutive and
    the match is then verified at every (origin, lead) by :meth:`Benchmark.validate`.
    """
    t, _ = _flatten(target, 2)
    first = t[0, 0]
    diffs = np.array([np.nanmean(np.abs(obs.series[i] - first)) for i in range(obs.n_time)])
    t0 = int(np.nanargmin(diffs))
    if not np.isfinite(diffs[t0]) or diffs[t0] > atol:
        raise ValueError(
            f"no observed step matches the first target (best mean |diff| {diffs[t0]:.3g})"
        )
    origin0 = t0 - int(lead[0])
    if origin0 < 0:
        raise ValueError("first origin falls before the observed record")
    idx = origin0 + np.arange(t.shape[0])
    if idx[-1] >= obs.n_time:
        raise ValueError("origins run past the observed record")
    return obs.time[idx]


def forecasts_from_directory(
    pattern: str | Path,
    obs: Observations,
    lead: Leads | None = None,
    model: str = "model",
    features: str = "",
    targets_name: str = "targets.npy",
    seed_regex: str = r"SEED(\d+)",
    origin_time: np.ndarray | None = None,
) -> list[Forecast]:
    """Load ``predictions.npy`` arrays matched by a glob, one forecast per file.

    The seed is read from the path with ``seed_regex``; targets are read from the file
    next to each prediction; origins are inferred from the observations unless given.
    """
    paths = sorted(glob.glob(str(pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    out: list[Forecast] = []
    for p in paths:
        pp = Path(p)
        pred = np.load(pp)
        tgt = np.load(pp.parent / targets_name)
        n_lead = pred.shape[1]
        leads = np.arange(1, n_lead + 1) if lead is None else np.asarray(lead)
        ot = infer_origin_time(tgt, obs, leads) if origin_time is None else origin_time
        m = re.search(seed_regex, pp.as_posix())
        seed = int(m.group(1)) if m else None
        out.append(
            Forecast(
                pred=pred,
                target=tgt,
                origin_time=ot,
                lead=leads,
                model=model,
                features=features,
                seed=seed,
                lat=obs.lat,
                lon=obs.lon,
                mask=obs.mask,
                source=pp.as_posix(),
            )
        )
    return out


# --------------------------------------------------------------------------- Benchmark


@dataclass
class Benchmark:
    """Forecasts that verify against identical targets, with their observations."""

    forecasts: list[Forecast]
    observations: Observations
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.forecasts:
            raise ValueError("a benchmark needs at least one forecast")

    # ---------------------------------------------------------------- access
    @property
    def reference(self) -> Forecast:
        """The forecast whose grid defines the benchmark (all share it after validation)."""
        return self.forecasts[0]

    @property
    def origin_period(self) -> np.ndarray:
        return self.reference.origin_period

    @property
    def lead(self) -> np.ndarray:
        return self.reference.lead

    @property
    def target(self) -> np.ndarray:
        return self.reference.target

    @property
    def n_origin(self) -> int:
        return self.reference.n_origin

    @property
    def horizon(self) -> int:
        return self.reference.horizon

    def configurations(self) -> dict[tuple[str, str], list[Forecast]]:
        groups: dict[tuple[str, str], list[Forecast]] = {}
        for f in self.forecasts:
            groups.setdefault(f.configuration, []).append(f)
        return groups

    def cell_mask(self, mask: np.ndarray | str | None) -> np.ndarray | None:
        """Resolve ``None``, ``"domain"`` (the observations' mask) or an explicit array."""
        if mask is None:
            return None
        if isinstance(mask, str):
            if mask != "domain":
                raise ValueError(f"unknown mask {mask!r}")
            if self.observations.mask is None:
                raise ValueError("the observations carry no domain mask")
            return self.observations.mask
        m = np.asarray(mask).reshape(-1).astype(bool)
        if m.size != self.reference.n_cell:
            raise ValueError("mask size differs from the number of cells")
        return m

    # ------------------------------------------------------------ validation
    def validate(self, atol: float = 1e-4) -> list[str]:
        """Check the contract; raise on violation, return informational notes.

        Checks: one grid for every forecast; identical targets across forecasts;
        targets equal to the observations at their target periods; no target inside
        the training period; duplicate prediction arrays under different labels.
        """
        ref = self.reference
        notes: list[str] = []
        for f in self.forecasts[1:]:
            if f.pred.shape != ref.pred.shape:
                raise ValueError(f"{f.label}: shape {f.pred.shape} differs from {ref.pred.shape}")
            if not np.array_equal(f.origin_period, ref.origin_period) or not np.array_equal(
                f.lead, ref.lead
            ):
                raise ValueError(f"{f.label}: origin or lead axis differs from {ref.label}")
            if not _same(f.target, ref.target, atol):
                raise ValueError(
                    f"{f.label}: targets differ from {ref.label}; scores would not be comparable"
                )
        obs = self.observations
        tp = ref.target_period
        if tp.min() < obs.cutoff_period:
            raise ValueError("a verifying target falls inside the observations' training period")
        expected = obs.at(tp)
        if not _same(ref.target, expected, atol):
            d = np.nanmax(np.abs(ref.target - expected))
            raise ValueError(
                "targets do not match the observations at their target periods"
                f" (max |diff| {d:.3g}); check origin_time and lead"
            )
        seen: dict[str, str] = {}
        for f in self.forecasts:
            h = f.hash()
            if h in seen:
                notes.append(f"duplicate prediction array: {f.label} is identical to {seen[h]}")
            else:
                seen[h] = f.label
        for cfg, runs in self.configurations().items():
            seeds = [r.seed for r in runs]
            if len(seeds) != len(set(seeds)):
                notes.append(f"configuration {cfg}: repeated seed labels {seeds}")
            if any(s is None for s in seeds) and len(runs) > 1:
                notes.append(f"configuration {cfg}: mixes labelled seeds and unlabelled runs")
        self.notes = notes
        return notes

    def inputs(self) -> dict[str, str]:
        d = {f.label: f.hash() for f in self.forecasts}
        d["observations"] = array_hash(self.observations.series)
        return d


def _same(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return bool(np.allclose(a[fa], b[fb], rtol=0, atol=atol))


# ------------------------------------------------------------------ declaration loader


def benchmark_from_declaration(decl_data: dict[str, Any], root: str | Path = ".") -> Benchmark:
    """Build a benchmark from the ``data`` section of a declaration.

    ::

        data:
          observations: {format: netcdf, path: obs.nc}
          # or {format: field, path: dataset.nc, variable: total_precipitation, cutoff: "2016-07"}
          forecasts:
            - {format: netcdf, path: model_a_seed42.nc}
            - {format: directory, pattern: "out/SEED*/predictions.npy", model: A, features: BASIC}
            - {format: long_table, path: model_b.csv, model: B}
    """
    root = Path(root)
    o = decl_data.get("observations") or {}
    fmt = str(o.get("format", "netcdf"))
    if fmt == "netcdf":
        obs = Observations.from_netcdf(root / o["path"])
    elif fmt == "field":
        with xr.open_dataset(root / o["path"]) as ds:
            da = ds[o["variable"]].load()
        obs = Observations.from_field(da, o["cutoff"], time_dim=str(o.get("time_dim", "time")))
    else:
        raise ValueError(f"unknown observations format {fmt!r}")
    forecasts: list[Forecast] = []
    for spec in decl_data.get("forecasts") or []:
        f_fmt = str(spec.get("format", "netcdf"))
        labels = {k: spec[k] for k in ("model", "features", "seed") if k in spec}
        if f_fmt == "netcdf":
            forecasts.append(Forecast.from_netcdf(root / spec["path"], **labels))
        elif f_fmt == "long_table":
            forecasts.append(Forecast.from_long_table(root / spec["path"], **labels))
        elif f_fmt == "directory":
            forecasts.extend(
                forecasts_from_directory(
                    root / spec["pattern"],
                    obs,
                    lead=spec.get("lead"),
                    model=str(spec.get("model", "model")),
                    features=str(spec.get("features", "")),
                    targets_name=str(spec.get("targets", "targets.npy")),
                )
            )
        else:
            raise ValueError(f"unknown forecast format {f_fmt!r}")
    return Benchmark(forecasts=forecasts, observations=obs)


def from_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    origin_time: Iterable[object],
    lead: Sequence[int],
    **labels: Any,
) -> Forecast:
    """Convenience constructor mirroring :class:`Forecast`."""
    return Forecast(
        pred=pred,
        target=target,
        origin_time=np.asarray(list(origin_time)),
        lead=np.asarray(lead),
        **labels,
    )
