"""Synthetic data with known answers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from anchorgate import Benchmark, Forecast, Observations
from anchorgate.references import climatology_table

N_YEARS = 30
SHAPE = (3, 4)
CUTOFF = 24 * 12
LEADS = np.arange(1, 7)


def make_observations(
    seed: int = 0, n_years: int = N_YEARS, shape: tuple[int, int] = SHAPE, cutoff: int = CUTOFF
) -> Observations:
    rng = np.random.default_rng(seed)
    T = n_years * 12
    n_cell = int(np.prod(shape))
    month = np.arange(T) % 12
    cell_mean = rng.uniform(50, 200, n_cell)
    amp = rng.uniform(10, 60, n_cell)
    phase = rng.uniform(0, 2 * np.pi, n_cell)
    seasonal = cell_mean[None, :] + amp[None, :] * np.cos(
        2 * np.pi * month[:, None] / 12 + phase[None, :]
    )
    noise = rng.normal(0, 15, (T, n_cell))
    series = np.clip(seasonal + noise, 0, None).reshape(T, *shape)
    time = np.arange("1990-01", "1990-01", dtype="datetime64[M]")
    time = np.datetime64("1990-01", "M") + np.arange(T)
    mask = np.zeros(n_cell, bool)
    mask[: n_cell // 2] = True
    return Observations(series=series, time=time, cutoff=cutoff, mask=mask)


def origins_after_cutoff(obs: Observations, leads: np.ndarray = LEADS) -> np.ndarray:
    first = obs.cutoff - int(leads.min())
    last = obs.n_time - int(leads.max())
    return np.arange(first, last)


def targets_for(obs: Observations, origin_idx: np.ndarray, leads: np.ndarray = LEADS) -> np.ndarray:
    tp = obs.period[origin_idx][:, None] + leads[None, :]
    return obs.at(tp)


def make_forecast(
    obs: Observations,
    kind: str,
    seed: int | None = 1,
    model: str | None = None,
    features: str = "BASIC",
    leads: np.ndarray = LEADS,
    origin_idx: np.ndarray | None = None,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Forecast:
    """``kind`` in perfect | climatology | half | noisy | custom (uses ``transform``)."""
    oi = origins_after_cutoff(obs, leads) if origin_idx is None else origin_idx
    target = targets_for(obs, oi, leads)
    tp = obs.period[oi][:, None] + leads[None, :]
    clim = climatology_table(obs)[(tp % 12)]
    rng = np.random.default_rng(seed or 0)
    if kind == "perfect":
        pred = target.copy()
    elif kind == "climatology":
        pred = clim.copy()
    elif kind == "half":
        pred = clim + 0.5 * (target - clim)
    elif kind == "noisy":
        pred = target + rng.normal(0, 25, target.shape)
    elif kind == "custom" and transform is not None:
        pred = transform(target, clim)
    else:
        raise ValueError(kind)
    return Forecast(
        pred=pred,
        target=target,
        origin_time=obs.time[oi],
        lead=leads,
        model=model or kind,
        features=features,
        seed=seed,
        lat=obs.lat,
        lon=obs.lon,
        mask=obs.mask,
    )


@pytest.fixture
def obs() -> Observations:
    return make_observations()


@pytest.fixture
def benchmark(obs: Observations) -> Benchmark:
    fcs = [
        make_forecast(obs, "perfect", seed=1),
        make_forecast(obs, "perfect", seed=2),
        make_forecast(obs, "half", seed=1),
        make_forecast(obs, "half", seed=2),
        make_forecast(obs, "climatology", seed=1),
        make_forecast(obs, "climatology", seed=2),
        make_forecast(obs, "noisy", seed=None, model="single"),
    ]
    return Benchmark(fcs, obs)
