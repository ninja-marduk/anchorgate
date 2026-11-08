"""Synthetic monthly fields with a known seasonal cycle, for tests, tutorials and smoke
runs. Nothing here is a model of real precipitation; the point is that every answer is
known in advance (a perfect forecast, a climatology forecast, a half-way forecast)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .data import Forecast, Observations
from .references import climatology_table

DEFAULT_LEADS = np.arange(1, 7)


def synthetic_observations(
    seed: int = 0,
    n_years: int = 30,
    shape: tuple[int, int] = (3, 4),
    cutoff: int | None = None,
    start: str = "1990-01",
    noise_sd: float = 15.0,
    ar: float = 0.0,
) -> Observations:
    """A non-negative monthly field: per-cell mean plus a per-cell seasonal cosine plus
    noise, clipped at zero. With ``ar`` above zero the noise is an AR(1) process in time,
    so lagged values carry information about the anomaly (the case a model can exploit);
    at zero the anomalies are white and no model can beat the climatology. The training
    cutoff defaults to 80 % of the record. The domain mask marks the first half of the
    cells."""
    rng = np.random.default_rng(seed)
    n_time = n_years * 12
    n_cell = int(np.prod(shape))
    month = np.arange(n_time) % 12
    cell_mean = rng.uniform(50, 200, n_cell)
    amplitude = rng.uniform(10, 60, n_cell)
    phase = rng.uniform(0, 2 * np.pi, n_cell)
    seasonal = cell_mean[None, :] + amplitude[None, :] * np.cos(
        2 * np.pi * month[:, None] / 12 + phase[None, :]
    )
    noise = rng.normal(0, noise_sd, (n_time, n_cell))
    if ar:
        for t_ in range(1, n_time):
            noise[t_] = ar * noise[t_ - 1] + np.sqrt(1 - ar**2) * noise[t_]
    series = np.clip(seasonal + noise, 0, None)
    time = np.datetime64(start, "M") + np.arange(n_time)
    mask = np.zeros(n_cell, bool)
    mask[: n_cell // 2] = True
    cut = int(0.8 * n_time) if cutoff is None else cutoff
    return Observations(series=series.reshape(n_time, *shape), time=time, cutoff=cut, mask=mask)


def origins_after_cutoff(obs: Observations, leads: np.ndarray = DEFAULT_LEADS) -> np.ndarray:
    """Every origin whose first target is at or after the cutoff and whose last target
    is inside the record."""
    first = obs.cutoff - int(leads.min())
    last = obs.n_time - int(leads.max())
    return np.arange(first, last)


def targets_for(
    obs: Observations, origin_idx: np.ndarray, leads: np.ndarray = DEFAULT_LEADS
) -> np.ndarray:
    tp = obs.period[origin_idx][:, None] + leads[None, :]
    return obs.at(tp)


def synthetic_forecast(
    obs: Observations,
    kind: str,
    seed: int | None = 1,
    model: str | None = None,
    features: str = "BASIC",
    leads: np.ndarray = DEFAULT_LEADS,
    origin_idx: np.ndarray | None = None,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Forecast:
    """A forecast of known quality.

    ``kind`` is ``perfect`` (the target), ``climatology`` (the training climatology),
    ``half`` (half way from the climatology to the target), ``noisy`` (the target plus
    Gaussian noise of s.d. 25) or ``custom`` (``transform(target, climatology)``).
    """
    oi = origins_after_cutoff(obs, leads) if origin_idx is None else origin_idx
    target = targets_for(obs, oi, leads)
    tp = obs.period[oi][:, None] + leads[None, :]
    clim = climatology_table(obs)[tp % 12]
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
        raise ValueError(f"unknown kind {kind!r}")
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


__all__ = ["origins_after_cutoff", "synthetic_forecast", "synthetic_observations", "targets_for"]
