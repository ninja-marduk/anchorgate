"""Zero-cost references, built from the observations on the training period only."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._time import PERIODS_PER_YEAR, period_of_year
from .data import Observations

REFERENCE_NAMES = ("climatology", "persistence", "seasonal_naive")


def climatology_table(obs: Observations) -> np.ndarray:
    """``(12, n_cell)`` mean of each calendar month over the training steps."""
    cal = obs.calendar[: obs.cutoff]
    train = obs.training
    table = np.full((PERIODS_PER_YEAR, obs.n_cell), np.nan)
    for m in range(1, PERIODS_PER_YEAR + 1):
        sel = cal == m
        if sel.any():
            table[m - 1] = np.nanmean(train[sel], axis=0)
    return table


def training_terciles(obs: Observations) -> tuple[np.ndarray, np.ndarray]:
    """``(12, n_cell)`` lower and upper terciles of each calendar month, training only."""
    cal = obs.calendar[: obs.cutoff]
    train = obs.training
    lo = np.full((PERIODS_PER_YEAR, obs.n_cell), np.nan)
    hi = np.full((PERIODS_PER_YEAR, obs.n_cell), np.nan)
    for m in range(1, PERIODS_PER_YEAR + 1):
        sel = cal == m
        if sel.sum() >= 3:
            lo[m - 1] = np.nanquantile(train[sel], 1 / 3, axis=0)
            hi[m - 1] = np.nanquantile(train[sel], 2 / 3, axis=0)
    return lo, hi


def climatology(obs: Observations, origin_period: np.ndarray, lead: np.ndarray) -> np.ndarray:
    """Per-cell training climatology of each target's calendar month."""
    table = climatology_table(obs)
    tp = np.asarray(origin_period)[:, None] + np.asarray(lead)[None, :]
    return table[period_of_year(tp) - 1]


def persistence(obs: Observations, origin_period: np.ndarray, lead: np.ndarray) -> np.ndarray:
    """The last observed value at the origin, carried across every lead."""
    at_origin = obs.at(np.asarray(origin_period))
    return np.repeat(at_origin[:, None, :], len(lead), axis=1)


def seasonal_naive(obs: Observations, origin_period: np.ndarray, lead: np.ndarray) -> np.ndarray:
    """The observation one year before each target."""
    tp = np.asarray(origin_period)[:, None] + np.asarray(lead)[None, :]
    return obs.at(tp - PERIODS_PER_YEAR)


BUILDERS = {
    "climatology": climatology,
    "persistence": persistence,
    "seasonal_naive": seasonal_naive,
}


def build_references(
    obs: Observations,
    origin_period: np.ndarray,
    lead: np.ndarray,
    which: Iterable[str] = REFERENCE_NAMES,
) -> dict[str, np.ndarray]:
    """``name -> (n_origin, n_lead, n_cell)`` arrays on the benchmark's grid."""
    out: dict[str, np.ndarray] = {}
    for name in which:
        if name not in BUILDERS:
            raise ValueError(f"unknown reference {name!r}; known: {sorted(BUILDERS)}")
        out[name] = BUILDERS[name](obs, origin_period, lead)
    return out
