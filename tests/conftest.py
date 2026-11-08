"""Synthetic data with known answers; the generators live in ``anchorgate.synthetic``."""

from __future__ import annotations

import numpy as np
import pytest

from anchorgate import Benchmark, Observations
from anchorgate.synthetic import (
    origins_after_cutoff,
    synthetic_forecast,
    synthetic_observations,
    targets_for,
)

N_YEARS = 30
SHAPE = (3, 4)
CUTOFF = 24 * 12
LEADS = np.arange(1, 7)


def make_observations(
    seed: int = 0, n_years: int = N_YEARS, shape: tuple[int, int] = SHAPE, cutoff: int = CUTOFF
) -> Observations:
    return synthetic_observations(seed=seed, n_years=n_years, shape=shape, cutoff=cutoff)


make_forecast = synthetic_forecast

__all__ = ["make_forecast", "make_observations", "origins_after_cutoff", "targets_for"]


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
