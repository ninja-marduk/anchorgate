"""Period arithmetic on the time axis.

Version 1 supports a monthly period. Every time stamp is reduced to an integer period
index (``year * 12 + month - 1``) so that origins, leads and targets can be compared
with integer arithmetic and no calendar edge cases.
"""

from __future__ import annotations

import numpy as np

PERIODS_PER_YEAR = 12
FREQ = "M"


def to_period(times: np.ndarray) -> np.ndarray:
    """Integer monthly period index of each datetime64 value."""
    t = np.asarray(times).astype("datetime64[M]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    months = (t - t.astype("datetime64[Y]")).astype(int)  # 0..11
    return (years * PERIODS_PER_YEAR + months).astype(np.int64)


def period_to_datetime(periods: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_period`, as datetime64[M]."""
    p = np.asarray(periods).astype(np.int64)
    years = p // PERIODS_PER_YEAR
    months = p % PERIODS_PER_YEAR
    return (
        np.array(
            [f"{y:04d}-{m + 1:02d}" for y, m in zip(years.ravel(), months.ravel(), strict=True)]
        )
        .astype("datetime64[M]")
        .reshape(p.shape)
    )


def period_of_year(periods: np.ndarray) -> np.ndarray:
    """Calendar month, 1..12, of each period index."""
    return (np.asarray(periods).astype(np.int64) % PERIODS_PER_YEAR) + 1


def as_datetime_month(values: object) -> np.ndarray:
    """Coerce strings, datetimes or datetime64 values to datetime64[M]."""
    return np.asarray(values).astype("datetime64[M]")
