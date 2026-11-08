"""Block-bootstrap intervals over scored origins, shared by the gating components.

A verdict is read from an interval, not from a point. A configuration passes when the
lower bound of the interval of its seed-mean statistic clears the threshold, fails when
the upper bound does not, and is INSUFFICIENT when the interval straddles the threshold.
The resampling unit is the forecast origin, drawn in circular moving blocks so that the
month-to-month dependence of the targets survives in every resample; the same resamples
are used for every run, so seed means are taken over matched draws.
"""

from __future__ import annotations

import numpy as np

from .result import Verdict


def block_indices(n: int, block: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """``(n_boot, n)`` origin indices drawn as circular moving blocks of ``block`` origins."""
    if n < 1:
        raise ValueError("at least one origin is needed")
    rng = np.random.default_rng(seed)
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx % n


def skill_from_sums(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``1 - sum(num[idx]) / sum(den[idx])`` per resample.

    ``num`` and ``den`` are per-origin sums of squared errors of a forecast and of the
    reference, shaped ``(n,)`` or ``(n, k)``; the result is ``(n_boot,)`` or ``(n_boot, k)``.
    """
    num_s = np.asarray(num, dtype=float)[idx].sum(axis=1)
    den_s = np.asarray(den, dtype=float)[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - num_s / den_s
    return np.where(den_s > 0, out, np.nan)


def squared_error_sums(pred: np.ndarray, target: np.ndarray, per_lead: bool) -> np.ndarray:
    """Per-origin sums of squared errors over cells (and over leads unless ``per_lead``),
    ignoring missing pairs: ``(n,)`` or ``(n, k)`` for inputs ``(n, k, cells)``."""
    d = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    d = np.where(np.isfinite(d), d, 0.0) ** 2
    return d.sum(axis=-1) if per_lead else d.sum(axis=(1, 2))


def correlation_sums(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-origin sufficient statistics of the per-cell correlation between ``x`` and
    ``y`` shaped ``(n, k, cells)``: counts and the five sums, each ``(n, cells)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x0 = np.where(m, x, 0.0)
    y0 = np.where(m, y, 0.0)
    return {
        "n": m.sum(axis=1).astype(float),
        "sx": x0.sum(axis=1),
        "sy": y0.sum(axis=1),
        "sxx": (x0 * x0).sum(axis=1),
        "syy": (y0 * y0).sum(axis=1),
        "sxy": (x0 * y0).sum(axis=1),
    }


def median_correlation(
    sums: dict[str, np.ndarray], idx: np.ndarray, chunk: int = 200
) -> np.ndarray:
    """Median over cells of the per-cell correlation, per resample ``(n_boot,)``."""
    n = sums["n"].shape[0]
    out = np.empty(idx.shape[0])
    for start in range(0, idx.shape[0], chunk):
        sub = idx[start : start + chunk]
        w = np.zeros((sub.shape[0], n))
        rows = np.repeat(np.arange(sub.shape[0]), sub.shape[1])
        np.add.at(w, (rows, sub.ravel()), 1.0)
        cnt = w @ sums["n"]
        sx, sy = w @ sums["sx"], w @ sums["sy"]
        sxx, syy, sxy = w @ sums["sxx"], w @ sums["syy"], w @ sums["sxy"]
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy - sx * sy / cnt
            vx = sxx - sx * sx / cnt
            vy = syy - sy * sy / cnt
            r = cov / np.sqrt(vx * vy)
        r = np.where((cnt >= 3) & (vx > 0) & (vy > 0), r, np.nan)
        with np.errstate(all="ignore"):
            out[start : start + chunk] = np.nanmedian(r, axis=1) if r.size else np.nan
    return out


def interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Central percentile interval of ``samples`` at ``level`` (NaN when nothing is finite)."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(s, tail)), float(np.quantile(s, 1.0 - tail))


def verdict_from_interval(lower: float, upper: float, threshold: float) -> Verdict:
    """PASS when the lower bound clears the threshold, FAIL when the upper bound does not
    (or the statistic is undefined), INSUFFICIENT when the interval straddles it."""
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return Verdict.FAIL
    if lower > threshold:
        return Verdict.PASS
    if upper <= threshold:
        return Verdict.FAIL
    return Verdict.INSUFFICIENT


def skill_horizon(lower_by_lead: np.ndarray, leads: np.ndarray, threshold: float) -> int:
    """The last lead of the unbroken run, from the shortest lead, whose lower bound clears
    the threshold; 0 when the shortest lead does not."""
    order = np.argsort(leads)
    horizon = 0
    for j in order:
        if np.isfinite(lower_by_lead[j]) and lower_by_lead[j] > threshold:
            horizon = int(leads[j])
        else:
            break
    return horizon
