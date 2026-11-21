"""The deterministic scores the protocol needs, and nothing more.

Each function takes ``pred`` and ``obs`` arrays of identical shape whose last axis is
the spatial cell; leading axes (origin, lead) are pooled unless the function says
otherwise. NaN in either array removes the pair.
"""

from __future__ import annotations

import numpy as np


def _pairs(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    if p.shape != o.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs obs {o.shape}")
    m = np.isfinite(p) & np.isfinite(o)
    return p[m], o[m]


def mse(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _pairs(pred, obs)
    return float(np.mean((o - p) ** 2)) if o.size else float("nan")


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, obs)))


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _pairs(pred, obs)
    return float(np.mean(np.abs(o - p))) if o.size else float("nan")


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean of prediction minus observation."""
    p, o = _pairs(pred, obs)
    return float(np.mean(p - o)) if o.size else float("nan")


def pooled_r2(pred: np.ndarray, obs: np.ndarray) -> float:
    """Coefficient of determination against one global mean over every pooled pair.

    Dominated by between-cell and seasonal variance, so a climatology scores high on
    it almost anywhere. Report it together with :func:`per_cell_nse`.
    """
    p, o = _pairs(pred, obs)
    if o.size == 0:
        return float("nan")
    ss = float(np.sum((o - o.mean()) ** 2))
    return float(1 - np.sum((o - p) ** 2) / ss) if ss > 0 else float("nan")


def per_cell_nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean over cells of the Nash-Sutcliffe efficiency against each cell's own mean.

    Removes the between-cell variance that inflates :func:`pooled_r2`; a per-cell
    monthly climatology still scores above zero here through the seasonal cycle it
    carries. Only the deseasonalised anomaly test (component 4) removes that too.
    """
    p: np.ndarray = np.asarray(pred, dtype=np.float64).reshape(-1, np.shape(pred)[-1])
    o: np.ndarray = np.asarray(obs, dtype=np.float64).reshape(-1, np.shape(obs)[-1])
    m = np.isfinite(p) & np.isfinite(o)
    p = np.where(m, p, np.nan)
    o = np.where(m, o, np.nan)
    n = m.sum(0)
    good = n >= 2
    if not good.any():
        return float("nan")
    cell_mean = np.nanmean(o[:, good], axis=0)
    ss_res = np.nansum((o[:, good] - p[:, good]) ** 2, axis=0)
    ss_tot = np.nansum((o[:, good] - cell_mean) ** 2, axis=0)
    ok = ss_tot > 0
    return float(np.mean(1 - ss_res[ok] / ss_tot[ok])) if ok.any() else float("nan")


def msess(pred: np.ndarray, obs: np.ndarray, ref: np.ndarray) -> float:
    """Mean-squared-error skill score against a reference, ``1 - MSE / MSE_ref``.

    Pairs are kept only where prediction, observation and reference are all finite,
    so the two errors are computed on identical targets.
    """
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if not (p.shape == o.shape == r.shape):
        raise ValueError("pred, obs and ref must share a shape")
    m = np.isfinite(p) & np.isfinite(o) & np.isfinite(r)
    if not m.any():
        return float("nan")
    e_model = float(np.mean((o[m] - p[m]) ** 2))
    e_ref = float(np.mean((o[m] - r[m]) ** 2))
    return float(1 - e_model / e_ref) if e_ref > 0 else float("nan")


def score_table(
    pred: np.ndarray, obs: np.ndarray, ref: np.ndarray | None = None
) -> dict[str, float]:
    """The standard row of scores; ``msess`` is present only when ``ref`` is given."""
    out = {
        "pooled_r2": pooled_r2(pred, obs),
        "per_cell_nse": per_cell_nse(pred, obs),
        "rmse": rmse(pred, obs),
        "mae": mae(pred, obs),
        "bias": bias(pred, obs),
    }
    if ref is not None:
        out["msess"] = msess(pred, obs, ref)
    return out


# ---------------------------------------------------------------------------
# Categorical scores on binary events
# ---------------------------------------------------------------------------


def tercile_events(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, np.ndarray]:
    """Dry (``< lo``) and wet (``> hi``) events; NaN where any input is missing."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi)
    dry = np.where(finite, (v < lo).astype(np.float64), np.nan)
    wet = np.where(finite, (v > hi).astype(np.float64), np.nan)
    return {"dry": dry, "wet": wet}


def contingency(f_event: np.ndarray, o_event: np.ndarray) -> dict[str, int]:
    """Hits, misses, false alarms and correct negatives over finite pairs."""
    f = np.asarray(f_event, dtype=np.float64)
    o = np.asarray(o_event, dtype=np.float64)
    m = np.isfinite(f) & np.isfinite(o)
    f_b, o_b = f[m] > 0.5, o[m] > 0.5
    return {
        "hits": int(np.sum(f_b & o_b)),
        "misses": int(np.sum(~f_b & o_b)),
        "false_alarms": int(np.sum(f_b & ~o_b)),
        "correct_negatives": int(np.sum(~f_b & ~o_b)),
    }


def categorical_scores(f_event: np.ndarray, o_event: np.ndarray) -> dict[str, float]:
    """POD, FAR, CSI, frequency bias, Heidke, Peirce and equitable threat scores.

    HSS, PSS and ETS are zero for a random or constant forecast with the same marginals
    and one for a perfect forecast.
    """
    c = contingency(f_event, o_event)
    a, b, cc, d = c["hits"], c["false_alarms"], c["misses"], c["correct_negatives"]
    n = a + b + cc + d

    def div(x: float, y: float) -> float:
        return float(x / y) if y else float("nan")

    pod = div(a, a + cc)
    far = div(b, a + b)
    csi = div(a, a + b + cc)
    fbias = div(a + b, a + cc)
    expected = div((a + cc) * (a + b) + (d + cc) * (d + b), n)
    hss = div(2 * (a * d - b * cc), (a + cc) * (cc + d) + (a + b) * (b + d))
    pss = div(a * d - b * cc, (a + cc) * (b + d))
    a_rand = div((a + b) * (a + cc), n)
    ets = div(a - a_rand, a + b + cc - a_rand)
    return {
        "n": float(n),
        "pod": pod,
        "far": far,
        "csi": csi,
        "frequency_bias": fbias,
        "hss": hss,
        "pss": pss,
        "ets": ets,
        "proportion_correct": div(a + d, n),
        "expected_correct": div(expected, n),
    }
