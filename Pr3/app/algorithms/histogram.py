from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def histogram(plane):
    h = np.zeros(256, dtype=np.int64)
    rows, cols = plane.shape
    for y in range(rows):
        for x in range(cols):
            h[plane[y, x]] += 1
    return h


@njit(cache=True)
def cumulative(h):
    c = np.empty(256, dtype=np.int64)
    s = 0
    for i in range(256):
        s += h[i]
        c[i] = s
    return c


@njit(cache=True)
def stats_from_hist(h):
    n = 0
    s = 0.0
    lo = -1
    hi = -1
    for i in range(256):
        if h[i] > 0:
            if lo < 0:
                lo = i
            hi = i
        n += h[i]
        s += i * h[i]
    if n == 0:
        return 0, 0, 0.0, 0.0
    mean = s / n
    d = 0.0
    for i in range(256):
        d += h[i] * (i - mean) * (i - mean)
    return lo, hi, mean, np.sqrt(d / n)


def stats(plane: np.ndarray) -> dict:
    h = histogram(plane)
    lo, hi, mean, sigma = stats_from_hist(h)
    levels = int(np.count_nonzero(h))
    return {"min": int(lo), "max": int(hi), "mean": float(mean),
            "sigma": float(sigma), "levels": levels}
