
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..core.buffer import PixelBuffer
from ..core.parallel import Stage, run_bands


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    r = max(1, int(math.ceil(3.0 * sigma)))
    k = np.array([math.exp(-(t * t) / (2.0 * sigma * sigma)) for t in range(-r, r + 1)],
                 dtype=np.float32)
    return (k / k.sum()).astype(np.float32)


def box_kernel_1d(size: int) -> np.ndarray:
    size = max(1, int(size) | 1)
    return np.full(size, 1.0 / size, dtype=np.float32)


@njit(cache=True, parallel=True)
def _rows_pass(pad, kx, tmp, y0, y1):
    _, w = tmp.shape
    kn = kx.shape[0]
    for y in prange(y0, y1):
        for x in range(w):
            s = 0.0
            for k in range(kn):
                s += kx[k] * pad[y, x + k]
            tmp[y, x] = s


@njit(cache=True, parallel=True)
def _cols_pass(tmp, ky, out, y0, y1):
    _, w = out.shape
    kn = ky.shape[0]
    for y in prange(y0, y1):
        for x in range(w):
            s = 0.0
            for k in range(kn):
                s += ky[k] * tmp[y + k, x]
            out[y, x] = s


def separable_filter(plane: np.ndarray, kx: np.ndarray, ky: np.ndarray, border: int,
                     progress=None) -> np.ndarray:
    kx = np.ascontiguousarray(kx, dtype=np.float32)
    ky = np.ascontiguousarray(ky, dtype=np.float32)
    rx, ry = kx.shape[0] // 2, ky.shape[0] // 2
    h, w = plane.shape
    pad = PixelBuffer.padded(plane, ry, rx, border)
    tmp = np.empty((h + 2 * ry, w), dtype=np.float32)
    out = np.empty((h, w), dtype=np.float32)
    st = Stage(progress)
    run_bands(_rows_pass, h + 2 * ry, st.sub(0.0, 0.5), pad, kx, tmp)
    run_bands(_cols_pass, h, st.sub(0.5, 1.0), tmp, ky, out)
    return out


@njit(cache=True, parallel=True)
def _direct_pass(pad, kern, out, y0, y1):
    _, w = out.shape
    kh, kw = kern.shape
    for y in prange(y0, y1):
        for x in range(w):
            s = 0.0
            for i in range(kh):
                for j in range(kw):
                    s += kern[i, j] * pad[y + i, x + j]
            out[y, x] = s


def filter2d(plane: np.ndarray, kern: np.ndarray, border: int, progress=None) -> np.ndarray:
    kern = np.ascontiguousarray(kern, dtype=np.float32)
    ry, rx = kern.shape[0] // 2, kern.shape[1] // 2
    h, w = plane.shape
    pad = PixelBuffer.padded(plane, ry, rx, border)
    out = np.empty((h, w), dtype=np.float32)
    run_bands(_direct_pass, h, progress, pad, kern, out)
    return out


def gaussian_blur(plane: np.ndarray, sigma: float, border: int, progress=None) -> np.ndarray:
    k = gaussian_kernel_1d(sigma)
    return separable_filter(plane, k, k, border, progress)
