
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..core.buffer import PixelBuffer
from .color import hsv_to_rgb, rgb_to_hsv
from .histogram import cumulative, histogram

COLOR_RGB = "rgb"
COLOR_HSV = "hsv"


@njit(cache=True, parallel=True)
def apply_lut(plane, lut):
    h, w = plane.shape
    out = np.empty((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            out[y, x] = lut[plane[y, x]]
    return out


@njit(cache=True)
def linear_lut(a, b):
    lut = np.empty(256, dtype=np.uint8)
    if b <= a:
        for i in range(256):
            lut[i] = i
        return lut
    k = 255.0 / (b - a)
    for i in range(256):
        v = (i - a) * k
        if v < 0.0:
            v = 0.0
        elif v > 255.0:
            v = 255.0
        lut[i] = np.uint8(int(v + 0.5))
    return lut


@njit(cache=True)
def equalize_lut(h):
    c = cumulative(h)
    n = c[255]
    cdf_min = 0
    for i in range(256):
        if h[i] > 0:
            cdf_min = c[i]
            break
    lut = np.empty(256, dtype=np.uint8)
    if n == cdf_min:
        for i in range(256):
            lut[i] = i
        return lut
    for i in range(256):
        if c[i] <= cdf_min:
            lut[i] = 0
        else:
            lut[i] = np.uint8(int((c[i] - cdf_min) * 255.0 / (n - cdf_min) + 0.5))
    return lut


def percentile_range(h: np.ndarray, percent: float) -> tuple[int, int]:
    c = cumulative(h)
    n = c[255]
    cut = n * percent / 100.0
    a = int(np.searchsorted(c, cut, side="right")) if cut > 0 else int(np.argmax(h > 0))
    b = int(np.searchsorted(c, n - cut, side="left"))
    a = min(max(a, 0), 255)
    b = min(max(b, 0), 255)
    return a, b


def _range_for(plane: np.ndarray, mode: str, a: int, b: int, percent: float) -> tuple[int, int]:
    h = histogram(plane)
    if mode == "auto":
        nz = np.nonzero(h)[0]
        return int(nz[0]), int(nz[-1])
    if mode == "percent":
        return percentile_range(h, percent)
    return int(a), int(b)


def process_planes(buf: PixelBuffer, color_mode: str, fn) -> tuple[PixelBuffer, list]:
    if buf.is_gray:
        p, info = fn(buf.channel(0))
        return PixelBuffer.from_plane(p), [info]
    if color_mode == COLOR_HSV:
        H, S, V = rgb_to_hsv(buf.data)
        V2, info = fn(V)
        return PixelBuffer(hsv_to_rgb(H, S, V2)), [info]
    out = np.empty_like(buf.data)
    infos = []
    for c in range(3):
        p, info = fn(buf.channel(c))
        out[:, :, c] = p
        infos.append(info)
    return PixelBuffer(out), infos


def linear_contrast(buf: PixelBuffer, mode: str = "auto", a: int = 0, b: int = 255,
                    percent: float = 1.0, color_mode: str = COLOR_HSV):
    def fn(plane):
        lo, hi = _range_for(plane, mode, a, b, percent)
        return apply_lut(plane, linear_lut(lo, hi)), (lo, hi)
    out, infos = process_planes(buf, color_mode, fn)
    return out, {"ranges": infos}


def equalize(buf: PixelBuffer, color_mode: str = COLOR_HSV):
    def fn(plane):
        return apply_lut(plane, equalize_lut(histogram(plane))), None
    out, _ = process_planes(buf, color_mode, fn)
    return out, {}


def gamma(buf: PixelBuffer, g: float = 1.0, color_mode: str = COLOR_HSV):
    lut = np.array([int(255.0 * (i / 255.0) ** g + 0.5) for i in range(256)], dtype=np.uint8)
    out, _ = process_planes(buf, color_mode, lambda p: (apply_lut(p, lut), None))
    return out, {}


def negative(buf: PixelBuffer):
    lut = np.array([255 - i for i in range(256)], dtype=np.uint8)
    out, _ = process_planes(buf, COLOR_RGB, lambda p: (apply_lut(p, lut), None))
    return out, {}
