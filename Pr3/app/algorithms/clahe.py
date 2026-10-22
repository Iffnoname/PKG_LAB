
from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..core.buffer import PixelBuffer
from ..core.parallel import run_bands
from .contrast import COLOR_HSV, process_planes


@njit(cache=True, parallel=True)
def _clahe_sliding_rows(pad, out, r, limit, y0, y1):
    h_out, w_out = out.shape
    win = 2 * r + 1
    n = win * win
    for y in prange(y0, y1):
        hist = np.zeros(256, dtype=np.int32)
        for dy in range(win):
            for dx in range(win):
                hist[pad[y + dy, dx]] += 1
        for x in range(w_out):
            if x > 0:
                xl = x - 1          # уходящий столбец
                xr = x + 2 * r      # входящий столбец
                for dy in range(win):
                    hist[pad[y + dy, xl]] -= 1
                    hist[pad[y + dy, xr]] += 1
            v = pad[y + r, x + r]
            total = 0.0
            part = 0.0
            for k in range(256):
                c = hist[k] if hist[k] < limit else limit
                total += c
                if k <= v:
                    part += c
            excess = n - total
            cdf = part + excess * (v + 1) / 256.0
            val = 255.0 * cdf / n
            if val > 255.0:
                val = 255.0
            out[y, x] = np.uint8(int(val + 0.5))


def clahe_sliding_plane(plane: np.ndarray, radius: int, clip: float, border: int,
                        progress=None) -> np.ndarray:
    radius = max(1, int(radius))
    win = 2 * radius + 1
    limit = max(1.0, clip * win * win / 256.0)
    pad = PixelBuffer.padded_u8(plane, radius, border)
    out = np.empty_like(plane)
    run_bands(_clahe_sliding_rows, plane.shape[0], progress, pad, out, radius, limit)
    return out


@njit(cache=True)
def _tile_lut(plane, y0, y1, x0, x1, clip):
    hist = np.zeros(256, dtype=np.int64)
    for y in range(y0, y1):
        for x in range(x0, x1):
            hist[plane[y, x]] += 1
    n = (y1 - y0) * (x1 - x0)
    limit = max(1, int(clip * n / 256.0))
    excess = 0
    for k in range(256):
        if hist[k] > limit:
            excess += hist[k] - limit
            hist[k] = limit
    add = excess // 256
    rest = excess - add * 256
    for k in range(256):
        hist[k] += add
    if rest > 0:
        step = max(1, 256 // rest)
        k = 0
        while k < 256 and rest > 0:
            hist[k] += 1
            rest -= 1
            k += step
    lut = np.empty(256, dtype=np.float32)
    s = 0
    for k in range(256):
        s += hist[k]
        lut[k] = min(255.0, 255.0 * s / n)
    return lut


@njit(cache=True, parallel=True)
def _clahe_tiles(plane, gx, gy, clip):
    h, w = plane.shape
    luts = np.empty((gy, gx, 256), dtype=np.float32)
    for t in prange(gx * gy):
        ty = t // gx
        tx = t % gx
        y0 = ty * h // gy
        y1 = (ty + 1) * h // gy
        x0 = tx * w // gx
        x1 = (tx + 1) * w // gx
        luts[ty, tx, :] = _tile_lut(plane, y0, y1, x0, x1, clip)
    out = np.empty((h, w), dtype=np.uint8)
    th = h / gy
    tw = w / gx
    for y in prange(h):
        fy = (y + 0.5) / th - 0.5
        ty0 = int(np.floor(fy))
        ay = fy - ty0
        if ty0 < 0:
            ty0, ay = 0, 0.0
        if ty0 >= gy - 1:
            ty0, ay = gy - 1, 0.0
        ty1 = min(ty0 + 1, gy - 1)
        for x in range(w):
            fx = (x + 0.5) / tw - 0.5
            tx0 = int(np.floor(fx))
            ax = fx - tx0
            if tx0 < 0:
                tx0, ax = 0, 0.0
            if tx0 >= gx - 1:
                tx0, ax = gx - 1, 0.0
            tx1 = min(tx0 + 1, gx - 1)
            v = plane[y, x]
            top = (1.0 - ax) * luts[ty0, tx0, v] + ax * luts[ty0, tx1, v]
            bot = (1.0 - ax) * luts[ty1, tx0, v] + ax * luts[ty1, tx1, v]
            out[y, x] = np.uint8(int((1.0 - ay) * top + ay * bot + 0.5))
    return out


def clahe(buf: PixelBuffer, variant: str = "sliding", radius: int = 32, tiles: int = 8,
          clip: float = 2.0, border: int = 1, color_mode: str = COLOR_HSV, progress=None):
    planes_total = 1 if (buf.is_gray or color_mode == COLOR_HSV) else 3
    counter = [0]

    def fn(plane):
        k = counter[0]
        counter[0] += 1
        sub = None
        if progress is not None:
            sub = lambda f: progress((k + f) / planes_total)
        if variant == "tiles":
            res = _clahe_tiles(plane, int(tiles), int(tiles), float(clip))
            if sub:
                sub(1.0)
            return res, None
        return clahe_sliding_plane(plane, radius, clip, border, sub), None

    out, _ = process_planes(buf, color_mode, fn)
    return out, {}
