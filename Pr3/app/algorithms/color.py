from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def rgb_to_hsv(rgb):
    h_, w_, _ = rgb.shape
    H = np.empty((h_, w_), dtype=np.float32)
    S = np.empty((h_, w_), dtype=np.float32)
    V = np.empty((h_, w_), dtype=np.uint8)
    for y in prange(h_):
        for x in range(w_):
            r = float(rgb[y, x, 0])
            g = float(rgb[y, x, 1])
            b = float(rgb[y, x, 2])
            mx = max(r, max(g, b))
            mn = min(r, min(g, b))
            d = mx - mn
            if d == 0.0:
                hue = 0.0
            elif mx == r:
                hue = 60.0 * ((g - b) / d)
                if hue < 0.0:
                    hue += 360.0
            elif mx == g:
                hue = 60.0 * ((b - r) / d) + 120.0
            else:
                hue = 60.0 * ((r - g) / d) + 240.0
            H[y, x] = hue
            S[y, x] = 0.0 if mx == 0.0 else d / mx
            V[y, x] = np.uint8(mx)
    return H, S, V


@njit(cache=True, inline="always")
def hsv_pixel_to_rgb(hue, s, v):
    c = v * s
    hp = (hue % 360.0) / 60.0
    xx = c * (1.0 - abs(hp % 2.0 - 1.0))
    if hp < 1.0:
        r, g, b = c, xx, 0.0
    elif hp < 2.0:
        r, g, b = xx, c, 0.0
    elif hp < 3.0:
        r, g, b = 0.0, c, xx
    elif hp < 4.0:
        r, g, b = 0.0, xx, c
    elif hp < 5.0:
        r, g, b = xx, 0.0, c
    else:
        r, g, b = c, 0.0, xx
    m = v - c
    return r + m, g + m, b + m


@njit(cache=True, parallel=True)
def hsv_to_rgb(H, S, V):
    h_, w_ = V.shape
    out = np.empty((h_, w_, 3), dtype=np.uint8)
    for y in prange(h_):
        for x in range(w_):
            r, g, b = hsv_pixel_to_rgb(H[y, x], S[y, x], float(V[y, x]))
            out[y, x, 0] = np.uint8(min(255.0, max(0.0, r) + 0.5))
            out[y, x, 1] = np.uint8(min(255.0, max(0.0, g) + 0.5))
            out[y, x, 2] = np.uint8(min(255.0, max(0.0, b) + 0.5))
    return out
