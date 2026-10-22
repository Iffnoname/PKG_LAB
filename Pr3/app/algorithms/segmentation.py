
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..core.buffer import PixelBuffer
from ..core.parallel import Stage, run_bands
from .color import hsv_pixel_to_rgb
from .convolution import filter2d, gaussian_blur, separable_filter

POINT_MASK = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float32)

LINE_MASKS = {
    "h": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),   # горизонтальная —
    "d45": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32),  # +45°  /
    "v": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),   # вертикальная |
    "dm45": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),  # -45°  \
}
LINE_ORDER = ["h", "d45", "v", "dm45"]
LINE_COLORS = np.array([[255, 60, 60], [60, 220, 60], [70, 120, 255], [255, 220, 40]],
                       dtype=np.uint8)
LINE_TITLES = {"h": "Горизонтальные (0°)", "d45": "+45° ( / )", "v": "Вертикальные (90°)",
               "dm45": "−45° ( \\ )", "all": "Все направления (максимум отклика)"}

LAPLACE4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

SMOOTH = {"sobel": np.array([1, 2, 1], np.float32), "prewitt": np.array([1, 1, 1], np.float32)}
DERIV = np.array([-1, 0, 1], np.float32)


@njit(cache=True, parallel=True)
def _abs_or_polar(r, polarity):
    h, w = r.shape
    out = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            v = r[y, x]
            if polarity == 1:
                v = v if v > 0 else 0.0
            elif polarity == 2:
                v = -v if v < 0 else 0.0
            else:
                v = abs(v)
            out[y, x] = v
    return out


@njit(cache=True, parallel=True)
def _threshold(a, t):
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            if a[y, x] >= t and a[y, x] > 0:
                out[y, x] = 255
    return out


@njit(cache=True)
def _max(a):
    m = 0.0
    h, w = a.shape
    for y in range(h):
        for x in range(w):
            if a[y, x] > m:
                m = a[y, x]
    return m


@njit(cache=True)
def count_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=np.uint8)
    stack = np.empty((h * w, 2), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or seen[y, x]:
                continue
            n += 1
            top = 0
            stack[0, 0] = y
            stack[0, 1] = x
            seen[y, x] = 1
            top = 1
            while top > 0:
                top -= 1
                cy = stack[top, 0]
                cx = stack[top, 1]
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny = cy + dy
                        nx = cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = 1
                            stack[top, 0] = ny
                            stack[top, 1] = nx
                            top += 1
    return n


def normalize_u8(a: np.ndarray, vmax: float | None = None) -> np.ndarray:
    from ..core.buffer import float_to_u8
    m = float(vmax if vmax is not None else _max(a))
    return float_to_u8(np.ascontiguousarray(a, dtype=np.float32), 255.0 / m if m > 0 else 0.0, 0.0)


# --------------------------------------------------------- обнаружение точек
def detect_points(plane_f: np.ndarray, threshold_pct: float, border: int,
                  polarity: int = 0, progress=None) -> dict:
    r = filter2d(plane_f, POINT_MASK, border, progress)
    resp = _abs_or_polar(r, polarity)
    m = _max(resp)
    t = m * threshold_pct / 100.0
    mask = _threshold(resp, t)
    return {"response": resp, "mask": mask, "T": t, "max": m,
            "count": int(count_components(mask)), "pixels": int(np.count_nonzero(mask))}


# ---------------------------------------------------------- обнаружение линий
@njit(cache=True, parallel=True)
def _argmax_lines(r0, r1, r2, r3):
    h, w = r0.shape
    best = np.empty((h, w), dtype=np.float32)
    idx = np.empty((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            b = r0[y, x]
            k = 0
            if r1[y, x] > b:
                b = r1[y, x]
                k = 1
            if r2[y, x] > b:
                b = r2[y, x]
                k = 2
            if r3[y, x] > b:
                b = r3[y, x]
                k = 3
            best[y, x] = b
            idx[y, x] = k
    return best, idx


def detect_lines(plane_f: np.ndarray, direction: str, threshold_pct: float, border: int,
                 polarity: int = 1, progress=None) -> dict:
    st = Stage(progress)
    if direction != "all":
        r = filter2d(plane_f, LINE_MASKS[direction], border, st)
        resp = _abs_or_polar(r, polarity)
        idx = np.full(resp.shape, LINE_ORDER.index(direction), np.uint8)
    else:
        rs = []
        for i, key in enumerate(LINE_ORDER):
            rs.append(_abs_or_polar(filter2d(plane_f, LINE_MASKS[key], border, st.sub(i / 4, (i + 1) / 4)),
                                    polarity))
        resp, idx = _argmax_lines(*rs)
    m = _max(resp)
    t = m * threshold_pct / 100.0
    mask = _threshold(resp, t)
    counts = {LINE_ORDER[k]: int(np.count_nonzero(mask[idx == k])) for k in range(4)}
    return {"response": resp, "mask": mask, "index": idx, "T": t, "max": m, "per_dir": counts,
            "pixels": int(np.count_nonzero(mask))}


@njit(cache=True, parallel=True)
def color_lines(mask, idx, colors):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            if mask[y, x]:
                k = idx[y, x]
                out[y, x, 0] = colors[k, 0]
                out[y, x, 1] = colors[k, 1]
                out[y, x, 2] = colors[k, 2]
    return out


# --------------------------------------------------------- градиент и угол
@njit(cache=True, parallel=True)
def _roberts(pad, gx, gy, y0, y1):
    _, w = gx.shape
    for y in prange(y0, y1):
        for x in range(w):
            z5 = pad[y + 1, x + 1]
            z6 = pad[y + 1, x + 2]
            z8 = pad[y + 2, x + 1]
            z9 = pad[y + 2, x + 2]
            d1 = z9 - z5
            d2 = z8 - z6
            gx[y, x] = 0.5 * (d1 - d2)
            gy[y, x] = 0.5 * (d1 + d2)


@njit(cache=True, parallel=True)
def magnitude_angle(gx, gy, l1):

    h, w = gx.shape
    mag = np.empty((h, w), dtype=np.float32)
    ang = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            a = gx[y, x]
            b = gy[y, x]
            mag[y, x] = (abs(a) + abs(b)) if l1 else math.sqrt(a * a + b * b)
            ang[y, x] = math.degrees(math.atan2(b, a))
    return mag, ang


def gradient(plane_f: np.ndarray, operator: str, border: int, progress=None):
    st = Stage(progress)
    if operator == "roberts":
        h, w = plane_f.shape
        pad = PixelBuffer.padded(plane_f, 1, 1, border)
        gx = np.empty((h, w), np.float32)
        gy = np.empty((h, w), np.float32)
        run_bands(_roberts, h, st, pad, gx, gy)
        return gx, gy
    s = SMOOTH[operator]
    gx = separable_filter(plane_f, DERIV, s, border, st.sub(0, 0.5))
    gy = separable_filter(plane_f, s, DERIV, border, st.sub(0.5, 1))
    return gx, gy


@njit(cache=True, parallel=True)
def direction_map(mag, ang, vmax, mask, use_mask):

    h, w = mag.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            if use_mask:
                if mask[y, x] == 0:
                    continue
                v = 255.0
            else:
                v = 255.0 * mag[y, x] / vmax if vmax > 0 else 0.0
                if v > 255.0:
                    v = 255.0
            hue = ang[y, x]
            if hue < 0.0:
                hue += 360.0
            r, g, b = hsv_pixel_to_rgb(hue, 1.0, v)
            out[y, x, 0] = np.uint8(min(255.0, r + 0.5))
            out[y, x, 1] = np.uint8(min(255.0, g + 0.5))
            out[y, x, 2] = np.uint8(min(255.0, b + 0.5))
    return out


def gradient_edges(plane_f: np.ndarray, operator: str, threshold_pct: float, border: int,
                   sigma: float = 0.0, l1: bool = False, progress=None) -> dict:
    st = Stage(progress)
    src = gaussian_blur(plane_f, sigma, border, st.sub(0, 0.4)) if sigma > 0 else plane_f
    gx, gy = gradient(src, operator, border, st.sub(0.4, 0.9))
    mag, ang = magnitude_angle(gx, gy, l1)
    m = _max(mag)
    t = m * threshold_pct / 100.0
    mask = _threshold(mag, t)
    st(1.0)
    return {"gx": gx, "gy": gy, "magnitude": mag, "angle": ang, "mask": mask, "T": t, "max": m,
            "pixels": int(np.count_nonzero(mask))}


@njit(cache=True, parallel=True)
def _zero_cross(pad, out, t, y0, y1):
    _, w = out.shape
    for y in prange(y0, y1):
        for x in range(w):
            cy = y + 1
            cx = x + 1
            e = 0
            for k in range(4):
                if k == 0:
                    a = pad[cy - 1, cx]
                    b = pad[cy + 1, cx]
                elif k == 1:
                    a = pad[cy, cx - 1]
                    b = pad[cy, cx + 1]
                elif k == 2:
                    a = pad[cy - 1, cx - 1]
                    b = pad[cy + 1, cx + 1]
                else:
                    a = pad[cy - 1, cx + 1]
                    b = pad[cy + 1, cx - 1]
                if a * b < 0 and abs(a - b) >= t:
                    e = 1
                    break
            out[y, x] = 255 if e else 0


def log_edges(plane_f: np.ndarray, sigma: float, threshold_pct: float, border: int,
              progress=None) -> dict:
    st = Stage(progress)
    g = gaussian_blur(plane_f, sigma, border, st.sub(0, 0.5))
    lap = filter2d(g, LAPLACE4, border, st.sub(0.5, 0.8))
    m = _max(_abs_or_polar(lap, 0))
    t = m * threshold_pct / 100.0
    pad = PixelBuffer.padded(lap, 1, 1, border)
    mask = np.empty(plane_f.shape, np.uint8)
    run_bands(_zero_cross, plane_f.shape[0], st.sub(0.8, 1.0), pad, mask, t)
    return {"log": lap, "mask": mask, "T": t, "max": m, "pixels": int(np.count_nonzero(mask))}


# ---------------------------------------------------------------- Кэнни
@njit(cache=True, inline="always")
def _bilinear(a, x, y):
    h, w = a.shape
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    if x > w - 1.0:
        x = w - 1.0
    if y > h - 1.0:
        y = h - 1.0
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


@njit(cache=True, parallel=True)
def _nms(mag, gx, gy, nms, offs, y0, y1):

    h, w = mag.shape
    for y in prange(y0, y1):
        for x in range(w):
            m = mag[y, x]
            nms[y, x] = 0.0
            offs[y, x] = 0.0
            if m <= 0.0:
                continue
            ux = gx[y, x] / m
            uy = gy[y, x] / m
            mp = _bilinear(mag, x + ux, y + uy)
            mm = _bilinear(mag, x - ux, y - uy)
            if m >= mp and m > mm:
                nms[y, x] = m
                den = mm - 2.0 * m + mp
                d = 0.0
                if den < 0.0:
                    d = 0.5 * (mm - mp) / den
                    if d > 0.5:
                        d = 0.5
                    elif d < -0.5:
                        d = -0.5
                offs[y, x] = d


@njit(cache=True)
def _hysteresis(nms, high, low):

    h, w = nms.shape
    out = np.zeros((h, w), dtype=np.uint8)
    stack = np.empty((h * w, 2), dtype=np.int32)
    top = 0
    for y in range(h):
        for x in range(w):
            if nms[y, x] >= high and nms[y, x] > 0:
                out[y, x] = 255
                stack[top, 0] = y
                stack[top, 1] = x
                top += 1
    while top > 0:
        top -= 1
        cy = stack[top, 0]
        cx = stack[top, 1]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny = cy + dy
                nx = cx + dx
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx] == 0 and nms[ny, nx] >= low \
                        and nms[ny, nx] > 0:
                    out[ny, nx] = 255
                    stack[top, 0] = ny
                    stack[top, 1] = nx
                    top += 1
    return out


def canny(plane_f: np.ndarray, sigma: float, high_pct: float, low_ratio_pct: float, border: int,
          progress=None) -> dict:

    st = Stage(progress)
    src = gaussian_blur(plane_f, sigma, border, st.sub(0, 0.35)) if sigma > 0 else plane_f
    gx, gy = gradient(src, "sobel", border, st.sub(0.35, 0.65))
    mag, ang = magnitude_angle(gx, gy, False)
    nms = np.empty_like(mag)
    offs = np.empty_like(mag)
    run_bands(_nms, mag.shape[0], st.sub(0.65, 0.9), mag, gx, gy, nms, offs)
    m = _max(nms)
    high = m * high_pct / 100.0
    low = high * low_ratio_pct / 100.0
    mask = _hysteresis(nms, high, low)
    ys, xs = np.nonzero(mask)
    mg = mag[ys, xs]
    d = offs[ys, xs]
    ux = gx[ys, xs] / np.maximum(mg, 1e-12)
    uy = gy[ys, xs] / np.maximum(mg, 1e-12)
    sub_x = xs + d * ux
    sub_y = ys + d * uy
    st(1.0)
    return {"gx": gx, "gy": gy, "magnitude": mag, "angle": ang, "nms": nms, "mask": mask,
            "high": high, "low": low, "max": m, "pixels": int(len(xs)),
            "subpixel": np.stack([sub_x, sub_y], axis=1).astype(np.float32)}


@njit(cache=True, parallel=True)
def overlay(gray, mask, r, g, b):
    h, w = gray.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            if mask[y, x]:
                out[y, x, 0] = r
                out[y, x, 1] = g
                out[y, x, 2] = b
            else:
                v = np.uint8(gray[y, x] * 0.6)
                out[y, x, 0] = v
                out[y, x, 1] = v
                out[y, x, 2] = v
    return out
