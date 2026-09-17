"""
Математические преобразования цветовых моделей RGB, HSV и CIE L*a*b*.

Вариант №4: RGB ↔ HSV ↔ LAB (через RGB как опорную модель).
"""

from __future__ import annotations

import math
from typing import Tuple

# Стандартный белый D65 (CIE 1931), Y = 100
D65_XN = 95.047
D65_YN = 100.0
D65_ZN = 108.883

# Рабочий диапазон a*, b* в интерфейсе (стандартный охват sRGB в Lab)
LAB_A_MIN = -128.0
LAB_A_MAX = 127.0
LAB_B_MIN = -128.0
LAB_B_MAX = 127.0

# Порог для функции f(t) в Lab (CIE)
_EPSILON = 216.0 / 24389.0  # ~0.008856
_KAPPA = 24389.0 / 27.0  # ~903.3


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _f_lab(t: float) -> float:
    if t > _EPSILON:
        return t ** (1.0 / 3.0)
    return (_KAPPA * t + 16.0) / 116.0


def _f_inv_lab(t: float) -> float:
    t3 = t ** 3
    if t3 > _EPSILON:
        return t3
    return (116.0 * t - 16.0) / _KAPPA


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    RGB (0–255) → HSV.
    H: 0–360°, S и V: 0–100%.
    """
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    c_max = max(rf, gf, bf)
    c_min = min(rf, gf, bf)
    delta = c_max - c_min

    if delta == 0:
        h = 0.0
    elif c_max == rf:
        h = 60.0 * (((gf - bf) / delta) % 6.0)
    elif c_max == gf:
        h = 60.0 * (((bf - rf) / delta) + 2.0)
    else:
        h = 60.0 * (((rf - gf) / delta) + 4.0)

    if h < 0:
        h += 360.0

    s = 0.0 if c_max == 0 else (delta / c_max) * 100.0
    v = c_max * 100.0
    return h, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int, bool]:
    """
    HSV → RGB (0–255).
    Возвращает (r, g, b, clipped), где clipped=True, если вход был ограничен.
    """
    clipped = False
    h = h % 360.0
    if h < 0:
        h += 360.0
    orig_s, orig_v = s, v
    s = _clip(s, 0.0, 100.0)
    v = _clip(v, 0.0, 100.0)
    if s != orig_s or v != orig_v:
        clipped = True

    s_f = s / 100.0
    v_f = v / 100.0
    c = v_f * s_f
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v_f - c

    if 0 <= h < 60:
        rf, gf, bf = c, x, 0.0
    elif 60 <= h < 120:
        rf, gf, bf = x, c, 0.0
    elif 120 <= h < 180:
        rf, gf, bf = 0.0, c, x
    elif 180 <= h < 240:
        rf, gf, bf = 0.0, x, c
    elif 240 <= h < 300:
        rf, gf, bf = x, 0.0, c
    else:
        rf, gf, bf = c, 0.0, x

    r = int(round((rf + m) * 255.0))
    g = int(round((gf + m) * 255.0))
    b = int(round((bf + m) * 255.0))
    return r, g, b, clipped


def _linearize_srgb(channel: float) -> float:
    """sRGB (0–1) → линейный RGB."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _gamma_srgb(linear: float) -> float:
    """Линейный RGB → sRGB (0–1)."""
    if linear <= 0.0031308:
        return 12.92 * linear
    return 1.055 * (linear ** (1.0 / 2.4)) - 0.055


def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    sRGB (D65) → XYZ. X, Y, Z в масштабе Y=100 для белого.
    """
    rs = _linearize_srgb(r / 255.0)
    gs = _linearize_srgb(g / 255.0)
    bs = _linearize_srgb(b / 255.0)

    x = (rs * 0.4124564 + gs * 0.3575761 + bs * 0.1804375) * 100.0
    y = (rs * 0.2126729 + gs * 0.7151522 + bs * 0.0721750) * 100.0
    z = (rs * 0.0193339 + gs * 0.1191920 + bs * 0.9503041) * 100.0
    return x, y, z


def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[int, int, int, bool]:
    """
    XYZ (Y=100) → sRGB. Возвращает clipped=True, если каналы обрезаны до 0–255.
    """
    xn, yn, zn = x / 100.0, y / 100.0, z / 100.0

    rl = xn * 3.2404542 + yn * -1.5371385 + zn * -0.4985314
    gl = xn * -0.9692660 + yn * 1.8760108 + zn * 0.0415560
    bl = xn * 0.0556434 + yn * -0.2040259 + zn * 1.0572252

    rs = _gamma_srgb(rl)
    gs = _gamma_srgb(gl)
    bs = _gamma_srgb(bl)

    r_f = rs * 255.0
    g_f = gs * 255.0
    b_f = bs * 255.0

    clipped = r_f < 0 or r_f > 255 or g_f < 0 or g_f > 255 or b_f < 0 or b_f > 255

    r = int(round(_clip(r_f, 0.0, 255.0)))
    g = int(round(_clip(g_f, 0.0, 255.0)))
    b = int(round(_clip(b_f, 0.0, 255.0)))
    return r, g, b, clipped


def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """XYZ (D65, Y=100) → CIE L*a*b*."""
    xr = x / D65_XN
    yr = y / D65_YN
    zr = z / D65_ZN

    fx = _f_lab(xr)
    fy = _f_lab(yr)
    fz = _f_lab(zr)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def lab_to_xyz(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """CIE L*a*b* → XYZ (D65, Y=100)."""
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xr = _f_inv_lab(fx)
    yr = _f_inv_lab(fy)
    zr = _f_inv_lab(fz)

    x = xr * D65_XN
    y = yr * D65_YN
    z = zr * D65_ZN
    return x, y, z


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """RGB → LAB через XYZ."""
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int, bool]:
    """LAB → RGB через XYZ."""
    x, y, z = lab_to_xyz(L, a, b)
    return xyz_to_rgb(x, y, z)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"
