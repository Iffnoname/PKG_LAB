
from __future__ import annotations

import numpy as np
from numba import njit, prange


BORDER_CLAMP = 0
BORDER_REFLECT = 1

BORDER_NAMES = {
    BORDER_REFLECT: "Reflect (зеркальное отражение)",
    BORDER_CLAMP: "Clamp (дублирование граничных пикселей)",
}


@njit(cache=True, inline="always")
def border_index(i, n, mode):

    if 0 <= i < n:
        return i
    if mode == BORDER_CLAMP or n == 1:
        return 0 if i < 0 else n - 1

    period = 2 * n - 2
    if i < 0:
        i = -i
    i = i % period
    if i >= n:
        i = period - i
    return i


@njit(cache=True, parallel=True)
def pad_plane(plane, ry, rx, mode):

    h, w = plane.shape
    out = np.empty((h + 2 * ry, w + 2 * rx), dtype=np.float32)
    for y in prange(h + 2 * ry):
        sy = border_index(y - ry, h, mode)
        for x in range(w + 2 * rx):
            sx = border_index(x - rx, w, mode)
            out[y, x] = plane[sy, sx]
    return out


@njit(cache=True, parallel=True)
def pad_plane_u8(plane, r, mode):
    h, w = plane.shape
    out = np.empty((h + 2 * r, w + 2 * r), dtype=np.uint8)
    for y in prange(h + 2 * r):
        sy = border_index(y - r, h, mode)
        for x in range(w + 2 * r):
            sx = border_index(x - r, w, mode)
            out[y, x] = plane[sy, sx]
    return out


@njit(cache=True, parallel=True)
def _extract_channel(data, c):
    h, w, _ = data.shape
    out = np.empty((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            out[y, x] = data[y, x, c]
    return out


@njit(cache=True, parallel=True)
def _luminance(data):
    h, w, ch = data.shape
    out = np.empty((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            if ch == 1:
                out[y, x] = data[y, x, 0]
            else:
                v = 0.299 * data[y, x, 0] + 0.587 * data[y, x, 1] + 0.114 * data[y, x, 2]
                out[y, x] = np.uint8(min(255.0, v + 0.5))
    return out


@njit(cache=True, parallel=True)
def _to_float(plane):
    h, w = plane.shape
    out = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            out[y, x] = plane[y, x]
    return out


@njit(cache=True, parallel=True)
def float_to_u8(plane, scale, offset):
    h, w = plane.shape
    out = np.empty((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            v = plane[y, x] * scale + offset
            if v < 0.0:
                v = 0.0
            elif v > 255.0:
                v = 255.0
            out[y, x] = np.uint8(int(v + 0.5))
    return out


@njit(cache=True, parallel=True)
def _stack_gray(plane):
    h, w = plane.shape
    out = np.empty((h, w, 1), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            out[y, x, 0] = plane[y, x]
    return out


class PixelBuffer:

    def __init__(self, data: np.ndarray):
        if data.ndim == 2:
            data = data[:, :, None]
        if data.dtype != np.uint8:
            raise TypeError("PixelBuffer ожидает массив uint8")
        self.data = np.ascontiguousarray(data)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        return self.data.shape[2]

    @property
    def is_gray(self) -> bool:
        return self.channels == 1

    @property
    def stride(self) -> int:
        return self.width * self.channels

    def copy(self) -> "PixelBuffer":
        return PixelBuffer(self.data.copy())

    def channel(self, c: int) -> np.ndarray:
        return _extract_channel(self.data, c)

    def luminance(self) -> np.ndarray:
        return _luminance(self.data)

    def luminance_f32(self) -> np.ndarray:
        return _to_float(self.luminance())

    @staticmethod
    def from_plane(plane_u8: np.ndarray) -> "PixelBuffer":
        return PixelBuffer(_stack_gray(np.ascontiguousarray(plane_u8)))

    @staticmethod
    def padded(plane_f32: np.ndarray, radius_y: int, radius_x: int, mode: int) -> np.ndarray:

        return pad_plane(np.ascontiguousarray(plane_f32, dtype=np.float32),
                         radius_y, radius_x, mode)

    @staticmethod
    def padded_u8(plane_u8: np.ndarray, radius: int, mode: int) -> np.ndarray:
        return pad_plane_u8(np.ascontiguousarray(plane_u8), radius, mode)

    @staticmethod
    def load(path: str) -> "PixelBuffer":
        from PIL import Image
        with Image.open(path) as im:
            if im.mode in ("L", "I;16", "I", "F", "1"):
                arr = np.asarray(im.convert("L"), dtype=np.uint8)
            else:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return PixelBuffer(np.array(arr, dtype=np.uint8))

    def save(self, path: str) -> None:
        from PIL import Image
        if self.is_gray:
            Image.fromarray(self.data[:, :, 0], "L").save(path)
        else:
            Image.fromarray(self.data, "RGB").save(path)
