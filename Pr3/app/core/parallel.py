
from __future__ import annotations

from typing import Callable, Optional

import numba

ProgressFn = Optional[Callable[[float], None]]


class Cancelled(Exception):
    pass

def set_threads(n: int) -> None:
    numba.set_num_threads(max(1, min(int(n), numba.config.NUMBA_NUM_THREADS)))


def get_threads() -> int:
    return numba.get_num_threads()


def max_threads() -> int:
    return numba.config.NUMBA_NUM_THREADS


class Stage:

    def __init__(self, progress: ProgressFn, lo: float = 0.0, hi: float = 1.0):
        self.progress, self.lo, self.hi = progress, lo, hi

    def __call__(self, f: float) -> None:
        if self.progress is not None:
            self.progress(self.lo + (self.hi - self.lo) * f)

    def sub(self, a: float, b: float) -> "Stage":
        span = self.hi - self.lo
        return Stage(self.progress, self.lo + span * a, self.lo + span * b)


def run_bands(kernel, height: int, progress: ProgressFn, *args, bands: int = 16) -> None:
    bands = max(1, min(bands, height))
    step = (height + bands - 1) // bands
    y0 = 0
    while y0 < height:
        y1 = min(height, y0 + step)
        kernel(*args, y0, y1)
        y0 = y1
        if progress is not None:
            progress(y0 / height)
