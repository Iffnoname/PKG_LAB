"""Фоновая обработка: вычисления идут в отдельном потоке QThread, окно не
зависает; прогресс передаётся сигналами, отмена — через флаг."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ..algorithms.operations import run_operation
from ..core.parallel import Cancelled


class Worker(QThread):
    progressChanged = Signal(int)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, name, buffer, params, parent=None):
        super().__init__(parent)
        self.name, self.buffer, self.params = name, buffer, params
        self._stop = False

    def stop(self):
        self._stop = True

    def _progress(self, f: float):
        if self._stop:
            raise Cancelled()
        self.progressChanged.emit(int(100 * min(1.0, max(0.0, f))))

    def run(self):
        try:
            res = run_operation(self.name, self.buffer, self.params, self._progress)
            if self._stop:
                self.cancelled.emit()
            else:
                self.done.emit(res)
        except Cancelled:
            self.cancelled.emit()
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class WarmUp(QThread):
    """Однократная JIT-компиляция ядер на маленьком изображении (результат
    кэшируется numba на диске, последующие запуски мгновенные)."""
    finishedOk = Signal(float)

    def run(self):
        import time

        import numpy as np

        from ..core.buffer import PixelBuffer
        t = time.perf_counter()
        rgb = PixelBuffer(np.random.default_rng(0).integers(0, 256, (24, 24, 3), dtype=np.uint8))
        ops = [("linear", {}), ("equalize", {}), ("equalize", {"color": "rgb"}),
               ("clahe", {"radius": 3}), ("clahe", {"variant": "tiles", "tiles": 2}),
               ("points", {}), ("lines", {}), ("gradient", {"output": "direction"}),
               ("gradient", {"operator": "roberts"}), ("log", {}),
               ("canny", {"output": "overlay"}), ("blur", {})]
        for name, p in ops:
            try:
                run_operation(name, rgb, p)
            except Exception:  # noqa: BLE001
                pass
        self.finishedOk.emit(time.perf_counter() - t)
