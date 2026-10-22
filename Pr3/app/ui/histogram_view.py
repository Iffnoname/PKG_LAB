"""Отрисовка гистограмм «до» (серые столбцы) и «после» (зелёная линия)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class HistogramView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before = None
        self.after = None
        self.cumulative = False
        self.log_scale = False
        self.caption = ""
        self.setMinimumHeight(170)

    def set_data(self, before, after, caption=""):
        self.before = None if before is None else np.asarray(before, dtype=np.float64)
        self.after = None if after is None else np.asarray(after, dtype=np.float64)
        self.caption = caption
        self.update()

    def _prep(self, h):
        if h is None:
            return None
        h = h / max(h.sum(), 1)
        if self.cumulative:
            return np.cumsum(h)
        if self.log_scale:
            return np.log1p(h * 1e4)
        return h

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(250, 250, 250))
        left, right, top, bottom = 34, 10, 22, 22
        plot = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)
        p.setPen(QPen(QColor(180, 180, 180)))
        p.drawRect(plot)
        b, a = self._prep(self.before), self._prep(self.after)
        vals = [v for v in (b, a) if v is not None]
        if not vals:
            p.end()
            return
        vmax = max(float(v.max()) for v in vals) or 1.0
        bw = plot.width() / 256.0
        if b is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(150, 150, 150))
            for i in range(256):
                hh = plot.height() * b[i] / vmax
                p.drawRect(QRectF(plot.left() + i * bw, plot.bottom() - hh, max(bw, 1.0), hh))
        if a is not None:
            path = QPainterPath()
            for i in range(256):
                pt = QPointF(plot.left() + (i + 0.5) * bw, plot.bottom() - plot.height() * a[i] / vmax)
                path.moveTo(pt) if i == 0 else path.lineTo(pt)
            p.setPen(QPen(QColor(20, 160, 60), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        p.setPen(Qt.black)
        for v in (0, 64, 128, 192, 255):
            x = plot.left() + (v + 0.5) * bw
            p.drawText(QRectF(x - 15, plot.bottom() + 2, 30, 16), Qt.AlignCenter, str(v))
        p.setPen(QColor(110, 110, 110))
        p.fillRect(QRectF(plot.left() + 6, 5, 12, 10), QColor(150, 150, 150))
        p.drawText(QPointF(plot.left() + 22, 15), "до")
        p.setPen(QPen(QColor(20, 160, 60), 2))
        p.drawLine(QPointF(plot.left() + 50, 10), QPointF(plot.left() + 64, 10))
        p.setPen(QColor(110, 110, 110))
        p.drawText(QPointF(plot.left() + 68, 15), "после")
        p.drawText(QPointF(plot.left() + 120, 15), self.caption)
        p.end()
