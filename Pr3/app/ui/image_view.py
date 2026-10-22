"""Панель просмотра изображения: масштаб колесом мыши, перетаскивание,
синхронизация масштаба и положения между панелями «До» и «После»."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def to_qimage(data: np.ndarray) -> QImage:
    data = np.ascontiguousarray(data)
    h, w = data.shape[:2]
    if data.ndim == 2 or data.shape[2] == 1:
        img = QImage(data.data, w, h, w, QImage.Format_Grayscale8)
    else:
        img = QImage(data.data, w, h, 3 * w, QImage.Format_RGB888)
    return img.copy()  # отвязываемся от памяти numpy


class ImageView(QGraphicsView):
    viewChanged = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.FastTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setBackgroundBrush(Qt.darkGray)
        self._fit = True
        self.horizontalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())

    def set_image(self, data: np.ndarray | None):
        if data is None:
            self.item.setPixmap(QPixmap())
            return
        first = self.item.pixmap().isNull() or self.item.pixmap().size() != \
            QPixmap.fromImage(to_qimage(data)).size()
        self.item.setPixmap(QPixmap.fromImage(to_qimage(data)))
        self.scene().setSceneRect(self.item.boundingRect())
        if first or self._fit:
            self._fit = True
            self.fit()
            QTimer.singleShot(0, self.fit)

    def fit(self):
        self._fit = True
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.KeepAspectRatio)
        self.viewChanged.emit()

    def showEvent(self, e):
        super().showEvent(e)
        if self._fit:
            QTimer.singleShot(0, self.fit)

    def wheelEvent(self, e):
        self._fit = False
        k = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(k, k)
        self.viewChanged.emit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self.viewport())
        p.setPen(Qt.white)
        p.fillRect(4, 4, 8 * len(self.title) + 12, 20, Qt.black)
        p.drawText(10, 19, self.title)
        p.end()

    def sync_from(self, other: "ImageView"):
        if other._fit:
            self.fit()
            return
        self._fit = False
        self.setTransform(other.transform())
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())
