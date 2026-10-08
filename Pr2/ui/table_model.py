
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor

from models import (ImageMetadata, STATUS_CORRUPT, STATUS_ERROR, STATUS_NOT_IMAGE, STATUS_OK,
                    STATUS_WARNING)

COLUMNS = ["Файл", "Формат", "Размер, px", "DPI (X × Y)", "Бит/пиксель", "Сжатие", "Статус"]
SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 1

_STATUS_COLOR = {
    STATUS_OK: None,
    STATUS_WARNING: QColor(255, 170, 0, 55),
    STATUS_CORRUPT: QColor(230, 40, 40, 60),
    STATUS_NOT_IMAGE: QColor(150, 150, 150, 60),
    STATUS_ERROR: QColor(230, 40, 40, 60),
}


class ImageTableModel(QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: List[ImageMetadata] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        m = self._rows[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return (m.name, m.format or "—", m.size_text, m.dpi_text, m.bpp_text,
                    m.compression or "—", m.status)[col]
        if role == SORT_ROLE:
            return (m.name.casefold(), m.format, (m.width or 0) * (m.height or 0),
                    -1.0 if m.dpi_x is None else m.dpi_x,
                    -1 if m.bits_per_pixel is None else m.bits_per_pixel,
                    m.compression, m.status)[col]
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == 0:
                return m.path
            if col == 6 and m.message:
                return m.message
            return None
        if role == Qt.ItemDataRole.BackgroundRole:
            return _STATUS_COLOR.get(m.status)
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (2, 3, 4):
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    def append(self, batch: List[ImageMetadata]) -> None:
        if not batch:
            return
        first = len(self._rows)
        self.beginInsertRows(QModelIndex(), first, first + len(batch) - 1)
        self._rows.extend(batch)
        self.endInsertRows()

    def clear(self) -> None:
        self.beginResetModel()
        self._rows = []
        self.endResetModel()

    def metadata(self, row: int) -> Optional[ImageMetadata]:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def all_rows(self) -> List[ImageMetadata]:
        return self._rows


class ImageProxyModel(QSortFilterProxyModel):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        self.setDynamicSortFilter(False)   # не пересортировывать на каждую порцию данных
        self.only_problems = False

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if not self.only_problems:
            return True
        m = self.sourceModel().metadata(source_row)
        return m is not None and m.status != STATUS_OK
