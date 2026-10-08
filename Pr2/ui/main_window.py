
from __future__ import annotations

import csv
import threading
import time
from typing import List

from PySide6.QtCore import QThread, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSplitter, QTableView,
                               QVBoxLayout, QWidget)

from engine.scanner import IMAGE_EXTENSIONS, MAX_FILES, ScanSummary, run_scan
from ui.preview import PreviewPanel
from ui.table_model import COLUMNS, ImageProxyModel, ImageTableModel


class ScanWorker(QThread):

    found = Signal(int)
    batch = Signal(object)
    progress = Signal(int, int)
    finished_scan = Signal(object)

    def __init__(self, sources: List[str], parent=None) -> None:
        super().__init__(parent)
        self.sources = sources
        self.cancel_event = threading.Event()

    def run(self) -> None:
        summary = run_scan(self.sources, cancel=self.cancel_event, on_found=self.found.emit,
                           on_batch=self.batch.emit, on_progress=self.progress.emit)
        self.finished_scan.emit(summary)

    def stop(self) -> None:
        self.cancel_event.set()


def _human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Лаб2 — чтение метаданных графических файлов")
        self.resize(1280, 760)
        self.setAcceptDrops(True)
        self._worker: ScanWorker | None = None
        self._started = 0.0

        self.model = ImageTableModel(self)
        self.proxy = ImageProxyModel(self)
        self.proxy.setSourceModel(self.model)

        # --- кнопки ---
        self.btn_folder = QPushButton("Выбрать папку…")
        self.btn_files = QPushButton("Выбрать файлы…")
        self.btn_stop = QPushButton("Остановить")
        self.btn_export = QPushButton("Экспорт в CSV…")
        self.chk_problems = QCheckBox("Только проблемные файлы")
        self.btn_stop.setEnabled(False)
        top = QHBoxLayout()
        for w in (self.btn_folder, self.btn_files, self.btn_stop, self.btn_export):
            top.addWidget(w)
        top.addSpacing(16)
        top.addWidget(self.chk_problems)
        top.addStretch(1)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        for col, width in enumerate((250, 110, 105, 110, 90, 260, 130)):
            self.table.setColumnWidth(col, width)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.setSortingEnabled(True)

        self.preview = PreviewPanel(QThreadPool.globalInstance())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 440])

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(320)
        limit_text = f"{MAX_FILES:,}".replace(",", " ")
        self.status = QLabel("Выберите папку или файлы (можно перетащить их в окно). "
                             f"Форматы: JPEG, GIF, TIFF, BMP, PNG, PCX. Лимит — {limit_text} файлов.")
        bottom = QHBoxLayout()
        bottom.addWidget(self.progress)
        bottom.addWidget(self.status, 1)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(splitter, 1)
        lay.addLayout(bottom)
        self.setCentralWidget(central)

        self.btn_folder.clicked.connect(self.choose_folder)
        self.btn_files.clicked.connect(self.choose_files)
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_export.clicked.connect(self.export_csv)
        self.chk_problems.toggled.connect(self._toggle_filter)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if folder:
            self.start_scan([folder])

    def choose_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите файлы", "",
                                                f"Изображения ({exts});;Все файлы (*)")
        if files:
            self.start_scan(files)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.start_scan(paths)

    def start_scan(self, sources: List[str]) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "Идёт обработка", "Дождитесь окончания или нажмите «Остановить».")
            return
        self.model.clear()
        self.preview.show_file(None)
        self.table.setSortingEnabled(False)
        self.proxy.sort(-1)
        self.progress.setRange(0, 0)               # «бегущий» индикатор, пока ищем файлы
        self.status.setText("Поиск файлов…")
        self.btn_folder.setEnabled(False)
        self.btn_files.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._started = time.perf_counter()

        self._worker = ScanWorker(list(sources), self)
        self._worker.found.connect(self._on_found)
        self._worker.batch.connect(self._on_batch)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_scan.connect(self._on_finished)
        self._worker.start()

    def stop_scan(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self.status.setText("Остановка…")

    def _on_found(self, n: int) -> None:
        self.status.setText(f"Поиск файлов… найдено: {n}")

    def _on_batch(self, batch) -> None:
        self.model.append(batch)

    def _on_progress(self, done: int, total: int) -> None:
        if self.progress.maximum() != total:
            self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.status.setText(f"Обработано {done} из {total}")

    def _on_finished(self, s: ScanSummary) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if s.processed else 0)
        self.btn_folder.setEnabled(True)
        self.btn_files.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.proxy.invalidateFilter()
        self.table.setSortingEnabled(True)
        text = (f"{'Остановлено' if s.cancelled else 'Готово'}: {s.processed} файл(ов) за {s.elapsed:.2f} с. "
                f"OK: {s.ok}, предупреждений: {s.warnings}, повреждено: {s.corrupt}, прочее: {s.other}. "
                f"Прочитано {_human(s.bytes_read)} из {_human(s.bytes_total)}.")
        if s.limit_reached:
            text += f" Достигнут предел {MAX_FILES} файлов — остальные пропущены."
        if s.skipped_dirs:
            text += f" Недоступных папок: {s.skipped_dirs}."
        self.status.setText(text)

    def _on_row_changed(self, current, _previous) -> None:
        if not current.isValid():
            return
        src = self.proxy.mapToSource(current)
        self.preview.show_file(self.model.metadata(src.row()))

    def _toggle_filter(self, checked: bool) -> None:
        self.proxy.only_problems = checked
        self.proxy.invalidateFilter()

    def export_csv(self) -> None:
        if self.model.rowCount() == 0:
            QMessageBox.information(self, "Экспорт", "Таблица пуста.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить таблицу", "metadata.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM — чтобы Excel узнал UTF-8
                w = csv.writer(f, delimiter=";")
                w.writerow(COLUMNS + ["Путь", "Сообщение"])
                for row in range(self.proxy.rowCount()):
                    m = self.model.metadata(self.proxy.mapToSource(self.proxy.index(row, 0)).row())
                    w.writerow([m.name, m.format, m.size_text, m.dpi_text, m.bpp_text,
                                m.compression, m.status, m.path, m.message])
        except OSError as e:
            QMessageBox.warning(self, "Экспорт", f"Не удалось сохранить файл: {e}")

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(5000)
        super().closeEvent(event)
