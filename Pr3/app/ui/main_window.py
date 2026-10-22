"""Главное окно (Presentation Layer). Только отображение и сбор параметров;
вся обработка выполняется функциями слоя algorithms в фоновом потоке."""
from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox,
                               QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from ..algorithms import segmentation as seg
from ..algorithms.color import rgb_to_hsv
from ..algorithms.histogram import histogram, stats
from ..core.buffer import BORDER_CLAMP, BORDER_REFLECT, PixelBuffer
from ..core import parallel
from .histogram_view import HistogramView
from .image_view import ImageView, to_qimage
from .worker import WarmUp, Worker

# ----------------------------------------------------------- описание параметров
OUT_EDGES = [("Бинарная карта краёв", "mask"), ("Модуль градиента |∇f|", "response"),
             ("Карта направлений (все пиксели)", "direction"),
             ("Карта направлений (только края)", "direction_edges"),
             ("Наложение краёв на исходное", "overlay")]

CONTRAST_METHODS = [
    ("linear", "Линейное контрастирование", [
        ("mode", "Границы a, b", "combo", "auto", [("Авто: min … max изображения", "auto"),
                                                  ("Отсечение % с каждого края", "percent"),
                                                  ("Ручной ввод a и b", "manual")]),
        ("percent", "Отсечение, % с края", "float", 1.0, (0.0, 20.0, 0.1)),
        ("a", "a (нижняя граница)", "int", 0, (0, 255)),
        ("b", "b (верхняя граница)", "int", 255, (0, 255)),
    ]),
    ("equalize", "Эквализация гистограммы", []),
    ("clahe", "CLAHE (адаптивная эквализация)", [
        ("variant", "Вариант", "combo", "sliding", [("Скользящее окно", "sliding"),
                                                    ("Тайлы + билинейная интерполяция", "tiles")]),
        ("radius", "Радиус окна r (окно 2r+1)", "int", 32, (2, 128)),
        ("tiles", "Тайлов по стороне", "int", 8, (2, 32)),
        ("clip", "Ограничение контраста", "float", 2.0, (1.0, 10.0, 0.1)),
    ]),
    ("gamma", "Гамма-коррекция (подготовка)", [("g", "γ", "float", 1.0, (0.1, 5.0, 0.05))]),
    ("negative", "Негатив", []),
]

SEG_METHODS = [
    ("points", "Обнаружение точек (маска-лапласиан)", [
        ("threshold", "Порог T, % от max|R|", "float", 50.0, (1.0, 100.0, 0.5)),
        ("polarity", "Тип точек", "combo", 0, [("Любые (|R|)", 0), ("Светлые на тёмном (R > 0)", 1),
                                               ("Тёмные на светлом (R < 0)", 2)]),
        ("output", "Вывод", "combo", "mask", [("Бинарная карта точек", "mask"),
                                               ("Отклик |R|", "response"),
                                               ("Наложение на исходное", "overlay")]),
    ]),
    ("lines", "Обнаружение линий (4 маски)", [
        ("direction", "Направление", "combo", "all",
         [(seg.LINE_TITLES[k], k) for k in ["all"] + seg.LINE_ORDER]),
        ("threshold", "Порог T, % от max R", "float", 50.0, (1.0, 100.0, 0.5)),
        ("polarity", "Тип линий", "combo", 1, [("Светлые на тёмном (R > 0)", 1),
                                               ("Тёмные на светлом (R < 0)", 2), ("Любые (|R|)", 0)]),
        ("output", "Вывод", "combo", "mask", [("Карта линий (цвет = направление)", "mask"),
                                               ("Отклик R", "response"),
                                               ("Наложение на исходное", "overlay")]),
    ]),
    ("gradient", "Перепады: Собель / Превитт / Робертс", [
        ("operator", "Оператор", "combo", "sobel", [("Собель", "sobel"), ("Превитт", "prewitt"),
                                                    ("Робертс", "roberts")]),
        ("sigma", "Сглаживание Гаусса σ (0 — нет)", "float", 0.0, (0.0, 5.0, 0.1)),
        ("threshold", "Порог T, % от max|∇f|", "float", 20.0, (1.0, 100.0, 0.5)),
        ("l1", "|gx| + |gy| вместо √(gx² + gy²)", "check", False, None),
        ("output", "Вывод", "combo", "mask", OUT_EDGES),
    ]),
    ("log", "Перепады: LoG (Марр — Хилдрет)", [
        ("sigma", "σ гауссиана", "float", 2.0, (0.5, 8.0, 0.1)),
        ("threshold", "Порог перехода, % от max", "float", 5.0, (0.0, 50.0, 0.5)),
        ("output", "Вывод", "combo", "mask", [("Бинарная карта краёв", "mask"),
                                               ("|LoG|", "response"),
                                               ("Наложение на исходное", "overlay")]),
    ]),
    ("canny", "Перепады: детектор Кэнни", [
        ("sigma", "Сглаживание Гаусса σ", "float", 1.4, (0.0, 5.0, 0.1)),
        ("high", "Верхний порог, % от max", "float", 20.0, (1.0, 100.0, 0.5)),
        ("low", "Нижний порог, % от верхнего", "float", 40.0, (5.0, 100.0, 1.0)),
        ("output", "Вывод", "combo", "mask", OUT_EDGES),
    ]),
]

PREP_METHODS = [
    ("noise", "Добавить гауссов шум", [
        ("sigma", "σ шума", "float", 20.0, (1.0, 80.0, 1.0)),
        ("seed", "Зерно генератора", "int", 1, (0, 9999)),
    ]),
    ("blur", "Сглаживание (разделяемая свёртка)", [
        ("kind", "Фильтр", "combo", "gauss", [("Гаусс", "gauss"), ("Бокс-фильтр", "box")]),
        ("sigma", "σ (Гаусс)", "float", 1.5, (0.3, 10.0, 0.1)),
        ("size", "Размер (бокс)", "int", 5, (3, 51)),
    ]),
]


def app_root() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SliderEdit(QWidget):
    """Ползунок + поле ввода для числового параметра."""

    def __init__(self, kind, default, rng, on_change):
        super().__init__()
        lo, hi = rng[0], rng[1]
        self.k = 1.0 if kind == "int" else 1.0 / rng[2]
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(lo * self.k)), int(round(hi * self.k)))
        if kind == "int":
            self.spin = QSpinBox()
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(2 if rng[2] < 0.1 else 1)
            self.spin.setSingleStep(rng[2])
        self.spin.setRange(lo, hi)
        self.spin.setValue(default)
        self.slider.setValue(int(round(default * self.k)))
        self.spin.setFixedWidth(70)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / self.k))
        self.spin.valueChanged.connect(lambda v: self.slider.setValue(int(round(v * self.k))))
        self.spin.valueChanged.connect(lambda _: on_change())

    def value(self):
        return self.spin.value()


class MethodPanel(QWidget):
    """Выбор метода + форма его параметров (строится по описанию)."""

    def __init__(self, methods, on_change):
        super().__init__()
        self.methods = methods
        self.on_change = on_change
        lay = QVBoxLayout(self)
        self.combo = QComboBox()
        for key, title, _ in methods:
            self.combo.addItem(title, key)
        lay.addWidget(self.combo)
        self.stack = QStackedWidget()
        self.widgets = []
        for key, _, params in methods:
            page = QWidget()
            form = QFormLayout(page)
            form.setContentsMargins(0, 4, 0, 0)
            ws = {}
            for pkey, label, kind, default, opt in params:
                if kind == "combo":
                    w = QComboBox()
                    for t, v in opt:
                        w.addItem(t, v)
                    w.setCurrentIndex([v for _, v in opt].index(default))
                    w.currentIndexChanged.connect(lambda _: self.on_change())
                elif kind == "check":
                    w = QCheckBox()
                    w.setChecked(default)
                    w.toggled.connect(lambda _: self.on_change())
                else:
                    w = SliderEdit(kind, default, opt, self.on_change)
                lbl = QLabel(label)
                lbl.setWordWrap(True)
                form.addRow(lbl, w)
                ws[pkey] = (kind, w)
            if not params:
                form.addRow(QLabel("Параметров нет"))
            self.stack.addWidget(page)
            self.widgets.append(ws)
        lay.addWidget(self.stack)
        lay.addStretch(1)
        self.combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo.currentIndexChanged.connect(lambda _: self.on_change())

    def current(self):
        i = self.combo.currentIndex()
        params = {}
        for pkey, (kind, w) in self.widgets[i].items():
            if kind == "combo":
                params[pkey] = w.currentData()
            elif kind == "check":
                params[pkey] = w.isChecked()
            elif kind == "int":
                params[pkey] = int(w.value())
            else:
                params[pkey] = float(w.value())
        return self.methods[i][0], params


def direction_legend(size=120) -> np.ndarray:
    """Цветовое колесо направлений, построенное той же функцией, что и карта."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    dx, dy = xx - c, yy - c
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)).astype(np.float32)   # ось y вниз, как в изображении
    mag = np.where((r <= c) & (r >= c * 0.35), 1.0, 0.0).astype(np.float32)
    img = seg.direction_map(mag, ang, 1.0, np.zeros((size, size), np.uint8), False)
    img[mag == 0] = 255
    return img


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЛР 3, вариант 4 — гистограммы, контрастирование, сегментация")
        self.resize(1500, 920)
        self.original: PixelBuffer | None = None
        self.current: PixelBuffer | None = None
        self.result = None
        self.worker: Worker | None = None
        self.pending = False
        self._build_ui()
        self._build_menu()
        self.timer = QTimer(self, singleShot=True, interval=350, timeout=self.run)
        self.status("Подготовка: компиляция вычислительных ядер (только при первом запуске)…")
        self.warm = WarmUp(self)
        self.warm.finishedOk.connect(lambda t: self.status(f"Готово. Ядра подготовлены за {t:.1f} с. "
                                                           "Откройте изображение (Файл → Открыть)."))
        self.warm.start()
        default = os.path.join(app_root(), "images", "01_moon_low_contrast.png")
        if os.path.exists(default):
            self.open_path(default)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.view_before = ImageView("До")
        self.view_after = ImageView("После")
        self.view_before.viewChanged.connect(lambda: self._sync(self.view_before, self.view_after))
        self.view_after.viewChanged.connect(lambda: self._sync(self.view_after, self.view_before))
        self._syncing = False

        images = QSplitter(Qt.Horizontal)
        images.addWidget(self.view_before)
        images.addWidget(self.view_after)
        images.setSizes([500, 500])
        images.setStretchFactor(0, 1)
        images.setStretchFactor(1, 1)

        self.hist = HistogramView()
        self.hist_channel = QComboBox()
        for t, v in [("Яркость Y", "y"), ("Канал R", 0), ("Канал G", 1), ("Канал B", 2),
                     ("Яркость V (HSV)", "v")]:
            self.hist_channel.addItem(t, v)
        self.hist_cdf = QCheckBox("Накопленная (CDF)")
        self.hist_log = QCheckBox("Лог. шкала")
        for w in (self.hist_channel,):
            w.currentIndexChanged.connect(self.update_histogram)
        self.hist_cdf.toggled.connect(self._hist_mode)
        self.hist_log.toggled.connect(self._hist_mode)
        self.stats_label = QLabel()
        self.stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        hbar = QHBoxLayout()
        hbar.addWidget(QLabel("Гистограмма:"))
        hbar.addWidget(self.hist_channel)
        hbar.addWidget(self.hist_cdf)
        hbar.addWidget(self.hist_log)
        hbar.addStretch(1)
        hwrap = QWidget()
        hl = QVBoxLayout(hwrap)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addLayout(hbar)
        hl.addWidget(self.hist, 1)
        hl.addWidget(self.stats_label)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(500)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(hwrap)
        bottom.addWidget(self.log)
        bottom.setSizes([800, 500])

        center = QSplitter(Qt.Vertical)
        center.addWidget(images)
        center.addWidget(bottom)
        center.setSizes([620, 280])

        # ---- панель управления
        self.tabs = QTabWidget()
        self.p_contrast = MethodPanel(CONTRAST_METHODS, self.params_changed)
        self.p_seg = MethodPanel(SEG_METHODS, self.params_changed)
        self.p_prep = MethodPanel(PREP_METHODS, self.params_changed)
        self.color_mode = QComboBox()
        self.color_mode.addItem("HSV: только яркость V", "hsv")
        self.color_mode.addItem("RGB: каждый канал отдельно", "rgb")
        self.color_mode.currentIndexChanged.connect(self.params_changed)
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        cm = QFormLayout()
        cm.addRow("Цветное изображение:", self.color_mode)
        l1.addLayout(cm)
        l1.addWidget(self.p_contrast)
        self.tabs.addTab(t1, "Гистограмма и контраст")
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        l2.addWidget(QLabel("Сегментация выполняется по яркости Y."))
        l2.addWidget(self.p_seg)
        self.legend = QLabel()
        self.legend.setPixmap(QPixmap.fromImage(to_qimage(direction_legend())))
        self.legend_box = QGroupBox("Легенда направлений градиента θ")
        lb = QVBoxLayout(self.legend_box)
        lb.addWidget(self.legend, alignment=Qt.AlignCenter)
        lb.addWidget(QLabel("0° → (вправо): красный; 90° ↓: жёлто-зелёный;\n"
                            "±180° ←: голубой; −90° ↑: фиолетовый.\nЯркость — модуль |∇f|."))
        l2.addWidget(self.legend_box)
        self.tabs.addTab(t2, "Сегментация")
        self.tabs.addTab(self.p_prep, "Подготовка")
        self.tabs.currentChanged.connect(self.params_changed)

        self.border = QComboBox()
        self.border.addItem("Reflect (зеркальное отражение)", BORDER_REFLECT)
        self.border.addItem("Clamp (дублирование граничных)", BORDER_CLAMP)
        self.border.currentIndexChanged.connect(self.params_changed)
        self.threads = QSpinBox()
        self.threads.setRange(1, parallel.max_threads())
        self.threads.setValue(parallel.max_threads())
        self.threads.valueChanged.connect(lambda v: parallel.set_threads(v))
        self.auto = QCheckBox("Пересчитывать при изменении параметров")
        self.auto.setChecked(True)
        common = QGroupBox("Общие настройки")
        cf = QFormLayout(common)
        cf.addRow("Обработка краёв:", self.border)
        cf.addRow("Потоков:", self.threads)
        cf.addRow(self.auto)

        self.btn_run = QPushButton("Выполнить  (F5)")
        self.btn_run.clicked.connect(self.run)
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_apply = QPushButton("Применить результат как исходное")
        self.btn_apply.clicked.connect(self.apply_result)
        self.btn_reset = QPushButton("Сброс к оригиналу")
        self.btn_reset.clicked.connect(self.reset)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.time_label = QLabel("—")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.tabs, 1)
        rl.addWidget(common)
        row = QHBoxLayout()
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        rl.addLayout(row)
        rl.addWidget(self.btn_apply)
        rl.addWidget(self.btn_reset)
        rl.addWidget(self.progress)
        rl.addWidget(self.time_label)
        right.setMinimumWidth(380)
        right.setMaximumWidth(460)

        main = QSplitter(Qt.Horizontal)
        main.addWidget(center)
        main.addWidget(right)
        main.setStretchFactor(0, 1)
        self.setCentralWidget(main)
        self._update_legend()

    def _build_menu(self):
        m = self.menuBar().addMenu("Файл")
        a = QAction("Открыть…", self, shortcut=QKeySequence.Open, triggered=self.open_dialog)
        m.addAction(a)
        tm = m.addMenu("Тестовые изображения")
        folder = os.path.join(app_root(), "images")
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                    tm.addAction(QAction(f, self, triggered=lambda _=False, p=os.path.join(folder, f):
                                         self.open_path(p)))
        m.addAction(QAction("Сохранить результат…", self, shortcut=QKeySequence.Save,
                            triggered=self.save_dialog))
        m.addAction(QAction("Сохранить субпиксельные точки Кэнни (CSV)…", self,
                            triggered=self.save_subpixel))
        m.addSeparator()
        m.addAction(QAction("Выход", self, triggered=self.close))
        v = self.menuBar().addMenu("Вид")
        v.addAction(QAction("Вписать в окно", self, shortcut="Ctrl+0",
                            triggered=lambda: (self.view_before.fit(), self.view_after.fit())))
        h = self.menuBar().addMenu("Справка")
        h.addAction(QAction("О программе", self, triggered=self.about))
        self.addAction(QAction(self, shortcut="F5", triggered=self.run))

    # ------------------------------------------------------------ helpers
    def status(self, text):
        self.statusBar().showMessage(text)

    def _sync(self, src, dst):
        if self._syncing:
            return
        self._syncing = True
        dst.sync_from(src)
        self._syncing = False

    def _hist_mode(self):
        self.hist.cumulative = self.hist_cdf.isChecked()
        self.hist.log_scale = self.hist_log.isChecked()
        self.hist.update()

    def _plane(self, buf: PixelBuffer, ch):
        if ch == "y" or buf.is_gray:
            return buf.luminance()
        if ch == "v":
            return rgb_to_hsv(buf.data)[2]
        return buf.channel(int(ch))

    def update_histogram(self):
        if self.current is None:
            return
        ch = self.hist_channel.currentData()
        pb = self._plane(self.current, ch)
        hb = histogram(pb)
        sb = stats(pb)
        txt = (f"До: min {sb['min']}, max {sb['max']}, среднее {sb['mean']:.2f}, σ {sb['sigma']:.2f}, "
               f"уровней {sb['levels']}")
        ha = None
        if self.result is not None:
            pa = self._plane(self.result.buffer, ch)
            ha = histogram(pa)
            sa = stats(pa)
            txt += (f"   |   После: min {sa['min']}, max {sa['max']}, среднее {sa['mean']:.2f}, "
                    f"σ {sa['sigma']:.2f}, уровней {sa['levels']}")
        self.hist.set_data(hb, ha, self.hist_channel.currentText())
        self.stats_label.setText(txt)

    def _update_legend(self):
        _, p = self.p_seg.current()
        self.legend_box.setVisible(str(p.get("output", "")).startswith("direction"))

    # ------------------------------------------------------------- файлы
    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение",
                                              os.path.join(app_root(), "images"),
                                              "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            self.original = PixelBuffer.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл:\n{e}")
            return
        self.current = self.original
        self.result = None
        self.view_before.set_image(self.current.data)
        self.view_before.fit()
        self.view_after.set_image(None)
        self.color_mode.setEnabled(not self.current.is_gray)
        b = self.current
        self.setWindowTitle(f"ЛР 3, вариант 4 — {os.path.basename(path)} "
                            f"({b.width}×{b.height}, {'серое' if b.is_gray else 'RGB'})")
        self.log.appendPlainText(f"Открыто: {os.path.basename(path)} — {b.width}×{b.height}, "
                                 f"каналов: {b.channels}")
        self.update_histogram()
        self.params_changed()

    def save_dialog(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", "result.png", "PNG (*.png)")
        if path:
            self.result.buffer.save(path)
            self.status(f"Сохранено: {path}")

    def save_subpixel(self):
        if self.result is None or "subpixel" not in self.result.extra:
            QMessageBox.information(self, "Субпиксельные точки", "Сначала выполните детектор Кэнни.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить точки", "canny_subpixel.csv", "CSV (*.csv)")
        if path:
            np.savetxt(path, self.result.extra["subpixel"], delimiter=";", fmt="%.3f",
                       header="x;y", comments="")
            self.status(f"Сохранено точек: {len(self.result.extra['subpixel'])}")

    # ------------------------------------------------------------- обработка
    def params_changed(self, *_):
        self._update_legend()
        if self.auto.isChecked() and self.current is not None:
            self.timer.start()

    def _current_op(self):
        idx = self.tabs.currentIndex()
        panel = [self.p_contrast, self.p_seg, self.p_prep][idx]
        name, p = panel.current()
        p["border"] = self.border.currentData()
        p["color"] = self.color_mode.currentData()
        return name, p

    def run(self):
        if self.current is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self.pending = True
            self.worker.stop()
            return
        name, p = self._current_op()
        self.worker = Worker(name, self.current, p, self)
        self.worker.progressChanged.connect(self.progress.setValue)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.on_finished)
        self.progress.setValue(0)
        self.btn_cancel.setEnabled(True)
        self.status("Обработка…")
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.pending = False
            self.worker.stop()

    def on_done(self, res):
        self.result = res
        self.view_after.set_image(res.buffer.data)
        self._sync(self.view_before, self.view_after)
        self.update_histogram()
        msg = f"{res.info}\nВремя: {res.seconds * 1000:.1f} мс, потоков: {parallel.get_threads()}"
        self.log.appendPlainText(msg + "\n")
        self.time_label.setText(f"Время обработки: {res.seconds * 1000:.1f} мс "
                                f"({self.current.width}×{self.current.height}, потоков {parallel.get_threads()})")
        self.status("Готово")

    def on_failed(self, tb):
        self.log.appendPlainText("ОШИБКА:\n" + tb)
        self.status("Ошибка обработки — подробности в журнале")

    def on_cancelled(self):
        self.status("Обработка отменена")
        self.progress.setValue(0)

    def on_finished(self):
        self.btn_cancel.setEnabled(False)
        if self.pending:
            self.pending = False
            QTimer.singleShot(0, self.run)

    def apply_result(self):
        if self.result is None:
            return
        self.current = self.result.buffer
        self.result = None
        self.view_before.set_image(self.current.data)
        self.view_after.set_image(None)
        self.log.appendPlainText("Результат применён: дальнейшие операции выполняются над ним.\n")
        self.update_histogram()

    def reset(self):
        if self.original is None:
            return
        self.current = self.original
        self.result = None
        self.view_before.set_image(self.current.data)
        self.view_after.set_image(None)
        self.update_histogram()
        self.log.appendPlainText("Сброс к исходному изображению.\n")

    def about(self):
        QMessageBox.about(self, "О программе",
                          "Лабораторная работа 3, вариант 4.\n\n"
                          "Построение и эквализация гистограммы, линейное контрастирование, CLAHE;\n"
                          "сегментация: обнаружение точек, линий и перепадов яркости\n"
                          "(Робертс, Превитт, Собель, LoG, Кэнни с субпиксельной точностью).\n\n"
                          "Архитектура: ui (представление) → core (буфер пикселей, края, потоки) → "
                          "algorithms (чистые функции обработки).")

    def closeEvent(self, e):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        if self.warm.isRunning():
            self.warm.wait(30000)
        super().closeEvent(e)
