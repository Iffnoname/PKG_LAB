
from __future__ import annotations

import html

from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

from models import ImageMetadata, STATUS_OK
from parsers import inspect_file

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 400_000_000
except Exception:
    Image = None

MAX_PREVIEW = 640


class _Signals(QObject):
    ready = Signal(int, object, object, str)


class _Job(QRunnable):
    def __init__(self, token: int, path: str, signals: _Signals) -> None:
        super().__init__()
        self.token, self.path, self.signals = token, path, signals

    def run(self) -> None:
        meta = inspect_file(self.path, detailed=True)
        image, error = None, ""
        if Image is None:
            error = "Pillow не установлен — предпросмотр недоступен"
        else:
            try:
                with Image.open(self.path) as im:
                    if im.format == "JPEG":
                        im.draft("RGB", (MAX_PREVIEW * 2, MAX_PREVIEW * 2))
                    im.thumbnail((MAX_PREVIEW, MAX_PREVIEW))
                    if im.mode.startswith("I"):
                        im = im.point(lambda v: v * (1 / 256)).convert("L")
                    rgba = im.convert("RGBA")
                    w, h = rgba.size
                    image = QImage(rgba.tobytes("raw", "RGBA"), w, h, w * 4,
                                   QImage.Format.Format_RGBA8888).copy()
            except Exception as e:     # битые файлы, неподдерживаемые варианты
                error = f"Предпросмотр недоступен: {type(e).__name__}: {e}"
        self.signals.ready.emit(self.token, image, meta, error)


class PreviewPanel(QWidget):
    def __init__(self, pool, parent=None) -> None:
        super().__init__(parent)
        self._pool = pool
        self._token = 0
        self._signals = _Signals()
        self._signals.ready.connect(self._on_ready)

        self.picture = QLabel("Выберите файл в таблице")
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(300, 240)
        self.picture.setStyleSheet("QLabel { border: 1px solid palette(mid); background: palette(base); }")
        self.details = QTextBrowser()
        self.details.setOpenLinks(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.picture, 3)
        layout.addWidget(self.details, 4)

    def show_file(self, meta: ImageMetadata | None) -> None:
        self._token += 1
        if meta is None:
            self.picture.setPixmap(QPixmap())
            self.picture.setText("Выберите файл в таблице")
            self.details.clear()
            return
        self.picture.setPixmap(QPixmap())
        self.picture.setText("Загрузка…")
        self.details.setHtml(self._render(meta, ""))
        self._pool.start(_Job(self._token, meta.path, self._signals))

    def _on_ready(self, token: int, image, meta: ImageMetadata, error: str) -> None:
        if token != self._token:
            return
        if image is not None:
            pix = QPixmap.fromImage(image)
            self.picture.setPixmap(pix.scaled(self.picture.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
        else:
            self.picture.setText(error or "Нет изображения")
        self.details.setHtml(self._render(meta, error))

    @staticmethod
    def _render(m: ImageMetadata, error: str) -> str:
        e = html.escape
        out = [f"<h3 style='margin:0'>{e(m.name)}</h3><div style='color:gray'>{e(m.path)}</div>"]
        color = "green" if m.status == STATUS_OK else "#c0392b"
        out.append(f"<p><b>Статус:</b> <span style='color:{color}'>{e(m.status)}</span>")
        if m.message:
            out.append(f"<br>{e(m.message)}")
        out.append("</p>")
        out.append(f"<p>Размер файла: {m.file_size:,} байт".replace(",", " ")
                   + (f"<br>Прочитано при разборе: {m.bytes_read:,} байт".replace(",", " ") if m.bytes_read else "")
                   + "</p>")
        if m.extras:
            out.append("<table cellspacing='0' cellpadding='3' width='100%'>")
            for name, value, descr in m.extras:
                val = (f"<pre style='margin:0;font-family:Consolas,monospace'>{e(value.strip())}</pre>"
                       if "\n" in value else e(value))
                out.append(f"<tr><td valign='top'><b>{e(name)}</b></td><td valign='top'>{val}</td></tr>"
                           f"<tr><td></td><td style='color:gray'><i>{e(descr)}</i></td></tr>")
            out.append("</table>")
        return "".join(out)
