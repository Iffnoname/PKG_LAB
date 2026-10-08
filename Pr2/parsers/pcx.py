"""PCX: заголовок 128 байт, затем RLE-данные; для 256 цветов в конце файла палитра (0x0C + 768 байт)."""
from __future__ import annotations

from models import ImageMetadata
from .errors import CorruptFileError
from .reader import BinaryReader

VERSIONS = {0: "2.5", 2: "2.8 (с палитрой)", 3: "2.8 (без палитры)", 4: "Paintbrush для Windows", 5: "3.0 и выше"}


def parse(r: BinaryReader, meta: ImageMetadata) -> None:
    hdr = r.read_at(0, 128)
    version, encoding, bits = hdr[1], hdr[2], hdr[3]
    xmin = int.from_bytes(hdr[4:6], "little")
    ymin = int.from_bytes(hdr[6:8], "little")
    xmax = int.from_bytes(hdr[8:10], "little")
    ymax = int.from_bytes(hdr[10:12], "little")
    hdpi = int.from_bytes(hdr[12:14], "little")
    vdpi = int.from_bytes(hdr[14:16], "little")
    planes = hdr[65]
    bpl = int.from_bytes(hdr[66:68], "little")

    if xmax < xmin or ymax < ymin:
        raise CorruptFileError("Границы изображения в заголовке некорректны")
    width, height = xmax - xmin + 1, ymax - ymin + 1
    meta.width, meta.height = width, height
    meta.dpi_x = float(hdpi) if hdpi else None
    meta.dpi_y = float(vdpi) if vdpi else None
    meta.bits_per_pixel = bits * planes if bits and planes else None
    meta.compression = "RLE (PCX)" if encoding == 1 else f"Неизвестный метод ({encoding})"

    if bits not in (1, 2, 4, 8) or not 1 <= planes <= 4:
        raise CorruptFileError(f"Недопустимое сочетание: {bits} бит × {planes} плоскостей")
    if encoding != 1:
        meta.warn(f"Поле кодирования равно {encoding}, ожидалось 1 (RLE)")
    if bpl == 0 or bpl * 8 < width * bits:
        raise CorruptFileError("Поле BytesPerLine слишком мало для заданной ширины")
    if bpl % 2:
        meta.warn("BytesPerLine нечётно (по спецификации должно быть чётным)")

    has_vga = version == 5 and bits == 8 and planes == 1
    if has_vga:
        if r.size < 128 + 769 or r.read_at(r.size - 769, 1) != b"\x0c":
            meta.warn("Нет 256-цветной палитры в конце файла (байт 0x0C)")
            has_vga = False
    raw = bpl * planes * height
    avail = r.size - 128 - (769 if has_vga else 0)
    # Серия RLE кодирует не более 63 байт двумя байтами — нижняя граница размера сжатых данных.
    minimum = 2 * -(-raw // 63) if encoding == 1 else raw
    if avail < minimum:
        raise CorruptFileError("Пиксельные данные обрезаны: файл меньше минимально возможного")

    meta.add_extra("Версия", VERSIONS.get(version, version), "Байт 1 заголовка (байт 0 — сигнатура 0x0A).")
    meta.add_extra("Плоскостей × бит", f"{planes} × {bits}",
                   "Байты 65 и 3: бит/пиксель = число плоскостей × бит на плоскость.")
    meta.add_extra("Байт на строку", bpl, "Поле BytesPerLine (66–67): длина одной строки одной плоскости после распаковки.")
    if has_vga:
        meta.add_extra("Палитра", "256 цветов в конце файла", "Блок из 769 байт: маркер 0x0C и 256×RGB.")
    elif bits * planes <= 4:
        meta.add_extra("Палитра", "16 цветов в заголовке", "Байты 16–63 заголовка.")
    meta.add_extra("DPI", f"{hdpi} × {vdpi}",
                   "Поля HDpi/VDpi (12–15) — разрешение устройства создания; нередко содержат значения экрана.")
