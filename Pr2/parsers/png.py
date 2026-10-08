"""PNG: сигнатура + последовательность чанков (длина, тип, данные, CRC)."""
from __future__ import annotations

import struct
import zlib

from models import ImageMetadata
from .errors import CorruptFileError
from .reader import BinaryReader

SIGNATURE = b"\x89PNG\r\n\x1a\n"
IEND_TAIL = b"\x00\x00\x00\x00IEND\xaeB`\x82"  # 12 последних байт корректного PNG

CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ALLOWED_DEPTH = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
COLOR_NAMES = {0: "оттенки серого", 2: "RGB", 3: "индексированный (палитра)",
               4: "серый + альфа", 6: "RGBA"}


def _check_crc(r: BinaryReader, ctype: bytes, data: bytes, name: str) -> None:
    stored = struct.unpack(">I", r.read(4))[0]
    if zlib.crc32(ctype + data) & 0xFFFFFFFF != stored:
        raise CorruptFileError(f"Контрольная сумма (CRC) чанка {name} не совпадает")


def parse(r: BinaryReader, meta: ImageMetadata) -> None:
    r.seek(8)
    length, ctype = r.unpack(">I4s")
    if ctype != b"IHDR" or length != 13:
        raise CorruptFileError("Первым чанком должен быть IHDR длиной 13 байт")
    ihdr = r.read(13)
    _check_crc(r, ctype, ihdr, "IHDR")
    width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", ihdr)

    meta.width, meta.height = width, height
    if color not in CHANNELS:
        raise CorruptFileError(f"Недопустимый тип цвета в IHDR: {color}")
    meta.bits_per_pixel = depth * CHANNELS[color]
    meta.compression = "Deflate (zlib); фильтрация: адаптивная" if comp == 0 and filt == 0 else ""
    if width == 0 or height == 0 or width > 0x7FFFFFFF or height > 0x7FFFFFFF:
        raise CorruptFileError(f"Недопустимые размеры в IHDR: {width} × {height}")
    if depth not in ALLOWED_DEPTH[color]:
        raise CorruptFileError(f"Глубина {depth} бит недопустима для типа цвета {color}")
    if comp != 0:
        raise CorruptFileError(f"Неизвестный метод сжатия в IHDR: {comp}")
    if filt != 0:
        raise CorruptFileError(f"Неизвестный метод фильтрации в IHDR: {filt}")
    if interlace not in (0, 1):
        raise CorruptFileError(f"Неизвестный метод развёртки в IHDR: {interlace}")

    meta.add_extra("Тип цвета", f"{color} — {COLOR_NAMES[color]}",
                   "Байт Color type в IHDR; вместе с Bit depth определяет число бит на пиксель.")
    meta.add_extra("Глубина канала", f"{depth} бит × {CHANNELS[color]} кан.",
                   "Байт Bit depth в IHDR — бит на один канал; бит/пиксель = глубина × число каналов.")
    meta.add_extra("Фильтрация", "метод 0: адаптивная (None, Sub, Up, Average, Paeth)",
                   "Байт Filter method в IHDR. Перед сжатием каждая строка пикселей преобразуется одним из 5 фильтров.")
    meta.add_extra("Развёртка", "Adam7 (7 проходов)" if interlace else "нет",
                   "Байт Interlace method в IHDR: Adam7 позволяет показать грубую картинку до полной загрузки.")

    names = []
    palette = False
    seen_idat = False
    while True:
        if r.pos + 12 > r.size:
            raise CorruptFileError("Отсутствует завершающий чанк IEND (файл обрезан)")
        length, ctype = r.unpack(">I4s")
        if length > 0x7FFFFFFF:
            raise CorruptFileError(f"Недопустимая длина чанка {ctype!r}: {length}")
        if r.pos + length + 4 > r.size:
            raise CorruptFileError(
                f"Чанк {ctype.decode('latin-1')} выходит за пределы файла (файл обрезан)")
        tname = ctype.decode("latin-1")

        if ctype == b"IDAT":
            if not seen_idat:
                seen_idat = True
                if color == 3 and not palette:
                    raise CorruptFileError("У индексированного PNG нет чанка PLTE")
                # Всё, что нужно из заголовка, уже прочитано (pHYs обязан стоять до IDAT).
                # Быстрый путь: проверяем конец файла, не проходя по всем IDAT.
                data_pos = r.pos
                if r.size >= 12 and r.read_at(r.size - 12, 12) == IEND_TAIL:
                    meta.add_extra("Чанки до IDAT", ", ".join(names) or "нет",
                                   "Служебные чанки (PLTE, pHYs, tRNS, gAMA, iCCP, tEXt…) между IHDR и данными.")
                    return
                r.seek(data_pos)  # вернуться и продолжить медленным обходом
            r.skip(length + 4)
            continue

        data = b""
        if ctype in (b"pHYs", b"PLTE"):
            data = r.read(length)
            _check_crc(r, ctype, data, tname)
        else:
            r.skip(length + 4)

        if ctype == b"PLTE":
            if length % 3 or length == 0 or length // 3 > 256:
                raise CorruptFileError("Некорректный размер палитры PLTE")
            palette = True
            meta.add_extra("Цветов в палитре", length // 3,
                           "Длина чанка PLTE делится на 3 (R, G, B на каждый цвет).")
        elif ctype == b"pHYs":
            if length != 9:
                raise CorruptFileError("Чанк pHYs должен иметь длину 9 байт")
            ppux, ppuy, unit = struct.unpack(">IIB", data)
            if unit == 1:  # пикселей на метр
                meta.dpi_x = round(ppux * 0.0254, 2) if ppux else None
                meta.dpi_y = round(ppuy * 0.0254, 2) if ppuy else None
            else:
                meta.add_extra("Единица pHYs", "не определена (только пропорции)",
                               "Если Unit specifier = 0, чанк задаёт лишь соотношение сторон пикселя, но не DPI.")
        elif ctype == b"IEND":
            if r.size > r.pos:
                meta.warn(f"После IEND есть лишние данные ({r.size - r.pos} байт)")
            return

        if tname not in names:
            names.append(tname)
