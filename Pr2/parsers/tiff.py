"""TIFF / BigTIFF: ручной разбор каталогов IFD. Также используется для Exif в JPEG."""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from models import ImageMetadata
from .errors import CorruptFileError
from .reader import BinaryReader

# тип поля -> (размер значения в байтах, struct-код)
TYPES = {
    1: (1, "B"), 2: (1, "c"), 3: (2, "H"), 4: (4, "I"), 5: (8, "RATIONAL"),
    6: (1, "b"), 7: (1, "B"), 8: (2, "h"), 9: (4, "i"), 10: (8, "SRATIONAL"),
    11: (4, "f"), 12: (8, "d"), 13: (4, "I"), 16: (8, "Q"), 17: (8, "q"), 18: (8, "Q"),
}

COMPRESSION = {
    1: "Без сжатия", 2: "CCITT RLE (Modified Huffman)", 3: "CCITT Group 3 (T.4)",
    4: "CCITT Group 4 (T.6)", 5: "LZW", 6: "JPEG (старый)", 7: "JPEG", 8: "Deflate (Adobe)",
    9: "JBIG (T.85)", 10: "JBIG (T.43)", 32766: "NeXT RLE 2-bit", 32771: "Без сжатия (выравнивание по слову)",
    32773: "PackBits (RLE)", 32946: "Deflate (PKZIP)", 34661: "JBIG", 34712: "JPEG 2000",
    34925: "LZMA", 50000: "Zstandard", 50001: "WebP",
}
PHOTOMETRIC = {0: "WhiteIsZero (0 — белый)", 1: "BlackIsZero (0 — чёрный)", 2: "RGB", 3: "палитра",
               4: "маска прозрачности", 5: "CMYK", 6: "YCbCr", 8: "CIE L*a*b*"}
MAX_PAGES = 100_000
MAX_VALUES = 4096

Entry = Tuple[int, int, bytes]  # (тип, количество, 4/8 байт «значение или смещение»)


def read_ifd(r: BinaryReader, offset: int, endian: str, big: bool, base: int = 0
             ) -> Tuple[Dict[int, Entry], int]:
    """Читает один каталог IFD целиком одним чтением. Возвращает (теги, смещение следующего IFD)."""
    if offset < 8 or base + offset >= r.size:
        raise CorruptFileError(f"Смещение IFD {offset} выходит за пределы файла")
    r.seek(base + offset)
    count = struct.unpack(endian + ("Q" if big else "H"), r.read(8 if big else 2))[0]
    esize, nsize = (20, 8) if big else (12, 4)
    if count == 0:
        raise CorruptFileError("Каталог IFD не содержит ни одного тега")
    if r.pos + count * esize + nsize > r.size:
        raise CorruptFileError("Каталог IFD выходит за пределы файла")
    block = r.read(count * esize)
    next_off = struct.unpack(endian + ("Q" if big else "I"), r.read(nsize))[0]
    fmt = endian + ("HHQ8s" if big else "HHI4s")
    entries: Dict[int, Entry] = {}
    for i in range(count):
        tag, typ, cnt, raw = struct.unpack_from(fmt, block, i * esize)
        entries.setdefault(tag, (typ, cnt, raw))
    return entries, next_off


def _decode(typ: int, data: bytes, n: int, endian: str):
    code = TYPES[typ][1]
    if code == "c":
        return data[:n]
    if code in ("RATIONAL", "SRATIONAL"):
        c = "I" if code == "RATIONAL" else "i"
        flat = struct.unpack(endian + c * (2 * n), data[: 8 * n])
        return [(flat[i], flat[i + 1]) for i in range(0, 2 * n, 2)]
    return list(struct.unpack(endian + code * n, data[: TYPES[typ][0] * n]))


def get_values(r: BinaryReader, entry: Optional[Entry], endian: str, big: bool, base: int = 0,
               limit: int = MAX_VALUES):
    """Значения тега (не более limit штук). None, если тега нет или тип неизвестен."""
    if entry is None:
        return None
    typ, cnt, raw = entry
    if typ not in TYPES:
        return None
    size = TYPES[typ][0]
    n = min(cnt, limit)
    inline = 8 if big else 4
    if size * cnt <= inline:
        data = raw
    else:
        off = struct.unpack(endian + ("Q" if big else "I"), raw)[0]
        if base + off + size * n > r.size:
            raise CorruptFileError("Значение тега выходит за пределы файла")
        data = r.read_at(base + off, size * n)
    return _decode(typ, data, n, endian)


def _element(r: BinaryReader, entry: Entry, idx: int, endian: str, big: bool) -> Optional[int]:
    """Один элемент числового массива без чтения всего массива (для проверки первой/последней полосы)."""
    typ, cnt, raw = entry
    if typ not in TYPES or idx >= cnt or TYPES[typ][1] in ("RATIONAL", "SRATIONAL", "c", "f", "d"):
        return None
    size = TYPES[typ][0]
    inline = 8 if big else 4
    if size * cnt <= inline:
        return _decode(typ, raw, cnt, endian)[idx]
    off = struct.unpack(endian + ("Q" if big else "I"), raw)[0]
    if off + size * (idx + 1) > r.size:
        raise CorruptFileError("Массив смещений полос выходит за пределы файла")
    return _decode(typ, r.read_at(off + size * idx, size), 1, endian)[0]


def _first(vals) -> Optional[int]:
    return vals[0] if vals else None


def _resolution(vals, unit: int) -> Optional[float]:
    if not vals or unit not in (2, 3):
        return None
    num, den = vals[0]
    if den == 0 or num == 0:
        return None
    dpi = num / den
    return round(dpi * 2.54, 2) if unit == 3 else round(dpi, 2)


def read_exif_resolution(r: BinaryReader, base: int):
    """Разрешение из Exif-блока JPEG: TIFF-заголовок начинается со смещения base. -> (x_dpi, y_dpi) или None."""
    head = r.read_at(base, 8)
    if head[:2] == b"II":
        endian = "<"
    elif head[:2] == b"MM":
        endian = ">"
    else:
        return None
    if struct.unpack(endian + "H", head[2:4])[0] != 42:
        return None
    ifd0 = struct.unpack(endian + "I", head[4:8])[0]
    entries, _ = read_ifd(r, ifd0, endian, False, base)
    unit = _first(get_values(r, entries.get(0x0128), endian, False, base)) or 2
    xs = get_values(r, entries.get(0x011A), endian, False, base)
    ys = get_values(r, entries.get(0x011B), endian, False, base)
    x, y = _resolution(xs, unit), _resolution(ys, unit)
    return (x, y) if (x or y) else None


def parse(r: BinaryReader, meta: ImageMetadata) -> None:
    head = r.read_at(0, 8)
    endian = "<" if head[:2] == b"II" else ">"
    version = struct.unpack(endian + "H", head[2:4])[0]
    big = version == 43
    if big:
        osize, zero = struct.unpack(endian + "HH", head[4:8])
        if osize != 8 or zero != 0:
            raise CorruptFileError("Некорректный заголовок BigTIFF")
        first_ifd = struct.unpack(endian + "Q", r.read(8))[0]
    else:
        first_ifd = struct.unpack(endian + "I", head[4:8])[0]
    if first_ifd == 0:
        raise CorruptFileError("Смещение первого IFD равно нулю")

    visited = set()
    offset = first_ifd
    pages = 0
    while offset:
        if offset in visited:
            raise CorruptFileError("Цепочка каталогов IFD зациклена")
        visited.add(offset)
        entries, next_off = read_ifd(r, offset, endian, big)
        if pages == 0:
            _first_page(r, meta, entries, endian, big)
        pages += 1
        if pages >= MAX_PAGES:
            meta.warn(f"Достигнут предел {MAX_PAGES} страниц, остальные не учитываются")
            break
        offset = next_off

    if pages > 1:
        meta.format = f"TIFF ({pages} стр.)"
    meta.add_extra("Порядок байтов", "little-endian (II)" if endian == "<" else "big-endian (MM)",
                   "Первые два байта файла: «II» (Intel) или «MM» (Motorola); определяют, как читать все числа.")
    meta.add_extra("Версия", "BigTIFF (43)" if big else "классический TIFF (42)",
                   "Число после порядка байтов; BigTIFF использует 64-битные смещения (файлы > 4 ГБ).")
    meta.add_extra("Страниц (IFD)", pages,
                   "Каталоги IFD образуют цепочку; в таблице показаны свойства первой страницы.")


def _first_page(r, meta, entries, endian, big) -> None:
    def vals(tag, limit=MAX_VALUES):
        return get_values(r, entries.get(tag), endian, big, 0, limit)

    width = _first(vals(256))
    height = _first(vals(257))
    if not width or not height:
        raise CorruptFileError("В IFD нет обязательных тегов ImageWidth / ImageLength")
    meta.width, meta.height = width, height

    spp = _first(vals(277)) or 1
    bps = vals(258) or [1]
    if len(bps) == 1 and spp > 1:
        bits = bps[0] * spp
    else:
        bits = sum(bps[:spp]) if len(bps) >= spp else sum(bps)
    meta.bits_per_pixel = bits or None

    comp = _first(vals(259)) or 1
    meta.compression = COMPRESSION.get(comp, f"Неизвестный код сжатия {comp}")

    unit = _first(vals(296)) or 2
    meta.dpi_x = _resolution(vals(282), unit)
    meta.dpi_y = _resolution(vals(283), unit)

    # Проверка: данные изображения не выходят за пределы файла (первая и последняя полоса/плитка).
    for off_tag, cnt_tag in ((273, 279), (324, 325)):
        offs = entries.get(off_tag)
        if offs is None:
            continue
        cnts = entries.get(cnt_tag)
        for idx in {0, offs[1] - 1}:
            o = _element(r, offs, idx, endian, big)
            c = _element(r, cnts, idx, endian, big) if cnts else 0
            if o is not None and (o >= r.size or o + (c or 0) > r.size):
                raise CorruptFileError(
                    "Данные изображения выходят за пределы файла (файл обрезан)")

    if not meta.detailed:
        return
    photo = _first(vals(262))
    meta.add_extra("Цветовая модель", PHOTOMETRIC.get(photo, photo if photo is not None else "не указана"),
                   "Тег PhotometricInterpretation (262): как интерпретировать значения пикселей.")
    meta.add_extra("Каналов на пиксель", spp, "Тег SamplesPerPixel (277).")
    meta.add_extra("Бит по каналам", "+".join(map(str, bps)), "Тег BitsPerSample (258); сумма даёт бит/пиксель.")
    meta.add_extra("Единица разрешения",
                   {1: "нет", 2: "дюйм", 3: "сантиметр"}.get(unit, unit),
                   "Тег ResolutionUnit (296); DPI пересчитывается из XResolution/YResolution (282/283).")
    rps = _first(vals(278))
    if 273 in entries:
        meta.add_extra("Полос (strips)", entries[273][1],
                       "Тег StripOffsets (273): изображение хранится кусками; RowsPerStrip = %s." % (rps or "—"))
    if 324 in entries:
        meta.add_extra("Плиток (tiles)", entries[324][1], "Теги TileOffsets/TileByteCounts (324/325).")
    pred = _first(vals(317))
    if pred:
        meta.add_extra("Предсказатель", {2: "горизонтальный", 3: "с плавающей точкой"}.get(pred, pred),
                       "Тег Predictor (317): разностное кодирование перед LZW/Deflate.")
    if photo == 3:
        meta.add_extra("Цветов в палитре", 1 << bps[0], "Тег ColorMap (320) содержит 2^BitsPerSample записей.")
