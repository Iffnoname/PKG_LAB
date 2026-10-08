"""BMP: BITMAPFILEHEADER (14 байт) + DIB-заголовок (BITMAPCOREHEADER / BITMAPINFOHEADER и новее)."""
from __future__ import annotations

from models import ImageMetadata
from .errors import CorruptFileError
from .reader import BinaryReader

COMPRESSION = {
    0: "BI_RGB (без сжатия)",
    1: "BI_RLE8 (RLE, 8 бит)",
    2: "BI_RLE4 (RLE, 4 бита)",
    3: "BI_BITFIELDS (без сжатия, маски каналов)",
    4: "BI_JPEG (JPEG внутри BMP)",
    5: "BI_PNG (PNG внутри BMP)",
    6: "BI_ALPHABITFIELDS (без сжатия, маски RGBA)",
    11: "BI_CMYK (без сжатия)",
    12: "BI_CMYKRLE8 (RLE, 8 бит)",
    13: "BI_CMYKRLE4 (RLE, 4 бита)",
}
OS2_COMPRESSION = {3: "Huffman 1D (OS/2)", 4: "RLE24 (OS/2)"}
HEADER_NAMES = {
    12: "BITMAPCOREHEADER", 40: "BITMAPINFOHEADER", 52: "BITMAPV2INFOHEADER",
    56: "BITMAPV3INFOHEADER", 64: "OS22XBITMAPHEADER", 108: "BITMAPV4HEADER",
    124: "BITMAPV5HEADER",
}
VALID_BPP = (0, 1, 2, 4, 8, 16, 24, 32, 48, 64)


def parse(r: BinaryReader, meta: ImageMetadata) -> None:
    r.seek(0)
    _sig, declared_size, _r1, _r2, data_offset = r.unpack("<2sIHHI")
    hdr_size = r.unpack("<I")[0]

    x_ppm = y_ppm = 0
    comp = 0
    img_size = 0
    clr_used = 0
    entry_size = 4
    if hdr_size == 12:  # BITMAPCOREHEADER (OS/2 1.x)
        width, height, planes, bpp = r.unpack("<HHHH")
        entry_size = 3
    elif 40 <= hdr_size <= 124:
        (width, height, planes, bpp, comp, img_size,
         x_ppm, y_ppm, clr_used, _clr_imp) = r.unpack("<iiHHIIiiII")
    else:
        raise CorruptFileError(f"Неизвестный размер DIB-заголовка: {hdr_size}")

    top_down = height < 0
    meta.width = width
    meta.height = abs(height)
    meta.bits_per_pixel = bpp if bpp else None
    if hdr_size == 64 and comp in OS2_COMPRESSION:
        meta.compression = OS2_COMPRESSION[comp]
    else:
        meta.compression = COMPRESSION.get(comp, f"Неизвестный код сжатия {comp}")
    meta.dpi_x = round(abs(x_ppm) * 0.0254, 2) if x_ppm else None
    meta.dpi_y = round(abs(y_ppm) * 0.0254, 2) if y_ppm else None

    # ---- проверки целостности ----
    if width <= 0 or meta.height == 0:
        raise CorruptFileError(f"Недопустимые размеры изображения: {width} × {height}")
    if bpp not in VALID_BPP:
        raise CorruptFileError(f"Недопустимая глубина цвета: {bpp} бит")
    if planes != 1:
        meta.warn(f"Число плоскостей равно {planes}, а должно быть 1")
    if declared_size and r.size < declared_size:
        raise CorruptFileError(
            f"Размер файла ({r.size} байт) меньше указанного в заголовке ({declared_size} байт)"
        )
    if data_offset < 14 + hdr_size or data_offset > r.size:
        raise CorruptFileError(f"Смещение пиксельных данных {data_offset} выходит за пределы файла")

    # Палитра лежит между DIB-заголовком и пикселями.
    colors = 0
    if bpp and bpp <= 8:
        colors = clr_used if clr_used else (1 << bpp)
        if clr_used > (1 << bpp):
            meta.warn(f"В палитре указано {clr_used} цветов, максимум для {bpp} бит — {1 << bpp}")
            colors = 1 << bpp
    masks = 0
    if hdr_size == 40 and comp == 3:
        masks = 12
    elif hdr_size == 40 and comp == 6:
        masks = 16
    palette_end = 14 + hdr_size + masks + colors * entry_size
    if palette_end > r.size:
        raise CorruptFileError("Палитра выходит за пределы файла")
    if palette_end > data_offset:
        meta.warn("Палитра не помещается между заголовком и пиксельными данными")

    rows = meta.height
    if comp in (0, 3, 6, 11) and bpp:
        stride = ((width * bpp + 31) // 32) * 4
        needed = data_offset + stride * (rows - 1) + (width * bpp + 7) // 8
        if needed > r.size:
            raise CorruptFileError(
                f"Пиксельные данные обрезаны: нужно {needed} байт, в файле {r.size}"
            )
    elif img_size and data_offset + img_size > r.size:
        raise CorruptFileError("Сжатые пиксельные данные выходят за пределы файла")

    # ---- дополнительные сведения ----
    meta.add_extra("DIB-заголовок", f"{HEADER_NAMES.get(hdr_size, 'неизвестный')} ({hdr_size} байт)",
                   "Поле biSize сразу после 14-байтного BITMAPFILEHEADER; по нему определяется версия структуры.")
    meta.add_extra("Смещение пикселей", f"{data_offset} байт",
                   "Поле bfOffBits файлового заголовка: где в файле начинается растр.")
    meta.add_extra("Порядок строк", "сверху вниз" if top_down else "снизу вверх",
                   "Знак высоты biHeight: положительная — строки хранятся снизу вверх (обычный случай BMP).")
    if bpp and bpp <= 8:
        meta.add_extra("Цветов в палитре", colors,
                       "Поле biClrUsed; если 0 — палитра полная (2^бит). Каждая запись палитры — 4 байта B, G, R, 0; "
                       "пиксель хранит лишь индекс в этой таблице.")
