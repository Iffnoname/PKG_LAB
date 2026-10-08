"""Выбор парсера по сигнатуре файла и общая точка входа inspect_file()."""
from __future__ import annotations

import os
import struct
from typing import Optional

from models import (ImageMetadata, STATUS_CORRUPT, STATUS_ERROR, STATUS_NOT_IMAGE, STATUS_OK,
                    STATUS_WARNING)
from . import bmp, gif, jpeg, pcx, png, tiff
from .errors import CorruptFileError
from .reader import BinaryReader

PARSERS = {"BMP": bmp.parse, "PNG": png.parse, "JPEG": jpeg.parse, "GIF": gif.parse,
           "TIFF": tiff.parse, "PCX": pcx.parse}

FORMAT_BY_EXT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".jpe": "JPEG", ".jfif": "JPEG",
    ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF",
    ".bmp": "BMP", ".dib": "BMP", ".png": "PNG", ".pcx": "PCX",
}


def detect_format(head: bytes) -> Optional[str]:
    """Формат по первым байтам (сигнатуре), а не по расширению."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "TIFF"
    if head[:2] == b"BM":
        return "BMP"
    if len(head) >= 4 and head[0] == 0x0A and head[1] in (0, 2, 3, 4, 5) and head[2] in (0, 1) \
            and head[3] in (1, 2, 4, 8):
        return "PCX"
    return None


def inspect_file(path: str, detailed: bool = False) -> ImageMetadata:
    """Читает метаданные одного файла. Никогда не бросает исключений: ошибки превращаются в статус."""
    meta = ImageMetadata(path=path, name=os.path.basename(path), detailed=detailed)
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            meta.file_size = size
            r = BinaryReader(f, size)
            try:
                if size == 0:
                    raise CorruptFileError("Пустой файл (0 байт)")
                fmt = detect_format(r.read(min(size, 16)))
                if fmt is None:
                    meta.status = STATUS_NOT_IMAGE
                    expected = FORMAT_BY_EXT.get(ext)
                    meta.message = (f"Сигнатура {expected} не найдена: файл не является изображением "
                                    f"или повреждён (расширение подменено)" if expected
                                    else "Сигнатура не соответствует ни одному из поддерживаемых форматов")
                    return meta
                meta.format = fmt
                if ext in FORMAT_BY_EXT and FORMAT_BY_EXT[ext] != fmt:
                    meta.warn(f"Расширение {ext} не соответствует содержимому ({fmt})")
                PARSERS[fmt](r, meta)
                if meta.warnings:
                    meta.status = STATUS_WARNING
                    meta.message = "; ".join(meta.warnings)
            except CorruptFileError as e:
                meta.status = STATUS_CORRUPT
                meta.message = "; ".join(meta.warnings + [str(e)])
            except (struct.error, ValueError, IndexError, KeyError, OverflowError, TypeError, AttributeError) as e:
                # Страховка: любое неожиданное содержимое не должно ронять программу.
                meta.status = STATUS_CORRUPT
                meta.message = f"Структура файла не поддаётся разбору ({type(e).__name__}: {e})"
            finally:
                meta.bytes_read = r.bytes_read
    except OSError as e:
        meta.status = STATUS_ERROR
        meta.message = f"Не удалось прочитать файл: {e.strerror or e}"
    except MemoryError:
        meta.status = STATUS_ERROR
        meta.message = "Недостаточно памяти"
    if not meta.name:
        meta.name = path
    return meta
