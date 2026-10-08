"""Безопасное двоичное чтение: все выходы за границы файла превращаются в CorruptFileError."""
from __future__ import annotations

import struct

from .errors import CorruptFileError


class BinaryReader:
    """Обёртка над файловым объектом.

    * read/seek проверяют границы, поэтому «битый» файл не приводит к падению;
    * seek ленивый: реальный системный seek выполняется только перед чтением;
    * bytes_read — сколько байт реально запросили у файла (для статистики).
    """

    __slots__ = ("_f", "size", "pos", "_fpos", "bytes_read")

    def __init__(self, f, size: int) -> None:
        self._f = f
        self.size = size
        self.pos = 0
        self._fpos = 0
        self.bytes_read = 0

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > self.size:
            raise CorruptFileError(
                f"Неожиданный конец файла: по смещению {self.pos} нужно {n} байт, "
                f"а размер файла {self.size} байт"
            )
        if n == 0:
            return b""
        if self._fpos != self.pos:
            self._f.seek(self.pos)
        data = self._f.read(n)
        if len(data) != n:
            raise CorruptFileError("Файл изменился или оборван во время чтения")
        self.pos += n
        self._fpos = self.pos
        self.bytes_read += n
        return data

    def seek(self, offset: int) -> None:
        if offset < 0:
            raise CorruptFileError(f"Отрицательное смещение {offset}")
        self.pos = offset

    def skip(self, n: int) -> None:
        self.seek(self.pos + n)

    def read_at(self, offset: int, n: int) -> bytes:
        self.seek(offset)
        return self.read(n)

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def u8(self) -> int:
        return self.read(1)[0]

    def u16be(self) -> int:
        return struct.unpack(">H", self.read(2))[0]
