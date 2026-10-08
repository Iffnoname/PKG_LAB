
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


STATUS_OK = "OK"
STATUS_WARNING = "Предупреждение"
STATUS_CORRUPT = "Файл поврежден"
STATUS_NOT_IMAGE = "Не изображение"
STATUS_ERROR = "Ошибка чтения"

ExtraItem = Tuple[str, str, str]


@dataclass(slots=True)
class ImageMetadata:
    path: str
    name: str = ""
    file_size: int = 0
    format: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    dpi_x: Optional[float] = None
    dpi_y: Optional[float] = None
    bits_per_pixel: Optional[int] = None
    compression: str = ""
    status: str = STATUS_OK
    message: str = ""
    bytes_read: int = 0
    detailed: bool = False
    warnings: List[str] = field(default_factory=list)
    extras: List[ExtraItem] = field(default_factory=list)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def add_extra(self, name: str, value, description: str) -> None:
        if self.detailed:
            self.extras.append((name, str(value), description))

    @property
    def size_text(self) -> str:
        if self.width is None or self.height is None:
            return "—"
        return f"{self.width} × {self.height}"

    @staticmethod
    def format_dpi(value: Optional[float]) -> str:
        if value is None:
            return "не указано"
        if abs(value - round(value)) < 0.01:
            return str(int(round(value)))
        return f"{value:.2f}"

    @property
    def dpi_text(self) -> str:
        if self.dpi_x is None and self.dpi_y is None:
            return "не указано"
        return f"{self.format_dpi(self.dpi_x)} × {self.format_dpi(self.dpi_y)}"

    @property
    def bpp_text(self) -> str:
        return "—" if self.bits_per_pixel is None else str(self.bits_per_pixel)

    @property
    def is_problem(self) -> bool:
        return self.status != STATUS_OK
