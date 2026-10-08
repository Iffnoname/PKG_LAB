"""GIF: заголовок + логический экран + (глобальная палитра) + блоки (расширения / кадры) + трейлер 3B."""
from __future__ import annotations

from models import ImageMetadata
from .errors import CorruptFileError
from .reader import BinaryReader


def _skip_subblocks(r: BinaryReader, keep: bool = False) -> bytes:
    """Пропускает цепочку подблоков (длина, данные ... 0). keep=True — вернуть данные."""
    out = []
    while True:
        n = r.u8()
        if n == 0:
            return b"".join(out)
        if keep:
            out.append(r.read(n))
        else:
            r.skip(n)
            if r.pos > r.size:
                raise CorruptFileError("Блок данных GIF выходит за пределы файла")


def parse(r: BinaryReader, meta: ImageMetadata) -> None:
    r.seek(0)
    version = r.read(6)[3:].decode("ascii", "replace")
    width, height, packed, _bg, aspect = r.unpack("<HHBBB")
    meta.width, meta.height = width, height
    meta.compression = "LZW"
    if width == 0 or height == 0:
        raise CorruptFileError("Нулевой размер логического экрана")

    has_gct = bool(packed & 0x80)
    gct_bits = (packed & 7) + 1
    color_res = ((packed >> 4) & 7) + 1
    if has_gct:
        r.skip(3 << gct_bits)
        if r.pos > r.size:
            raise CorruptFileError("Глобальная палитра выходит за пределы файла")

    frames = 0
    first_bits = None
    first_local = False
    first_interlace = False
    transparent = False
    loop = None
    comments = 0
    out_of_screen = False

    while True:
        b = r.u8()
        if b == 0x3B:  # трейлер
            break
        if b == 0x21:  # расширение
            label = r.u8()
            if label == 0xF9:
                data = _skip_subblocks(r, keep=True)
                if len(data) >= 4 and data[0] & 1:
                    transparent = True
            elif label == 0xFF:
                data = _skip_subblocks(r, keep=True)
                if data[:11] == b"NETSCAPE2.0" and len(data) >= 14:
                    loop = data[12] | (data[13] << 8)
            else:
                comments += label == 0xFE
                _skip_subblocks(r)
        elif b == 0x2C:  # кадр
            left, top, fw, fh, pk = r.unpack("<HHHHB")
            local = bool(pk & 0x80)
            lbits = (pk & 7) + 1
            if local:
                r.skip(3 << lbits)
                if r.pos > r.size:
                    raise CorruptFileError("Локальная палитра выходит за пределы файла")
            r.u8()  # LZW minimum code size
            _skip_subblocks(r)
            if frames == 0:
                first_local = local
                first_interlace = bool(pk & 0x40)
                first_bits = lbits if local else (gct_bits if has_gct else color_res)
            if left + fw > width or top + fh > height:
                out_of_screen = True
            frames += 1
        elif b == 0x00:
            continue
        else:
            raise CorruptFileError(f"Неизвестный блок 0x{b:02X} по смещению {r.pos - 1}")

    if frames == 0:
        raise CorruptFileError("В файле нет ни одного кадра")
    meta.bits_per_pixel = first_bits
    if frames > 1:
        meta.format = f"GIF ({frames} кадр.)"
    if out_of_screen:
        meta.warn("Кадр выходит за границы логического экрана")
    if r.pos < r.size:
        meta.warn(f"После трейлера есть лишние данные ({r.size - r.pos} байт)")

    meta.add_extra("Версия", f"GIF{version}", "Первые 6 байт файла: GIF87a или GIF89a (89a добавляет анимацию и прозрачность).")
    meta.add_extra("Кадров", frames, "Количество блоков «Image Descriptor» (0x2C).")
    meta.add_extra("Глобальная палитра", f"{1 << gct_bits} цветов" if has_gct else "нет",
                   "Флаг и размер в байте «Packed» логического экрана: размер = 2^(N+1) записей по 3 байта.")
    meta.add_extra("Палитра 1-го кадра", "локальная" if first_local else "глобальная",
                   "Если у кадра есть локальная таблица, она заменяет глобальную.")
    meta.add_extra("Прозрачность", "есть" if transparent else "нет", "Флаг в блоке Graphic Control Extension (0x21 0xF9).")
    meta.add_extra("Чересстрочность 1-го кадра", "да" if first_interlace else "нет", "Бит Interlace в байте Packed дескриптора кадра.")
    if loop is not None:
        meta.add_extra("Повторов анимации", "бесконечно" if loop == 0 else loop, "Расширение NETSCAPE2.0.")
    if aspect:
        meta.add_extra("Соотношение сторон пикселя", f"{(aspect + 15) / 64:.2f}", "Поле Pixel Aspect Ratio: (значение + 15) / 64.")
    meta.add_extra("DPI", "не хранится", "В формате GIF нет поля физического разрешения.")
