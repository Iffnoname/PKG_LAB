"""JPEG: последовательность сегментов «маркер FFxx + длина»; читаем только заголовочную часть до SOS."""
from __future__ import annotations

from models import ImageMetadata
from .errors import CorruptFileError
from .reader import BinaryReader
from .tiff import read_exif_resolution

SOF_NAMES = {
    0xC0: "Baseline DCT, Huffman", 0xC1: "Extended sequential DCT, Huffman",
    0xC2: "Progressive DCT, Huffman", 0xC3: "Lossless, Huffman",
    0xC5: "Differential sequential DCT", 0xC6: "Differential progressive DCT",
    0xC7: "Differential lossless", 0xC9: "Extended sequential DCT, арифметическое кодирование",
    0xCA: "Progressive DCT, арифметическое кодирование", 0xCB: "Lossless, арифметическое кодирование",
    0xCD: "Differential sequential, арифметическое", 0xCE: "Differential progressive, арифметическое",
    0xCF: "Differential lossless, арифметическое",
}
ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
          20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
          58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]


def _subsampling(comps) -> str:
    if len(comps) < 3:
        return "нет (одноканальное)"
    (h0, v0), (h1, v1) = comps[0][1:3], comps[1][1:3]
    if h1 and v1 and h0 % h1 == 0 and v0 % v1 == 0:
        key = (h0 // h1, v0 // v1)
        names = {(1, 1): "4:4:4", (2, 1): "4:2:2", (2, 2): "4:2:0", (4, 1): "4:1:1",
                 (1, 2): "4:4:0", (4, 2): "4:1:0"}
        if key in names:
            return names[key]
    return f"Y {h0}×{v0}, Cb {h1}×{v1}"


def parse(r: BinaryReader, meta: ImageMetadata) -> None:
    r.seek(2)  # после SOI (FF D8)
    comps = []
    precision = 0
    sof = None
    jfif = None          # (единицы, X, Y)
    exif = None          # (x_dpi, y_dpi)
    dqt_count = 0
    dqt_first = None
    dht_count = 0
    restart = 0
    adobe = None

    while True:
        if r.u8() != 0xFF:
            raise CorruptFileError(f"По смещению {r.pos - 1} ожидался маркер (байт FF)")
        m = r.u8()
        while m == 0xFF:  # допустимые байты-заполнители
            m = r.u8()
        if m == 0x00 or m == 0xD8:
            raise CorruptFileError(f"Недопустимый маркер FF{m:02X} в заголовочной части")
        if m == 0xD9:
            raise CorruptFileError("Маркер конца изображения (EOI) встретился до начала данных")
        if 0xD0 <= m <= 0xD7 or m == 0x01:
            continue
        length = r.u16be()
        if length < 2:
            raise CorruptFileError(f"Недопустимая длина сегмента FF{m:02X}: {length}")
        start = r.pos
        end = start + length - 2
        if end > r.size:
            raise CorruptFileError(f"Сегмент FF{m:02X} выходит за пределы файла (файл обрезан)")

        if m in SOF_NAMES:
            if sof is None:
                precision, h, w, n = r.unpack(">BHHB")
                if n == 0 or n > 4 or length != 8 + 3 * n:
                    raise CorruptFileError("Некорректный заголовок кадра SOF")
                raw = r.read(3 * n)
                comps = [(raw[i], raw[i + 1] >> 4, raw[i + 1] & 15, raw[i + 2]) for i in range(0, 3 * n, 3)]
                sof = m
                meta.width, meta.height = w, h
                meta.bits_per_pixel = precision * n
                meta.compression = "JPEG: " + SOF_NAMES[m]
                if w == 0:
                    raise CorruptFileError("Ширина в SOF равна нулю")
                if h == 0:
                    meta.warn("Высота в SOF равна нулю (определяется маркером DNL)")
        elif m == 0xE0 and length >= 16:
            if r.read(5) == b"JFIF\x00":
                _ver, units, xd, yd = r.unpack(">HBHH")
                jfif = (units, xd, yd)
        elif m == 0xE1 and length >= 14:
            if r.read(6) == b"Exif\x00\x00":
                try:
                    exif = read_exif_resolution(r, start + 6)
                except CorruptFileError:
                    exif = None
        elif m == 0xDB:
            pos = start
            while pos < end:
                pq_tq = r.read_at(pos, 1)[0]
                size = 64 * ((pq_tq >> 4) + 1)
                if dqt_first is None and meta.detailed and pos + 1 + size <= end:
                    dqt_first = (pq_tq, r.read(size))
                dqt_count += 1
                pos += 1 + size
        elif m == 0xC4:
            dht_count += 1
        elif m == 0xDD and length == 4:
            restart = r.u16be()
        elif m == 0xEE and length >= 14:
            if r.read(5) == b"Adobe":
                r.skip(6)
                adobe = r.u8()
        elif m == 0xDA:
            break
        r.seek(end)

    if sof is None:
        raise CorruptFileError("Не найден заголовок кадра SOF (нет размеров изображения)")

    # ---- DPI: JFIF приоритетнее, иначе Exif ----
    if jfif and jfif[0] in (1, 2) and jfif[1] and jfif[2]:
        k = 1.0 if jfif[0] == 1 else 2.54
        meta.dpi_x, meta.dpi_y = round(jfif[1] * k, 2), round(jfif[2] * k, 2)
    elif exif:
        meta.dpi_x, meta.dpi_y = exif

    # ---- проверка конца файла: маркер EOI (FF D9) ----
    if r.read_at(r.size - 2, 2) != b"\xff\xd9":
        n = min(r.size, 65536)
        idx = r.read_at(r.size - n, n).rfind(b"\xff\xd9")
        if idx < 0:
            raise CorruptFileError("Отсутствует маркер конца изображения EOI (FF D9): файл обрезан")
        meta.warn(f"После маркера EOI есть лишние данные ({n - idx - 2} байт)")

    # ---- дополнительные сведения ----
    meta.add_extra("Тип кодирования", SOF_NAMES[sof] + f" (маркер FF{sof:02X})",
                   "Маркер SOFn: определяет алгоритм (baseline / progressive) и способ энтропийного кодирования.")
    meta.add_extra("Компонент", len(comps), "Поле Nf в SOF: 1 — серое, 3 — YCbCr/RGB, 4 — CMYK.")
    meta.add_extra("Точность", f"{precision} бит/компонент",
                   "Поле P в SOF; бит/пиксель = точность × число компонентов.")
    if len(comps) >= 3:
        meta.add_extra("Субдискретизация цветности", _subsampling(comps),
                       "Коэффициенты H×V компонентов в SOF: цвет часто хранят с меньшим разрешением, чем яркость.")
    meta.add_extra("Таблиц квантования (DQT)", dqt_count,
                   "Сегменты FFDB: таблицы 8×8, на которые делятся DCT-коэффициенты; чем больше числа — тем сильнее потери.")
    meta.add_extra("Таблиц Хаффмана (DHT)", dht_count, "Сегменты FFC4: коды для энтропийного кодирования.")
    if restart:
        meta.add_extra("Интервал рестартов", f"{restart} MCU", "Сегмент FFDD: через сколько блоков ставятся маркеры RSTn.")
    if jfif:
        meta.add_extra("JFIF", {0: "плотность — только пропорции", 1: "точек на дюйм", 2: "точек на см"}[jfif[0]]
                       if jfif[0] in (0, 1, 2) else f"единицы {jfif[0]}",
                       "Сегмент APP0 «JFIF»: единицы и плотность Xdensity/Ydensity — источник DPI.")
    if exif:
        meta.add_extra("Exif-разрешение", f"{exif[0]} × {exif[1]}",
                       "Сегмент APP1 «Exif»: теги XResolution/YResolution/ResolutionUnit в формате TIFF.")
    if adobe is not None:
        meta.add_extra("Adobe APP14", f"transform={adobe}", "Сегмент FFEE: преобразование цвета (0 — RGB/CMYK, 1 — YCbCr, 2 — YCCK).")
    if dqt_first:
        pq, raw = dqt_first
        step = 2 if pq >> 4 else 1
        vals = [int.from_bytes(raw[i * step:(i + 1) * step], "big") for i in range(64)]
        grid = [0] * 64
        for i, v in enumerate(vals):
            grid[ZIGZAG[i]] = v
        rows = "\n".join(" ".join(f"{v:3d}" for v in grid[i * 8:(i + 1) * 8]) for i in range(8))
        meta.add_extra("Матрица квантования №1", "\n" + rows,
                       "Первая таблица DQT, развёрнута из зигзагообразного порядка в матрицу 8×8 (обычно яркость).")
