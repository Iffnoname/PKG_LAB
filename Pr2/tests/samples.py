"""Генерация тестовых файлов. Pillow используется ТОЛЬКО здесь, для создания образцов и сверки."""
from __future__ import annotations

import os
import struct

from PIL import Image


def gradient(w, h, mode="RGB"):
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = ((x * 5) % 256, (y * 7) % 256, (x + y) % 256)
    return im.convert(mode) if mode != "RGB" else im


def pillow_samples(d):
    """-> список (путь, режим Pillow) корректных файлов разных форматов и вариантов."""
    out = []

    def save(name, im, **kw):
        p = os.path.join(d, name)
        im.save(p, **kw)
        out.append(p)

    for mode in ("1", "L", "P", "RGB", "RGBA"):
        im = gradient(61, 37, mode)
        save(f"png_{mode}.png", im, dpi=(300, 300))
        if mode != "RGBA":
            save(f"bmp_{mode}.bmp", im, dpi=(96, 96))
    save("png_16.png", Image.new("I;16", (20, 10), 1000))
    save("png_nodpi.png", gradient(33, 21))
    save("jpg_rgb.jpg", gradient(97, 53), dpi=(200, 100), quality=80)
    save("jpg_gray.jpg", gradient(50, 50, "L"), dpi=(72, 72))
    save("jpg_prog.jpg", gradient(80, 60), progressive=True, dpi=(150, 150))
    save("jpg_nodpi.jpg", gradient(33, 21), subsampling=0)
    save("jpg_cmyk.jpg", gradient(30, 30, "CMYK"))
    ex = Image.Exif()
    ex[0x011A], ex[0x011B], ex[0x0128] = 240.0, 240.0, 2
    save("jpg_exif.jpg", gradient(64, 48), exif=ex)
    save("gif_p.gif", gradient(70, 40, "P"))
    save("gif_bw.gif", gradient(70, 40, "1").convert("P"))
    frames = [gradient(30, 30, "P"), Image.new("P", (30, 30), 3), Image.new("P", (30, 30), 5)]
    p = os.path.join(d, "gif_anim.gif")
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=50, loop=0)
    out.append(p)
    for comp in (None, "tiff_lzw", "packbits", "tiff_adobe_deflate", "jpeg"):
        save(f"tif_{comp}.tif", gradient(75, 45), dpi=(72, 72), compression=comp)
    save("tif_gray.tif", gradient(75, 45, "L"), dpi=(150, 300))
    save("tif_bw_g4.tif", gradient(75, 45, "1"), compression="group4")
    save("tif_rgba.tif", gradient(75, 45, "RGBA"))
    pages = [gradient(40, 40), gradient(20, 20), gradient(10, 10)]
    p = os.path.join(d, "tif_pages.tif")
    pages[0].save(p, save_all=True, append_images=pages[1:])
    out.append(p)
    save("pcx_rgb.pcx", gradient(61, 37), dpi=(150, 150))
    save("pcx_p.pcx", gradient(61, 37, "P"))
    save("pcx_gray.pcx", gradient(61, 37, "L"))
    return out


def bmp_rle8(path, w=16, h=8):
    """BMP с BI_RLE8: каждая строка — одна серия (w пикселей индекса 5) + конец строки."""
    palette = b"".join(bytes([i, i, i, 0]) for i in range(256))
    data = b""
    for _ in range(h):
        data += bytes([w, 5, 0, 0])
    data += bytes([0, 1])
    off = 14 + 40 + len(palette)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 8, 1, len(data), 3780, 3780, 256, 0)
    with open(path, "wb") as f:
        f.write(b"BM" + struct.pack("<IHHI", off + len(data), 0, 0, off) + dib + palette + data)


def tiff_be(path, w=10, h=6):
    """Big-endian TIFF, без сжатия, 8-бит серый, 300 dpi."""
    pix = bytes(range(w * h))
    entries = [
        (256, 3, 1, struct.pack(">HH", w, 0)), (257, 3, 1, struct.pack(">HH", h, 0)),
        (258, 3, 1, struct.pack(">HH", 8, 0)), (259, 3, 1, struct.pack(">HH", 1, 0)),
        (262, 3, 1, struct.pack(">HH", 1, 0)), (273, 4, 1, struct.pack(">I", 8)),
        (277, 3, 1, struct.pack(">HH", 1, 0)), (279, 4, 1, struct.pack(">I", len(pix))),
        (282, 5, 1, None), (283, 5, 1, None), (296, 3, 1, struct.pack(">HH", 2, 0)),
    ]
    ifd_off = 8 + len(pix)
    res_off = ifd_off + 2 + 12 * len(entries) + 4
    body = struct.pack(">H", len(entries))
    for tag, typ, cnt, val in entries:
        if val is None:
            val = struct.pack(">I", res_off if tag == 282 else res_off + 8)
        body += struct.pack(">HHI", tag, typ, cnt) + val
    body += struct.pack(">I", 0) + struct.pack(">II", 300, 1) + struct.pack(">II", 300, 1)
    with open(path, "wb") as f:
        f.write(b"MM\x00*" + struct.pack(">I", ifd_off) + pix + body)


def bigtiff(path, w=12, h=5):
    pix = bytes(w * h)
    entries = [
        (256, 4, 1, struct.pack("<I4x", w)), (257, 4, 1, struct.pack("<I4x", h)),
        (258, 3, 1, struct.pack("<H6x", 8)), (259, 3, 1, struct.pack("<H6x", 5)),
        (262, 3, 1, struct.pack("<H6x", 1)), (273, 16, 1, struct.pack("<Q", 16)),
        (277, 3, 1, struct.pack("<H6x", 1)), (279, 16, 1, struct.pack("<Q", len(pix))),
    ]
    ifd_off = 16 + len(pix)
    body = struct.pack("<Q", len(entries))
    for tag, typ, cnt, val in entries:
        body += struct.pack("<HHQ", tag, typ, cnt) + val
    body += struct.pack("<Q", 0)
    with open(path, "wb") as f:
        f.write(b"II+\x00" + struct.pack("<HHQ", 8, 0, ifd_off) + pix + body)


def tiff_cycle(path):
    """Корректная первая страница, но следующий IFD указывает на самого себя."""
    pix = bytes(20)
    entries = [(256, 3, 1, 4), (257, 3, 1, 5), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1),
               (273, 4, 1, 8), (277, 3, 1, 1), (279, 4, 1, 20)]
    ifd_off = 8 + len(pix)
    body = struct.pack("<H", len(entries))
    for tag, typ, cnt, v in entries:
        body += struct.pack("<HHI", tag, typ, cnt) + (struct.pack("<I", v) if typ == 4 else struct.pack("<HH", v, 0))
    body += struct.pack("<I", ifd_off)
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_off) + pix + body)


def fake_jpeg_header(path, w, h, size, dpi=96):
    """«Разреженный» JPEG: настоящий заголовок, ноль-байты вместо данных и EOI в конце. Занимает ~0 на диске."""
    jfif = b"JFIF\x00" + struct.pack(">HBHH", 0x0101, 1, dpi, dpi) + b"\x00\x00"
    sof = struct.pack(">BHHB", 8, h, w, 3) + bytes([1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1])
    sos = bytes([3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0])
    head = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", len(jfif) + 2) + jfif
            + b"\xff\xdb" + struct.pack(">H", 67) + b"\x00" + bytes(range(1, 65))
            + b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
            + b"\xff\xda" + struct.pack(">H", len(sos) + 2) + sos)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(size)
        f.seek(size - 2)
        f.write(b"\xff\xd9")
