import os
import random
import shutil
import tempfile
import unittest

from PIL import Image

from models import (STATUS_CORRUPT, STATUS_NOT_IMAGE, STATUS_OK, STATUS_WARNING)
from parsers import inspect_file
from . import samples

def rb(p):
    with open(p, "rb") as f:
        return f.read()


BPP = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "I;16": 16, "LA": 16, "I": 32}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="lab2_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def write(self, name, data):
        p = self.path(name)
        with open(p, "wb") as f:
            f.write(data)
        return p


class CrossCheckWithPillow(Base):
    """Корректные файлы: результат ручных парсеров совпадает с эталоном Pillow."""

    def test_all_samples(self):
        d = os.path.join(self.dir, "ok")
        os.mkdir(d)
        files = samples.pillow_samples(d)
        self.assertGreater(len(files), 30)
        for p in files:
            with self.subTest(file=os.path.basename(p)):
                m = inspect_file(p, detailed=True)
                self.assertIn(m.status, (STATUS_OK, STATUS_WARNING), m.message)
                with Image.open(p) as im:
                    self.assertEqual((m.width, m.height), im.size)
                    ref_bpp = BPP[im.mode]
                    if p.endswith(".gif"):
                        ref_bpp = None  # у GIF глубина = размер палитры, сверяем отдельно
                    if p.endswith(".pcx") and im.mode == "P":
                        ref_bpp = 8
                    if ref_bpp is not None:
                        # Pillow конвертирует JPEG-CMYK/TIFF-JPEG в иные режимы — допускаем только известные
                        self.assertEqual(m.bits_per_pixel, ref_bpp, f"{im.mode}")
                    dpi = im.info.get("dpi")
                    if p.endswith((".tif", ".tiff")) and 282 not in im.tag_v2:
                        dpi = None  # Pillow подставляет (1, 1), хотя в файле тегов разрешения нет
                    if dpi and not p.endswith(".pcx"):
                        self.assertIsNotNone(m.dpi_x, p)
                        self.assertAlmostEqual(m.dpi_x, float(dpi[0]), delta=1.0)
                        self.assertAlmostEqual(m.dpi_y, float(dpi[1]), delta=1.0)
                    elif not p.endswith((".pcx", ".bmp")):
                        self.assertIsNone(m.dpi_x, p)

    def test_png_fields(self):
        m = inspect_file(self.path("ok/png_RGBA.png"), detailed=True)
        self.assertEqual((m.format, m.bits_per_pixel, m.dpi_x), ("PNG", 32, 300))
        self.assertIn("Deflate", m.compression)

    def test_compression_names(self):
        expect = {"tif_None.tif": "Без сжатия", "tif_tiff_lzw.tif": "LZW", "tif_packbits.tif": "PackBits",
                  "tif_tiff_adobe_deflate.tif": "Deflate", "tif_jpeg.tif": "JPEG", "tif_bw_g4.tif": "Group 4"}
        for f, text in expect.items():
            with self.subTest(f=f):
                self.assertIn(text, inspect_file(self.path("ok/" + f)).compression)

    def test_bmp_dpi_and_compression(self):
        m = inspect_file(self.path("ok/bmp_RGB.bmp"))
        self.assertIn("BI_RGB", m.compression)
        self.assertAlmostEqual(m.dpi_x, 96, delta=0.1)

    def test_jpeg_variants(self):
        self.assertIn("Progressive", inspect_file(self.path("ok/jpg_prog.jpg")).compression)
        self.assertIn("Baseline", inspect_file(self.path("ok/jpg_rgb.jpg")).compression)
        self.assertEqual(inspect_file(self.path("ok/jpg_gray.jpg")).bits_per_pixel, 8)
        m = inspect_file(self.path("ok/jpg_exif.jpg"))
        self.assertEqual((m.dpi_x, m.dpi_y), (240, 240))   # Exif, без JFIF-плотности
        m = inspect_file(self.path("ok/jpg_rgb.jpg"))
        self.assertEqual((m.dpi_x, m.dpi_y), (200, 100))
        self.assertIsNone(inspect_file(self.path("ok/jpg_nodpi.jpg")).dpi_x)

    def test_jpeg_quant_matrix_detailed(self):
        m = inspect_file(self.path("ok/jpg_rgb.jpg"), detailed=True)
        names = [e[0] for e in m.extras]
        self.assertIn("Матрица квантования №1", names)
        self.assertEqual(inspect_file(self.path("ok/jpg_rgb.jpg")).extras, [])

    def test_gif(self):
        m = inspect_file(self.path("ok/gif_anim.gif"))
        with Image.open(self.path("ok/gif_anim.gif")) as ref:
            self.assertEqual(m.format, f"GIF ({ref.n_frames} кадр.)")
        self.assertGreater(ref.n_frames, 1)
        self.assertEqual(m.compression, "LZW")
        self.assertIsNone(m.dpi_x)

    def test_tiff_pages(self):
        m = inspect_file(self.path("ok/tif_pages.tif"))
        self.assertEqual((m.format, m.width, m.height), ("TIFF (3 стр.)", 40, 40))

    def test_pcx(self):
        m = inspect_file(self.path("ok/pcx_rgb.pcx"))
        self.assertEqual((m.width, m.height, m.bits_per_pixel), (61, 37, 24))
        with open(self.path("ok/pcx_rgb.pcx"), "rb") as f:
            hdr = f.read(16)
        self.assertEqual(m.dpi_x, int.from_bytes(hdr[12:14], "little"))
        self.assertEqual(m.compression, "RLE (PCX)")
        self.assertEqual(inspect_file(self.path("ok/pcx_p.pcx")).bits_per_pixel, 8)


class HandcraftedFormats(Base):
    def test_bmp_rle8(self):
        p = self.path("rle8.bmp")
        samples.bmp_rle8(p, 16, 8)
        m = inspect_file(p)
        self.assertEqual(m.status, STATUS_OK, m.message)
        self.assertEqual((m.width, m.height, m.bits_per_pixel), (16, 8, 8))
        self.assertIn("BI_RLE8", m.compression)
        self.assertAlmostEqual(m.dpi_x, 96, delta=0.1)

    def test_tiff_big_endian(self):
        p = self.path("be.tif")
        samples.tiff_be(p)
        m = inspect_file(p, detailed=True)
        self.assertEqual(m.status, STATUS_OK, m.message)
        self.assertEqual((m.width, m.height, m.bits_per_pixel, m.dpi_x), (10, 6, 8, 300))
        self.assertTrue(any("big-endian" in e[1] for e in m.extras))

    def test_bigtiff(self):
        p = self.path("big.tif")
        samples.bigtiff(p)
        m = inspect_file(p)
        self.assertEqual(m.status, STATUS_OK, m.message)
        self.assertEqual((m.width, m.height, m.bits_per_pixel), (12, 5, 8))
        self.assertEqual(m.compression, "LZW")

    def test_tiff_ifd_cycle(self):
        p = self.path("cycle.tif")
        samples.tiff_cycle(p)
        m = inspect_file(p)
        self.assertEqual(m.status, STATUS_CORRUPT)
        self.assertIn("зациклена", m.message)
        self.assertEqual((m.width, m.height), (4, 5))   # данные первой страницы сохранены


class CorruptFiles(Base):
    def test_text_renamed_to_jpg(self):
        m = inspect_file(self.write("fake.jpg", b"Hello, this is a text file\n" * 10))
        self.assertEqual(m.status, STATUS_NOT_IMAGE)

    def test_empty_file(self):
        m = inspect_file(self.write("empty.png", b""))
        self.assertEqual(m.status, STATUS_CORRUPT)

    def test_missing_file(self):
        m = inspect_file(self.path("no_such.png"))
        self.assertNotEqual(m.status, STATUS_OK)

    def test_png_without_iend(self):
        d = os.path.join(self.dir, "c1"); os.mkdir(d)
        p = samples.pillow_samples(d)[0]
        data = rb(p)
        m = inspect_file(self.write("noiend.png", data[:-12]))
        self.assertEqual(m.status, STATUS_CORRUPT)
        self.assertIn("IEND", m.message)
        self.assertEqual(inspect_file(self.write("trunc.png", data[: len(data) // 2])).status, STATUS_CORRUPT)
        m = inspect_file(self.write("trail.png", data + b"JUNK"))
        self.assertEqual(m.status, STATUS_WARNING)

    def test_png_bad_crc(self):
        d = os.path.join(self.dir, "c2"); os.mkdir(d)
        data = bytearray(rb(samples.pillow_samples(d)[0]))
        data[20] ^= 0xFF  # внутри IHDR
        self.assertEqual(inspect_file(self.write("crc.png", bytes(data))).status, STATUS_CORRUPT)

    def test_jpeg_without_eoi(self):
        p = self.path("sparse.jpg")
        samples.fake_jpeg_header(p, 100, 50, 5000)
        data = rb(p)
        self.assertEqual(inspect_file(p).status, STATUS_OK)
        m = inspect_file(self.write("noeoi.jpg", data[:-2]))
        self.assertEqual(m.status, STATUS_CORRUPT)
        self.assertIn("EOI", m.message)
        self.assertEqual((m.width, m.height), (100, 50))
        m = inspect_file(self.write("padded.jpg", data + b"\x00" * 10))
        self.assertEqual(m.status, STATUS_WARNING)

    def test_jpeg_cut_in_header(self):
        p = self.path("sparse2.jpg")
        samples.fake_jpeg_header(p, 100, 50, 5000)
        data = rb(p)
        for cut in (3, 10, 30, 100):
            self.assertEqual(inspect_file(self.write("cut.jpg", data[:cut])).status, STATUS_CORRUPT)

    def test_bmp_truncated(self):
        d = os.path.join(self.dir, "c3"); os.mkdir(d)
        p = [x for x in samples.pillow_samples(d) if x.endswith("bmp_RGB.bmp")][0]
        data = rb(p)
        m = inspect_file(self.write("t.bmp", data[: len(data) - 500]))
        self.assertEqual(m.status, STATUS_CORRUPT)
        self.assertIn("меньше", m.message)

    def test_gif_without_trailer(self):
        d = os.path.join(self.dir, "c4"); os.mkdir(d)
        p = [x for x in samples.pillow_samples(d) if x.endswith("gif_p.gif")][0]
        data = rb(p)
        self.assertEqual(inspect_file(self.write("nt.gif", data[:-1])).status, STATUS_CORRUPT)

    def test_pcx_truncated(self):
        d = os.path.join(self.dir, "c5"); os.mkdir(d)
        p = [x for x in samples.pillow_samples(d) if x.endswith("pcx_rgb.pcx")][0]
        data = rb(p)
        self.assertEqual(inspect_file(self.write("t.pcx", data[:140])).status, STATUS_CORRUPT)

    def test_tiff_truncated_strip(self):
        d = os.path.join(self.dir, "c6"); os.mkdir(d)
        p = [x for x in samples.pillow_samples(d) if x.endswith("tif_None.tif")][0]
        data = rb(p)
        self.assertEqual(inspect_file(self.write("t.tif", data[:300])).status, STATUS_CORRUPT)

    def test_extension_mismatch(self):
        d = os.path.join(self.dir, "c7"); os.mkdir(d)
        data = rb(samples.pillow_samples(d)[0])   # PNG
        m = inspect_file(self.write("really_png.jpg", data))
        self.assertEqual(m.status, STATUS_WARNING)
        self.assertEqual(m.format, "PNG")
        self.assertIn("не соответствует", m.message)
        self.assertEqual(m.width, 61)


class Fuzz(Base):
    """Любые обрезки и случайные повреждения не должны вызывать исключений."""

    def test_truncation_and_bitflips(self):
        d = os.path.join(self.dir, "fz"); os.mkdir(d)
        files = samples.pillow_samples(d)
        for n, fn in (("rle8.bmp", samples.bmp_rle8), ("be.tif", samples.tiff_be),
                      ("big.tif", samples.bigtiff), ("cyc.tif", samples.tiff_cycle)):
            p = os.path.join(d, n); fn(p); files.append(p)
        rnd = random.Random(1234)
        checked = 0
        for p in files:
            data = rb(p)
            ext = os.path.splitext(p)[1]
            cuts = sorted(set(list(range(0, min(len(data), 200))) +
                              [rnd.randrange(len(data)) for _ in range(25)]))
            for cut in cuts:
                m = inspect_file(self.write("fz" + ext, data[:cut]), detailed=True)
                self.assertTrue(m.status)
                checked += 1
            for _ in range(60):
                b = bytearray(data)
                for _ in range(rnd.randint(1, 8)):
                    b[rnd.randrange(len(b))] = rnd.randrange(256)
                m = inspect_file(self.write("fz" + ext, bytes(b)), detailed=True)
                self.assertTrue(m.status)
                checked += 1
        self.assertGreater(checked, 5000)

    def test_random_garbage_with_signatures(self):
        rnd = random.Random(7)
        sigs = [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF89a", b"II*\x00", b"MM\x00*", b"II+\x00", b"BM",
                b"\x0a\x05\x01\x08"]
        for i in range(600):
            data = rnd.choice(sigs) + bytes(rnd.randrange(256) for _ in range(rnd.randrange(0, 300)))
            m = inspect_file(self.write("g.bin", data))
            self.assertTrue(m.status)


if __name__ == "__main__":
    unittest.main()
