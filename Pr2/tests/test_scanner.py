import os
import shutil
import tempfile
import threading
import time
import unittest

from engine import scanner
from engine.scanner import run_scan
from models import STATUS_CORRUPT, STATUS_NOT_IMAGE, STATUS_OK
from . import samples


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="lab2s_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def collect(self, sources, **kw):
        rows, progress = [], []
        s = run_scan(sources, on_batch=rows.extend, on_progress=lambda d, t: progress.append((d, t)), **kw)
        return s, rows, progress

    def test_recursive_scan_and_filters(self):
        sub = os.path.join(self.root, "sub", "deep")
        os.makedirs(sub)
        os.makedirs(os.path.join(self.root, ".hidden"))
        samples.pillow_samples(self.root)
        base = len([f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))])
        samples.fake_jpeg_header(os.path.join(sub, "x.jpg"), 10, 10, 1000)
        samples.fake_jpeg_header(os.path.join(self.root, ".hidden", "h.jpg"), 10, 10, 1000)
        samples.fake_jpeg_header(os.path.join(self.root, ".dot.jpg"), 10, 10, 1000)
        with open(os.path.join(self.root, "notes.txt"), "w") as f:
            f.write("not an image")
        with open(os.path.join(self.root, "fake.png"), "w") as f:
            f.write("text renamed to png")
        if hasattr(os, "symlink"):
            os.symlink(sub, os.path.join(self.root, "link_dir"))
            os.symlink(os.path.join(sub, "x.jpg"), os.path.join(self.root, "link.jpg"))
        s, rows, progress = self.collect([self.root])
        names = {r.name for r in rows}
        self.assertIn("x.jpg", names)
        self.assertNotIn("h.jpg", names)         # скрытая папка
        self.assertNotIn(".dot.jpg", names)      # скрытый файл
        self.assertNotIn("link.jpg", names)      # символическая ссылка
        self.assertNotIn("notes.txt", names)
        self.assertEqual(sum(r.name == "x.jpg" for r in rows), 1)  # не удвоилось через link_dir
        self.assertEqual(s.total_found, base + 2)    # + x.jpg + fake.png
        self.assertEqual(s.processed, len(rows))
        self.assertEqual(progress[-1], (s.processed, s.total_found))
        fake = [r for r in rows if r.name == "fake.png"][0]
        self.assertEqual(fake.status, STATUS_NOT_IMAGE)
        self.assertEqual(s.ok + s.warnings + s.corrupt + s.other, s.processed)

    def test_explicit_files(self):
        p = os.path.join(self.root, "a.jpg")
        samples.fake_jpeg_header(p, 20, 10, 800)
        t = os.path.join(self.root, "b.dat")
        with open(t, "wb") as f:
            f.write(b"xx")
        s, rows, _ = self.collect([p, t])
        self.assertEqual([r.name for r in rows], ["a.jpg", "b.dat"])

    def test_limit(self):
        for i in range(30):
            samples.fake_jpeg_header(os.path.join(self.root, f"{i:03d}.jpg"), 8, 8, 600)
        s, rows, _ = self.collect([self.root], limit=10)
        self.assertEqual(len(rows), 10)
        self.assertTrue(s.limit_reached)
        s, rows, _ = self.collect([self.root], limit=30)
        self.assertFalse(s.limit_reached)

    def test_order_is_stable_with_threads(self):
        for i in range(300):
            samples.fake_jpeg_header(os.path.join(self.root, f"{i:04d}.jpg"), 8 + i, 8, 600)
        s, rows, _ = self.collect([self.root], workers=4, chunk_size=7)
        self.assertEqual([r.name for r in rows], [f"{i:04d}.jpg" for i in range(300)])
        self.assertEqual([r.width for r in rows], [8 + i for i in range(300)])

    def test_worker_threads_used(self):
        seen = set()
        orig = scanner.inspect_file

        def spy(p):
            seen.add(threading.current_thread().name)
            time.sleep(0.002)
            return orig(p)

        scanner.inspect_file = spy
        try:
            for i in range(200):
                samples.fake_jpeg_header(os.path.join(self.root, f"{i}.jpg"), 8, 8, 600)
            self.collect([self.root], workers=4, chunk_size=5)
        finally:
            scanner.inspect_file = orig
        self.assertGreaterEqual(len(seen), 2)
        self.assertTrue(all(n.startswith("meta") for n in seen))
        self.assertNotIn("MainThread", seen)

    def test_cancel_keeps_partial_results(self):
        for i in range(2000):
            samples.fake_jpeg_header(os.path.join(self.root, f"{i:05d}.jpg"), 8, 8, 600)
        cancel = threading.Event()
        rows = []

        def on_batch(b):
            rows.extend(b)
            if len(rows) >= 100:
                cancel.set()

        s = run_scan([self.root], cancel=cancel, on_batch=on_batch, chunk_size=10)
        self.assertTrue(s.cancelled)
        self.assertGreaterEqual(len(rows), 100)
        self.assertLess(len(rows), 2000)
        self.assertEqual(s.processed, len(rows))
        self.assertTrue(all(r.status == STATUS_OK for r in rows))

    def test_mixed_folder_never_crashes(self):
        d = os.path.join(self.root, "mix")
        os.mkdir(d)
        for p in samples.pillow_samples(d):
            with open(p, "rb") as f:
                data = f.read()
            with open(p, "wb") as f:           # половина файлов — обрезки
                f.write(data[: len(data) // 2] if hash(p) % 2 else data)
        s, rows, _ = self.collect([self.root])
        self.assertEqual(s.processed, s.total_found)
        self.assertGreater(s.corrupt, 0)

    def test_stress_600_large_files(self):
        """600 «файлов по 4 МБ» (разреженные, >2 ГБ логически): читаются только заголовки."""
        d = os.path.join(self.root, "big")
        os.mkdir(d)
        size = 4 * 1024 * 1024
        for i in range(600):
            samples.fake_jpeg_header(os.path.join(d, f"IMG_{i:04d}.jpg"), 4000 + i, 3000, size)
        t = time.perf_counter()
        s, rows, _ = self.collect([d])
        dt = time.perf_counter() - t
        print(f"\n[stress] {s.processed} файлов, логический объём {s.bytes_total / 2**30:.2f} ГБ, "
              f"прочитано {s.bytes_read} байт, {dt:.2f} с")
        self.assertEqual(s.processed, 600)
        self.assertEqual(s.ok, 600)
        self.assertGreater(s.bytes_total, 2 * 2**30)
        self.assertLess(s.bytes_read, 600 * 2048)       # меньше 2 КБ на файл
        self.assertLess(dt, 10)
        self.assertEqual((rows[5].width, rows[5].height), (4005, 3000))


if __name__ == "__main__":
    unittest.main()
