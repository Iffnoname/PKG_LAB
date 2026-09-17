"""
Тесты преобразований RGB ↔ HSV ↔ LAB.
"""

import unittest

import color_converter as cc


def _round_trip_rgb_hsv(r: int, g: int, b: int) -> tuple[int, int, int]:
    h, s, v = cc.rgb_to_hsv(r, g, b)
    r2, g2, b2, _ = cc.hsv_to_rgb(h, s, v)
    return r2, g2, b2


def _round_trip_rgb_lab(r: int, g: int, b: int) -> tuple[int, int, int]:
    L, a, b_lab = cc.rgb_to_lab(r, g, b)
    r2, g2, b2, _ = cc.lab_to_rgb(L, a, b_lab)
    return r2, g2, b2


class TestColorConversion(unittest.TestCase):
    BASIC_COLORS = [
        (0, 0, 0, "black"),
        (255, 255, 255, "white"),
        (255, 0, 0, "red"),
        (0, 255, 0, "green"),
        (0, 0, 255, "blue"),
        (128, 128, 128, "gray"),
    ]

    EXTRA_COLORS = [
        (255, 128, 0),
        (64, 120, 200),
        (200, 50, 180),
    ]

    def test_rgb_hsv_known_values(self) -> None:
        h, s, v = cc.rgb_to_hsv(255, 0, 0)
        self.assertAlmostEqual(h, 0.0, places=1)
        self.assertAlmostEqual(s, 100.0, places=1)
        self.assertAlmostEqual(v, 100.0, places=1)

        h, s, v = cc.rgb_to_hsv(0, 255, 0)
        self.assertAlmostEqual(h, 120.0, places=1)
        self.assertAlmostEqual(s, 100.0, places=1)
        self.assertAlmostEqual(v, 100.0, places=1)

        h, s, v = cc.rgb_to_hsv(0, 0, 255)
        self.assertAlmostEqual(h, 240.0, places=1)
        self.assertAlmostEqual(s, 100.0, places=1)
        self.assertAlmostEqual(v, 100.0, places=1)

        h, s, v = cc.rgb_to_hsv(0, 0, 0)
        self.assertAlmostEqual(s, 0.0, places=1)
        self.assertAlmostEqual(v, 0.0, places=1)

        h, s, v = cc.rgb_to_hsv(255, 255, 255)
        self.assertAlmostEqual(s, 0.0, places=1)
        self.assertAlmostEqual(v, 100.0, places=1)

    def test_rgb_lab_white_black(self) -> None:
        L, a, b = cc.rgb_to_lab(255, 255, 255)
        self.assertAlmostEqual(L, 100.0, delta=0.5)
        self.assertAlmostEqual(a, 0.0, delta=0.5)
        self.assertAlmostEqual(b, 0.0, delta=0.5)

        L, a, b = cc.rgb_to_lab(0, 0, 0)
        self.assertAlmostEqual(L, 0.0, delta=0.5)
        self.assertAlmostEqual(a, 0.0, delta=1.0)
        self.assertAlmostEqual(b, 0.0, delta=1.0)

    def test_round_trip_rgb_hsv(self) -> None:
        for r, g, b, _ in self.BASIC_COLORS:
            with self.subTest(rgb=(r, g, b)):
                r2, g2, b2 = _round_trip_rgb_hsv(r, g, b)
                self.assertLessEqual(abs(r - r2), 1)
                self.assertLessEqual(abs(g - g2), 1)
                self.assertLessEqual(abs(b - b2), 1)

        for r, g, b in self.EXTRA_COLORS:
            with self.subTest(rgb=(r, g, b)):
                r2, g2, b2 = _round_trip_rgb_hsv(r, g, b)
                self.assertLessEqual(abs(r - r2), 1)
                self.assertLessEqual(abs(g - g2), 1)
                self.assertLessEqual(abs(b - b2), 1)

    def test_round_trip_rgb_lab(self) -> None:
        for r, g, b, _ in self.BASIC_COLORS:
            with self.subTest(rgb=(r, g, b)):
                r2, g2, b2 = _round_trip_rgb_lab(r, g, b)
                self.assertLessEqual(abs(r - r2), 1)
                self.assertLessEqual(abs(g - g2), 1)
                self.assertLessEqual(abs(b - b2), 1)

        for r, g, b in self.EXTRA_COLORS:
            with self.subTest(rgb=(r, g, b)):
                r2, g2, b2 = _round_trip_rgb_lab(r, g, b)
                self.assertLessEqual(abs(r - r2), 1)
                self.assertLessEqual(abs(g - g2), 1)
                self.assertLessEqual(abs(b - b2), 1)

    def test_lab_out_of_gamut_clipping(self) -> None:
        # Насыщенный зелёный в Lab, затем обратно — clipping не должен падать
        r, g, b, clipped = cc.lab_to_rgb(50.0, -80.0, 80.0)
        self.assertTrue(clipped)
        for ch in (r, g, b):
            self.assertGreaterEqual(ch, 0)
            self.assertLessEqual(ch, 255)

    def test_xyz_round_trip(self) -> None:
        x, y, z = cc.rgb_to_xyz(128, 128, 128)
        r, g, b, _ = cc.xyz_to_rgb(x, y, z)
        self.assertLessEqual(abs(128 - r), 1)
        self.assertLessEqual(abs(128 - g), 1)
        self.assertLessEqual(abs(128 - b), 1)


if __name__ == "__main__":
    unittest.main()
