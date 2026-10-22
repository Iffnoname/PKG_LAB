
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.buffer import BORDER_CLAMP, BORDER_REFLECT, PixelBuffer, border_index  # noqa: E402
from app.algorithms import contrast, segmentation as seg  # noqa: E402
from app.algorithms.clahe import clahe_sliding_plane  # noqa: E402
from app.algorithms.color import hsv_to_rgb, rgb_to_hsv  # noqa: E402
from app.algorithms.convolution import filter2d, gaussian_kernel_1d, separable_filter  # noqa: E402
from app.algorithms.histogram import histogram, stats  # noqa: E402
from app.core.parallel import set_threads, max_threads  # noqa: E402

cv2 = pytest.importorskip("cv2")
CV_BORDER = {BORDER_REFLECT: cv2.BORDER_REFLECT_101, BORDER_CLAMP: cv2.BORDER_REPLICATE}
RNG = np.random.default_rng(0)
GRAY = RNG.integers(0, 256, (97, 131), dtype=np.uint8)
RGB = RNG.integers(0, 256, (61, 83, 3), dtype=np.uint8)


def test_border_index():
    n = 5
    assert [border_index(i, n, BORDER_CLAMP) for i in range(-3, 8)] == [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
    assert [border_index(i, n, BORDER_REFLECT) for i in range(-3, 8)] == [3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]


@pytest.mark.parametrize("mode", [BORDER_CLAMP, BORDER_REFLECT])
def test_padding_matches_opencv(mode):
    f = GRAY.astype(np.float32)
    ours = PixelBuffer.padded(f, 4, 6, mode)
    ref = cv2.copyMakeBorder(f, 4, 4, 6, 6, CV_BORDER[mode])
    assert np.array_equal(ours, ref)


def test_histogram_and_stats():
    h = histogram(GRAY)
    assert np.array_equal(h, np.bincount(GRAY.ravel(), minlength=256))
    s = stats(GRAY)
    assert s["min"] == GRAY.min() and s["max"] == GRAY.max()
    assert abs(s["mean"] - GRAY.mean()) < 1e-9 and abs(s["sigma"] - GRAY.std()) < 1e-9


def test_linear_contrast_auto():
    img = (GRAY // 4 + 80).astype(np.uint8)  # диапазон 80..143
    out, meta = contrast.linear_contrast(PixelBuffer(img), "auto")
    ref = np.clip(np.round((img.astype(float) - img.min()) * 255 / (img.max() - img.min())), 0, 255)
    assert np.array_equal(out.data[:, :, 0], ref.astype(np.uint8))
    assert out.data.min() == 0 and out.data.max() == 255


def test_equalize_matches_opencv():
    img = (GRAY // 3 + 40).astype(np.uint8)
    out, _ = contrast.equalize(PixelBuffer(img))
    ref = cv2.equalizeHist(img)
    assert np.max(np.abs(out.data[:, :, 0].astype(int) - ref.astype(int))) <= 1


def test_hsv_roundtrip_and_reference():
    H, S, V = rgb_to_hsv(RGB)
    back = hsv_to_rgb(H, S, V)
    assert np.max(np.abs(back.astype(int) - RGB.astype(int))) <= 1
    ref = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV_FULL)
    assert np.array_equal(V, ref[:, :, 2])


def test_hsv_equalization_keeps_hue():
    img = (RGB // 3 + 60).astype(np.uint8)
    out, _ = contrast.equalize(PixelBuffer(img), contrast.COLOR_HSV)
    H0, S0, _ = rgb_to_hsv(img)
    H1, S1, V1 = rgb_to_hsv(out.data)
    ok = (V1 > 40) & (S0 > 0.2)
    dh = np.abs(H0 - H1)[ok]
    dh = np.minimum(dh, 360 - dh)
    assert np.median(dh) < 2.0


@pytest.mark.parametrize("mode", [BORDER_CLAMP, BORDER_REFLECT])
def test_separable_equals_direct_and_opencv(mode):
    f = GRAY.astype(np.float32)
    k = gaussian_kernel_1d(2.0)
    sep = separable_filter(f, k, k, mode)
    direct = filter2d(f, np.outer(k, k), mode)
    ref = cv2.sepFilter2D(f, -1, k, k, borderType=CV_BORDER[mode])
    assert np.max(np.abs(sep - direct)) < 1e-3
    assert np.max(np.abs(sep - ref)) < 1e-3


@pytest.mark.parametrize("mode", [BORDER_CLAMP, BORDER_REFLECT])
def test_sobel_matches_opencv(mode):
    f = GRAY.astype(np.float32)
    gx, gy = seg.gradient(f, "sobel", mode)
    rx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3, borderType=CV_BORDER[mode])
    ry = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3, borderType=CV_BORDER[mode])
    assert np.max(np.abs(gx - rx)) < 1e-3 and np.max(np.abs(gy - ry)) < 1e-3


def test_gradient_angle_is_exact():
    for a in (-150, -90, -30, 0, 37, 90, 135, 180):
        r = np.deg2rad(a)
        yy, xx = np.mgrid[0:40, 0:40].astype(np.float32)
        f = 100 + 2 * (xx * np.cos(r) + yy * np.sin(r))
        gx, gy = seg.gradient(f.astype(np.float32), "sobel", BORDER_REFLECT)
        _, ang = seg.magnitude_angle(gx, gy, False)
        d = (ang[10:30, 10:30] - a + 180) % 360 - 180
        assert np.max(np.abs(d)) < 0.01


def test_roberts_angle():
    yy, xx = np.mgrid[0:30, 0:30].astype(np.float32)
    f = (3 * xx + 1 * yy).astype(np.float32)
    gx, gy = seg.gradient(f, "roberts", BORDER_REFLECT)
    assert np.allclose(gx[5:25, 5:25], 3) and np.allclose(gy[5:25, 5:25], 1)


def test_point_detection_finds_isolated_points():
    img = np.full((64, 64), 100, np.float32)
    pts = [(10, 12), (30, 40), (50, 20), (55, 55)]
    for y, x in pts:
        img[y, x] = 200
    r = seg.detect_points(img, 90, BORDER_REFLECT)
    ys, xs = np.nonzero(r["mask"])
    assert sorted(zip(ys.tolist(), xs.tolist())) == sorted(pts)
    assert r["count"] == 4


def test_line_detection_directions():
    img = np.zeros((41, 41), np.float32)
    img[20, :] = 255
    r = seg.detect_lines(img, "all", 90, BORDER_REFLECT)
    ys, xs = np.nonzero(r["mask"])
    assert set(ys.tolist()) == {20} and set(r["index"][r["mask"] > 0].tolist()) == {0}
    img = np.zeros((41, 41), np.float32)
    for i in range(41):
        img[40 - i, i] = 255   # линия «/»
    r = seg.detect_lines(img, "all", 90, BORDER_REFLECT)
    assert set(r["index"][r["mask"] > 0].tolist()) == {1}


def test_clahe_sliding_matches_bruteforce():
    img = RNG.integers(60, 140, (23, 29), dtype=np.uint8)
    rad, clip = 3, 2.0
    for mode in (BORDER_CLAMP, BORDER_REFLECT):
        out = clahe_sliding_plane(img, rad, clip, mode)
        pad = PixelBuffer.padded_u8(img, rad, mode)
        n = (2 * rad + 1) ** 2
        lim = max(1.0, clip * n / 256)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                win = pad[y:y + 2 * rad + 1, x:x + 2 * rad + 1]
                h = np.minimum(np.bincount(win.ravel(), minlength=256), lim)
                exc = n - h.sum()
                v = img[y, x]
                cdf = h[:v + 1].sum() + exc * (v + 1) / 256
                assert out[y, x] == int(min(255, 255 * cdf / n) + 0.5)


def test_canny_close_to_opencv():
    yy, xx = np.mgrid[0:120, 0:120]
    img = (((xx - 60) ** 2 + (yy - 60) ** 2) < 35 ** 2).astype(np.uint8) * 150 + 50
    r = seg.canny(img.astype(np.float32), 0.0, 30, 50, BORDER_REFLECT)
    ref = cv2.Canny(img, 100, 200, L2gradient=True)
    assert abs(r["pixels"] - np.count_nonzero(ref)) / np.count_nonzero(ref) < 0.15


def test_canny_subpixel_circle():
    R, c = 30.3, 64.37
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)
    acc = np.zeros((128, 128))
    for sy in range(8):
        for sx in range(8):
            d = np.hypot(xx + (sx + 0.5) / 8 - 0.5 - c, yy + (sy + 0.5) / 8 - 0.5 - c)
            acc += d < R
    img = (50 + 150 * acc / 64).astype(np.float32)
    r = seg.canny(img, 1.0, 30, 40, BORDER_REFLECT)
    ys, xs = np.nonzero(r["mask"])
    pix_err = np.abs(np.hypot(xs - c, ys - c) - R)
    sp = r["subpixel"]
    sub_err = np.abs(np.hypot(sp[:, 0] - c, sp[:, 1] - c) - R)
    assert sub_err.mean() < 0.1 < pix_err.mean()


def test_multithreaded_equals_single():
    f = GRAY.astype(np.float32)
    set_threads(1)
    a = seg.canny(f, 1.4, 20, 40, BORDER_REFLECT)["mask"]
    b = clahe_sliding_plane(GRAY, 5, 2.0, BORDER_REFLECT)
    set_threads(max_threads())
    a2 = seg.canny(f, 1.4, 20, 40, BORDER_REFLECT)["mask"]
    b2 = clahe_sliding_plane(GRAY, 5, 2.0, BORDER_REFLECT)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
