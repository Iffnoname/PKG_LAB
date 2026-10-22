"""Генерация базы тестовых изображений (папка images/).

Исходные фотографии — из набора skimage.data (свободная лицензия),
искажения (сжатие диапазона, затемнение, шум, неравномерное освещение)
и синтетические сцены строятся здесь. Запуск: python tools/make_test_images.py
"""
import os

import numpy as np
from PIL import Image
import skimage.data as sd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "images")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)


def save(name, a):
    a = np.clip(np.round(a), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(os.path.join(OUT, name))
    print(name, a.shape, a.min(), a.max(), round(float(a.std()), 1))


def squeeze(a, lo, hi):
    a = a.astype(np.float64)
    return lo + (a - a.min()) * (hi - lo) / (a.max() - a.min())


def gauss_blur(a, s):
    r = int(3 * s)
    t = np.arange(-r, r + 1)
    k = np.exp(-t ** 2 / (2 * s * s))
    k /= k.sum()
    a = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="edge"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="edge"), k, "valid"), 1, a)


# ---------- 1. гистограммы и контраст
save("01_moon_low_contrast.png", sd.moon())                           # узкая гистограмма + выбросы
save("02_camera_low_contrast.png", squeeze(sd.camera(), 90, 160))      # сжатый диапазон
save("03_dark_coffee.png", 255 * (sd.coffee() / 255.0) ** 2.6)         # тёмное цветное
save("04_color_low_contrast.png", squeeze(sd.chelsea(), 85, 175))      # малоконтрастное цветное
save("05_overexposed_astronaut.png", 255 * (sd.astronaut() / 255.0) ** 0.35)  # пересвеченное
cam = squeeze(sd.camera(), 100, 150)
m = rng.random(cam.shape)
cam[m < 0.002] = 0
cam[m > 0.998] = 255
save("06_outliers_low_contrast.png", cam)                              # выбросы: авто min/max бессилен
save("07_uneven_light_page.png", sd.page())                            # неравномерное освещение
h, w = 512, 512
yy, xx = np.mgrid[0:h, 0:w]
light = 0.12 + 0.88 * np.clip((xx / w) ** 1.6, 0, 1)
save("08_uneven_light_color.png", sd.astronaut() * light[:, :, None])  # тень слева, свет справа

# ---------- 2. обнаружение точек
save("09_stars_hubble.png", sd.hubble_deep_field())
h, w = 400, 600
yy, xx = np.mgrid[0:h, 0:w]
bg = 110 + 40 * np.sin(xx / 90.0) * np.cos(yy / 70.0) + 0.06 * xx     # плавный неоднородный фон
blob = np.zeros((h, w))
for cy, cx, r, v in [(120, 160, 60, 50), (260, 420, 80, -40)]:
    blob += v * (np.hypot(xx - cx, yy - cy) < r)
img = bg + gauss_blur(blob, 3) + rng.normal(0, 3.0, (h, w))
pts = []
while len(pts) < 40:
    py, px = rng.integers(10, h - 10), rng.integers(10, w - 10)
    if all(abs(py - a) + abs(px - b) > 12 for a, b in pts):
        pts.append((py, px))
amp = np.linspace(18, 60, len(pts))                                  # разная «глубина» пор
for i, (py, px) in enumerate(pts):
    img[py, px] += amp[i] if i % 4 else -amp[i]                       # 30 светлых и 10 тёмных пор
save("10_porosity_synthetic.png", img)
np.savetxt(os.path.join(OUT, "10_porosity_points.csv"), np.array(pts), fmt="%d", delimiter=";",
           header="y;x", comments="")

# ---------- 3. обнаружение линий
h, w = 400, 400
img = np.full((h, w), 60.0)
img[100, 20:380] = 200                                    # горизонтальная 1 px
img[20:380, 300] = 200                                    # вертикальная 1 px
for t in range(20, 380):
    img[399 - t, t] = 200                                 # +45° (/)
    if 0 <= t - 60 < h:
        img[t - 60, t] = 200                              # −45° (\)
img[250:254, 30:200] = 200                                # толстая горизонтальная (4 px)
img[300:400, 20:120] = 150                                # прямоугольник (перепад, не линия)
for t in range(0, 180):                                   # линия под 30°
    y = int(round(380 - t * np.tan(np.radians(30))))
    img[y, 200 + t] = 200
save("11_lines_synthetic.png", img + rng.normal(0, 3, (h, w)))
h, w = 360, 480
img = np.full((h, w), 40.0)
for k in range(-6, 10):                                   # «проволочная схема»: сетка тонких линий
    for t in range(w):
        y = t + k * 45 - 120
        if 0 <= y < h:
            img[h - 1 - y, t] = 190                       # +45°
for k in range(0, 8):
    img[20 + k * 45, :] = 170                             # горизонтальные
img = gauss_blur(img, 0.6)
save("12_wire_grid_synthetic.png", img + rng.normal(0, 4, (h, w)))
ret = sd.retina()[200:1200, 200:1200]
save("13_retina_vessels.png", ret)

# ---------- 4. перепады яркости (края)
h, w = 420, 560
yy, xx = np.mgrid[0:h, 0:w]
img = 70 + 50 * xx / w
img[(np.hypot(xx - 140, yy - 140) < 90)] = 200
img[60:200, 300:470] = 30
tri = (yy > 250) & (yy < 390) & (np.abs(xx - 380) < (yy - 250) * 0.8)
img[tri] = 220
img[260:380, 60:240] = 160
ring = (np.hypot(xx - 150, yy - 320) < 45) & (np.hypot(xx - 150, yy - 320) > 25)
img[ring] = 40
shapes = img.copy()
save("14_shapes.png", shapes)
save("15_shapes_noisy.png", shapes + rng.normal(0, 25, shapes.shape))
save("16_camera.png", sd.camera())
save("17_coins.png", sd.coins())
# диск с известным радиусом и антиалиасингом (8x8 подвыборок) — тест субпиксельной точности
R, cx, cy = 80.37, 128.61, 127.23
yy, xx = np.mgrid[0:256, 0:256].astype(np.float64)
acc = np.zeros((256, 256))
for sy in range(8):
    for sx in range(8):
        acc += np.hypot(xx + (sx + 0.5) / 8 - 0.5 - cx, yy + (sy + 0.5) / 8 - 0.5 - cy) < R
save("18_disk_subpixel.png", 60 + 140 * acc / 64)
save("19_astronaut_color.png", sd.astronaut())
