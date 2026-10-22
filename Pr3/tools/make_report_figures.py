"""Расчёт измерений и построение рисунков для отчёта (папка report/).
Использует только функции приложения (app.*); matplotlib — только для графиков."""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app.core.buffer import BORDER_CLAMP, BORDER_REFLECT, PixelBuffer  # noqa: E402
from app.core import parallel  # noqa: E402
from app.algorithms import contrast, segmentation as seg  # noqa: E402
from app.algorithms.clahe import clahe  # noqa: E402
from app.algorithms.color import rgb_to_hsv  # noqa: E402
from app.algorithms.convolution import filter2d, gaussian_kernel_1d, separable_filter  # noqa: E402
from app.algorithms.histogram import histogram, stats  # noqa: E402
from app.algorithms.operations import run_operation  # noqa: E402

FIG = os.path.join(ROOT, "report", "fig")
os.makedirs(FIG, exist_ok=True)
IMG = os.path.join(ROOT, "images")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "figure.dpi": 150})
M = {}


def load(name):
    return PixelBuffer.load(os.path.join(IMG, name))


def show(ax, data, title):
    d = data[:, :, 0] if data.ndim == 3 and data.shape[2] == 1 else data
    ax.imshow(d, cmap="gray" if d.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def hist_plot(ax, before, after, title, ylim=None):
    hb = histogram(before) / before.size
    ha = histogram(after) / after.size
    ax.bar(np.arange(256), hb, width=1.0, color="#999999", label="до")
    ax.plot(np.arange(256), ha, color="#159a3c", lw=1.1, label="после")
    ax.set_xlim(0, 255)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("уровень яркости")
    ax.set_ylabel("доля пикселей")
    ax.legend(fontsize=8, loc="upper right")


def cdf_plot(ax, before, after):
    ax.plot(np.cumsum(histogram(before)) / before.size, color="#777777", label="CDF до")
    ax.plot(np.cumsum(histogram(after)) / after.size, color="#159a3c", label="CDF после")
    ax.set_xlim(0, 255)
    ax.legend(fontsize=8)
    ax.set_title("Накопленные гистограммы", fontsize=9)


def st(buf, plane=None):
    p = buf.luminance() if plane is None else plane
    s = stats(p)
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


# =============================== 1. Линейное контрастирование и эквализация
b = load("02_camera_low_contrast.png")
lin = run_operation("linear", b, {"mode": "auto"}).buffer
eq = run_operation("equalize", b, {}).buffer
M["camera"] = {"src": st(b), "linear": st(lin), "equalize": st(eq)}
fig, ax = plt.subplots(2, 3, figsize=(10, 6.2))
show(ax[0, 0], b.data, "Исходное (диапазон 90…160)")
show(ax[0, 1], lin.data, "Линейное контрастирование")
show(ax[0, 2], eq.data, "Эквализация гистограммы")
hist_plot(ax[1, 0], b.luminance(), lin.luminance(), "Гистограмма: линейное контрастирование")
hist_plot(ax[1, 1], b.luminance(), eq.luminance(), "Гистограмма: эквализация")
cdf_plot(ax[1, 2], b.luminance(), eq.luminance())
save(fig, "f02_linear_vs_eq.png")

# выбросы
b = load("06_outliers_low_contrast.png")
auto = run_operation("linear", b, {"mode": "auto"}).buffer
pc = run_operation("linear", b, {"mode": "percent", "percent": 1.0}).buffer
_, meta = contrast.linear_contrast(b, "percent", percent=1.0)
eq = run_operation("equalize", b, {}).buffer
M["outliers"] = {"src": st(b), "auto": st(auto), "percent": st(pc), "percent_range": meta["ranges"][0],
                 "equalize": st(eq)}
fig, ax = plt.subplots(2, 3, figsize=(10, 6.2))
show(ax[0, 0], b.data, "Исходное с выбросами 0 и 255")
show(ax[0, 1], auto.data, "Авто (min…max): без изменений")
show(ax[0, 2], pc.data, f"Отсечение 1 %: a={meta['ranges'][0][0]}, b={meta['ranges'][0][1]}")
hist_plot(ax[1, 0], b.luminance(), auto.luminance(), "Авто min…max")
hist_plot(ax[1, 1], b.luminance(), pc.luminance(), "Отсечение 1 %")
hist_plot(ax[1, 2], b.luminance(), eq.luminance(), "Эквализация")
save(fig, "f03_outliers.png")

# луна и тёмное фото
for key, name in [("moon", "01_moon_low_contrast.png"), ("coffee", "03_dark_coffee.png"),
                  ("over", "05_overexposed_astronaut.png")]:
    b = load(name)
    M[key] = {"src": st(b),
              "linear": st(run_operation("linear", b, {"mode": "auto"}).buffer),
              "linear_pc": st(run_operation("linear", b, {"mode": "percent", "percent": 1.0}).buffer),
              "equalize": st(run_operation("equalize", b, {}).buffer)}
b = load("01_moon_low_contrast.png")
eq = run_operation("equalize", b, {}).buffer
pc = run_operation("linear", b, {"mode": "percent", "percent": 1.0}).buffer
fig, ax = plt.subplots(2, 3, figsize=(10, 6.2))
show(ax[0, 0], b.data, "Исходное (луна)")
show(ax[0, 1], pc.data, "Линейное, отсечение 1 %")
show(ax[0, 2], eq.data, "Эквализация")
hist_plot(ax[1, 0], b.luminance(), pc.luminance(), "Линейное, отсечение 1 %")
hist_plot(ax[1, 1], b.luminance(), eq.luminance(), "Эквализация")
cdf_plot(ax[1, 2], b.luminance(), eq.luminance())
save(fig, "f04_moon.png")

# =============================== 2. RGB против HSV
b = load("04_color_low_contrast.png")
e_rgb = run_operation("equalize", b, {"color": "rgb"}).buffer
e_hsv = run_operation("equalize", b, {"color": "hsv"}).buffer


def hue_shift(a, c):
    H0, S0, V0 = rgb_to_hsv(a.data)
    H1, S1, V1 = rgb_to_hsv(c.data)
    ok = (S0 > 0.15) & (V0 > 30)
    d = np.abs(H0 - H1)[ok]
    d = np.minimum(d, 360 - d)
    return round(float(d.mean()), 2), round(float(np.abs(S0 - S1)[ok].mean()), 3)


M["rgb_hsv"] = {"src": st(b, rgb_to_hsv(b.data)[2]), "rgb": st(e_rgb, rgb_to_hsv(e_rgb.data)[2]),
                "hsv": st(e_hsv, rgb_to_hsv(e_hsv.data)[2]),
                "hue_shift_rgb": hue_shift(b, e_rgb), "hue_shift_hsv": hue_shift(b, e_hsv)}
for key, name in [("coffee_rgbhsv", "03_dark_coffee.png")]:
    bb = load(name)
    r1 = run_operation("equalize", bb, {"color": "rgb"}).buffer
    r2 = run_operation("equalize", bb, {"color": "hsv"}).buffer
    M[key] = {"hue_shift_rgb": hue_shift(bb, r1), "hue_shift_hsv": hue_shift(bb, r2)}
cof = load("03_dark_coffee.png")
c_rgb = run_operation("equalize", cof, {"color": "rgb"}).buffer
c_hsv = run_operation("equalize", cof, {"color": "hsv"}).buffer
fig, ax = plt.subplots(3, 3, figsize=(10, 8.6))
show(ax[0, 0], b.data, "Малоконтрастное (сжатый диапазон)")
show(ax[0, 1], e_rgb.data, "Эквализация R, G, B по отдельности")
show(ax[0, 2], e_hsv.data, "Эквализация только V (HSV)")
show(ax[1, 0], cof.data, "Тёмное (гамма 2.6)")
show(ax[1, 1], c_rgb.data, "Эквализация R, G, B по отдельности")
show(ax[1, 2], c_hsv.data, "Эквализация только V (HSV)")
cols = ["#d62728", "#2ca02c", "#1f77b4"]
for k, (img, t) in enumerate([(cof, "до"), (c_rgb, "RGB"), (c_hsv, "HSV")]):
    for c in range(3):
        ax[2, k].plot(histogram(img.channel(c)) / img.width / img.height, color=cols[c], lw=1)
    ax[2, k].set_xlim(0, 255)
    ax[2, k].set_ylim(0, 0.03)
    ax[2, k].set_title(f"Тёмное фото: гистограммы R, G, B ({t})", fontsize=9)
save(fig, "f05_rgb_hsv.png")

# =============================== 3. CLAHE
b = load("08_uneven_light_color.png")
g = run_operation("equalize", b, {}).buffer
cs = run_operation("clahe", b, {"variant": "sliding", "radius": 32, "clip": 2.5}).buffer
ct = run_operation("clahe", b, {"variant": "tiles", "tiles": 8, "clip": 2.5}).buffer
M["clahe_color"] = {"src": st(b), "global": st(g), "sliding": st(cs), "tiles": st(ct)}


def region_sigma(buf, sl):
    return round(float(buf.luminance()[sl].std()), 2)


left = (slice(0, 512), slice(0, 160))
M["clahe_color"]["left_sigma"] = {k: region_sigma(v, left) for k, v in
                                  [("src", b), ("global", g), ("sliding", cs), ("tiles", ct)]}
fig, ax = plt.subplots(2, 4, figsize=(12, 6.2))
show(ax[0, 0], b.data, "Неравномерное освещение")
show(ax[0, 1], g.data, "Глобальная эквализация (V)")
show(ax[0, 2], cs.data, "CLAHE: окно 65×65, лимит 2.5")
show(ax[0, 3], ct.data, "CLAHE: тайлы 8×8, лимит 2.5")
hist_plot(ax[1, 0], b.luminance(), b.luminance(), "Исходная гистограмма Y", 0.03)
hist_plot(ax[1, 1], b.luminance(), g.luminance(), "Глобальная эквализация", 0.03)
hist_plot(ax[1, 2], b.luminance(), cs.luminance(), "CLAHE (окно)", 0.03)
hist_plot(ax[1, 3], b.luminance(), ct.luminance(), "CLAHE (тайлы)", 0.03)
save(fig, "f06_clahe.png")

b = load("07_uneven_light_page.png")
g = run_operation("equalize", b, {}).buffer
cs = run_operation("clahe", b, {"variant": "sliding", "radius": 16, "clip": 3.0}).buffer
noclip = run_operation("clahe", b, {"variant": "sliding", "radius": 16, "clip": 100.0}).buffer
M["clahe_page"] = {"src": st(b), "global": st(g), "sliding": st(cs), "noclip": st(noclip)}
fig, ax = plt.subplots(2, 2, figsize=(10, 4.4))
show(ax[0, 0], b.data, "Страница при неравномерном освещении")
show(ax[0, 1], g.data, "Глобальная эквализация")
show(ax[1, 0], noclip.data, "Локальная эквализация без ограничения (лимит 100)")
show(ax[1, 1], cs.data, "CLAHE, окно 33×33, лимит 3")
save(fig, "f07_clahe_page.png")

# =============================== 4. Обнаружение точек
b = load("10_porosity_synthetic.png")
truth = np.loadtxt(os.path.join(IMG, "10_porosity_points.csv"), delimiter=";", skiprows=1).astype(int)
f = b.luminance_f32()
rows = []
for T in (10, 15, 20, 25, 30, 40, 50):
    r = seg.detect_points(f, T, BORDER_REFLECT)
    ys, xs = np.nonzero(r["mask"])
    det = set(zip(ys.tolist(), xs.tolist()))
    tp = sum((int(y), int(x)) in det for y, x in truth)
    rows.append({"T": T, "Tabs": round(r["T"], 1), "found": r["count"], "tp": tp, "fp": r["count"] - tp})
M["points"] = rows
r = seg.detect_points(f, 25, BORDER_REFLECT)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
show(ax[0], b.data, "Неоднородный фон + 40 одиночных пор")
show(ax[1], seg.normalize_u8(r["response"], r["T"] * 1.2), "Отклик |R| (шкала 0…1.2·T)")
ov = seg.overlay(b.luminance(), r["mask"], 255, 40, 40)
show(ax[2], ov, f"Порог 25 %: найдено {r['count']} точек")
for y, x in zip(*np.nonzero(r["mask"])):
    ax[2].add_patch(plt.Circle((x, y), 6, fill=False, color="red", lw=0.8))
save(fig, "f08_points.png")
# глобальный порог по яркости для сравнения — поры не отделимы от фона
lum = b.luminance()
M["points_global_threshold"] = {"bg_min": int(lum.min()), "bg_max": int(lum.max())}

b = load("09_stars_hubble.png")
r = seg.detect_points(b.luminance_f32(), 25, BORDER_REFLECT, polarity=1)
M["stars"] = {"count": r["count"], "pixels": r["pixels"]}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
crop = (slice(0, 436), slice(0, 500))
show(ax[0], b.data[crop], "Фрагмент снимка Hubble Deep Field")
ov = seg.overlay(b.luminance(), r["mask"], 255, 255, 0)
show(ax[1], ov[crop], f"Светлые точки, порог 25 % (всего {r['count']})")
save(fig, "f09_stars.png")

# =============================== 5. Обнаружение линий
b = load("11_lines_synthetic.png")
f = b.luminance_f32()
fig, ax = plt.subplots(2, 3, figsize=(11, 7.4))
show(ax[0, 0], b.data, "Исходное: линии 1 px, толстая линия, перепад")
lines_rows = {}
for i, key in enumerate(seg.LINE_ORDER):
    r = seg.detect_lines(f, key, 60, BORDER_REFLECT, polarity=1)
    lines_rows[key] = r["pixels"]
    a = ax[(i + 1) // 3, (i + 1) % 3]
    show(a, r["mask"], f"Маска {seg.LINE_TITLES[key]}, T = 60 %")
r = seg.detect_lines(f, "all", 60, BORDER_REFLECT, polarity=1)
lines_rows["all"] = r["per_dir"]
show(ax[1, 2], seg.color_lines(r["mask"], r["index"], seg.LINE_COLORS), "Все 4 маски (цвет = направление)")
save(fig, "f10_lines.png")
M["lines"] = lines_rows

b = load("12_wire_grid_synthetic.png")
r = seg.detect_lines(b.luminance_f32(), "d45", 25, BORDER_REFLECT, polarity=1)
r2 = seg.detect_lines(b.luminance_f32(), "d45", 25, BORDER_REFLECT, polarity=0)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
show(ax[0], b.data, "Сетка тонких проводников")
show(ax[1], r2["mask"], "+45°, |R| (двойные отклики по бокам)")
show(ax[2], r["mask"], "+45°, только R > 0")
save(fig, "f11_wire.png")
M["wire"] = {"abs": r2["pixels"], "pos": r["pixels"]}

b = load("13_retina_vessels.png")
f = b.luminance_f32()
from app.algorithms.convolution import gaussian_blur  # noqa: E402
fs = gaussian_blur(f, 1.0, BORDER_REFLECT)
r = seg.detect_lines(fs, "all", 12, BORDER_REFLECT, polarity=2)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
show(ax[0], b.data, "Глазное дно (сосуды темнее фона)")
show(ax[1], seg.color_lines(r["mask"], r["index"], seg.LINE_COLORS), "Тёмные линии (R < 0), σ = 1, T = 12 %")
save(fig, "f12_retina.png")
M["retina"] = r["per_dir"]

# =============================== 6. Перепады яркости
b = load("16_camera.png")
f = b.luminance_f32()
res = {
    "Робертс": seg.gradient_edges(f, "roberts", 12, BORDER_REFLECT),
    "Превитт": seg.gradient_edges(f, "prewitt", 12, BORDER_REFLECT),
    "Собель": seg.gradient_edges(f, "sobel", 12, BORDER_REFLECT),
    "LoG, σ = 2": seg.log_edges(f, 2.0, 8, BORDER_REFLECT),
    "Кэнни, σ = 1.4": seg.canny(f, 1.4, 20, 40, BORDER_REFLECT),
}
fig, ax = plt.subplots(2, 3, figsize=(11, 7.4))
show(ax[0, 0], b.data, "Исходное")
for i, (k, r) in enumerate(res.items()):
    show(ax[(i + 1) // 3, (i + 1) % 3], r["mask"], k)
save(fig, "f13_edges_camera.png")
M["edges_camera"] = {k: r["pixels"] for k, r in res.items()}

# карта направлений
b = load("14_shapes.png")
r = seg.gradient_edges(b.luminance_f32(), "sobel", 10, BORDER_REFLECT)
c = seg.canny(b.luminance_f32(), 1.4, 20, 40, BORDER_REFLECT)
cam = seg.gradient_edges(load("16_camera.png").luminance_f32(), "sobel", 10, BORDER_REFLECT, sigma=1.0)
from app.ui.main_window import direction_legend  # noqa: E402
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [4, 4, 3.2, 1.6]})
vmax = float(np.percentile(r["magnitude"], 99.5))
show(ax[0], seg.direction_map(r["magnitude"], r["angle"], vmax, r["mask"], False),
     "Собель: H = θ, V = |∇f|")
show(ax[1], seg.direction_map(c["magnitude"], c["angle"], 1, c["mask"], True), "Кэнни: направления на краях")
vm = float(np.percentile(cam["magnitude"], 99.5))
show(ax[2], seg.direction_map(cam["magnitude"], cam["angle"], vm, cam["mask"], False), "Фото: карта направлений")
show(ax[3], direction_legend(160), "")
ax[3].set_xlim(-45, 205)
ax[3].set_ylim(200, -40)
ax[3].text(163, 80, "0°", fontsize=8, ha="left", va="center")
ax[3].text(80, 165, "90°", fontsize=8, ha="center", va="top")
ax[3].text(-3, 80, "180°", fontsize=8, ha="right", va="center")
ax[3].text(80, -5, "−90°", fontsize=8, ha="center", va="bottom")
ax[3].set_title("Легенда θ", fontsize=9, pad=2)
save(fig, "f14_direction.png")
# проверка угла: на границе диска с антиалиасингом градиент направлен к центру
dsk = load("18_disk_subpixel.png").luminance_f32()
cd = seg.canny(dsk, 1.0, 30, 40, BORDER_REFLECT)
yy, xx = np.nonzero(cd["mask"])
cx, cy = 128.61, 127.23
exp = np.degrees(np.arctan2(cy - yy, cx - xx))
got = cd["angle"][yy, xx]
d = (got - exp + 180) % 360 - 180
M["angle_disk"] = {"n": int(len(xx)), "mean_abs": round(float(np.abs(d).mean()), 2),
                   "max_abs": round(float(np.abs(d).max()), 2)}

# шум: Собель без сглаживания, Собель с Гауссом, LoG, Кэнни; метрика F1 с допуском 1 px
clean = load("14_shapes.png").luminance().astype(np.int32)
gt = np.zeros(clean.shape, bool)
gt[:, 1:] |= clean[:, 1:] != clean[:, :-1]
gt[1:, :] |= clean[1:, :] != clean[:-1, :]
gt[:, :-1] |= clean[:, 1:] != clean[:, :-1]
gt[:-1, :] |= clean[1:, :] != clean[:-1, :]
# фон-градиент 70+50x/w даёт изменение на 1 уровень — это не край
strong = np.zeros(clean.shape, bool)
gxx = np.abs(np.diff(clean, axis=1))
gyy = np.abs(np.diff(clean, axis=0))
strong[:, 1:] |= gxx > 5
strong[:, :-1] |= gxx > 5
strong[1:, :] |= gyy > 5
strong[:-1, :] |= gyy > 5
gt = strong


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def f1(mask):
    e = mask > 0
    prec = (e & dilate(gt, 1)).sum() / max(e.sum(), 1)
    rec = (gt & dilate(e, 1)).sum() / max(gt.sum(), 1)
    return round(float(prec), 3), round(float(rec), 3), round(float(2 * prec * rec / max(prec + rec, 1e-9)), 3)


b = load("15_shapes_noisy.png")
f = b.luminance_f32()
noisy = {
    "Собель без сглаживания, T = 35 %": seg.gradient_edges(f, "sobel", 35, BORDER_REFLECT),
    "Собель + Гаусс σ = 2, T = 25 %": seg.gradient_edges(f, "sobel", 25, BORDER_REFLECT, sigma=2.0),
    "LoG, σ = 3, T = 15 %": seg.log_edges(f, 3.0, 15, BORDER_REFLECT),
    "Кэнни, σ = 2, 25 % / 40 %": seg.canny(f, 2.0, 25, 40, BORDER_REFLECT),
}
fig, ax = plt.subplots(1, 5, figsize=(15, 2.9))
show(ax[0], b.data, "Шум σ = 25")
M["noisy"] = {}
for i, (k, r) in enumerate(noisy.items()):
    show(ax[i + 1], r["mask"], k)
    M["noisy"][k] = f1(r["mask"])
save(fig, "f15_noisy.png")

# субпиксельная точность
b = load("18_disk_subpixel.png")
R, cx, cy = 80.37, 128.61, 127.23
c = seg.canny(b.luminance_f32(), 1.0, 30, 40, BORDER_REFLECT)
ys, xs = np.nonzero(c["mask"])
pe = np.hypot(xs - cx, ys - cy) - R
sp = c["subpixel"]
se = np.hypot(sp[:, 0] - cx, sp[:, 1] - cy) - R
M["subpixel"] = {"n": int(len(xs)), "pix_mean": round(float(np.abs(pe).mean()), 3),
                 "pix_rms": round(float(np.sqrt((pe ** 2).mean())), 3), "pix_max": round(float(np.abs(pe).max()), 3),
                 "sub_mean": round(float(np.abs(se).mean()), 3), "sub_rms": round(float(np.sqrt((se ** 2).mean())), 3),
                 "sub_max": round(float(np.abs(se).max()), 3),
                 "R_est_pix": round(float(np.hypot(xs - cx, ys - cy).mean()), 3),
                 "R_est_sub": round(float(np.hypot(sp[:, 0] - cx, sp[:, 1] - cy).mean()), 3)}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
show(ax[0], b.data, f"Диск R = {R}, центр ({cx}; {cy})")
x0, y0 = 196, 92
ax[1].imshow(b.data[y0:y0 + 24, x0:x0 + 24, 0], cmap="gray", vmin=0, vmax=255, interpolation="nearest",
             extent=(x0 - 0.5, x0 + 23.5, y0 + 23.5, y0 - 0.5))
sel = (xs >= x0) & (xs < x0 + 24) & (ys >= y0) & (ys < y0 + 24)
ax[1].plot(xs[sel], ys[sel], "s", mfc="none", mec="red", ms=6, label="пиксельные края")
ax[1].plot(sp[sel, 0], sp[sel, 1], "o", color="#00c8ff", ms=3, label="субпиксельные точки")
t = np.linspace(-0.6, 0.2, 200)
ax[1].plot(cx + R * np.cos(t), cy + R * np.sin(t), "y--", lw=1, label="истинная окружность")
ax[1].set_xlim(x0 - 0.5, x0 + 23.5)
ax[1].set_ylim(y0 + 23.5, y0 - 0.5)
ax[1].legend(fontsize=7, loc="lower left")
ax[1].set_title("Фрагмент 24×24: Кэнни с субпиксельным уточнением", fontsize=9)
save(fig, "f16_subpixel.png")

# края: Zero / Clamp / Reflect
b = load("16_camera.png")
f = b.luminance_f32()
k = gaussian_kernel_1d(6.0)
r = len(k) // 2
# нулевое дополнение (запрещённый вариант) — только для демонстрации артефакта:
# изображение помещается в рамку из нулей и фильтруется той же функцией
big = np.zeros((f.shape[0] + 2 * r, f.shape[1] + 2 * r), np.float32)
big[r:-r, r:-r] = f
zero = separable_filter(big, k, k, BORDER_CLAMP)[r:-r, r:-r]
cl = separable_filter(f, k, k, BORDER_CLAMP)
rf = separable_filter(f, k, k, BORDER_REFLECT)
M["border"] = {"zero_edge_mean": round(float(zero[:, :3].mean()), 1), "clamp_edge_mean": round(float(cl[:, :3].mean()), 1),
               "reflect_edge_mean": round(float(rf[:, :3].mean()), 1), "orig_edge_mean": round(float(f[:, :3].mean()), 1)}
gz = seg.magnitude_angle(*seg.gradient(np.ascontiguousarray(big), "sobel", BORDER_CLAMP), False)[0][r:-r, r:-r]
gc = seg.gradient_edges(f, "sobel", 15, BORDER_CLAMP)["magnitude"]
gr = seg.gradient_edges(f, "sobel", 15, BORDER_REFLECT)["magnitude"]
fig, ax = plt.subplots(2, 3, figsize=(10, 6.8))
for i, (img, t) in enumerate([(zero, "Гаусс σ = 6: нули за краем"), (cl, "Гаусс σ = 6: Clamp"),
                              (rf, "Гаусс σ = 6: Reflect")]):
    show(ax[0, i], np.clip(img, 0, 255), t)
vm = float(np.percentile(gr, 99.5))
for i, (img, t) in enumerate([(gz, "Собель: нули за краем"), (gc, "Собель: Clamp"), (gr, "Собель: Reflect")]):
    show(ax[1, i], np.clip(img * 255 / vm, 0, 255), t)
save(fig, "f17_borders.png")
M["border_sobel_frame"] = {"zero": round(float(gz[:, 0].mean()), 1), "clamp": round(float(gc[:, 0].mean()), 1),
                           "reflect": round(float(gr[:, 0].mean()), 1)}

# =============================== 7. Производительность
astr = load("19_astronaut_color.png").luminance()
big = np.kron(astr, np.ones((4, 4), np.uint8))  # 2048 x 2048
bf = big.astype(np.float32)
perf = {"size": f"{big.shape[1]}×{big.shape[0]}"}


def tm(fn, rep=3):
    fn()
    best = 1e9
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return round(best * 1000, 1)


sep_rows = []
for s in (1.0, 2.0, 3.0, 5.0):
    k = gaussian_kernel_1d(s)
    K = len(k)
    t_sep = tm(lambda: separable_filter(bf, k, k, BORDER_REFLECT))
    t_dir = tm(lambda: filter2d(bf, np.outer(k, k), BORDER_REFLECT), rep=1)
    sep_rows.append({"sigma": s, "K": K, "sep": t_sep, "direct": t_dir, "ratio": round(t_dir / t_sep, 1),
                     "theory": round(K / 2, 1)})
perf["separable"] = sep_rows
thr = []
nmax = parallel.max_threads()
for name, fn in [("Гаусс σ = 3 (разделяемый)", lambda: separable_filter(bf, gaussian_kernel_1d(3), gaussian_kernel_1d(3), BORDER_REFLECT)),
                 ("Собель + модуль + угол", lambda: seg.magnitude_angle(*seg.gradient(bf, "sobel", BORDER_REFLECT), False)),
                 ("Кэнни σ = 1.4", lambda: seg.canny(bf, 1.4, 20, 40, BORDER_REFLECT)),
                 ("CLAHE, окно 65×65", lambda: clahe(PixelBuffer(big), "sliding", 32, 8, 2.0, BORDER_REFLECT)),
                 ("CLAHE, тайлы 8×8", lambda: clahe(PixelBuffer(big), "tiles", 32, 8, 2.0, BORDER_REFLECT)),
                 ("Эквализация", lambda: contrast.equalize(PixelBuffer(big))),
                 ("Линейное контрастирование", lambda: contrast.linear_contrast(PixelBuffer(big)))]:
    parallel.set_threads(1)
    t1 = tm(fn, rep=2)
    parallel.set_threads(nmax)
    tn = tm(fn, rep=2)
    thr.append({"op": name, "t1": t1, "tn": tn, "speedup": round(t1 / tn, 2)})
perf["threads"] = thr
perf["nthreads"] = nmax
M["perf"] = perf

with open(os.path.join(ROOT, "report", "measurements.json"), "w", encoding="utf-8") as fp:
    json.dump(M, fp, ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1))

# =============================== 8. Локальный контраст (среднее локальное σ в окне 15×15)
from app.algorithms.convolution import box_kernel_1d  # noqa: E402


def local_sigma(buf):
    f = buf.luminance_f32()
    k = box_kernel_1d(15)
    m1 = separable_filter(f, k, k, BORDER_REFLECT)
    m2 = separable_filter(f * f, k, k, BORDER_REFLECT)
    return round(float(np.sqrt(np.maximum(m2 - m1 * m1, 0)).mean()), 2)


loc = {}
for key, name, ops in [
    ("page", "07_uneven_light_page.png", [("global", "equalize", {}),
                                          ("noclip", "clahe", {"radius": 16, "clip": 100.0}),
                                          ("clahe", "clahe", {"radius": 16, "clip": 3.0})]),
    ("color", "08_uneven_light_color.png", [("global", "equalize", {}),
                                            ("sliding", "clahe", {"radius": 32, "clip": 2.5}),
                                            ("tiles", "clahe", {"variant": "tiles", "tiles": 8, "clip": 2.5})]),
    ("noisy", "15_shapes_noisy.png", [("global", "equalize", {}),
                                      ("clahe", "clahe", {"radius": 32, "clip": 2.0})])]:
    b = load(name)
    loc[key] = {"src": local_sigma(b)}
    for k2, op, p in ops:
        loc[key][k2] = local_sigma(run_operation(op, b, p).buffer)
M["local_sigma"] = loc
with open(os.path.join(ROOT, "report", "measurements.json"), "w", encoding="utf-8") as fp:
    json.dump(M, fp, ensure_ascii=False, indent=1)
print(loc)
