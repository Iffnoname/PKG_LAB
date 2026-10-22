
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.buffer import PixelBuffer, float_to_u8
from . import clahe as clahe_mod
from . import contrast, segmentation as seg
from .convolution import box_kernel_1d, gaussian_blur, separable_filter
from .histogram import stats


@dataclass
class OpResult:
    buffer: PixelBuffer
    info: str = ""
    extra: dict = field(default_factory=dict)
    seconds: float = 0.0


def _fmt_stats(s: dict) -> str:
    return (f"min={s['min']}, max={s['max']}, среднее={s['mean']:.2f}, "
            f"σ={s['sigma']:.2f}, уровней={s['levels']}")


def _seg_output(res: dict, gray_u8: np.ndarray, output: str, kind: str) -> PixelBuffer:
    mask = res["mask"]
    if output == "overlay":
        return PixelBuffer(seg.overlay(gray_u8, mask, 255, 40, 40))
    if output == "response":
        key = {"points": "response", "lines": "response", "log": "log"}.get(kind, "magnitude")
        a = res[key]
        if kind == "log":
            a = np.abs(a)
        return PixelBuffer.from_plane(seg.normalize_u8(a))
    if output in ("direction", "direction_edges"):
        mag = res["magnitude"]
        vmax = float(np.percentile(mag, 99.5)) or 1.0
        return PixelBuffer(seg.direction_map(mag, res["angle"], vmax, mask,
                                             output == "direction_edges"))
    if kind == "lines":
        return PixelBuffer(seg.color_lines(mask, res["index"], seg.LINE_COLORS))
    return PixelBuffer.from_plane(mask)


def run_operation(name: str, buf: PixelBuffer, p: dict, progress=None) -> OpResult:
    t0 = time.perf_counter()
    info = []
    extra = {}
    color = p.get("color", contrast.COLOR_HSV)
    border = int(p.get("border", 1))

    if name == "linear":
        out, meta = contrast.linear_contrast(buf, p.get("mode", "auto"), p.get("a", 0),
                                             p.get("b", 255), p.get("percent", 1.0), color)
        rng = ", ".join(f"[{a}; {b}]" for a, b in meta["ranges"])
        info.append(f"Линейное контрастирование, входной диапазон a..b: {rng} → [0; 255]")
    elif name == "equalize":
        out, _ = contrast.equalize(buf, color)
        info.append("Эквализация гистограммы " +
                    ("(серое)" if buf.is_gray else "(по каналам RGB)" if color == "rgb"
                     else "(только яркость V в HSV)"))
    elif name == "clahe":
        out, _ = clahe_mod.clahe(buf, p.get("variant", "sliding"), p.get("radius", 32),
                                 p.get("tiles", 8), p.get("clip", 2.0), border, color, progress)
        if p.get("variant", "sliding") == "tiles":
            info.append(f"CLAHE (тайлы {p.get('tiles', 8)}×{p.get('tiles', 8)}, лимит {p.get('clip', 2.0)})")
        else:
            w = 2 * int(p.get("radius", 32)) + 1
            info.append(f"CLAHE (скользящее окно {w}×{w}, лимит {p.get('clip', 2.0)})")
    elif name == "gamma":
        out, _ = contrast.gamma(buf, p.get("g", 1.0), color)
        info.append(f"Гамма-коррекция, γ = {p.get('g', 1.0)}")
    elif name == "negative":
        out, _ = contrast.negative(buf)
        info.append("Негатив y = 255 − x")
    elif name == "blur":
        planes = [buf.channel(c) for c in range(buf.channels)]
        res = []
        for i, pl in enumerate(planes):
            f = pl.astype(np.float32)
            if p.get("kind", "gauss") == "box":
                k = box_kernel_1d(p.get("size", 5))
                g = separable_filter(f, k, k, border)
            else:
                g = gaussian_blur(f, p.get("sigma", 1.5), border)
            res.append(float_to_u8(g, 1.0, 0.0))
            if progress:
                progress((i + 1) / len(planes))
        out = PixelBuffer(np.stack(res, axis=2))
        info.append("Сглаживание (разделяемая свёртка, два 1D-прохода)")
    elif name == "noise":
        rng = np.random.default_rng(p.get("seed", 1))
        n = rng.normal(0, p.get("sigma", 20.0), buf.data.shape)
        out = PixelBuffer(np.clip(buf.data + n, 0, 255).round().astype(np.uint8))
        info.append(f"Добавлен гауссов шум σ = {p.get('sigma', 20.0)}")
    else:
        gray = buf.luminance()
        f = gray.astype(np.float32)
        output = p.get("output", "mask")
        if name == "points":
            r = seg.detect_points(f, p.get("threshold", 50.0), border, p.get("polarity", 0), progress)
            info.append(f"Обнаружение точек: max|R| = {r['max']:.0f}, T = {r['T']:.0f}; "
                        f"найдено точек (8-связных областей): {r['count']}, пикселей: {r['pixels']}")
            extra["count"] = r["count"]
            out = _seg_output(r, gray, output, "points")
        elif name == "lines":
            r = seg.detect_lines(f, p.get("direction", "all"), p.get("threshold", 50.0), border,
                                 p.get("polarity", 1), progress)
            per = ", ".join(f"{seg.LINE_TITLES[k]}: {v}" for k, v in r["per_dir"].items() if v)
            info.append(f"Обнаружение линий ({seg.LINE_TITLES[p.get('direction', 'all')]}): "
                        f"T = {r['T']:.0f} из max {r['max']:.0f}; пикселей линий: {r['pixels']} ({per})")
            out = _seg_output(r, gray, output, "lines")
        elif name == "gradient":
            op = p.get("operator", "sobel")
            r = seg.gradient_edges(f, op, p.get("threshold", 20.0), border, p.get("sigma", 0.0),
                                   p.get("l1", False), progress)
            info.append(f"Оператор {op.capitalize()}: max|∇f| = {r['max']:.1f}, T = {r['T']:.1f}; "
                        f"пикселей края: {r['pixels']}")
            out = _seg_output(r, gray, output, "gradient")
            extra["angle"] = r["angle"]
        elif name == "log":
            r = seg.log_edges(f, p.get("sigma", 2.0), p.get("threshold", 5.0), border, progress)
            info.append(f"LoG (Марр — Хилдрет), σ = {p.get('sigma', 2.0)}: T = {r['T']:.2f}; "
                        f"пикселей края: {r['pixels']}")
            out = _seg_output(r, gray, output, "log")
        elif name == "canny":
            r = seg.canny(f, p.get("sigma", 1.4), p.get("high", 20.0), p.get("low", 40.0), border,
                          progress)
            info.append(f"Кэнни, σ = {p.get('sigma', 1.4)}: T_high = {r['high']:.1f}, "
                        f"T_low = {r['low']:.1f}; пикселей края: {r['pixels']}")
            out = _seg_output(r, gray, output, "canny")
            extra["subpixel"] = r["subpixel"]
            extra["angle"] = r["angle"]
        else:
            raise ValueError(f"Неизвестная операция {name}")

    if name in ("linear", "equalize", "clahe", "gamma", "negative", "blur", "noise"):
        info.append("До:    " + _fmt_stats(stats(buf.luminance())))
        info.append("После: " + _fmt_stats(stats(out.luminance())))
    dt = time.perf_counter() - t0
    if progress:
        progress(1.0)
    return OpResult(out, "\n".join(info), extra, dt)
