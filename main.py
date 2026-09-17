#!/usr/bin/env python3
"""
Лабораторная работа 1 — цветовые модели.
Вариант №4: RGB ↔ HSV ↔ LAB.

Запуск: python main.py
"""

from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, ttk
from typing import Callable, Dict, List, Optional, Tuple

import color_converter as cc


class ColorApp:
    """GUI-приложение для интерактивной работы с RGB, HSV и LAB."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Color Model Converter — Вариант №4 (RGB ↔ HSV ↔ LAB)")
        self.root.minsize(720, 680)

        self._updating = False
        self._active_source: Optional[str] = None  # 'rgb' | 'hsv' | 'lab'

        # Внутреннее состояние (точные значения, не округлённые из GUI)
        self._r = 128
        self._g = 128
        self._b = 128
        self._h = 0.0
        self._s = 0.0
        self._v = 50.0
        self._L = 53.59
        self._a = 0.0
        self._b_lab = 0.0

        self._sync_from_rgb(initial=True)

        self._rgb_entries: Dict[str, tk.StringVar] = {}
        self._hsv_entries: Dict[str, tk.StringVar] = {}
        self._lab_entries: Dict[str, tk.StringVar] = {}
        self._rgb_scales: Dict[str, tk.Scale] = {}
        self._hsv_scales: Dict[str, tk.Scale] = {}
        self._lab_scales: Dict[str, tk.Scale] = {}

        self.status_var = tk.StringVar(value="Статус: готово")
        self.hex_var = tk.StringVar(value=cc.rgb_to_hex(self._r, self._g, self._b))

        self._build_ui()
        self._refresh_ui_from_state()

    # --- Преобразования (делегирование в color_converter) ---

    def rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        return cc.rgb_to_hsv(r, g, b)

    def hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int, bool]:
        return cc.hsv_to_rgb(h, s, v)

    def rgb_to_xyz(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        return cc.rgb_to_xyz(r, g, b)

    def xyz_to_rgb(self, x: float, y: float, z: float) -> Tuple[int, int, int, bool]:
        return cc.xyz_to_rgb(x, y, z)

    def xyz_to_lab(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return cc.xyz_to_lab(x, y, z)

    def lab_to_xyz(self, L: float, a: float, b: float) -> Tuple[float, float, float]:
        return cc.lab_to_xyz(L, a, b)

    def rgb_to_lab(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        return cc.rgb_to_lab(r, g, b)

    def lab_to_rgb(self, L: float, a: float, b: float) -> Tuple[int, int, int, bool]:
        return cc.lab_to_rgb(L, a, b)

    # --- Синхронизация состояния ---

    def _sync_from_rgb(self, initial: bool = False) -> None:
        self._h, self._s, self._v = self.rgb_to_hsv(self._r, self._g, self._b)
        self._L, self._a, self._b_lab = self.rgb_to_lab(self._r, self._g, self._b)

    def _sync_from_hsv(self) -> bool:
        r, g, b, clipped = self.hsv_to_rgb(self._h, self._s, self._v)
        self._r, self._g, self._b = r, g, b
        self._L, self._a, self._b_lab = self.rgb_to_lab(r, g, b)
        return clipped

    def _sync_from_lab(self) -> bool:
        r, g, b, clipped = self.lab_to_rgb(self._L, self._a, self._b_lab)
        self._r, self._g, self._b = r, g, b
        self._h, self._s, self._v = self.rgb_to_hsv(r, g, b)
        return clipped

    def update_from_rgb(self) -> None:
        self._sync_from_rgb()
        self._refresh_ui_from_state()
        self._set_status_ok()

    def update_from_hsv(self, report_clip: bool = False) -> None:
        clipped = self._sync_from_hsv()
        self._refresh_ui_from_state()
        if report_clip or clipped:
            self._set_status_clip()
        else:
            self._set_status_ok()

    def update_from_lab(self, report_clip: bool = False) -> None:
        clipped = self._sync_from_lab()
        self._refresh_ui_from_state()
        if report_clip or clipped:
            self._set_status_clip()
        else:
            self._set_status_ok()

    def choose_color(self) -> None:
        rgb, hex_color = colorchooser.askcolor(
            color=f"#{self._r:02x}{self._g:02x}{self._b:02x}",
            title="Выберите цвет",
        )
        if rgb is None:
            return
        self._r, self._g, self._b = (int(round(c)) for c in rgb)
        self._active_source = "rgb"
        self.update_from_rgb()

    def _set_status_ok(self) -> None:
        self.status_var.set("Статус: готово")

    def _set_status_clip(self) -> None:
        self.status_var.set(
            "Статус: некоторые значения вышли за допустимый диапазон и были ограничены."
        )

    def _set_status_error(self, message: str) -> None:
        self.status_var.set(f"Статус: {message}")

    # --- UI ---

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            outer,
            text="COLOR MODEL CONVERTER",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(0, 8))

        preview_frame = ttk.LabelFrame(outer, text="CURRENT COLOR", padding=10)
        preview_frame.pack(fill=tk.X, pady=6)

        self.preview = tk.Canvas(preview_frame, height=100, highlightthickness=1, highlightbackground="#888")
        self.preview.pack(fill=tk.X)
        ttk.Label(preview_frame, textvariable=self.hex_var, font=("Menlo", 12)).pack(pady=(6, 0))

        self._add_model_section(
            outer,
            "RGB",
            [("R", 0, 255, True), ("G", 0, 255, True), ("B", 0, 255, True)],
            self._rgb_entries,
            self._rgb_scales,
            self._on_rgb_slider,
            self._commit_rgb_entry,
        )
        self._add_model_section(
            outer,
            "HSV",
            [("H", 0, 360, False), ("S", 0, 100, False), ("V", 0, 100, False)],
            self._hsv_entries,
            self._hsv_scales,
            self._on_hsv_slider,
            self._commit_hsv_entry,
        )
        self._add_model_section(
            outer,
            "CIE LAB",
            [
                ("L", 0, 100, False),
                ("a", cc.LAB_A_MIN, cc.LAB_A_MAX, False),
                ("b", cc.LAB_B_MIN, cc.LAB_B_MAX, False),
            ],
            self._lab_entries,
            self._lab_scales,
            self._on_lab_slider,
            self._commit_lab_entry,
        )

        ttk.Button(outer, text="Выбрать цвет", command=self.choose_color).pack(pady=10)
        ttk.Label(outer, textvariable=self.status_var, wraplength=680).pack(anchor=tk.W, pady=(4, 0))

    def _add_model_section(
        self,
        parent: ttk.Frame,
        title: str,
        components: List[Tuple[str, float, float, bool]],
        entries: Dict[str, tk.StringVar],
        scales: Dict[str, tk.Scale],
        slider_cmd: Callable[[str, float], None],
        entry_commit: Callable[[str], None],
    ) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, pady=6)

        for name, vmin, vmax, is_int in components:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=3)

            ttk.Label(row, text=f"{name}:", width=3).pack(side=tk.LEFT)

            var = tk.StringVar()
            entries[name] = var
            entry = ttk.Entry(row, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=(4, 8))
            entry.bind("<Return>", lambda _e, n=name: entry_commit(n))
            entry.bind("<FocusOut>", lambda _e, n=name: entry_commit(n))

            resolution = 1 if is_int else 0.01
            scale = tk.Scale(
                row,
                from_=vmin,
                to=vmax,
                orient=tk.HORIZONTAL,
                length=420,
                resolution=resolution,
                showvalue=False,
                command=lambda val, n=name: slider_cmd(n, float(val)),
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            scales[name] = scale

    def _refresh_ui_from_state(self) -> None:
        self._updating = True
        try:
            self._rgb_entries["R"].set(str(self._r))
            self._rgb_entries["G"].set(str(self._g))
            self._rgb_entries["B"].set(str(self._b))
            self._rgb_scales["R"].set(self._r)
            self._rgb_scales["G"].set(self._g)
            self._rgb_scales["B"].set(self._b)

            self._hsv_entries["H"].set(f"{self._h:.2f}")
            self._hsv_entries["S"].set(f"{self._s:.2f}")
            self._hsv_entries["V"].set(f"{self._v:.2f}")
            self._hsv_scales["H"].set(self._h)
            self._hsv_scales["S"].set(self._s)
            self._hsv_scales["V"].set(self._v)

            self._lab_entries["L"].set(f"{self._L:.2f}")
            self._lab_entries["a"].set(f"{self._a:.2f}")
            self._lab_entries["b"].set(f"{self._b_lab:.2f}")
            self._lab_scales["L"].set(self._L)
            self._lab_scales["a"].set(self._a)
            self._lab_scales["b"].set(self._b_lab)

            hex_val = cc.rgb_to_hex(self._r, self._g, self._b)
            self.hex_var.set(f"HEX: {hex_val}")
            self.preview.configure(bg=hex_val)
        finally:
            self._updating = False

    def update_ui(self) -> None:
        """Публичный метод обновления интерфейса (алиас)."""
        self._refresh_ui_from_state()

    # --- Обработчики RGB ---

    def _on_rgb_slider(self, name: str, value: float) -> None:
        if self._updating:
            return
        self._active_source = "rgb"
        iv = int(round(value))
        if name == "R":
            self._r = iv
        elif name == "G":
            self._g = iv
        else:
            self._b = iv
        self.update_from_rgb()

    def _commit_rgb_entry(self, _name: str) -> None:
        if self._updating:
            return
        parsed, err = self._parse_rgb_entries()
        if err:
            self._set_status_error(err)
            self._refresh_ui_from_state()
            return
        self._r, self._g, self._b = parsed
        self._active_source = "rgb"
        self.update_from_rgb()

    def _parse_rgb_entries(self) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
        try:
            r = self._parse_int_field(self._rgb_entries["R"].get(), "R", 0, 255)
            g = self._parse_int_field(self._rgb_entries["G"].get(), "G", 0, 255)
            b = self._parse_int_field(self._rgb_entries["B"].get(), "B", 0, 255)
            return (r, g, b), None
        except ValueError as e:
            return None, str(e)

    # --- Обработчики HSV ---

    def _on_hsv_slider(self, name: str, value: float) -> None:
        if self._updating:
            return
        self._active_source = "hsv"
        if name == "H":
            self._h = value
        elif name == "S":
            self._s = value
        else:
            self._v = value
        self.update_from_hsv(report_clip=False)

    def _commit_hsv_entry(self, _name: str) -> None:
        if self._updating:
            return
        parsed, err = self._parse_hsv_entries()
        if err:
            self._set_status_error(err)
            self._refresh_ui_from_state()
            return
        self._h, self._s, self._v = parsed
        self._active_source = "hsv"
        self.update_from_hsv(report_clip=True)

    def _parse_hsv_entries(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
        try:
            h = self._parse_float_field(self._hsv_entries["H"].get(), "H", 0, 360)
            s = self._parse_float_field(self._hsv_entries["S"].get(), "S", 0, 100)
            v = self._parse_float_field(self._hsv_entries["V"].get(), "V", 0, 100)
            return (h, s, v), None
        except ValueError as e:
            return None, str(e)

    # --- Обработчики LAB ---

    def _on_lab_slider(self, name: str, value: float) -> None:
        if self._updating:
            return
        self._active_source = "lab"
        if name == "L":
            self._L = value
        elif name == "a":
            self._a = value
        else:
            self._b_lab = value
        self.update_from_lab(report_clip=False)

    def _commit_lab_entry(self, _name: str) -> None:
        if self._updating:
            return
        parsed, err = self._parse_lab_entries()
        if err:
            self._set_status_error(err)
            self._refresh_ui_from_state()
            return
        self._L, self._a, self._b_lab = parsed
        self._active_source = "lab"
        self.update_from_lab(report_clip=True)

    def _parse_lab_entries(self) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
        try:
            L = self._parse_float_field(self._lab_entries["L"].get(), "L", 0, 100)
            a = self._parse_float_field(self._lab_entries["a"].get(), "a", cc.LAB_A_MIN, cc.LAB_A_MAX)
            b = self._parse_float_field(self._lab_entries["b"].get(), "b", cc.LAB_B_MIN, cc.LAB_B_MAX)
            return (L, a, b), None
        except ValueError as e:
            return None, str(e)

    # --- Валидация ввода ---

    @staticmethod
    def _parse_int_field(text: str, label: str, lo: int, hi: int) -> int:
        text = text.strip()
        if not text:
            raise ValueError(f"Введите числовое значение для {label}.")
        try:
            val = int(text)
        except ValueError:
            raise ValueError(f"Введите числовое значение для {label}.")
        if val < lo or val > hi:
            raise ValueError(f"Значение {label} должно находиться в диапазоне {lo}–{hi}.")
        return val

    @staticmethod
    def _parse_float_field(text: str, label: str, lo: float, hi: float) -> float:
        text = text.strip().replace(",", ".")
        if not text:
            raise ValueError(f"Введите числовое значение для {label}.")
        try:
            val = float(text)
        except ValueError:
            raise ValueError(f"Введите числовое значение для {label}.")
        if val < lo or val > hi:
            raise ValueError(f"Значение {label} должно находиться в диапазоне {lo:g}–{hi:g}.")
        return val


def main() -> None:
    root = tk.Tk()
    ColorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
