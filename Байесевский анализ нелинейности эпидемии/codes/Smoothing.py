# -*- coding: utf-8 -*-
"""
GUI для сглаживания и фильтрации временного ряда из JSON формата:

    {"days":[{"total":[{"index":"0","real":"...", ...}, ...], "day":"1"}, ...]}

Поддерживаются:
    - скользящее среднее (окно по умолчанию ±3 дня, т.е. 7 точек)
    - фильтр Савицкого–Голея (окно/порядок настраиваются)
    - фильтр Калмана (1D, Q/R настраиваются)
    - комбинация Savitzky–Golay → Kalman

Серия выбирается по полю "index", поле значения — из real / predicted / bestrun.
Запятые в числах интерпретируются как десятичный разделитель.

На графике ВСЕГДА одна синяя линия:
    - до применения фильтра — исходный ряд,
    - после применения — сглаженный ряд.
Кнопка «↺ Сброс» возвращает график к исходному ряду.

Дополнительная вкладка «Сравнение» показывает сразу все методы поверх оригинала.
"""

import os
import json
import math
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ============================================================
# Вспомогательные функции
# ============================================================

def _to_float(v):
    """'0,8' -> 0.8; '0.8' -> 0.8; None/''/NaN -> nan."""
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().replace("\u00a0", "").replace(" ", "")
    if s == "" or s.lower() in ("nan", "none", "null"):
        return float("nan")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_series_from_json(path, series_index=1, value_field="real"):
    """
    Читает JSON и возвращает (days, values) — плотные массивы,
    без пропусков (дни с нечисловым значением ПРОПУСКАЮТСЯ).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    days_json = data.get("days")
    if not isinstance(days_json, list) or not days_json:
        raise ValueError("В JSON нет массива 'days' или он пуст.")

    days = []
    values = []
    for entry in days_json:
        totals = entry.get("total") or []
        if series_index < 0 or series_index >= len(totals):
            continue
        it = totals[series_index]
        if not isinstance(it, dict):
            continue
        v = _to_float(it.get(value_field, float("nan")))
        if math.isnan(v):
            continue
        try:
            day = int(entry.get("day"))
        except (TypeError, ValueError):
            day = len(days) + 1
        days.append(day)
        values.append(v)

    if not values:
        raise ValueError(
            "Не удалось извлечь ни одной точки из days[].total[]."
        )

    return np.asarray(days, dtype=int), np.asarray(values, dtype=float)


def scan_series(path, value_field="real"):
    """Сканирует все серии: {index: {'n_total':..., 'n_nonzero':..., 'first_day':..., 'last_val':...}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    days = data.get("days") or []

    stats = {}
    for entry in days:
        try:
            day = int(entry.get("day"))
        except (TypeError, ValueError):
            continue
        for it in (entry.get("total") or []):
            try:
                idx = int(it.get("index"))
            except (TypeError, ValueError):
                continue
            v = _to_float(it.get(value_field, float("nan")))
            s = stats.setdefault(idx, {
                "n_total": 0, "n_nonzero": 0,
                "first_day": None, "last_val": None,
            })
            s["n_total"] += 1
            if not math.isnan(v) and v != 0:
                s["n_nonzero"] += 1
                if s["first_day"] is None:
                    s["first_day"] = day
                s["last_val"] = v
    return stats


# ============================================================
# Сглаживание / фильтрация
# ============================================================

def moving_average(y, window_half=3):
    """
    Скользящее среднее с окном [t-half, t+half] включительно.
    window_half=3 -> окно шириной 7 (3 дня до, 3 после, текущий).
    На границах окно сужается до доступных точек.
    """
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window_half)
        hi = min(n - 1, i + window_half)
        out[i] = np.mean(y[lo:hi + 1])
    return out


def savitzky_golay_filter(y, window_length=7, polyorder=2):
    """
    Фильтр Савицкого–Голея.
    window_length — обязательно нечётное, > polyorder.
    """
    if not HAS_SCIPY:
        raise RuntimeError(
            "scipy не установлен. Установите: pip install scipy"
        )
    n = len(y)
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    if wl < 3:
        return y.copy()
    po = min(int(polyorder), wl - 1)
    return savgol_filter(y, window_length=wl, polyorder=po, mode="interp")


def kalman_1d(y, Q=1e-4, R=1e-2, x0=None, P0=1.0):
    """
    Простой 1D-фильтр Калмана для скалярного ряда.

    Модель: x_k = x_{k-1} + w, w ~ N(0, Q)
            z_k = x_k + v,     v ~ N(0, R)

    Q — дисперсия шума процесса (насколько сигнал "дышит"),
    R — дисперсия шума измерений (насколько шумные данные).
    """
    n = len(y)
    x = np.zeros(n)
    if x0 is None:
        x[0] = y[0]
    else:
        x[0] = float(x0)
    P = float(P0)

    for k in range(1, n):
        # Prediction
        x_pred = x[k - 1]
        P_pred = P + Q
        # Update
        K = P_pred / (P_pred + R)
        x[k] = x_pred + K * (y[k] - x_pred)
        P = (1.0 - K) * P_pred

    return x


def apply_pipeline(y, mode,
                   ma_half=3,
                   sg_window=7, sg_order=2,
                   kalman_Q=1e-4, kalman_R=1e-2,
                   comb_sg_window=7, comb_sg_order=2,
                   comb_kalman_Q=1e-4, comb_kalman_R=1e-2):
    """
    Применяет выбранный метод к ряду y.

    mode:
        "ma"        — скользящее среднее
        "sg"        — Савицкий–Голей
        "kalman"    — Калман
        "sg_kalman" — SG, затем Калман
    """
    mode = mode.lower()

    if mode == "ma":
        return moving_average(y, window_half=int(ma_half))

    if mode == "sg":
        return savitzky_golay_filter(y, sg_window, sg_order)

    if mode == "kalman":
        return kalman_1d(y, Q=kalman_Q, R=kalman_R)

    if mode == "sg_kalman":
        y1 = savitzky_golay_filter(y, comb_sg_window, comb_sg_order)
        return kalman_1d(y1, Q=comb_kalman_Q, R=comb_kalman_R)

    raise ValueError(f"Неизвестный режим: {mode}")


# ============================================================
# GUI
# ============================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Сглаживание и фильтрация временного ряда (JSON)")
        self.geometry("1250x880")
        self.minsize(1100, 760)

        self.path_json = tk.StringVar()
        self.path_out = tk.StringVar(value=os.path.abspath("smoothed.xlsx"))
        self.series_idx = tk.IntVar(value=1)
        self.value_field = tk.StringVar(value="real")

        self.mode = tk.StringVar(value="ma")

        # Параметры скользящего среднего
        self.ma_half = tk.IntVar(value=3)

        # Параметры Савицкого–Голея
        self.sg_window = tk.IntVar(value=7)
        self.sg_order = tk.IntVar(value=2)

        # Параметры Калмана
        self.kalman_Q = tk.DoubleVar(value=1e-4)
        self.kalman_R = tk.DoubleVar(value=1e-2)

        # Параметры комбинированного режима
        self.comb_sg_window = tk.IntVar(value=7)
        self.comb_sg_order = tk.IntVar(value=2)
        self.comb_kalman_Q = tk.DoubleVar(value=1e-4)
        self.comb_kalman_R = tk.DoubleVar(value=1e-2)

        self.days = None
        self.y_raw = None
        self.y_filtered = None

        self._build_widgets()

    def _build_widgets(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        # ----- Файл и серия -----
        file_frame = ttk.LabelFrame(top, text=" Входной JSON ", padding=8)
        file_frame.pack(side=tk.TOP, fill=tk.X)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="JSON-файл:").grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Entry(file_frame, textvariable=self.path_json).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(file_frame, text="Обзор…",
                   command=self._pick_json).grid(row=0, column=2, padx=4)
        ttk.Button(file_frame, text="Загрузить",
                   command=self._load_json).grid(row=0, column=3, padx=4)

        ttk.Label(file_frame, text="Серия (index):").grid(
            row=1, column=0, sticky="w", pady=2)
        series_row = ttk.Frame(file_frame)
        series_row.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Spinbox(series_row, from_=0, to=50, width=6,
                    textvariable=self.series_idx).pack(side=tk.LEFT)
        ttk.Label(series_row, text="   поле:").pack(side=tk.LEFT)
        ttk.Combobox(series_row, width=12, state="readonly",
                     textvariable=self.value_field,
                     values=("real", "predicted", "bestrun")
                     ).pack(side=tk.LEFT, padx=4)
        ttk.Button(series_row, text="Сканировать серии",
                   command=self._scan_series).pack(side=tk.LEFT, padx=8)

        ttk.Label(file_frame, text="Куда сохранить результат:").grid(
            row=2, column=0, sticky="w", pady=2)
        ttk.Entry(file_frame, textvariable=self.path_out).grid(
            row=2, column=1, sticky="ew", padx=4)
        ttk.Button(file_frame, text="Обзор…",
                   command=self._pick_out).grid(row=2, column=2, padx=4)

        # ----- Метод и параметры -----
        method_frame = ttk.LabelFrame(top, text=" Метод ", padding=8)
        method_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        for col, (val, text) in enumerate([
            ("ma", "Скользящее среднее (±N дней)"),
            ("sg", "Савицкий–Голей"),
            ("kalman", "Калман (1D)"),
            ("sg_kalman", "SG → Калман"),
        ]):
            ttk.Radiobutton(
                method_frame, text=text, value=val, variable=self.mode
            ).grid(row=0, column=col, sticky="w", padx=8, pady=2)

        # Параметры скользящего среднего
        ma_frame = ttk.LabelFrame(top, text=" Параметры скользящего среднего ", padding=8)
        ma_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Label(ma_frame, text="Полуширина окна (дней):").grid(
            row=0, column=0, sticky="w")
        ttk.Spinbox(ma_frame, from_=1, to=50, width=6,
                    textvariable=self.ma_half).grid(row=0, column=1, padx=6)
        ttk.Label(ma_frame, text="(окно = 2·N + 1 точек)").grid(
            row=0, column=2, sticky="w", padx=6)

        # Параметры Савицкого–Голея
        sg_frame = ttk.LabelFrame(top, text=" Параметры Савицкого–Голея ", padding=8)
        sg_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Label(sg_frame, text="Окно (нечётное):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(sg_frame, from_=3, to=501, increment=2, width=6,
                    textvariable=self.sg_window).grid(row=0, column=1, padx=6)
        ttk.Label(sg_frame, text="Порядок полинома:").grid(
            row=0, column=2, sticky="w", padx=6)
        ttk.Spinbox(sg_frame, from_=1, to=6, width=6,
                    textvariable=self.sg_order).grid(row=0, column=3, padx=6)

        # Параметры Калмана
        kf_frame = ttk.LabelFrame(top, text=" Параметры Калмана ", padding=8)
        kf_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Label(kf_frame, text="Q (шум процесса):").grid(row=0, column=0, sticky="w")
        ttk.Entry(kf_frame, textvariable=self.kalman_Q, width=12).grid(
            row=0, column=1, padx=6)
        ttk.Label(kf_frame, text="R (шум наблюдений):").grid(
            row=0, column=2, sticky="w", padx=6)
        ttk.Entry(kf_frame, textvariable=self.kalman_R, width=12).grid(
            row=0, column=3, padx=6)

        # Параметры комбинированного режима
        comb_frame = ttk.LabelFrame(top, text=" Параметры SG → Калман ", padding=8)
        comb_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Label(comb_frame, text="SG окно:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(comb_frame, from_=3, to=501, increment=2, width=6,
                    textvariable=self.comb_sg_window).grid(row=0, column=1, padx=6)
        ttk.Label(comb_frame, text="SG порядок:").grid(
            row=0, column=2, sticky="w", padx=6)
        ttk.Spinbox(comb_frame, from_=1, to=6, width=6,
                    textvariable=self.comb_sg_order).grid(row=0, column=3, padx=6)
        ttk.Label(comb_frame, text="Kalman Q:").grid(
            row=1, column=0, sticky="w", pady=4)
        ttk.Entry(comb_frame, textvariable=self.comb_kalman_Q, width=12).grid(
            row=1, column=1, padx=6)
        ttk.Label(comb_frame, text="Kalman R:").grid(
            row=1, column=2, sticky="w", padx=6)
        ttk.Entry(comb_frame, textvariable=self.comb_kalman_R, width=12).grid(
            row=1, column=3, padx=6)

        # ----- Кнопки -----
        btn_frame = ttk.Frame(self, padding=(8, 8))
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_run = ttk.Button(btn_frame, text="▶ Применить",
                                  command=self._run)
        self.btn_reset = ttk.Button(btn_frame, text="↺ Сброс",
                                    command=self._reset, state=tk.DISABLED)
        self.btn_save = ttk.Button(btn_frame, text="💾 Сохранить результат",
                                   command=self._save, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=4)
        self.btn_reset.pack(side=tk.LEFT, padx=4)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(btn_frame, text="Готов", foreground="#555")
        self.status.pack(side=tk.LEFT, padx=12)

        # ----- Notebook: график, сравнение, лог, таблица -----
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Вкладка 1: одиночный график
        gf = ttk.Frame(nb)
        nb.add(gf, text="График")
        self.fig = Figure(figsize=(11.5, 5.0), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, gf)
        toolbar.update()

        # Вкладка 2: сравнение всех методов
        cf = ttk.Frame(nb)
        nb.add(cf, text="Сравнение")
        self.fig_cmp = Figure(figsize=(11.5, 5.0), dpi=100)
        self.ax_cmp = self.fig_cmp.add_subplot(1, 1, 1)
        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=cf)
        self.canvas_cmp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_cmp = NavigationToolbar2Tk(self.canvas_cmp, cf)
        toolbar_cmp.update()

        # Вкладка 3: логи
        lf = ttk.Frame(nb, padding=8)
        nb.add(lf, text="Логи")
        self.log_text = scrolledtext.ScrolledText(
            lf, font=("Consolas", 10), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

        # Вкладка 4: таблица данных
        tf = ttk.Frame(nb, padding=8)
        nb.add(tf, text="Данные (первые 100 строк)")
        self.table_text = scrolledtext.ScrolledText(
            tf, font=("Consolas", 9), wrap=tk.NONE)
        self.table_text.pack(fill=tk.BOTH, expand=True)

    # ---------- операции с файлами ----------

    def _pick_json(self):
        p = filedialog.askopenfilename(
            title="Выберите JSON-файл",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if p:
            self.path_json.set(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"),
                       ("CSV files", "*.csv"),
                       ("All files", "*.*")],
            initialfile="smoothed.xlsx")
        if p:
            self.path_out.set(p)

    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, txt, color="#555"):
        self.status.configure(text=txt, foreground=color)
        self.update_idletasks()

    def _load_json(self):
        p = self.path_json.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Укажите существующий JSON-файл.")
            return
        try:
            idx = int(self.series_idx.get())
            field = self.value_field.get().strip() or "real"

            self.days, self.y_raw = load_series_from_json(
                p, series_index=idx, value_field=field)

            self._log(f"[load] Загружено {len(self.y_raw)} точек "
                      f"(серия index={idx}, поле='{field}')")
            self._log(f"[load] min={np.min(self.y_raw):.4g}, "
                      f"max={np.max(self.y_raw):.4g}, "
                      f"mean={np.mean(self.y_raw):.4g}")

            self.y_filtered = None
            self.btn_save.configure(state=tk.DISABLED)
            self.btn_reset.configure(state=tk.DISABLED)
            self._plot()
            self._plot_comparison()
            self._fill_table()
            self._set_status("Данные загружены", "#0a0")

        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            self._log("[load][ERROR] " + str(e))

    def _scan_series(self):
        p = self.path_json.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Сначала укажите JSON-файл.")
            return
        try:
            field = self.value_field.get().strip() or "real"
            stats = scan_series(p, value_field=field)
            self._log(f"[scan] поле='{field}', найдено серий: {len(stats)}")
            for idx in sorted(stats):
                s = stats[idx]
                self._log(
                    f"  index={idx:2d}: всего={s['n_total']:4d}, "
                    f"ненулевых={s['n_nonzero']:4d}, "
                    f"первый день={s['first_day']}, "
                    f"последнее значение={s['last_val']}")
            if stats:
                best = max(stats.items(), key=lambda kv: kv[1]["n_nonzero"])
                self._log(f"[scan] Похоже, данные в index={best[0]} "
                          f"(ненулевых={best[1]['n_nonzero']}). "
                          f"Подставил в поле «Серия».")
                self.series_idx.set(best[0])
        except Exception as e:
            messagebox.showerror("Ошибка сканирования", str(e))
            self._log("[scan][ERROR] " + str(e))

    # ---------- сглаживание ----------

    def _run(self):
        if self.y_raw is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите JSON.")
            return
        try:
            mode = self.mode.get()
            y = np.asarray(self.y_raw, dtype=float)

            self.y_filtered = apply_pipeline(
                y, mode,
                ma_half=int(self.ma_half.get()),
                sg_window=int(self.sg_window.get()),
                sg_order=int(self.sg_order.get()),
                kalman_Q=float(self.kalman_Q.get()),
                kalman_R=float(self.kalman_R.get()),
                comb_sg_window=int(self.comb_sg_window.get()),
                comb_sg_order=int(self.comb_sg_order.get()),
                comb_kalman_Q=float(self.comb_kalman_Q.get()),
                comb_kalman_R=float(self.comb_kalman_R.get()),
            )

            rmse = float(np.sqrt(np.mean((y - self.y_filtered) ** 2)))
            mae = float(np.mean(np.abs(y - self.y_filtered)))

            self._log(f"[filter] режим={mode}, "
                      f"RMSE(raw - filtered)={rmse:.4g}, MAE={mae:.4g}")

            self._plot()
            self._plot_comparison()
            self._fill_table()
            self.btn_save.configure(state=tk.NORMAL)
            self.btn_reset.configure(state=tk.NORMAL)
            self._set_status("Готово", "#0a0")

        except Exception as e:
            tb = traceback.format_exc()
            self._log("[filter][ERROR] " + str(e))
            self._log(tb)
            messagebox.showerror("Ошибка фильтрации", str(e))

    def _reset(self):
        """Убирает результат фильтрации: снова показывается исходный ряд."""
        if self.y_raw is None:
            return
        self.y_filtered = None
        self.btn_save.configure(state=tk.DISABLED)
        self.btn_reset.configure(state=tk.DISABLED)
        self._plot()
        self._plot_comparison()
        self._fill_table()
        self._log("[reset] Результат фильтрации сброшен, показан исходный ряд.")
        self._set_status("Сброшено", "#555")

    # ---------- отображение ----------

    def _plot(self):
        self.ax.clear()
        if self.y_raw is None:
            self.canvas.draw_idle()
            return

        x = np.arange(len(self.y_raw))

        if self.y_filtered is not None:
            # показываем ТОЛЬКО сглаженный ряд (одна синяя линия)
            self.ax.plot(x, self.y_filtered, color="C0", lw=1.8,
                         label=f"Обработанный ряд ({self.mode.get()})")
        else:
            # пока не применён фильтр — показываем исходный
            self.ax.plot(x, self.y_raw, color="C0", lw=1.5,
                         label="Исходный ряд")

        self.ax.set_xlabel("Индекс (точка)")
        self.ax.set_ylabel("Значение")
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc="best")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_comparison(self):
        """График сравнения: оригинал + все методы одновременно."""
        self.ax_cmp.clear()
        if self.y_raw is None:
            self.canvas_cmp.draw_idle()
            return

        x = np.arange(len(self.y_raw))
        y = np.asarray(self.y_raw, dtype=float)

        # Оригинал — чёрная жирная линия
        self.ax_cmp.plot(x, y, color="black", lw=2.0, alpha=0.85,
                         label="Исходный ряд")

        # Скользящее среднее
        try:
            y_ma = moving_average(y, window_half=int(self.ma_half.get()))
            self.ax_cmp.plot(x, y_ma, color="C0", lw=1.4,
                             label=f"Скользящее среднее (±{self.ma_half.get()})")
        except Exception as e:
            self._log(f"[cmp][MA][ERROR] {e}")

        # Савицкий–Голей
        try:
            y_sg = savitzky_golay_filter(
                y, int(self.sg_window.get()), int(self.sg_order.get()))
            self.ax_cmp.plot(x, y_sg, color="C1", lw=1.4,
                             label=f"Савицкий–Голей (окно={self.sg_window.get()}, "
                                   f"порядок={self.sg_order.get()})")
        except Exception as e:
            self._log(f"[cmp][SG][ERROR] {e}")

        # Калман
        try:
            y_kf = kalman_1d(
                y, Q=float(self.kalman_Q.get()), R=float(self.kalman_R.get()))
            self.ax_cmp.plot(x, y_kf, color="C2", lw=1.4,
                             label=f"Калман (Q={self.kalman_Q.get():g}, "
                                   f"R={self.kalman_R.get():g})")
        except Exception as e:
            self._log(f"[cmp][KF][ERROR] {e}")

        # SG → Калман
        try:
            y_sg1 = savitzky_golay_filter(
                y, int(self.comb_sg_window.get()), int(self.comb_sg_order.get()))
            y_comb = kalman_1d(
                y_sg1,
                Q=float(self.comb_kalman_Q.get()),
                R=float(self.comb_kalman_R.get()))
            self.ax_cmp.plot(x, y_comb, color="C3", lw=1.4, ls="--",
                             label="SG → Калман")
        except Exception as e:
            self._log(f"[cmp][SG+KF][ERROR] {e}")

        self.ax_cmp.set_xlabel("Индекс (точка)")
        self.ax_cmp.set_ylabel("Значение")
        self.ax_cmp.grid(alpha=0.3)
        self.ax_cmp.legend(loc="best", fontsize=9)
        self.fig_cmp.tight_layout()
        self.canvas_cmp.draw_idle()

    def _fill_table(self):
        self.table_text.delete("1.0", tk.END)
        if self.y_raw is None:
            return

        header = f"{'i':>6} {'day':>8} {'raw':>16}"
        if self.y_filtered is not None:
            header += f" {'filtered':>16}"
        self.table_text.insert(tk.END, header + "\n")
        self.table_text.insert(tk.END, "-" * len(header) + "\n")

        n = len(self.y_raw)
        for i in range(min(100, n)):
            day = int(self.days[i]) if self.days is not None else i
            line = f"{i:>6} {day:>8} {self.y_raw[i]:>16.6g}"
            if self.y_filtered is not None:
                line += f" {self.y_filtered[i]:>16.6g}"
            self.table_text.insert(tk.END, line + "\n")
        if n > 100:
            self.table_text.insert(tk.END, f"... ещё {n - 100} строк\n")

    # ---------- сохранение ----------

    def _save(self):
        if self.y_raw is None or self.y_filtered is None:
            messagebox.showwarning("Нет данных", "Сначала примените фильтр.")
            return
        path = self.path_out.get()
        if not path:
            messagebox.showwarning("Нет пути", "Укажите путь для сохранения.")
            return
        try:
            df = pd.DataFrame({
                "i":        np.arange(len(self.y_raw)),
                "day":      self.days if self.days is not None else np.arange(len(self.y_raw)),
                "raw":      self.y_raw,
                "filtered": self.y_filtered,
                "residual": self.y_raw - self.y_filtered,
            })

            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                df.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as xw:
                    df.to_excel(xw, sheet_name="filtered", index=False)

                    params = {
                        "mode": self.mode.get(),
                        "ma_half": self.ma_half.get(),
                        "sg_window": self.sg_window.get(),
                        "sg_order": self.sg_order.get(),
                        "kalman_Q": self.kalman_Q.get(),
                        "kalman_R": self.kalman_R.get(),
                        "comb_sg_window": self.comb_sg_window.get(),
                        "comb_sg_order": self.comb_sg_order.get(),
                        "comb_kalman_Q": self.comb_kalman_Q.get(),
                        "comb_kalman_R": self.comb_kalman_R.get(),
                    }
                    pd.DataFrame(
                        list(params.items()),
                        columns=["param", "value"]
                    ).to_excel(xw, sheet_name="params", index=False)

            self._log(f"[save] Сохранено: {path}")
            self._set_status("Сохранено", "#0a0")
            messagebox.showinfo("Успех", f"Сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))
            self._log("[save][ERROR] " + str(e))


if __name__ == "__main__":
    if not HAS_SCIPY:
        print("ВНИМАНИЕ: scipy не найден, режимы SG и SG→Калман "
              "будут недоступны. Установите: pip install scipy")
    App().mainloop()