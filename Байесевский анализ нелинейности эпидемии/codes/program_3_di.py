# -*- coding: utf-8 -*-
"""
GUI для решения системы ОДУ (АБСОЛЮТНЫЕ значения):

    dS/dt = -(R(t)*gamma) * S * I + mu * (1 - S) + (k + p) * (1 - S - I)
    dI/dt =  R(t)*gamma  * S * I - gamma * I - mu * I

    где S, I — ДОЛИ населения (0..1),
    EPOP    — численность популяции (чел.).

ВСЕ выводы и метрики — в АБСОЛЮТНЫХ значениях:
    S_abs(t)     = S(t) * EPOP          (чел.)
    I_abs(t)     = I(t) * EPOP          (чел.)
    I_new(t)     = R(t)*gamma*S(t)*I(t) (доля/день)
    I_new_abs(t) = I_new(t) * EPOP      (чел./день)

Коридоры неопределённости (из MCMC, ДИ = mean ± kσ):
    Основной:  lo3s / hi3s  (±3σ)
    Доп:       lo2s / hi2s  (±2σ)
               lo1s / hi1s  (±1σ)

НИКАКОЙ АГРЕГАЦИИ СЦЕНАРИЕВ НЕТ:
    в качестве основной траектории используется ТОЛЬКО сценарий main
    (R_mean, p_mean). Остальные сценарии рисуются как коридоры,
    но не усредняются и не смешиваются с main.

ФАКТ (JSON):
    Фактический ряд строится как в import json.txt:
      - берётся серия total[series_index] (по умолчанию 1);
      - из неё берётся поле value_field (по умолчанию "real");
      - дни с отсутствующим или нечисловым значением ПРОПУСКАЮТСЯ;
      - на выходе — плотный np.ndarray БЕЗ пропусков;
      - сравнение с моделью идёт ПО ПОЗИЦИИ (i-й день факта
        vs i-й день модели), без интерполяции по номерам дней.

Загрузка:
  drivers.xlsx : R_mean, R_lo3s, R_hi3s, R_lo2s, R_hi2s, R_lo1s, R_hi1s,
                 p_mean,   p_lo3s, p_hi3s, p_lo2s, p_hi2s, p_lo1s, p_hi1s
  params.xlsx  : ЛИСТ №2, «Параметр | Значение»
                 (S0, I0, gamma, mu, k, EPOP, ...).
  fact.json    : days[].total[]; серия и поле выбираются в UI.

Метрики соответствия факту (всё в абсолюте):
    R², RMSE, MAE, MAPE, sMAPE — для:
      1) I_new_abs main vs real       ← основная
      2) I_abs main     vs real
      3) cum_abs main   = (1−S)·EPOP  vs real
"""

import os
import json
import math
import threading
import traceback

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext



REQUIRED_DRIVER_COLS = [
    "R_mean",
    "R_lo3s", "R_hi3s",
    "R_lo2s", "R_hi2s",
    "R_lo1s", "R_hi1s",
    "p_mean",
    "p_lo3s", "p_hi3s",
    "p_lo2s", "p_hi2s",
    "p_lo1s", "p_hi1s",
]

REQUIRED_PARAM_KEYS = ["s0", "i0", "epop", "mu", "k", "gamma"]


def _to_float(v):
    """'0,8' -> 0.8; '0.8' -> 0.8; число -> float; ''/None/NaN -> nan."""
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




def load_params(path, sheet=1):
    """Читает ЛИСТ №2 (индекс 1) Excel: «Параметр | Значение»."""
    xls = pd.ExcelFile(path)
    if isinstance(sheet, int):
        if sheet >= len(xls.sheet_names):
            raise ValueError(
                f"В файле {os.path.basename(path)} нет листа №{sheet+1}. "
                f"Доступные листы: {xls.sheet_names}"
            )
        sheet_name = xls.sheet_names[sheet]
    else:
        sheet_name = sheet

    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)

    if raw.shape[1] < 2:
        raise ValueError(
            f"На листе '{sheet_name}' ожидалось минимум 2 столбца "
            f"(параметр | значение). Найдено: {raw.shape[1]}"
        )

    pairs = {}
    for _, row in raw.iterrows():
        key = row.iloc[0]
        val = row.iloc[1]
        if pd.isna(key) or pd.isna(val):
            continue
        key_s = str(key).strip().lower()
        if key_s in ("параметр", "parameter", "param", "name", "ключ", "key"):
            continue
        v = _to_float(val)
        if not math.isnan(v):
            pairs[key_s] = v

    if not pairs:
        raise ValueError(
            f"На листе '{sheet_name}' не найдено пар «параметр–значение»."
        )

    missing = [c for c in REQUIRED_PARAM_KEYS if c not in pairs]
    if missing:
        raise ValueError(
            f"На листе '{sheet_name}' отсутствуют параметры: {missing}. "
            f"Найдены: {list(pairs.keys())}"
        )

    params = {
        "s0":    pairs["s0"],
        "i0":    pairs["i0"],
        "epop":  pairs["epop"],
        "mu":    pairs["mu"],
        "k":     pairs["k"],
        "gamma": pairs["gamma"],
    }

    extras = {k: v for k, v in pairs.items() if k not in params}
    params["_extras"] = extras
    params["_sheet"] = sheet_name
    return params


def load_drivers(path, sheet=0):
    """
    Excel с колонками:
        R_mean, R_lo3s, R_hi3s, R_lo2s, R_hi2s, R_lo1s, R_hi1s,
        p_mean,   p_lo3s, p_hi3s, p_lo2s, p_hi2s, p_lo1s, p_hi1s
    """
    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[sheet] if isinstance(sheet, int) else sheet
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_DRIVER_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле драйверов (лист '{sheet_name}') нет колонок: {missing}. "
            f"Найдены: {list(df.columns)}"
        )

    df = df[pd.to_numeric(df["R_mean"], errors="coerce").notna()].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("В драйверах должно быть минимум 2 строки (дня).")
    return df




def load_json_fact(path, series_index=1, value_field="real"):
    """
    Читает JSON: {"days":[{"day":..., "total":[{...}, {...}]}]}.

    Логика КАК В import json.txt:
      - берётся серия total[series_index];
      - из неё берётся поле value_field;
      - дни с отсутствующим или нечисловым значением ПРОПУСКАЮТСЯ;
      - возвращается ПЛОТНЫЙ np.ndarray без пропусков.

    Никаких day/real DataFrame, никакой интерполяции.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    days = data.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("В JSON нет массива 'days' или он пуст.")

    values = []
    for d in days:
        totals = d.get("total") or []
        if series_index < 0 or series_index >= len(totals):
            continue
        entry = totals[series_index]
        if not isinstance(entry, dict):
            continue
        v = _to_float(entry.get(value_field, float("nan")))
        if not math.isnan(v):
            values.append(v)

    if not values:
        raise ValueError(
            "Не удалось извлечь ни одной строки из 'days[].total[]'."
        )

    return np.asarray(values, dtype=float)


def scan_json_series(path, value_field="real"):
    """{index: {...}} — какой index содержит данные (для UI-подсказки)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    days = data.get("days") or []

    stats = {}
    for d in days:
        try:
            day = int(d.get("day"))
        except (TypeError, ValueError):
            continue
        for it in (d.get("total") or []):
            try:
                idx = int(it.get("index"))
            except (TypeError, ValueError):
                continue
            v = _to_float(it.get(value_field, float("nan")))
            s = stats.setdefault(idx, {"n_total": 0, "n_nonzero": 0,
                                       "first_day": None, "last_val": None})
            s["n_total"] += 1
            if not math.isnan(v) and v != 0:
                s["n_nonzero"] += 1
                if s["first_day"] is None:
                    s["first_day"] = day
                s["last_val"] = v
    return stats




def compute_fit_metrics_from_arrays(fact_abs, model_abs):
    """
    Сравнивает фактический и модельный ряды ПО ПОЗИЦИИ.

    fact_abs  — плотный массив абсолютных новых случаев;
    model_abs — модельные значения той же длины.

    Логика КАК В import json.txt: без интерполяции по номерам дней.
    """
    if fact_abs is None or model_abs is None:
        return None

    fact_abs = np.asarray(fact_abs, dtype=float)
    model_abs = np.asarray(model_abs, dtype=float)

    n = min(len(fact_abs), len(model_abs))
    if n < 2:
        return {"n": n, "n_mape": 0,
                "r2": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "mape": float("nan"),
                "smape": float("nan")}

    y_true = fact_abs[:n]
    y_pred = model_abs[:n]

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = y_true.size

    if n < 2:
        return {"n": n, "n_mape": 0,
                "r2": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "mape": float("nan"),
                "smape": float("nan")}

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    eps = 1e-9
    mape_mask = np.abs(y_true) > eps
    n_mape = int(np.sum(mape_mask))

    if n_mape > 0:
        mape = float(
            np.mean(
                np.abs(
                    (y_true[mape_mask] - y_pred[mape_mask])
                    / y_true[mape_mask]
                )
            ) * 100.0
        )
    else:
        mape = float("nan")

    denom = np.abs(y_true) + np.abs(y_pred)
    ok = denom > eps

    if np.any(ok):
        smape = float(
            np.mean(
                2.0 * np.abs(y_true[ok] - y_pred[ok]) / denom[ok]
            ) * 100.0
        )
    else:
        smape = float("nan")

    return {
        "n": n,
        "n_mape": n_mape,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
    }




def make_interp(series):
    x = np.arange(len(series), dtype=float)
    y = np.array([_to_float(v) for v in series], dtype=float)

    def f(t):
        return np.interp(t, x, y)
    return f


def make_rhs(R_fun, p_fun, gamma, mu, k):
    def rhs(t, y):
        S, I = y
        S = max(S, 0.0)
        I = max(I, 0.0)
        R_t = R_fun(t)
        p_t = p_fun(t)
        beta = R_t * gamma
        dS = -beta * S * I + mu * (1.0 - S) + (k + p_t) * (1.0 - S - I)
        dI =  beta * S * I - gamma * I - mu * I
        return [dS, dI]
    return rhs


def solve_scenario(R_series, p_series, params):
    """
    Возвращает t, S (доли), I (доли), I_new (доля/день), scale_msg.
    """
    R_fun = make_interp(R_series)
    p_fun = make_interp(p_series)
    rhs = make_rhs(R_fun, p_fun,
                   gamma=params["gamma"],
                   mu=params["mu"],
                   k=params["k"])

    s0, i0, epop = params["s0"], params["i0"], params["epop"]
    if (s0 + i0) <= 1.0:
        y0 = [s0, i0]
        scale = "доли (S0+I0 ≤ 1)"
    else:
        if epop <= 0:
            raise ValueError("EPOP должен быть > 0.")
        y0 = [s0 / epop, i0 / epop]
        scale = "абсолютные, делим на EPOP"

    t_max = len(R_series) - 1
    t_eval = np.linspace(0, t_max, int(t_max) + 1)

    sol = solve_ivp(rhs, (0, t_max), y0, t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10, max_step=1.0)
    if not sol.success:
        raise RuntimeError("solve_ivp: " + str(sol.message))

    t = sol.t
    S = np.clip(sol.y[0], 0, 1)
    I = np.clip(sol.y[1], 0, 1)

    R_t = np.array([R_fun(ti) for ti in t])
    I_new = R_t * params["gamma"] * S * I

    return t, S, I, I_new, scale


def run_all(drivers, params):
    """
    Возвращает dict со сценариями.

    Сценарии — это независимые прогоны ОДУ с разными (R, p):
        main : R_mean, p_mean
        lo   : R_lo3s, p_lo3s
        hi   : R_hi3s, p_hi3s
        lo2  : R_lo2s, p_lo2s
        hi2  : R_hi2s, p_hi2s
        lo1  : R_lo1s, p_lo1s
        hi1  : R_hi1s, p_hi1s

    НИКАКОЙ АГРЕГАЦИИ НЕ ДЕЛАЕТСЯ.
    Основной траекторией считается сценарий "main".
    Остальные используются только для рисования коридоров ±1σ/±2σ/±3σ.
    """
    epop = params["epop"]

    scenarios = {
        "main": ("R_mean", "p_mean"),
        "lo":   ("R_lo3s",   "p_lo3s"),
        "hi":   ("R_hi3s",   "p_hi3s"),
        "lo2":  ("R_lo2s",   "p_lo2s"),
        "hi2":  ("R_hi2s",   "p_hi2s"),
        "lo1":  ("R_lo1s",   "p_lo1s"),
        "hi1":  ("R_hi1s",   "p_hi1s"),
    }

    results = {}
    scale_msg = None
    for name, (rcol, pcol) in scenarios.items():
        t, S, I, I_new, scale = solve_scenario(
            drivers[rcol], drivers[pcol], params
        )
        results[name] = {
            "t":         t,
            "S":         S,
            "I":         I,
            "I_new":     I_new,
            "S_abs":     S * epop,
            "I_abs":     I * epop,
            "I_new_abs": I_new * epop,
            "cum_abs":   (1.0 - S) * epop,
        }
        if scale_msg is None:
            scale_msg = scale

    return results, scale_msg




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SIR-ODE solver — main сценарий (R_mean, ±1σ/±2σ/±3σ)")
        self.geometry("1200x880")
        self.minsize(1000, 720)

        self.path_drivers = tk.StringVar()
        self.path_params  = tk.StringVar()
        self.path_out     = tk.StringVar(value=os.path.abspath("ode_solution_abs.xlsx"))
        self.path_fact    = tk.StringVar()

        self.fact_series_idx = tk.IntVar(value=1)
        self.fact_field      = tk.StringVar(value="real")

        self.drivers   = None
        self.params    = None
        self.results   = None
        self.scale_msg = ""
        self.fact_abs  = None
        self.fit_metrics = {}

        self._build_widgets()


    def _build_widgets(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        files_frame = ttk.LabelFrame(top, text="Входные файлы", padding=8)
        files_frame.pack(side=tk.TOP, fill=tk.X)
        files_frame.columnconfigure(1, weight=1)

        ttk.Label(files_frame, text="Драйверы (R_mean, ±1σ/±2σ/±3σ, p):").grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Entry(files_frame, textvariable=self.path_drivers).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(files_frame, text="Обзор…",
                   command=lambda: self._pick(self.path_drivers)).grid(
            row=0, column=2, padx=4)
        ttk.Button(files_frame, text="Загрузить",
                   command=self._load_drivers).grid(row=0, column=3, padx=4)

        ttk.Label(files_frame,
                  text="Параметры (2-й лист, «Параметр | Значение»):").grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Entry(files_frame, textvariable=self.path_params).grid(
            row=1, column=1, sticky="ew", padx=4)
        ttk.Button(files_frame, text="Обзор…",
                   command=lambda: self._pick(self.path_params)).grid(
            row=1, column=2, padx=4)
        ttk.Button(files_frame, text="Загрузить",
                   command=self._load_params).grid(row=1, column=3, padx=4)

        ttk.Label(files_frame, text="Куда сохранить результат:").grid(
            row=2, column=0, sticky="w", pady=2)
        ttk.Entry(files_frame, textvariable=self.path_out).grid(
            row=2, column=1, sticky="ew", padx=4)
        ttk.Button(files_frame, text="Обзор…",
                   command=lambda: self._pick(self.path_out, save=True)).grid(
            row=2, column=2, padx=4)

        ttk.Label(files_frame, text="Факт (JSON):").grid(
            row=3, column=0, sticky="w", pady=2)
        ttk.Entry(files_frame, textvariable=self.path_fact).grid(
            row=3, column=1, sticky="ew", padx=4)
        ttk.Button(files_frame, text="Обзор…",
                   command=lambda: self._pick(self.path_fact)).grid(
            row=3, column=2, padx=4)
        ttk.Button(files_frame, text="Загрузить",
                   command=self._load_fact).grid(row=3, column=3, padx=4)

        ttk.Label(files_frame, text="Серия (index):").grid(
            row=4, column=0, sticky="w", pady=2)
        series_row = ttk.Frame(files_frame)
        series_row.grid(row=4, column=1, sticky="w", padx=4)

        ttk.Spinbox(series_row, from_=0, to=50, width=5,
                    textvariable=self.fact_series_idx).pack(side=tk.LEFT)
        ttk.Label(series_row, text="   поле:").pack(side=tk.LEFT)
        ttk.Combobox(series_row, width=12, state="readonly",
                     textvariable=self.fact_field,
                     values=("real", "predicted", "bestrun")
                     ).pack(side=tk.LEFT, padx=4)
        ttk.Button(files_frame, text="Просканировать серии",
                   command=self._scan_fact_series).grid(
            row=4, column=3, padx=4, sticky="w")

        act = ttk.Frame(top, padding=(0, 8))
        act.pack(side=tk.TOP, fill=tk.X)
        self.btn_run  = ttk.Button(act, text="▶ Решить", command=self._run)
        self.btn_save = ttk.Button(act, text="💾 Сохранить Excel",
                                   command=self._save, state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, padx=4)
        self.btn_save.pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(act, text="Готов", foreground="#555")
        self.status.pack(side=tk.LEFT, padx=12)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        pf = ttk.Frame(nb, padding=8)
        nb.add(pf, text="Параметры и данные")
        self.param_text = scrolledtext.ScrolledText(pf, wrap=tk.NONE,
                                                    font=("Consolas", 10))
        self.param_text.pack(fill=tk.BOTH, expand=True)
        self._write_param_summary()

        lf = ttk.Frame(nb, padding=8)
        nb.add(lf, text="Логи")
        self.log_text = scrolledtext.ScrolledText(lf, wrap=tk.WORD,
                                                  font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

        gf = ttk.Frame(nb)
        nb.add(gf, text="График")
        self.fig = Figure(figsize=(11.5, 5.5), dpi=100)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, gf)
        toolbar.update()


    def _pick(self, var, save=False):
        if save:
            p = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=os.path.basename(var.get()) or "ode_solution_abs.xlsx")
        else:
            p = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx *.xls"),
                           ("JSON files", "*.json"),
                           ("All files", "*.*")])
        if p:
            var.set(p)

    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, txt, color="#555"):
        self.status.configure(text=txt, foreground=color)
        self.update_idletasks()


    def _load_drivers(self):
        p = self.path_drivers.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Укажите существующий файл драйверов.")
            return
        try:
            self.drivers = load_drivers(p)
            self._log(f"[drivers] Загружено {len(self.drivers)} строк из {p}")
            self._log(f"[drivers] Колонки: {list(self.drivers.columns)}")
            self._write_param_summary()
            self._set_status("Драйверы загружены", "#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки драйверов", str(e))
            self._log("[drivers][ERROR] " + str(e))

    def _load_params(self):
        p = self.path_params.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Укажите существующий файл параметров.")
            return
        try:
            self.params = load_params(p, sheet=1)
            extras = self.params.get("_extras") or {}
            self._log(
                f"[params] Лист '{self.params['_sheet']}'. Загружено: "
                + ", ".join(f"{k}={self.params[k]:.6g}"
                            for k in ["s0", "i0", "epop", "mu", "k", "gamma"])
                + (f"; доп: {extras}" if extras else "")
            )
            self._write_param_summary()
            self._set_status("Параметры загружены", "#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки параметров", str(e))
            self._log("[params][ERROR] " + str(e))

    def _load_fact(self):
        p = self.path_fact.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Укажите существующий JSON-файл с фактом.")
            return
        try:
            idx = int(self.fact_series_idx.get())
            field = self.fact_field.get().strip() or "real"

            # ПЛОТНЫЙ массив абсолютных новых случаев — КАК В import json.txt
            self.fact_abs = load_json_fact(p, series_index=idx, value_field=field)

            n = len(self.fact_abs)
            nz = int(np.sum(self.fact_abs != 0))

            self._log(f"[fact] Загружено {n} точек из {p}; "
                      f"серия index={idx}, поле='{field}'; "
                      f"ненулевых: {nz}")
            self._log(f"[fact] min={self.fact_abs.min():.4g}, "
                      f"max={self.fact_abs.max():.4g}, "
                      f"last={self.fact_abs[-1]:.4g}")

            if nz == 0:
                self._log("[fact] ВНИМАНИЕ: в этой серии нет ни одного ненулевого "
                          "значения. Нажмите «Просканировать серии».")

            self._write_param_summary()
            self._set_status("Факт загружен", "#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки JSON", str(e))
            self._log("[fact][ERROR] " + str(e))

    def _scan_fact_series(self):
        p = self.path_fact.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Сначала укажите JSON-файл.")
            return
        try:
            field = self.fact_field.get().strip() or "real"
            stats = scan_json_series(p, value_field=field)
            self._log(f"[scan] поле='{field}'. Найдено серий: {len(stats)}")
            for idx in sorted(stats):
                s = stats[idx]
                self._log(
                    f"  index={idx:2d}: всего={s['n_total']:4d}, "
                    f"ненулевых={s['n_nonzero']:4d}, "
                    f"первый день с данными={s['first_day']}, "
                    f"последнее значение={s['last_val']}"
                )
            if stats:
                best = max(stats.items(), key=lambda kv: kv[1]["n_nonzero"])
                self._log(f"[scan] Похоже, данные в серии index={best[0]} "
                          f"(ненулевых={best[1]['n_nonzero']}). "
                          f"Подставил в поле «Серия (index)».")
                self.fact_series_idx.set(best[0])
            self._write_param_summary()
        except Exception as e:
            messagebox.showerror("Ошибка сканирования", str(e))
            self._log("[scan][ERROR] " + str(e))


    def _write_param_summary(self):
        self.param_text.configure(state=tk.NORMAL)
        self.param_text.delete("1.0", tk.END)

        self.param_text.insert(tk.END, "=== ПАРАМЕТРЫ ===\n")
        if self.params is None:
            self.param_text.insert(tk.END, "  (не загружены)\n")
        else:
            self.param_text.insert(tk.END,
                f"  источник: лист '{self.params['_sheet']}'\n")
            for k in ["s0", "i0", "epop", "mu", "k", "gamma"]:
                self.param_text.insert(tk.END,
                    f"  {k:8s} = {self.params[k]:.6g}\n")

            extras = self.params.get("_extras") or {}
            if extras:
                self.param_text.insert(tk.END,
                    "\n  дополнительные поля из файла:\n")
                for k, v in extras.items():
                    self.param_text.insert(tk.END, f"    {k:8s} = {v:.6g}\n")

            s0, i0, epop = self.params["s0"], self.params["i0"], self.params["epop"]
            if (s0 + i0) <= 1.0:
                self.param_text.insert(tk.END,
                    "\n  режим: S0, I0 считаются ДОЛЯМИ (S0+I0 ≤ 1)\n"
                    f"  S0 = {s0:.6g}, I0 = {i0:.6g}\n"
                    f"  в абсолюте: S0*EPOP = {s0*epop:.6g} чел., "
                    f"I0*EPOP = {i0*epop:.6g} чел.\n")
            else:
                self.param_text.insert(tk.END,
                    "\n  режим: S0, I0 — абсолютные, делим на EPOP\n"
                    f"  S0/EPOP = {s0/epop:.6g}\n"
                    f"  I0/EPOP = {i0/epop:.6g}\n")
            self.param_text.insert(tk.END,
                "  beta(t)  = R(t) * gamma\n"
                "  I_new(t) = R(t) * gamma * S(t) * I(t)   (доля/день)\n"
                "  I_new_abs(t) = I_new(t) * EPOP          (чел./день)\n"
                "  Основная траектория — ТОЛЬКО сценарий main:\n"
                "    main = (R_mean, p_mean)\n"
                "  Остальные сценарии используются только для коридоров:\n"
                "    lo/hi    = (R_lo3s/R_hi3s, p_lo3s/p_hi3s)  — ±3σ\n"
                "    lo2/hi2  = (R_lo2s/R_hi2s, p_lo2s/p_hi2s)  — ±2σ\n"
                "    lo1/hi1  = (R_lo1s/R_hi1s, p_lo1s/p_hi1s)  — ±1σ\n"
                "  АГРЕГАЦИЯ СЦЕНАРИЕВ НЕ ВЫПОЛНЯЕТСЯ.\n")

        self.param_text.insert(tk.END,
            "\n=== ДРАЙВЕРЫ (R_mean, ±1σ/±2σ/±3σ, p) ===\n")
        if self.drivers is None:
            self.param_text.insert(tk.END, "  (не загружены)\n")
        else:
            n = len(self.drivers)
            self.param_text.insert(tk.END, f"  Строк (дней): {n}\n\n")
            head = self.drivers[REQUIRED_DRIVER_COLS].head(10)
            with pd.option_context("display.width", 260,
                                   "display.max_columns", 30):
                self.param_text.insert(tk.END, head.to_string(index=True) + "\n")
            if n > 10:
                self.param_text.insert(tk.END, f"  … ещё {n-10} строк\n")

            self.param_text.insert(tk.END, "\n  Средние за период:\n")
            for c in REQUIRED_DRIVER_COLS:
                mu_ = pd.to_numeric(self.drivers[c], errors="coerce").mean()
                self.param_text.insert(tk.END, f"    {c:14s} = {mu_:.6g}\n")

            if self.params is not None:
                g = self.params["gamma"]
                self.param_text.insert(
                    tk.END,
                    f"\n  gamma * R_mean(0) = "
                    f"{g * float(pd.to_numeric(self.drivers['R_mean'].iloc[0])):.6g}\n"
                )

        self.param_text.insert(tk.END, "\n=== ФАКТ (JSON) ===\n")
        if getattr(self, "fact_abs", None) is None:
            self.param_text.insert(tk.END, "  (не загружен)\n")
        else:
            f = np.asarray(self.fact_abs, dtype=float)
            n = len(f)
            nz = int(np.sum(f != 0))
            self.param_text.insert(tk.END,
                f"  источник: серия index={self.fact_series_idx.get()}, "
                f"поле='{self.fact_field.get()}'\n")
            self.param_text.insert(tk.END,
                f"  Точек (дней): {n}; ненулевых: {nz}\n")
            if n > 0:
                self.param_text.insert(tk.END,
                    f"  real: min={f.min():.4g}, max={f.max():.4g}, "
                    f"last={f[-1]:.4g}\n")
            # Первые 10 значений — просто по индексу
            self.param_text.insert(tk.END,
                "  первые 10 значений:\n")
            for i in range(min(10, n)):
                self.param_text.insert(
                    tk.END, f"    i={i:3d}  real_abs={f[i]:.4g}\n")
            if n > 10:
                self.param_text.insert(tk.END, f"  … ещё {n-10} точек\n")

        self.param_text.insert(tk.END,
            "\n=== КАЧЕСТВО ПОДГОНКИ (сценарий main, всё в абсолюте) ===\n")
        if not getattr(self, "fit_metrics", None):
            self.param_text.insert(tk.END,
                "  (нужно загрузить факт и нажать «▶ Решить»)\n")
        else:
            m = self.fit_metrics.get("I_new_abs_main")
            if m:
                self.param_text.insert(tk.END,
                    f"  I_new_abs main (чел./день):   "
                    f"R²={m['r2']:.4f}  RMSE={m['rmse']:.4g}  "
                    f"MAE={m['mae']:.4g}  "
                    f"MAPE={m.get('mape', float('nan')):.2f}%  "
                    f"n={m['n']} (n_mape={m.get('n_mape', 0)})\n")
            m = self.fit_metrics.get("I_abs_main")
            if m:
                self.param_text.insert(tk.END,
                    f"  I_abs main (чел.):            "
                    f"R²={m['r2']:.4f}  RMSE={m['rmse']:.4g}  "
                    f"MAE={m['mae']:.4g}  "
                    f"MAPE={m.get('mape', float('nan')):.2f}%  "
                    f"n={m['n']} (n_mape={m.get('n_mape', 0)})\n")
            m = self.fit_metrics.get("cum_abs_main")
            if m:
                self.param_text.insert(tk.END,
                    f"  cum_abs main ((1−S)·EPOP):    "
                    f"R²={m['r2']:.4f}  RMSE={m['rmse']:.4g}  "
                    f"MAE={m['mae']:.4g}  "
                    f"MAPE={m.get('mape', float('nan')):.2f}%  "
                    f"n={m['n']} (n_mape={m.get('n_mape', 0)})\n")
            self.param_text.insert(tk.END,
                "\n  (сравнение по позиции: i-й день факта vs i-й день модели)\n")

        self.param_text.configure(state=tk.DISABLED)


    def _run(self):
        if self.drivers is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите драйверы.")
            return
        if self.params is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите параметры.")
            return
        self.btn_run.configure(state=tk.DISABLED)
        self._set_status("Интегрирование…", "#c80")
        self._log("=== Запуск решения ОДУ (абсолютные значения, main сценарий) ===")
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        try:
            res, scale_msg = run_all(self.drivers, self.params)
            self.results = res
            self.scale_msg = scale_msg
            self._log(f"[solve] Готово. Режим S0/I0: {scale_msg}")
            self._log(f"[solve] Сценарии: main, lo(±3σ), hi(±3σ), "
                      f"lo2/hi2(±2σ), lo1/hi1(±1σ)")
            self._log(f"[solve] Основная траектория — ТОЛЬКО main. "
                      f"Агрегация сценариев НЕ выполняется.")

            self.fit_metrics = {}
            if getattr(self, "fact_abs", None) is not None and len(self.fact_abs) > 1:

                m0 = compute_fit_metrics_from_arrays(
                    self.fact_abs,
                    res["main"]["I_new_abs"],
                )
                if m0:
                    self.fit_metrics["I_new_abs_main"] = m0
                    self._log(f"[fit] I_new_abs main vs факт:  n={m0['n']}, "
                              f"R²={m0['r2']:.4f}, RMSE={m0['rmse']:.4g}, "
                              f"MAE={m0['mae']:.4g}, "
                              f"MAPE={m0.get('mape', float('nan')):.2f}% "
                              f"(n_mape={m0.get('n_mape', 0)})")

                m1 = compute_fit_metrics_from_arrays(
                    self.fact_abs,
                    res["main"]["I_abs"],
                )
                if m1:
                    self.fit_metrics["I_abs_main"] = m1
                    self._log(f"[fit] I_abs main vs факт:      n={m1['n']}, "
                              f"R²={m1['r2']:.4f}, RMSE={m1['rmse']:.4g}, "
                              f"MAE={m1['mae']:.4g}, "
                              f"MAPE={m1.get('mape', float('nan')):.2f}% "
                              f"(n_mape={m1.get('n_mape', 0)})")

                m2 = compute_fit_metrics_from_arrays(
                    self.fact_abs,
                    res["main"]["cum_abs"],
                )
                if m2:
                    self.fit_metrics["cum_abs_main"] = m2
                    self._log(f"[fit] cum_abs main vs факт:    n={m2['n']}, "
                              f"R²={m2['r2']:.4f}, RMSE={m2['rmse']:.4g}, "
                              f"MAE={m2['mae']:.4g}, "
                              f"MAPE={m2.get('mape', float('nan')):.2f}% "
                              f"(n_mape={m2.get('n_mape', 0)})")

            self.after(0, self._after_run_ok)
        except Exception as e:
            tb = traceback.format_exc()
            self._log("[solve][ERROR] " + str(e))
            self._log(tb)
            self.after(0, lambda: messagebox.showerror("Ошибка интегрирования", str(e)))
            self.after(0, lambda: self._set_status("Ошибка", "#c00"))
            self.after(0, lambda: self.btn_run.configure(state=tk.NORMAL))

    def _after_run_ok(self):
        self._plot()
        self._write_param_summary()
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_save.configure(state=tk.NORMAL)
        self._set_status("Решение готово", "#0a0")


    def _plot(self):
        r = self.results
        epop = self.params["epop"]
        self.ax1.clear()
        self.ax2.clear()

        t = r["main"]["t"]


        s_lo3 = r["lo"]["S_abs"];  s_hi3 = r["hi"]["S_abs"]
        s_lo2 = r["lo2"]["S_abs"]; s_hi2 = r["hi2"]["S_abs"]
        s_lo1 = r["lo1"]["S_abs"]; s_hi1 = r["hi1"]["S_abs"]

        i_lo3 = r["lo"]["I_abs"];  i_hi3 = r["hi"]["I_abs"]
        i_lo2 = r["lo2"]["I_abs"]; i_hi2 = r["hi2"]["I_abs"]
        i_lo1 = r["lo1"]["I_abs"]; i_hi1 = r["hi1"]["I_abs"]


        self.ax1.fill_between(t, s_lo3, s_hi3, color="C0", alpha=0.12,
                              label="S: ±3σ")
        self.ax1.fill_between(t, i_lo3, i_hi3, color="C3", alpha=0.12,
                              label="I: ±3σ")

        self.ax1.fill_between(t, s_lo2, s_hi2, color="C0", alpha=0.20)
        self.ax1.fill_between(t, i_lo2, i_hi2, color="C3", alpha=0.20)

        self.ax1.fill_between(t, s_lo1, s_hi1, color="C0", alpha=0.30)
        self.ax1.fill_between(t, i_lo1, i_hi1, color="C3", alpha=0.30)

        self.ax1.plot(t, r["main"]["S_abs"], color="C0",
                      linewidth=2.0, label="S_abs main (чел.)")
        self.ax1.plot(t, r["main"]["I_abs"], color="C3",
                      linewidth=2.0, label="I_abs main (чел.)")

        self.ax1.set_title(f"S_abs main, I_abs main — чел. (EPOP={epop:.0f})")
        self.ax1.set_xlabel("день")
        self.ax1.set_ylabel("чел.")
        self.ax1.legend(fontsize=7, ncol=2)
        self.ax1.grid(alpha=0.3)


        n_main = r["main"]["I_new_abs"]

        n_lo3 = r["lo"]["I_new_abs"];  n_hi3 = r["hi"]["I_new_abs"]
        n_lo2 = r["lo2"]["I_new_abs"]; n_hi2 = r["hi2"]["I_new_abs"]
        n_lo1 = r["lo1"]["I_new_abs"]; n_hi1 = r["hi1"]["I_new_abs"]

        self.ax2.fill_between(t, n_lo3, n_hi3, alpha=0.12, color="C3",
                              label="±3σ")
        self.ax2.fill_between(t, n_lo2, n_hi2, alpha=0.20, color="C3")
        self.ax2.fill_between(t, n_lo1, n_hi1, alpha=0.30, color="C3",
                              label="±1σ")

        self.ax2.plot(t, n_main, color="C3", linewidth=2.0,
                      label="I_new_abs main (чел./день)")

        self.ax2.set_title(
            f"I_new_abs main(t) = R_mean·γ·S·I·EPOP — новые заражения / день"
        )
        self.ax2.set_xlabel("день")
        self.ax2.set_ylabel("чел./день", color="C3")
        self.ax2.tick_params(axis="y", labelcolor="C3")
        self.ax2.grid(alpha=0.3)

        # Факт — плотный массив, ось X = индекс (КАК В import json.txt)
        if getattr(self, "fact_abs", None) is not None:
            f = np.asarray(self.fact_abs, dtype=float)
            n = min(len(f), len(t))
            if n > 0:
                t_fact = np.arange(n)
                self.ax2.plot(t_fact, f[:n],
                              color="black", marker="o", markersize=2,
                              linewidth=1.0, label="факт (JSON)")

        self.ax2.legend(fontsize=7, loc="upper left")


        m = getattr(self, "fit_metrics", {}).get("I_new_abs_main")
        if m and np.isfinite(m["r2"]):
            self.ax2.text(
                0.98, 0.98,
                f"I_new_abs main\n"
                f"R² = {m['r2']:.3f}\n"
                f"RMSE = {m['rmse']:.3g}\n"
                f"MAE = {m['mae']:.3g}\n"
                f"MAPE = {m.get('mape', float('nan')):.2f}%\n"
                f"n = {m['n']}",
                transform=self.ax2.transAxes,
                va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.85)
            )

        self.fig.tight_layout()
        self.canvas.draw_idle()


    def _save(self):
        if self.results is None:
            return
        path = self.path_out.get()
        try:
            r = self.results

            df_out = pd.DataFrame({"day": r["main"]["t"]})


            for name in ["main", "lo", "hi", "lo2", "hi2", "lo1", "hi1"]:
                df_out[f"S_{name}"]            = r[name]["S"]
                df_out[f"I_{name}"]            = r[name]["I"]
                df_out[f"Inew_{name}"]         = r[name]["I_new"]
                df_out[f"S_abs_{name}"]        = r[name]["S_abs"]
                df_out[f"I_abs_{name}"]        = r[name]["I_abs"]
                df_out[f"Inew_abs_{name}"]     = r[name]["I_new_abs"]
                df_out[f"cum_abs_{name}"]      = r[name]["cum_abs"]

            df_out.to_excel(path, index=False)

            params_clean = {k: v for k, v in self.params.items()
                            if not k.startswith("_")}
            with pd.ExcelWriter(path, engine="openpyxl", mode="a") as xw:
                pd.DataFrame([params_clean]).to_excel(
                    xw, sheet_name="params_used", index=False)
                self.drivers[REQUIRED_DRIVER_COLS].describe().to_excel(
                    xw, sheet_name="drivers_summary")

                # Факт — плотный массив, сохраняем как есть
                if getattr(self, "fact_abs", None) is not None:
                    f = np.asarray(self.fact_abs, dtype=float)
                    fact_df = pd.DataFrame({
                        "i":     np.arange(len(f)),
                        "real":  f,
                    })
                    fact_df.to_excel(
                        xw, sheet_name="fact_json", index=False)

                if getattr(self, "fit_metrics", None):
                    rows = []
                    for key, m in self.fit_metrics.items():
                        rows.append({
                            "series":  key,
                            "n":       m["n"],
                            "n_mape":  m.get("n_mape", 0),
                            "R2":      m["r2"],
                            "RMSE":    m["rmse"],
                            "MAE":     m["mae"],
                            "MAPE_%":  m.get("mape", float("nan")),
                            "sMAPE_%": m.get("smape", float("nan")),
                        })
                    pd.DataFrame(rows).to_excel(
                        xw, sheet_name="fit_metrics", index=False)

            extra = ", + лист fact_json" if getattr(self, "fact_abs", None) is not None else ""
            extra_m = ", + лист fit_metrics" if getattr(self, "fit_metrics", None) else ""
            self._log(f"[save] Результат сохранён: {path} "
                      f"(main + 6 сценариев-коридоров{extra}{extra_m})")
            self._set_status("Сохранено", "#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))
            self._log("[save][ERROR] " + str(e))


if __name__ == "__main__":
    App().mainloop()