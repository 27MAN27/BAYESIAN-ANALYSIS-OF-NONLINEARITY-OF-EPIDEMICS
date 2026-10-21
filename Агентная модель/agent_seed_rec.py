# -*- coding: utf-8 -*-

import os
import json
import math
import time
import traceback
import itertools

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


def _to_float(v):
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


def _to_int_or_none(v):
    f = _to_float(v)
    if math.isnan(f):
        return None
    return int(round(f))


def parse_param_node(node):
    if not isinstance(node, dict):
        return {"curr": _to_float(node), "specdays": {}}

    curr = _to_float(node.get("curr"))
    spec = {}
    for it in (node.get("specdays") or []):
        try:
            day = int(it.get("day"))
        except (TypeError, ValueError):
            continue
        v = _to_float(it.get("value"))
        if not math.isnan(v):
            spec[day] = v
    return {"curr": curr, "specdays": spec}


def param_value_at_day(node, day):
    if day in node["specdays"]:
        return node["specdays"][day]
    return node["curr"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    days_json = data.get("days")
    if not isinstance(days_json, list) or not days_json:
        raise ValueError("В JSON нет массива 'days' или он пуст.")

    day_numbers, fact_real, fact_predicted = [], [], []
    for entry in days_json:
        try:
            day = int(entry.get("day"))
        except (TypeError, ValueError):
            day = len(day_numbers) + 1
        day_numbers.append(day)

        totals = entry.get("total") or []
        real_val = float("nan")
        pred_val = float("nan")
        if len(totals) > 1:
            real_val = _to_float(totals[1].get("real"))
            pred_val = _to_float(totals[1].get("predicted"))
        fact_real.append(real_val)
        fact_predicted.append(pred_val)

    params = data.get("params") or {}
    parsed_params = {}
    for key in ("EPOP", "S0", "Y0", "CONTACTS", "PTI", "DTR",
                "IMM_LOSS", "MU"):
        parsed_params[key] = parse_param_node(params.get(key, {}))

    return {
        "day_numbers": np.asarray(day_numbers, dtype=int),
        "fact_real": np.asarray(fact_real, dtype=float),
        "fact_predicted": np.asarray(fact_predicted, dtype=float),
        "params": parsed_params,
    }


AC_S = 0
AC_I = 1
AC_R = 2


class AgentModel:
    def __init__(self, EPOP, S0, Y0):
        EPOP = int(round(EPOP))
        nS = int(round(S0 * EPOP))
        nI = int(round(Y0))
        nR = EPOP - nS - nI
        if nR < 0:
            nR = 0
        n = nS + nI + nR

        self.agent_class = np.empty(n, dtype=np.int8)
        self.agent_class[:nS] = AC_S
        self.agent_class[nS:nS + nI] = AC_I
        self.agent_class[nS + nI:] = AC_R

        self.agent_new_class = self.agent_class.copy()

        self.days_infected = np.zeros(n, dtype=np.int32)
        self.days_in_R = np.zeros(n, dtype=np.int32)

        self.days_infected[self.agent_class == AC_I] = 1
        self.days_in_R[self.agent_class == AC_R] = 0

        self.EPOP_initial = n
        self.EPOP = n

    def add_susceptibles(self, n_new):
        n_new = int(n_new)
        if n_new <= 0:
            return

        self.agent_class = np.concatenate(
            [self.agent_class, np.full(n_new, AC_S, dtype=np.int8)]
        )
        self.agent_new_class = np.concatenate(
            [self.agent_new_class, np.full(n_new, AC_S, dtype=np.int8)]
        )
        self.days_infected = np.concatenate(
            [self.days_infected, np.zeros(n_new, dtype=np.int32)]
        )
        self.days_in_R = np.concatenate(
            [self.days_in_R, np.zeros(n_new, dtype=np.int32)]
        )

        self.EPOP = self.agent_class.size

    def remove_random(self, n_out, mask):
        n_out = int(n_out)
        if n_out <= 0:
            return

        idx_in_mask = np.flatnonzero(mask)
        if idx_in_mask.size == 0:
            return

        n_out = min(n_out, idx_in_mask.size)
        chosen = np.random.choice(idx_in_mask, size=n_out, replace=False)

        keep_mask = np.ones(self.agent_class.size, dtype=bool)
        keep_mask[chosen] = False

        self.agent_class = self.agent_class[keep_mask]
        self.agent_new_class = self.agent_new_class[keep_mask]
        self.days_infected = self.days_infected[keep_mask]
        self.days_in_R = self.days_in_R[keep_mask]

        self.EPOP = self.agent_class.size

    def apply_emigration(self, mu):
        if mu is None or mu <= 0:
            return

        nS = int(np.sum(self.agent_class == AC_S))
        nI = int(np.sum(self.agent_class == AC_I))
        nR = int(np.sum(self.agent_class == AC_R))

        nS_out = int(round(mu * nS))
        nI_out = int(round(mu * nI))
        nR_out = int(round(mu * nR))

        if nS_out > 0:
            self.remove_random(nS_out, self.agent_class == AC_S)
        if nI_out > 0:
            self.remove_random(nI_out, self.agent_class == AC_I)
        if nR_out > 0:
            self.remove_random(nR_out, self.agent_class == AC_R)

    def contact(self, contacts, p_to_infect):
        infected_idx = np.where(self.agent_class == AC_I)[0]
        if len(infected_idx) == 0 or p_to_infect <= 0:
            return

        n = self.agent_class.size
        n_infected = infected_idx.size
        n_contacts_total = int(round(n_infected * contacts))

        if n_contacts_total <= 0:
            return

        targets = np.random.randint(0, n, size=n_contacts_total)
        is_S = (self.agent_class[targets] == AC_S)
        infected_events = (np.random.rand(n_contacts_total) < p_to_infect) & is_S

        newly_infected = np.unique(targets[infected_events])

        self.agent_new_class[newly_infected] = AC_I
        self.days_infected[newly_infected] = 0

    def regenerate(self, day_to_return):
        if day_to_return <= 0:
            return

        in_I = (self.agent_new_class == AC_I)
        self.days_infected[in_I] += 1

        recovered = (self.agent_new_class == AC_I) & (self.days_infected >= day_to_return)

        self.days_in_R[recovered] = 0
        self.agent_new_class[recovered] = AC_R

    def lose_immunity(self, imm_loss):
        if imm_loss is None or imm_loss <= 0:
            return

        in_R = (self.agent_new_class == AC_R)
        self.days_in_R[in_R] += 1

        back_to_S = (self.agent_new_class == AC_R) & (self.days_in_R >= imm_loss)
        self.agent_new_class[back_to_S] = AC_S
        self.days_in_R[back_to_S] = 0

    def step(self, contacts, p_to_infect, day_to_return,
             imm_loss, mu=0.0):
        if mu is not None and mu > 0:
            n_new = int(round(mu * self.EPOP_initial))
            self.add_susceptibles(n_new)

        if mu is not None and mu > 0:
            self.apply_emigration(mu)

        self.contact(contacts, p_to_infect)
        self.regenerate(day_to_return)
        self.lose_immunity(imm_loss)

        changed_to_I = (self.agent_new_class == AC_I) & (self.agent_class != AC_I)
        self.agent_class = self.agent_new_class.copy()

        nS = int(np.sum(self.agent_class == AC_S))
        nI = int(np.sum(self.agent_class == AC_I))
        nR = int(np.sum(self.agent_class == AC_R))
        n_new_I = int(np.sum(changed_to_I))

        return {
            "S_abs": nS,
            "I_abs": nI,
            "I_new_abs": n_new_I,
            "R_abs": nR,
        }


def _clone_agent_model(src):
    """Глубокая копия AgentModel для ветвления симуляций."""
    dst = AgentModel.__new__(AgentModel)
    dst.agent_class      = src.agent_class.copy()
    dst.agent_new_class  = src.agent_new_class.copy()
    dst.days_infected    = src.days_infected.copy()
    dst.days_in_R        = src.days_in_R.copy()
    dst.EPOP_initial     = src.EPOP_initial
    dst.EPOP             = src.EPOP
    return dst


def run_agent_model(days, params_parsed,
                    contacts=None,
                    pti=None,
                    dtr=None,
                    imm_loss=None,
                    mu=None,
                    epop=None,
                    s0=None,
                    y0=None,
                    seed=None):
    if seed is not None:
        np.random.seed(int(seed))

    if epop is not None and not (isinstance(epop, float) and math.isnan(epop)) and epop > 0:
        EPOP = float(epop)
    else:
        EPOP = params_parsed["EPOP"]["curr"]

    if y0 is not None and not (isinstance(y0, float) and math.isnan(y0)):
        Y0 = float(y0)
    else:
        Y0 = params_parsed["Y0"]["curr"]

    if s0 is not None and not (isinstance(s0, float) and math.isnan(s0)):
        raw = float(s0)
        if raw > 1.0:
            if EPOP <= 0:
                S0 = 1.0
            else:
                S0 = min(raw / EPOP, 1.0)
        else:
            S0 = raw
    else:
        S0 = params_parsed["S0"]["curr"]

    if math.isnan(EPOP) or EPOP <= 0:
        EPOP = 10000.0
    if math.isnan(Y0) or Y0 < 0:
        Y0 = 1.0
    if math.isnan(S0):
        S0 = 0.5

    S0 = max(0.0, min(S0, 1.0))

    if S0 * EPOP + Y0 > EPOP:
        S0 = max(0.0, (EPOP - Y0) / EPOP)

    model = AgentModel(EPOP, S0, Y0)

    S_list, I_list, Inew_list, R_list = [], [], [], []

    for day in range(days):
        c_val = contacts
        if c_val is None or (isinstance(c_val, float) and math.isnan(c_val)):
            c_val = param_value_at_day(params_parsed["CONTACTS"], day)

        p_val = pti
        if p_val is None or (isinstance(p_val, float) and math.isnan(p_val)):
            p_val = param_value_at_day(params_parsed["PTI"], day)

        d_val = dtr
        if d_val is None or (isinstance(d_val, float) and math.isnan(d_val)):
            d_val = param_value_at_day(params_parsed["DTR"], day)

        im_val = imm_loss
        if im_val is None or (isinstance(im_val, float) and math.isnan(im_val)):
            node = params_parsed.get("IMM_LOSS", {"curr": 0.0, "specdays": {}})
            im_val = param_value_at_day(node, day)

        mu_val = mu
        if mu_val is None or (isinstance(mu_val, float) and math.isnan(mu_val)):
            node = params_parsed.get("MU", {"curr": 0.0, "specdays": {}})
            mu_val = param_value_at_day(node, day)

        if math.isnan(c_val):
            c_val = 0
        if math.isnan(p_val):
            p_val = 0.0
        if math.isnan(d_val):
            d_val = 1
        if math.isnan(im_val):
            im_val = 0
        if math.isnan(mu_val):
            mu_val = 0.0

        d_val = int(round(d_val))
        im_val = int(round(im_val))

        stats = model.step(c_val, p_val, d_val, im_val, mu_val)
        S_list.append(stats["S_abs"])
        I_list.append(stats["I_abs"])
        Inew_list.append(stats["I_new_abs"])
        R_list.append(stats["R_abs"])

    return pd.DataFrame({
        "day":        np.arange(days),
        "S_abs":      S_list,
        "I_abs":      I_list,
        "I_new_abs":  Inew_list,
        "R_abs":      R_list,
    })


def compute_rmse(fact, model):
    fact = np.asarray(fact, dtype=float)
    model = np.asarray(model, dtype=float)
    n = min(len(fact), len(model))
    if n < 2:
        return float("nan")
    yt = fact[:n]
    yp = model[:n]
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def make_grid(vmin, vmax, step, log_scale=False):
    if step <= 0:
        raise ValueError("step должен быть > 0")
    if vmax < vmin:
        raise ValueError("max должен быть >= min")

    if log_scale:
        if vmin <= 0:
            raise ValueError("Для log-шкалы min должен быть > 0.")
        lo = math.log10(vmin)
        hi = math.log10(vmax)
        n = int(round((hi - lo) / step)) + 1
        return [10 ** (lo + i * step) for i in range(n)]
    else:
        n = int(round((vmax - vmin) / step)) + 1
        return [vmin + i * step for i in range(n)]


def recursive_grid_search(
    data,
    fixed_params,
    id_params,
    levels=3,
    shrink=0.2,
    seed=None,
    progress_cb=None,
    log_cb=None,
):
    days = len(data["day_numbers"])
    fact = data["fact_real"]
    params_parsed = data["params"]

    current = {}
    for name, spec in id_params.items():
        current[name] = {
            "min": float(spec["min"]),
            "max": float(spec["max"]),
            "step": float(spec["step"]),
            "log": bool(spec.get("log", False)),
        }

    all_rows = []
    best_rmse = float("inf")
    best_combo = None

    for level in range(levels):
        grids = {}
        names = list(id_params.keys())

        for name in names:
            spec = current[name]
            grids[name] = make_grid(
                spec["min"], spec["max"], spec["step"], spec["log"]
            )

        combos = list(itertools.product(*[grids[n] for n in names]))
        total = len(combos)

        level_best_rmse = float("inf")
        level_best_combo = None
        level_rows = []

        n_exceptions_logged = 0

        for i, values in enumerate(combos, 1):
            kw_lower = dict(fixed_params)
            for n, v in zip(names, values):
                kw_lower[n.lower()] = v

            try:
                df = run_agent_model(days, params_parsed, seed=seed, **kw_lower)
                rmse = compute_rmse(fact, df["I_new_abs"])
            except Exception as e:
                if n_exceptions_logged < 3 and log_cb is not None:
                    log_cb(f"[grid][except] {type(e).__name__}: {e}")
                    log_cb(f"  kwargs={kw_lower}")
                    import traceback as _tb
                    log_cb(_tb.format_exc())
                n_exceptions_logged += 1
                rmse = float("inf")

            row = {"level": level, "rmse": rmse}
            for n, v in zip(names, values):
                row[n] = v
            level_rows.append(row)

            if math.isfinite(rmse) and rmse < level_best_rmse:
                level_best_rmse = rmse
                level_best_combo = dict(zip(names, values))

            if progress_cb is not None:
                progress_cb(level, i, total)

        all_rows.extend(level_rows)

        if log_cb is not None:
            log_cb(f"[grid] уровень {level}: {total} комбинаций, "
                   f"лучшая RMSE = {level_best_rmse}")

        if level_best_combo is None:
            continue

        if level_best_rmse < best_rmse:
            best_rmse = level_best_rmse
            best_combo = level_best_combo

        for name in names:
            spec = current[name]
            if spec["log"]:
                if level_best_combo[name] <= 0:
                    continue
                center = math.log10(level_best_combo[name])
                width = math.log10(spec["max"]) - math.log10(spec["min"])
                new_min = 10 ** (center - width * shrink)
                new_max = 10 ** (center + width * shrink)
                new_step = spec["step"] * shrink
            else:
                center = level_best_combo[name]
                width = spec["max"] - spec["min"]
                new_min = max(center - width * shrink, 0.0)
                new_max = center + width * shrink
                new_step = spec["step"] * shrink

            current[name] = {
                "min": new_min,
                "max": new_max,
                "step": new_step,
                "log": spec["log"],
            }

    df_all = pd.DataFrame(all_rows).sort_values(
        "rmse", kind="stable").reset_index(drop=True)

    return {
        "best_combo": best_combo,
        "best_rmse": best_rmse,
        "rows": df_all,
    }


def recursive_pti_identification(
    data,
    base_params,
    pti_min=0.005,
    pti_max=0.15,
    pti_step=0.0025,
    window=15,
    seed=None,
    progress_cb=None,
    log_cb=None,
):
    """
    Рекурсивная идентификация PTI(t) методом полной сетки.

    На каждом дне t:
        1) строится сетка pti_min..pti_max с шагом pti_step
        2) для каждого кандидата клонируется текущая модель
           и прогоняется window шагов вперёд
        3) считается RMSE на окне [t, t+window)
        4) выбирается PTI с минимальным RMSE
        5) делается один шаг реальной модели с этим PTI

    Возвращает DataFrame:
        day, pti_t, rmse_local, S_abs, I_abs, I_new_abs, R_abs
    """

    days = len(data["day_numbers"])
    fact = data["fact_real"]
    params_parsed = data["params"]

    EPOP = base_params.get("epop")
    if EPOP is None or (isinstance(EPOP, float) and math.isnan(EPOP)):
        EPOP = params_parsed["EPOP"]["curr"]
    EPOP = int(round(EPOP))

    Y0 = base_params.get("y0")
    if Y0 is None or (isinstance(Y0, float) and math.isnan(Y0)):
        Y0 = params_parsed["Y0"]["curr"]
    Y0 = int(round(Y0))

    S0 = base_params.get("s0")
    if S0 is None or (isinstance(S0, float) and math.isnan(S0)):
        S0 = params_parsed["S0"]["curr"]
    S0 = float(S0)
    if S0 > 1.0:
        S0 = min(S0 / EPOP, 1.0)

    contacts = base_params.get("contacts")
    if contacts is None or (isinstance(contacts, float) and math.isnan(contacts)):
        contacts = params_parsed["CONTACTS"]["curr"]
    contacts = int(round(contacts))

    dtr = base_params.get("dtr")
    if dtr is None or (isinstance(dtr, float) and math.isnan(dtr)):
        dtr = params_parsed["DTR"]["curr"]
    dtr = int(round(dtr))

    imm_loss = base_params.get("imm_loss")
    if imm_loss is None or (isinstance(imm_loss, float) and math.isnan(imm_loss)):
        imm_loss = params_parsed.get("IMM_LOSS", {"curr": 0})["curr"]
    imm_loss = int(round(imm_loss))

    mu = base_params.get("mu")
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        mu = params_parsed.get("MU", {"curr": 0.0})["curr"]
    mu = float(mu)

    if seed is not None:
        np.random.seed(int(seed))

    if pti_step <= 0:
        raise ValueError("pti_step должен быть > 0")
    if window < 1:
        raise ValueError("window должен быть >= 1")

    n_pti = int(round((pti_max - pti_min) / pti_step)) + 1
    pti_grid = [pti_min + i * pti_step for i in range(n_pti)]

    if log_cb is not None:
        log_cb(f"[pti-id] Сетка PTI: {len(pti_grid)} точек, "
               f"диапазон [{pti_min}, {pti_max}], шаг {pti_step}")
        log_cb(f"[pti-id] Окно: {window} дней")

    model = AgentModel(EPOP, S0, Y0)

    rows = []

    for t in range(days):
        best_pti = pti_grid[0]
        best_rmse = float("inf")

        for pti_cand in pti_grid:
            sim_model = _clone_agent_model(model)
            preds = []
            for k in range(window):
                idx = t + k
                if idx >= days:
                    break
                stats = sim_model.step(contacts, pti_cand, dtr, imm_loss, mu)
                preds.append(stats["I_new_abs"])

            if len(preds) < 2:
                continue

            fact_slice = fact[t:t + len(preds)]
            mask = np.isfinite(fact_slice)
            if mask.sum() < 2:
                continue

            rmse = float(np.sqrt(np.mean(
                (fact_slice[mask] - np.array(preds)[mask]) ** 2
            )))

            if rmse < best_rmse:
                best_rmse = rmse
                best_pti = pti_cand

        stats = model.step(contacts, best_pti, dtr, imm_loss, mu)

        rows.append({
            "day":         t,
            "pti_t":       best_pti,
            "rmse_local":  best_rmse,
            "S_abs":       stats["S_abs"],
            "I_abs":       stats["I_abs"],
            "I_new_abs":   stats["I_new_abs"],
            "R_abs":       stats["R_abs"],
        })

        if progress_cb is not None:
            progress_cb(t + 1, days)
        if log_cb is not None and (t % 20 == 0 or t == days - 1):
            log_cb(f"[pti-id] день {t}: PTI={best_pti:.6f}, "
                   f"RMSE_local={best_rmse:.4g}")

    return pd.DataFrame(rows)


ID_PARAMS = [
    ("CONTACTS", "CONTACTS", "3.2", "5.0", "0.5", False),
    ("PTI",      "PTI",      "0.02", "0.10", "0.01", False),
    ("DTR",      "DTR",      "8",    "18",   "1",    False),
    ("IMM_LOSS", "IMM_LOSS", "90",   "270",  "30",   False),
    ("MU",       "MU",       "0.00005", "0.0003", "0.00005", False),
    ("EPOP",     "EPOP",     "3000", "10000", "1000", False),
    ("S0",       "S0",       "0.7",  "0.99", "0.05", False),
    ("Y0",       "Y0",       "1",    "20",   "3",    False),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Агентная SIRS-модель + идентификация параметров (RMSE)")
        self.geometry("1250x1000")
        self.minsize(1100, 820)

        self.path_json = tk.StringVar()
        self.path_out = tk.StringVar(value=os.path.abspath("agent_sirs.xlsx"))

        self.entry_epop = tk.StringVar(value="19100")
        self.entry_s0 = tk.StringVar(value="0.896582")
        self.entry_y0 = tk.StringVar(value="13")

        self.entry_contacts = tk.StringVar(value="3,7")
        self.entry_pti = tk.StringVar(value="0.04")
        self.entry_dtr = tk.StringVar(value="10")
        self.entry_imm_loss = tk.StringVar(value="180")
        self.entry_mu = tk.StringVar(value="0.000137")
        self.entry_seed = tk.StringVar(value="12345")

        # --- PTI(t): параметры сетки (редактируемые) ---
        self.entry_pti_min = tk.StringVar(value="0.005")
        self.entry_pti_max = tk.StringVar(value="0.15")
        self.entry_pti_step = tk.StringVar(value="0.0025")
        self.entry_window = tk.StringVar(value="15")

        self.id_enabled = {}
        self.id_min = {}
        self.id_max = {}
        self.id_step = {}
        self.id_log = {}

        for name, _lbl, vmin, vmax, vstep, vlog in ID_PARAMS:
            self.id_enabled[name] = tk.BooleanVar(value=False)
            self.id_min[name] = tk.StringVar(value=vmin)
            self.id_max[name] = tk.StringVar(value=vmax)
            self.id_step[name] = tk.StringVar(value=vstep)
            self.id_log[name] = tk.BooleanVar(value=vlog)

        self.id_levels = tk.StringVar(value="3")
        self.id_shrink = tk.StringVar(value="0.2")

        self.data = None
        self.model_df = None
        self.id_result = None
        self.pti_df = None

        self._build_widgets()

    def _build_widgets(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ff = ttk.LabelFrame(top, text=" Входной JSON ", padding=8)
        ff.pack(fill=tk.X)
        ff.columnconfigure(1, weight=1)

        ttk.Label(ff, text="JSON-файл:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(ff, textvariable=self.path_json).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(ff, text="Обзор…",
                   command=self._pick_json).grid(row=0, column=2, padx=4)
        ttk.Button(ff, text="Загрузить",
                   command=self._load).grid(row=0, column=3, padx=4)

        ttk.Label(ff, text="Куда сохранить результат:").grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Entry(ff, textvariable=self.path_out).grid(
            row=1, column=1, sticky="ew", padx=4)
        ttk.Button(ff, text="Обзор…",
                   command=self._pick_out).grid(row=1, column=2, padx=4)

        nf = ttk.LabelFrame(
            top,
            text=" Начальные условия ",
            padding=8)
        nf.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(nf, text="EPOP:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(nf, textvariable=self.entry_epop, width=12).grid(
            row=0, column=1, padx=4, pady=2)

        ttk.Label(nf, text="S0 (доля или абс. число):").grid(
            row=0, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(nf, textvariable=self.entry_s0, width=12).grid(
            row=0, column=3, padx=4, pady=2)

        ttk.Label(nf, text="Y0 (чел.):").grid(row=0, column=4, sticky="e", padx=4, pady=2)
        ttk.Entry(nf, textvariable=self.entry_y0, width=12).grid(
            row=0, column=5, padx=4, pady=2)

        ttk.Label(
            nf,
            text=("S0: ≤ 1 — доля; > 1 — абсолютное число восприимчивых. "
                  "При загрузке JSON поля заполняются.")
        ).grid(row=1, column=0, columnspan=6, sticky="w", padx=4, pady=(4, 0))

        pf = ttk.LabelFrame(
            top,
            text=" Эпидемиологические параметры ",
            padding=8)
        pf.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(pf, text="CONTACTS:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(pf, textvariable=self.entry_contacts, width=12).grid(
            row=0, column=1, padx=4, pady=2)

        ttk.Label(pf, text="PTI:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(pf, textvariable=self.entry_pti, width=12).grid(
            row=0, column=3, padx=4, pady=2)

        ttk.Label(pf, text="DTR:").grid(row=0, column=4, sticky="e", padx=4, pady=2)
        ttk.Entry(pf, textvariable=self.entry_dtr, width=12).grid(
            row=0, column=5, padx=4, pady=2)

        ttk.Label(pf, text="IMM_LOSS:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(pf, textvariable=self.entry_imm_loss, width=12).grid(
            row=1, column=1, padx=4, pady=2)

        ttk.Label(pf, text="MU:").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(pf, textvariable=self.entry_mu, width=12).grid(
            row=1, column=3, padx=4, pady=2)

        ttk.Label(pf, text="Seed:").grid(row=1, column=4, sticky="e", padx=4, pady=2)
        ttk.Entry(pf, textvariable=self.entry_seed, width=12).grid(
            row=1, column=5, padx=4, pady=2)

        ttk.Button(pf, text="↺ Сброс",
                   command=self._reset_overrides).grid(
            row=3, column=6, padx=6, pady=2)

        ttk.Button(pf, text="💾 Сохранить параметры",
                   command=self._export_ui_params).grid(
            row=3, column=7, padx=6, pady=2)

        # --- Параметры PTI(t)-идентификации (редактируемые) ---
        ptif = ttk.LabelFrame(
            top,
            text=" PTI(t): сетка и окно ",
            padding=8)
        ptif.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(ptif, text="pti_min:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(ptif, textvariable=self.entry_pti_min, width=10).grid(
            row=0, column=1, padx=4, pady=2)

        ttk.Label(ptif, text="pti_max:").grid(row=0, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(ptif, textvariable=self.entry_pti_max, width=10).grid(
            row=0, column=3, padx=4, pady=2)

        ttk.Label(ptif, text="pti_step:").grid(row=0, column=4, sticky="e", padx=4, pady=2)
        ttk.Entry(ptif, textvariable=self.entry_pti_step, width=10).grid(
            row=0, column=5, padx=4, pady=2)

        ttk.Label(ptif, text="WINDOW (дней):").grid(
            row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(ptif, textvariable=self.entry_window, width=10).grid(
            row=1, column=1, padx=4, pady=2)

        ttk.Label(
            ptif,
            text=("Сетка PTI(t): полный перебор [pti_min, pti_max] с шагом pti_step. "
                  "WINDOW — длина локального окна RMSE.")
        ).grid(row=2, column=0, columnspan=6,
               sticky="w", padx=4, pady=(4, 0))

        idf = ttk.LabelFrame(
            top,
            text=" Идентификация параметров (min / max / step) ",
            padding=8)
        idf.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(idf, text="Вкл").grid(row=0, column=0, padx=4)
        ttk.Label(idf, text="Параметр").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(idf, text="min").grid(row=0, column=2, padx=4)
        ttk.Label(idf, text="max").grid(row=0, column=3, padx=4)
        ttk.Label(idf, text="step").grid(row=0, column=4, padx=4)
        ttk.Label(idf, text="log").grid(row=0, column=5, padx=4)

        for i, (name, _lbl, _vmin, _vmax, _vstep, _vlog) in enumerate(ID_PARAMS, start=1):
            ttk.Checkbutton(
                idf, variable=self.id_enabled[name]
            ).grid(row=i, column=0, padx=4, pady=2)

            ttk.Label(idf, text=name).grid(row=i, column=1, sticky="w", padx=4, pady=2)

            ttk.Entry(idf, textvariable=self.id_min[name], width=10).grid(
                row=i, column=2, padx=4, pady=2)
            ttk.Entry(idf, textvariable=self.id_max[name], width=10).grid(
                row=i, column=3, padx=4, pady=2)
            ttk.Entry(idf, textvariable=self.id_step[name], width=10).grid(
                row=i, column=4, padx=4, pady=2)
            ttk.Checkbutton(
                idf, variable=self.id_log[name]
            ).grid(row=i, column=5, padx=4, pady=2)

        row_extra = len(ID_PARAMS) + 1
        ttk.Label(idf, text="Уровней:").grid(row=row_extra, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(idf, textvariable=self.id_levels, width=6).grid(
            row=row_extra, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(idf, text="shrink:").grid(row=row_extra, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(idf, textvariable=self.id_shrink, width=6).grid(
            row=row_extra, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(
            idf,
            text=("Рекурсивная сетка: на каждом уровне перебираем комбинации, "
                  "затем сужаем диапазон вокруг лучшей точки на shrink (0..1).")
        ).grid(row=row_extra + 1, column=0, columnspan=6,
               sticky="w", padx=4, pady=(4, 0))

        act = ttk.Frame(self, padding=(8, 8))
        act.pack(side=tk.TOP, fill=tk.X)

        self.btn_run = ttk.Button(act, text="▶ Запустить модель",
                                  command=self._run)
        self.btn_ident = ttk.Button(act, text="🔍 Идентифицировать",
                                    command=self._run_identification)
        self.btn_pti = ttk.Button(act, text="🔁 PTI(t) идентификация",
                                  command=self._run_pti_identification)
        self.btn_apply = ttk.Button(act, text="✔ Применить лучшие",
                                    command=self._apply_best, state=tk.DISABLED)
        self.btn_save = ttk.Button(act, text="💾 Сохранить результат",
                                   command=self._save, state=tk.DISABLED)

        self.btn_run.pack(side=tk.LEFT, padx=4)
        self.btn_ident.pack(side=tk.LEFT, padx=4)
        self.btn_pti.pack(side=tk.LEFT, padx=4)
        self.btn_apply.pack(side=tk.LEFT, padx=4)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(act, text="Готов", foreground="#555")
        self.status.pack(side=tk.LEFT, padx=12)

        self.progress = ttk.Progressbar(act, mode="determinate",
                                        length=220, maximum=1)
        self.progress.pack(side=tk.LEFT, padx=12)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        gf = ttk.Frame(nb)
        nb.add(gf, text="График")
        self.fig = Figure(figsize=(14, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax2 = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, gf).update()

        lf = ttk.Frame(nb, padding=8)
        nb.add(lf, text="Логи")
        self.log_text = scrolledtext.ScrolledText(lf, font=("Consolas", 10),
                                                  wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

        tf = ttk.Frame(nb, padding=8)
        nb.add(tf, text="Модель (первые 100 строк)")
        self.table_text = scrolledtext.ScrolledText(tf, font=("Consolas", 9),
                                                    wrap=tk.NONE)
        self.table_text.pack(fill=tk.BOTH, expand=True)

        gtf = ttk.Frame(nb, padding=8)
        nb.add(gtf, text="Идентификация (топ-200)")
        self.id_text = scrolledtext.ScrolledText(gtf, font=("Consolas", 9),
                                                 wrap=tk.NONE)
        self.id_text.pack(fill=tk.BOTH, expand=True)

        ptf = ttk.Frame(nb, padding=8)
        nb.add(ptf, text="PTI(t)")
        self.pti_text = scrolledtext.ScrolledText(ptf, font=("Consolas", 9),
                                                  wrap=tk.NONE)
        self.pti_text.pack(fill=tk.BOTH, expand=True)

    def _pick_json(self):
        p = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json *.jsonModel"),
                       ("All files", "*.*")])
        if p:
            self.path_json.set(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"),
                       ("CSV files", "*.csv"),
                       ("All files", "*.*")],
            initialfile="agent_sirs.xlsx")
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

    def _reset_overrides(self):
        self.entry_epop.set("")
        self.entry_s0.set("")
        self.entry_y0.set("")
        self.entry_contacts.set("")
        self.entry_pti.set("")
        self.entry_dtr.set("")
        self.entry_imm_loss.set("")
        self.entry_mu.set("")
        self.entry_seed.set("12345")
        self._log("[gui] Все поля очищены — значения будут взяты из JSON.")

    def _export_ui_params(self):
         """Выгрузка полей из блоков 'Начальные условия' и 'Эпидемиологические параметры'.
         Только колонки 'Параметр' и 'Значение'."""
         path = filedialog.asksaveasfilename(
             defaultextension=".xlsx",
             filetypes=[("Excel files", "*.xlsx"),
                        ("CSV files", "*.csv"),
                        ("All files", "*.*")],
             initialfile="ui_params.xlsx",
         )
         if not path:
             return
 
         rows = [
             # --- Начальные условия ---
             ("EPOP", self.entry_epop.get()),
             ("S0", self.entry_s0.get()),
             ("Y0", self.entry_y0.get()),
 
             # --- Эпидемиологические параметры ---
             ("CONTACTS", self.entry_contacts.get()),
             ("PTI", self.entry_pti.get()),
             ("DTR", self.entry_dtr.get()),
             ("IMM_LOSS", self.entry_imm_loss.get()),
             ("MU", self.entry_mu.get()),
             ("Seed", self.entry_seed.get()),
         ]
 
         df = pd.DataFrame(rows, columns=["Параметр", "Значение"])
 
         try:
             ext = os.path.splitext(path)[1].lower()
             if ext == ".csv":
                 df.to_csv(path, index=False, encoding="utf-8-sig")
             else:
                 with pd.ExcelWriter(path, engine="openpyxl") as xw:
                     df.to_excel(xw, sheet_name="UI_params", index=False)
 
             self._log(f"[export-ui] Сохранено: {path}")
             messagebox.showinfo("Успех", f"Параметры сохранены:\n{path}")
         except Exception as e:
             messagebox.showerror("Ошибка сохранения", str(e))
             self._log("[export-ui][ERROR] " + str(e))

    def _load(self):
        p = self.path_json.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Ошибка", "Укажите существующий JSON-файл.")
            return
        try:
            self.data = load_json(p)
            n = len(self.data["day_numbers"])
            self._log(f"[load] Загружено {n} дней из {p}")

            pr = self.data["params"]
            self._log(f"[load] EPOP={pr['EPOP']['curr']}, "
                      f"S0={pr['S0']['curr']}, "
                      f"Y0={pr['Y0']['curr']}")

            for k in ("CONTACTS", "PTI", "DTR", "IMM_LOSS", "MU"):
                node = pr.get(k, {"curr": 0.0, "specdays": {}})
                self._log(f"[load] {k}: curr={node['curr']}, "
                          f"specdays={len(node['specdays'])} записей")

            fact = self.data["fact_real"]
            n_fin = int(np.sum(np.isfinite(fact)))
            self._log(f"[load] fact_real: {len(fact)} точек, {n_fin} конечных")
            if n_fin == 0:
                self._log("[load] ВНИМАНИЕ: real пуст/NaN для всех дней. "
                          "Проверьте поле real в days[].total[1].")

            def _set_val(entry_var, value, is_int=False):
                cur = entry_var.get().strip()
                if cur and cur.lower() not in ("nan", "none"):
                    return
                if math.isnan(value):
                    return
                if is_int:
                    entry_var.set(str(int(round(value))))
                else:
                    entry_var.set(f"{value:.6g}")

            _set_val(self.entry_epop, pr["EPOP"]["curr"], is_int=True)
            _set_val(self.entry_s0, pr["S0"]["curr"])
            _set_val(self.entry_y0, pr["Y0"]["curr"], is_int=True)

            _set_val(self.entry_contacts, pr["CONTACTS"]["curr"])
            _set_val(self.entry_pti, pr["PTI"]["curr"])
            _set_val(self.entry_dtr, pr["DTR"]["curr"])
            _set_val(self.entry_imm_loss, pr["IMM_LOSS"]["curr"], is_int=True)
            _set_val(self.entry_mu, pr["MU"]["curr"])

            self._log("[load] Поля заполнены значениями из JSON.")

            self.model_df = None
            self.id_result = None
            self.pti_df = None
            self.btn_save.configure(state=tk.DISABLED)
            self.btn_apply.configure(state=tk.DISABLED)
            self._plot()
            self._fill_table()
            self._set_status("Данные загружены", "#0a0")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            self._log("[load][ERROR] " + str(e))

    def _collect_fixed_params(self):
        return {
            "epop": _to_float(self.entry_epop.get()),
            "s0": _to_float(self.entry_s0.get()),
            "y0": _to_float(self.entry_y0.get()),
            "contacts": _to_float(self.entry_contacts.get()),
            "pti": _to_float(self.entry_pti.get()),
            "dtr": _to_float(self.entry_dtr.get()),
            "imm_loss": _to_float(self.entry_imm_loss.get()),
            "mu": _to_float(self.entry_mu.get()),
        }

    def _run(self):
        if self.data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите JSON.")
            return
        try:
            days = len(self.data["day_numbers"])
            kw = self._collect_fixed_params()
            seed_val = _to_int_or_none(self.entry_seed.get())

            self._log(f"[model] Запуск модели на {days} дней…")
            self._log(f"[model] {kw}, seed={seed_val}")

            self.model_df = run_agent_model(
                days, self.data["params"],
                seed=seed_val,
                **kw
            )

            self._log(f"[model] Старт: S={int(self.model_df['S_abs'].iloc[0])}, "
                      f"I={int(self.model_df['I_abs'].iloc[0])}, "
                      f"R={int(self.model_df['R_abs'].iloc[0])}")

            rmse = compute_rmse(self.data["fact_real"], self.model_df["I_new_abs"])
            self._log(f"[fit] RMSE={rmse:.6g}")

            self._plot()
            self._fill_table()
            self.btn_save.configure(state=tk.NORMAL)
            self._set_status("Модель отработала", "#0a0")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("[model][ERROR] " + str(e))
            self._log(tb)
            messagebox.showerror("Ошибка модели", str(e))

    def _run_identification(self):
        if self.data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите JSON.")
            return

        try:
            id_spec = {}
            for name, _lbl, _vmin, _vmax, _vstep, _vlog in ID_PARAMS:
                if not self.id_enabled[name].get():
                    continue

                vmin = _to_float(self.id_min[name].get())
                vmax = _to_float(self.id_max[name].get())
                vstep = _to_float(self.id_step[name].get())
                vlog = bool(self.id_log[name].get())

                if math.isnan(vmin) or math.isnan(vmax) or math.isnan(vstep):
                    raise ValueError(f"Параметр {name}: min/max/step должны быть числами.")
                if vstep <= 0:
                    raise ValueError(f"Параметр {name}: step должен быть > 0.")
                if vmax < vmin:
                    raise ValueError(f"Параметр {name}: max должен быть >= min.")

                id_spec[name] = {
                    "min": vmin, "max": vmax, "step": vstep, "log": vlog
                }

            if not id_spec:
                messagebox.showwarning(
                    "Нечего искать",
                    "Отметьте хотя бы один параметр для идентификации.")
                return

            levels = int(self.id_levels.get())
            shrink = float(self.id_shrink.get().replace(",", "."))
            if levels < 1:
                raise ValueError("Уровней должно быть >= 1.")
            if not (0 < shrink < 1):
                raise ValueError("shrink должен быть в (0, 1).")

        except Exception as e:
            messagebox.showerror("Ошибка параметров идентификации", str(e))
            return

        fixed = self._collect_fixed_params()

        fixed_for_id = dict(fixed)
        for name in id_spec.keys():
            key = name.lower()
            if key in fixed_for_id:
                fixed_for_id[key] = float("nan")

        seed_val = _to_int_or_none(self.entry_seed.get())

        self._log("")
        self._log("=" * 60)
        self._log("ИДЕНТИФИКАЦИЯ ПАРАМЕТРОВ (RMSE)")
        self._log("=" * 60)
        self._log(f"  Ищем: {list(id_spec.keys())}")
        self._log(f"  Уровней: {levels}, shrink: {shrink}")
        self._log(f"  Фиксировано: {fixed_for_id}")

        t0 = time.time()

        def progress_cb(level, i, total):
            self.progress.configure(maximum=total, value=i)
            self._set_status(f"Идентификация: уровень {level}, {i}/{total}",
                             "#c80")
            if i % 20 == 0 or i == total:
                self._log(f"[id] уровень {level}: {i}/{total}")
            self.update_idletasks()

        try:
            result = recursive_grid_search(
                self.data,
                fixed_for_id,
                id_spec,
                levels=levels,
                shrink=shrink,
                seed=seed_val,
                progress_cb=progress_cb,
                log_cb=self._log,
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._log("[id][ERROR] " + str(e))
            self._log(tb)
            messagebox.showerror("Ошибка идентификации", str(e))
            return

        self.progress.configure(value=0)
        self.id_result = result

        dt = time.time() - t0
        best = result["best_combo"]
        best_rmse = result["best_rmse"]

        self._log(f"\n[id] Всего проверено комбинаций: {len(result['rows'])}")
        self._log(f"[id] Время: {dt:.1f} с")

        if best is None:
            self._log("[id] Не удалось найти ни одной рабочей комбинации.")
            self._log("[id] Проверьте диапазоны min/max/step и значения EPOP/S0/Y0.")
            return

        self._log(f"[id] Лучшая RMSE = {best_rmse:.6g}")
        for k, v in best.items():
            self._log(f"[id]   {k} = {v:.6g}")

        top = result["rows"].head(10)
        self._log("\n[id] Топ-10:")
        for _, row in top.iterrows():
            line = f"  lvl={int(row['level'])}  RMSE={row['rmse']:.6g}  "
            parts = [f"{k}={row[k]:.6g}" for k in id_spec.keys()]
            line += "  ".join(parts)
            self._log(line)

        self._fill_id_table(result["rows"])
        self.btn_apply.configure(state=tk.NORMAL)
        self._set_status("Идентификация завершена", "#0a0")

    def _run_pti_identification(self):
        if self.data is None:
            messagebox.showwarning("Нет данных", "Сначала загрузите JSON.")
            return

        base = self._collect_fixed_params()
        seed_val = _to_int_or_none(self.entry_seed.get())

        # --- параметры сетки PTI (из GUI) ---
        try:
            pti_min = float(self.entry_pti_min.get().replace(",", "."))
            pti_max = float(self.entry_pti_max.get().replace(",", "."))
            pti_step = float(self.entry_pti_step.get().replace(",", "."))
            window = int(self.entry_window.get())
        except Exception as e:
            messagebox.showerror("Ошибка параметров PTI(t)", str(e))
            return

        if pti_step <= 0:
            messagebox.showerror("Ошибка", "pti_step должен быть > 0")
            return
        if pti_max < pti_min:
            messagebox.showerror("Ошибка", "pti_max должен быть >= pti_min")
            return
        if window < 1:
            messagebox.showerror("Ошибка", "WINDOW должен быть >= 1")
            return

        self._log("")
        self._log("=" * 60)
        self._log("РЕКУРСИВНАЯ ИДЕНТИФИКАЦИЯ PTI(t) — ПОЛНАЯ СЕТКА")
        self._log("=" * 60)
        self._log(f"  pti_min={pti_min}, pti_max={pti_max}, "
                  f"pti_step={pti_step}")
        self._log(f"  window={window} дней")

        n_pti = int(round((pti_max - pti_min) / pti_step)) + 1
        n_days = len(self.data["day_numbers"])
        total_runs = n_pti * window * n_days
        self._log(f"  Точек сетки: {n_pti}")
        self._log(f"  Ожидаемое число шагов модели: {total_runs}")

        t0 = time.time()

        def progress_cb(i, total):
            self.progress.configure(maximum=total, value=i)
            self._set_status(f"PTI(t): {i}/{total}", "#c80")
            self.update_idletasks()

        try:
            df_pti = recursive_pti_identification(
                self.data,
                base,
                pti_min=pti_min,
                pti_max=pti_max,
                pti_step=pti_step,
                window=window,
                seed=seed_val,
                progress_cb=progress_cb,
                log_cb=self._log,
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._log("[pti-id][ERROR] " + str(e))
            self._log(tb)
            messagebox.showerror("Ошибка PTI(t) идентификации", str(e))
            return

        self.progress.configure(value=0)
        dt = time.time() - t0
        self.pti_df = df_pti
        self.model_df = df_pti.copy()

        self._log(f"[pti-id] Готово за {dt:.1f} с")
        self._log(f"[pti-id] Всего строк: {len(df_pti)}")
        self._log(f"[pti-id] PTI(t): min={df_pti['pti_t'].min():.6f}, "
                  f"max={df_pti['pti_t'].max():.6f}, "
                  f"mean={df_pti['pti_t'].mean():.6f}")

        self._plot_pti()
        self._fill_table()
        self._fill_pti_table()
        self.btn_save.configure(state=tk.NORMAL)
        self._set_status("PTI(t) готово", "#0a0")

    def _apply_best(self):
        if self.id_result is None or self.id_result["best_combo"] is None:
            return

        best = self.id_result["best_combo"]

        mapping = {
            "EPOP": self.entry_epop,
            "S0": self.entry_s0,
            "Y0": self.entry_y0,
            "CONTACTS": self.entry_contacts,
            "PTI": self.entry_pti,
            "DTR": self.entry_dtr,
            "IMM_LOSS": self.entry_imm_loss,
            "MU": self.entry_mu,
        }

        for k, v in best.items():
            if k in mapping:
                mapping[k].set(f"{v:.6g}")

        self._log(f"[apply] Перенесены лучшие параметры: {best}")

        self._run()

    def _fill_id_table(self, df):
        self.id_text.delete("1.0", tk.END)

        if df is None or len(df) == 0:
            return

        cols = list(df.columns)
        header = "  ".join(f"{c:>12}" for c in cols)
        self.id_text.insert(tk.END, header + "\n")
        self.id_text.insert(tk.END, "-" * len(header) + "\n")

        for _, row in df.head(200).iterrows():
            line = []
            for c in cols:
                v = row[c]
                if isinstance(v, (int, np.integer)):
                    line.append(f"{int(v):>12}")
                elif isinstance(v, (float, np.floating)):
                    line.append(f"{v:>12.6g}")
                else:
                    line.append(f"{str(v):>12}")
            self.id_text.insert(tk.END, "  ".join(line) + "\n")

    def _fill_pti_table(self):
        self.pti_text.delete("1.0", tk.END)

        if self.pti_df is None or len(self.pti_df) == 0:
            return

        header = (f"{'day':>6} {'PTI(t)':>12} {'RMSE_loc':>12} "
                  f"{'S_abs':>10} {'I_abs':>10} {'I_new':>10} {'R_abs':>10}")
        self.pti_text.insert(tk.END, header + "\n")
        self.pti_text.insert(tk.END, "-" * len(header) + "\n")

        for _, row in self.pti_df.head(300).iterrows():
            line = (f"{int(row['day']):>6} {row['pti_t']:>12.6f} "
                    f"{row['rmse_local']:>12.4g} "
                    f"{int(row['S_abs']):>10} {int(row['I_abs']):>10} "
                    f"{int(row['I_new_abs']):>10} {int(row['R_abs']):>10}")
            self.pti_text.insert(tk.END, line + "\n")

    def _plot(self):
        self.ax1.clear()

        if hasattr(self, "ax2") and self.ax2 is not None:
            self.ax2.remove()
            self.ax2 = None

        if self.data is None:
            self.canvas.draw_idle()
            return

        x = np.arange(len(self.data["day_numbers"]))

        if np.any(np.isfinite(self.data["fact_real"])):
            self.ax1.plot(
                x, self.data["fact_real"],
                color="green", lw=1.0,
                marker="o", markersize=2,
                label="Факт"
            )

        if self.model_df is not None and "pti_t" not in self.model_df.columns:
            m = self.model_df
            self.ax1.plot(
                x, m["I_new_abs"],
                color="red", lw=1.4,
                label="Прогноз (I_new_abs)"
            )

        self.ax1.set_xlabel("День")
        self.ax1.set_ylabel("Новые случаи")
        self.ax1.set_title("I_new_abs: модель и факт")
        self.ax1.grid(alpha=0.4)
        self.ax1.legend(fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_pti(self):
        self.ax1.clear()

        if hasattr(self, "ax2") and self.ax2 is not None:
            self.ax2.remove()
            self.ax2 = None

        if self.pti_df is None or len(self.pti_df) == 0:
            self.canvas.draw_idle()
            return

        df = self.pti_df
        x = df["day"].values

        if np.any(np.isfinite(self.data["fact_real"])):
            self.ax1.plot(x, self.data["fact_real"], color="green",
                          lw=1.0, marker="o", markersize=2, label="Факт")
        self.ax1.plot(x, df["I_new_abs"], color="red", lw=1.4,
                      label="Модель (I_new_abs)")

        self.ax1.set_xlabel("День")
        self.ax1.set_ylabel("Новые случаи")
        self.ax1.grid(alpha=0.4)
        self.ax1.legend(loc="upper left", fontsize=9)

        self.ax2 = self.ax1.twinx()
        self.ax2.plot(x, df["pti_t"], color="blue", lw=2.0,
                      linestyle="--", label="PTI(t)")
        self.ax2.set_ylabel("PTI(t)", color="blue")
        self.ax2.tick_params(axis="y", labelcolor="blue")
        self.ax2.legend(loc="upper right", fontsize=9)

        self.ax1.set_title("Рекурсивная идентификация PTI(t)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _fill_table(self):
        self.table_text.delete("1.0", tk.END)

        if self.model_df is None:
            return

        if "pti_t" in self.model_df.columns:
            header = (f"{'day':>6} {'PTI(t)':>12} {'RMSE_loc':>12} "
                      f"{'S_abs':>10} {'I_abs':>10} {'I_new':>10} {'R_abs':>10}")
            self.table_text.insert(tk.END, header + "\n")
            self.table_text.insert(tk.END, "-" * len(header) + "\n")

            df = self.model_df.head(100)
            for _, row in df.iterrows():
                line = (f"{int(row['day']):>6} {row['pti_t']:>12.6f} "
                        f"{row['rmse_local']:>12.4g} "
                        f"{int(row['S_abs']):>10} {int(row['I_abs']):>10} "
                        f"{int(row['I_new_abs']):>10} {int(row['R_abs']):>10}")
                self.table_text.insert(tk.END, line + "\n")
            return

        header = f"{'day':>6} {'S_abs':>10} {'I_abs':>10} {'I_new_abs':>12} {'R_abs':>10}"
        self.table_text.insert(tk.END, header + "\n")
        self.table_text.insert(tk.END, "-" * len(header) + "\n")

        df = self.model_df.head(100)
        for _, row in df.iterrows():
            line = (f"{int(row['day']):>6} {int(row['S_abs']):>10} "
                    f"{int(row['I_abs']):>10} {int(row['I_new_abs']):>12} "
                    f"{int(row['R_abs']):>10}")
            self.table_text.insert(tk.END, line + "\n")

    def _save(self):
        if self.model_df is None and self.id_result is None:
            messagebox.showwarning("Нет данных", "Сначала запустите модель.")
            return
        path = self.path_out.get()
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()

            if ext == ".csv":
                if self.model_df is not None:
                    self.model_df.to_csv(path, index=False,
                                         encoding="utf-8-sig")
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as xw:
                    if self.model_df is not None:
                        df = self.model_df.copy()
                        if "pti_t" not in df.columns:
                            fact = self.data["fact_real"]
                            n = min(len(df), len(fact))
                            df.loc[:n - 1, "I_fact_real"] = fact[:n]
                        df.to_excel(xw, sheet_name="agent_sirs", index=False)

                    if self.pti_df is not None:
                        self.pti_df.to_excel(xw, sheet_name="pti_recursive",
                                             index=False)

                    if self.id_result is not None:
                        self.id_result["rows"].to_excel(
                            xw, sheet_name="id_grid", index=False)
                        if self.id_result["best_combo"] is not None:
                            pd.DataFrame([self.id_result["best_combo"]]).to_excel(
                                xw, sheet_name="id_best", index=False)

                    used = {
                        "EPOP_override":     self.entry_epop.get() or "(JSON)",
                        "S0_override":       self.entry_s0.get() or "(JSON)",
                        "Y0_override":       self.entry_y0.get() or "(JSON)",
                        "CONTACTS_override": self.entry_contacts.get() or "(JSON)",
                        "PTI_override":      self.entry_pti.get() or "(JSON)",
                        "DTR_override":      self.entry_dtr.get() or "(JSON)",
                        "IMM_LOSS_override": self.entry_imm_loss.get() or "(JSON)",
                        "MU_override":       self.entry_mu.get() or "(JSON)",
                        "Seed":              self.entry_seed.get() or "(none)",
                        "PTI_min":           self.entry_pti_min.get(),
                        "PTI_max":           self.entry_pti_max.get(),
                        "PTI_step":          self.entry_pti_step.get(),
                        "WINDOW":            self.entry_window.get(),
                    }
                    pd.DataFrame(
                        list(used.items()),
                        columns=["param", "value"]
                    ).to_excel(xw, sheet_name="overrides", index=False)

            self._log(f"[save] Сохранено: {path}")
            messagebox.showinfo("Успех", f"Сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))
            self._log("[save][ERROR] " + str(e))


if __name__ == "__main__":
    App().mainloop()