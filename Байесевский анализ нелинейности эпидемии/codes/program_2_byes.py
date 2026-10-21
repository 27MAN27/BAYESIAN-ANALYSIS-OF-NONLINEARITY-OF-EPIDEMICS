# -*- coding: utf-8 -*-
"""
MCMC для R(t) и p(t)

ГЛАВНАЯ ЛОГИКА p(t)
--------------------
p_init загружается из Excel/CSV.

p_active_mask = (p_init > 0)

Если p_init[d] > 0:
    p[d] может адаптироваться MCMC.
    Начальное значение цепи:
        p_cur[d] = p_init[d]

Если p_init[d] == 0:
    p[d] ВСЕГДА равно 0.
    Этот день не адаптируется.

Никакого ограничения "с дня 430" нет.
Никакого определения активности p по росту I_obs нет.

ДИ (доверительные интервалы):
    Основной ДИ = mean ± 3σ  (по ансамблю цепей × сэмплов).
    Дополнительно сохраняются ±1σ и ±2σ.

ПАРАМЕТР ПЕРЕДАЧИ:
    beta(t) = R(t) * gamma
    gamma — единый параметр (и скорость выздоровления,
    и скорость передачи). alpha больше не используется.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


R_init = None
p_init = None
I_obs = None

days = 0

p_active_mask = None

mcmc_samples_R = None
mcmc_samples_p = None

mcmc_accept_R = None
mcmc_accept_p = None

mcmc_stats = None

diag_R = None
diag_p = None

ode_results = None

model_params = None


def get_float(value):
    return float(str(value).replace(",", ".").strip())


def r2_score_manual(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot <= 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def read_excel_smart(path):

    xls = pd.ExcelFile(path)

    best_df = None

    for sheet in xls.sheet_names:

        df = pd.read_excel(
            xls,
            sheet_name=sheet
        )

        cols = [
            str(c).lower()
            for c in df.columns
        ]

        has_r = any(
            (
                "r(" in c
                or c == "r"
                or "r_t" in c
                or "r0" in c
                or c == "rt"
            )
            for c in cols
        )

        if has_r and df.shape[1] >= 2:
            return df

        if best_df is None and df.shape[1] >= 2:
            best_df = df

    return best_df


def read_model_params_excel(path):

    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None

    target = None

    for sheet in xls.sheet_names:
        if "парам" in sheet.lower() or "model" in sheet.lower():
            target = sheet
            break

    if target is None and len(xls.sheet_names) >= 2:
        target = xls.sheet_names[1]

    if target is None:
        return None

    df = pd.read_excel(xls, sheet_name=target)

    if df.shape[1] < 2:
        return None

    keys = df.iloc[:, 0].astype(str).str.strip()
    vals = df.iloc[:, 1]

    params = {}

    for key, val in zip(keys, vals):
        try:
            params[key] = float(val)
        except (TypeError, ValueError):
            params[key] = val

    return params


def read_csv_smart(path):

    for sep in [",", ";", "\t", "|"]:

        try:

            df = pd.read_csv(
                path,
                sep=sep
            )

            if df.shape[1] >= 2:
                return df

        except Exception:
            pass

    raise ValueError(
        "Не удалось прочитать CSV."
    )


def extract_columns(df):

    cols_lower = [
        str(c).lower()
        for c in df.columns
    ]

    r_col = None
    p_col = None
    i_col = None

    for i, c in enumerate(cols_lower):

        if r_col is None:

            if (
                "r(" in c
                or c == "r"
                or "r_t" in c
                or "r0" in c
                or c == "rt"
            ):
                r_col = df.columns[i]

        if p_col is None:

            if (
                "p(" in c
                or c == "p"
                or "p_t" in c
                or "psi" in c
                or c == "pt"
            ):
                p_col = df.columns[i]

        if i_col is None:

            if (
                "i(" in c
                or c == "i"
                or "cases" in c
                or "new" in c
                or "obs" in c
                or "infected" in c
                or "заболев" in c
                or "случа" in c
            ):
                i_col = df.columns[i]

    if r_col is None:

        if df.shape[1] >= 2:
            r_col = df.columns[1]
        else:
            r_col = df.columns[0]

    if p_col is None and df.shape[1] >= 3:
        p_col = df.columns[2]

    R = (
        pd.to_numeric(
            df[r_col],
            errors="coerce"
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    if p_col is not None:

        p = (
            pd.to_numeric(
                df[p_col],
                errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    else:

        p = np.zeros_like(R)

    I = None

    if (
        i_col is not None
        and i_col != r_col
        and i_col != p_col
    ):

        candidate = (
            pd.to_numeric(
                df[i_col],
                errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        if not np.allclose(
            candidate,
            np.arange(
                1,
                len(candidate) + 1
            )
        ):
            I = candidate

    return R, p, I


def load_initial():

    global R_init
    global p_init
    global I_obs
    global days
    global p_active_mask

    global mcmc_samples_R
    global mcmc_samples_p
    global mcmc_stats
    global diag_R
    global diag_p
    global ode_results
    global model_params

    path = filedialog.askopenfilename(
        title=(
            "Выберите файл с R(t), p(t) "
            "и при наличии I_obs"
        ),
        filetypes=[
            (
                "Excel files",
                "*.xlsx *.xls"
            ),
            (
                "CSV files",
                "*.csv"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )

    if not path:
        return

    try:

        ext = os.path.splitext(
            path
        )[1].lower()

        if ext in (
            ".xlsx",
            ".xls"
        ):

            df = read_excel_smart(path)

        else:

            df = read_csv_smart(path)

        if df is None or df.shape[1] < 2:

            raise ValueError(
                "Не удалось распознать файл."
            )

        R_init, p_init, I_obs = (
            extract_columns(df)
        )

        days = len(R_init)

        if len(p_init) != days:

            raise ValueError(
                "R(t) и p(t) имеют разную длину."
            )

        R_init = np.nan_to_num(
            R_init,
            nan=1.0,
            posinf=20.0,
            neginf=0.05
        )

        R_init = np.clip(
            R_init,
            0.05,
            20.0
        )

        p_init = np.nan_to_num(
            p_init,
            nan=0.0,
            posinf=0.5,
            neginf=0.0
        )

        p_init = np.maximum(
            p_init,
            0.0
        )

        p_init = np.clip(
            p_init,
            0.0,
            0.5
        )

        p_active_mask = (
            p_init > 0.0
        )

        p_init[
            ~p_active_mask
        ] = 0.0

        mcmc_samples_R = None
        mcmc_samples_p = None
        mcmc_stats = None
        diag_R = None
        diag_p = None
        ode_results = None

        model_params = None

        if ext in (".xlsx", ".xls"):
            model_params = read_model_params_excel(path)

        if model_params:

            def _set(entry, key):
                if key in model_params:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(model_params[key]))

            _set(entry_S0, "S0")
            _set(entry_I0, "I0")
            _set(entry_gamma, "gamma")
            _set(entry_mu, "mu")
            _set(entry_k, "k")
            _set(entry_N, "EPOP")

        active_days = (
            np.where(
                p_active_mask
            )[0] + 1
        )

        log_text.insert(
            tk.END,
            "\n"
            + "=" * 70
            + "\n"
        )

        log_text.insert(
            tk.END,
            "ЗАГРУЗКА ДАННЫХ\n"
        )

        log_text.insert(
            tk.END,
            "=" * 70
            + "\n"
        )

        log_text.insert(
            tk.END,
            f"Файл: "
            f"{os.path.basename(path)}\n"
        )

        log_text.insert(
            tk.END,
            f"Дней: {days}\n"
        )

        log_text.insert(
            tk.END,
            f"R_init: "
            f"min={R_init.min():.6f}, "
            f"max={R_init.max():.6f}, "
            f"mean={R_init.mean():.6f}\n"
        )

        log_text.insert(
            tk.END,
            f"p_init: "
            f"min={p_init.min():.6f}, "
            f"max={p_init.max():.6f}, "
            f"mean={p_init.mean():.6f}\n"
        )

        log_text.insert(
            tk.END,
            f"\nАктивных дней p: "
            f"{len(active_days)} "
            f"из {days}\n"
        )

        if len(active_days) > 0:

            log_text.insert(
                tk.END,
                "Активные дни p:\n"
            )

            log_text.insert(
                tk.END,
                f"первый = "
                f"{active_days[0]}\n"
            )

            log_text.insert(
                tk.END,
                f"последний = "
                f"{active_days[-1]}\n"
            )

        else:

            log_text.insert(
                tk.END,
                "Активных дней p нет.\n"
            )

        log_text.insert(
            tk.END,
            "\n"
            "Правило:\n"
            "p_init > 0 -> p адаптируется\n"
            "p_init = 0 -> p фиксировано 0\n"
        )

        log_text.insert(
            tk.END,
            "\n"
            "Начальное состояние p каждой "
            "цепи = загруженный p_init.\n"
        )

        if I_obs is not None:

            log_text.insert(
                tk.END,
                f"I_obs: "
                f"{len(I_obs)} точек\n"
            )

        else:

            log_text.insert(
                tk.END,
                "I_obs: нет\n"
            )

        if model_params:

            log_text.insert(
                tk.END,
                "\nПараметры модели загружены со 2-го листа:\n"
            )

            for k_ in ("S0", "I0", "gamma", "mu", "k", "EPOP"):
                if k_ in model_params:
                    log_text.insert(
                        tk.END,
                        f"  {k_} = {model_params[k_]}\n"
                    )

        log_text.see(tk.END)

        plot_initial()

    except Exception as e:

        import traceback

        messagebox.showerror(
            "Ошибка загрузки",
            f"{e}\n\n"
            f"{traceback.format_exc()}"
        )


def plot_initial():

    if R_init is None:
        return

    t = np.arange(
        1,
        days + 1
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        t,
        R_init,
        lw=1.6,
        label="R_init(t)"
    )

    ax.axhline(
        1.0,
        ls="--",
        lw=1,
        label="R=1"
    )

    ax.set_xlabel("День")
    ax.set_ylabel("R(t)")

    ax.set_title(
        "Начальное R(t)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        t,
        p_init,
        lw=1.8,
        label="p_init(t)"
    )

    ax.axhline(
        0,
        ls="--",
        lw=1
    )

    if np.any(p_active_mask):

        ax.scatter(
            t[p_active_mask],
            p_init[p_active_mask],
            s=12,
            label="p_init > 0"
        )

    ax.set_xlabel("День")
    ax.set_ylabel("p(t)")

    ax.set_title(
        "Начальное p(t): "
        "адаптируются только ненулевые значения"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()
    plt.show()


def log_posterior(
    R_arr,
    p_arr,
    R_prior,
    p_prior,
    sigma_R,
    sigma_p,
    smooth_w,
    p_mask
):

    if np.any(R_arr < 0.05):
        return -np.inf

    if np.any(R_arr > 20.0):
        return -np.inf

    if np.any(p_arr < 0.0):
        return -np.inf

    if np.any(p_arr > 0.5):
        return -np.inf

    ll_R = -0.5 * np.sum(
        (
            (R_arr - R_prior)
            / sigma_R
        ) ** 2
    )

    if (
        p_mask is not None
        and np.any(p_mask)
    ):

        diff_p = (
            p_arr[p_mask]
            - p_prior[p_mask]
        )

        ll_p = -0.5 * np.sum(
            (
                diff_p
                / sigma_p
            ) ** 2
        )

    else:

        ll_p = 0.0

    smooth_R = (
        -smooth_w
        * np.sum(
            np.diff(R_arr) ** 2
        )
    )

    smooth_p = 0.0

    if (
        p_mask is not None
        and np.any(p_mask)
    ):

        idx = np.flatnonzero(
            p_mask
        )

        if len(idx) >= 2:

            consecutive = (
                np.diff(idx) == 1
            )

            if np.any(consecutive):

                a = idx[:-1][consecutive]
                b = idx[1:][consecutive]

                dp = (
                    p_arr[b]
                    - p_arr[a]
                )

                smooth_p = (
                    -smooth_w
                    * 10.0
                    * np.sum(dp ** 2)
                )

    return (
        ll_R
        + ll_p
        + smooth_R
        + smooth_p
    )


def mh_step(
    R_cur,
    p_cur,
    ll_cur,
    R_prior,
    p_prior,
    sigma_R,
    sigma_p,
    smooth_w,
    step_R,
    step_p,
    p_mask
):

    accepted_R = False
    accepted_p = False

    R_prop = np.clip(
        R_cur
        + np.random.normal(
            0.0,
            step_R,
            len(R_cur)
        ),
        0.05,
        20.0
    )

    ll_prop = log_posterior(
        R_prop,
        p_cur,
        R_prior,
        p_prior,
        sigma_R,
        sigma_p,
        smooth_w,
        p_mask
    )

    log_alpha = (ll_prop - ll_cur)

    if (
        np.log(np.random.rand())
        < log_alpha
    ):

        R_cur = R_prop
        ll_cur = ll_prop

        accepted_R = True

    if (
        p_mask is not None
        and np.any(p_mask)
    ):

        p_prop = p_cur.copy()

        n_active = int(np.sum(p_mask))

        noise = np.random.normal(
            0.0,
            step_p,
            n_active
        )

        p_prop[p_mask] = np.clip(
            p_cur[p_mask]
            + noise,
            0.0,
            0.5
        )

        p_prop[~p_mask] = 0.0

        ll_prop = log_posterior(
            R_cur,
            p_prop,
            R_prior,
            p_prior,
            sigma_R,
            sigma_p,
            smooth_w,
            p_mask
        )

        log_alpha = (ll_prop - ll_cur)

        if (
            np.log(np.random.rand())
            < log_alpha
        ):

            p_cur = p_prop
            ll_cur = ll_prop

            accepted_p = True

    return (
        R_cur,
        p_cur,
        ll_cur,
        accepted_R,
        accepted_p
    )


def run_mcmc():

    global mcmc_samples_R
    global mcmc_samples_p
    global mcmc_accept_R
    global mcmc_accept_p

    if R_init is None:

        messagebox.showwarning(
            "Нет данных",
            "Сначала загрузите Excel/CSV."
        )

        return

    try:

        n_chains = int(entry_n_chains.get())
        burnin = int(entry_burnin.get())
        n_samples = int(entry_n_samples.get())

        step_R = get_float(entry_step_R.get())
        step_p = get_float(entry_step_p.get())

        sigma_R = get_float(entry_sigma_R.get())
        sigma_p = get_float(entry_sigma_p.get())

        smooth_w = get_float(entry_smooth.get())

        if n_chains < 2:
            raise ValueError("Число цепей должно быть >= 2.")
        if burnin < 0:
            raise ValueError("Burn-in >= 0.")
        if n_samples < 1:
            raise ValueError("Число сэмплов >= 1.")
        if step_R <= 0:
            raise ValueError("Шаг R должен быть > 0.")
        if step_p <= 0:
            raise ValueError("Шаг p должен быть > 0.")
        if sigma_R <= 0:
            raise ValueError("sigma_R должен быть > 0.")
        if sigma_p <= 0:
            raise ValueError("sigma_p должен быть > 0.")
        if smooth_w < 0:
            raise ValueError("smooth_w >= 0.")

    except Exception as e:

        messagebox.showerror("Ошибка параметров", str(e))
        return

    n_active = int(np.sum(p_active_mask))

    log_text.insert(
        tk.END,
        "\n" + "=" * 70 + "\n"
        "ЗАПУСК MCMC\n"
        + "=" * 70 + "\n"
    )

    log_text.insert(
        tk.END,
        f"Цепей: {n_chains}\n"
        f"Burn-in: {burnin}\n"
        f"Сэмплов: {n_samples}\n"
        f"Шаг R: {step_R}\n"
        f"Шаг p: {step_p}\n"
        f"sigma_R: {sigma_R}\n"
        f"sigma_p: {sigma_p}\n"
        f"smooth: {smooth_w}\n"
        f"Активных дней p: {n_active}\n"
    )

    log_text.insert(
        tk.END,
        "\n"
        "p стартует каждой цепью "
        "с исходного загруженного p_init.\n"
    )

    log_text.insert(
        tk.END,
        "p_init=0 -> p всегда 0.\n"
    )

    log_text.see(tk.END)
    root.update()

    R_prior = R_init.copy()
    p_prior = p_init.copy()

    samples_R = np.zeros((n_chains, n_samples, days))
    samples_p = np.zeros((n_chains, n_samples, days))

    accept_R = np.zeros(n_chains)
    accept_p = np.zeros(n_chains)

    for chain in range(n_chains):

        R_cur = np.clip(
            R_init
            + np.random.normal(0.0, 0.1, days),
            0.05,
            20.0
        )

        p_cur = p_init.copy()
        p_cur[~p_active_mask] = 0.0

        ll_cur = log_posterior(
            R_cur,
            p_cur,
            R_prior,
            p_prior,
            sigma_R,
            sigma_p,
            smooth_w,
            p_active_mask
        )

        for _ in range(burnin):

            (R_cur, p_cur, ll_cur, _, _) = mh_step(
                R_cur, p_cur, ll_cur,
                R_prior, p_prior,
                sigma_R, sigma_p, smooth_w,
                step_R, step_p,
                p_active_mask
            )

        acc_r_count = 0
        acc_p_count = 0

        for s in range(n_samples):

            (R_cur, p_cur, ll_cur, acc_r, acc_p) = mh_step(
                R_cur, p_cur, ll_cur,
                R_prior, p_prior,
                sigma_R, sigma_p, smooth_w,
                step_R, step_p,
                p_active_mask
            )

            samples_R[chain, s, :] = R_cur
            samples_p[chain, s, :] = p_cur

            acc_r_count += int(acc_r)
            acc_p_count += int(acc_p)

        accept_R[chain] = acc_r_count / n_samples
        accept_p[chain] = acc_p_count / n_samples

        log_text.insert(
            tk.END,
            f"Цепь {chain + 1}: "
            f"accept R = {accept_R[chain] * 100:.1f}%   "
            f"accept p = {accept_p[chain] * 100:.1f}%\n"
        )

        log_text.see(tk.END)
        root.update()

    samples_p[:, :, ~p_active_mask] = 0.0

    mcmc_samples_R = samples_R
    mcmc_samples_p = samples_p

    mcmc_accept_R = accept_R
    mcmc_accept_p = accept_p

    log_text.insert(tk.END, "\nMCMC завершён.\n")
    log_text.insert(
        tk.END,
        f"Средний accept R: {accept_R.mean() * 100:.1f}%\n"
    )
    log_text.insert(
        tk.END,
        f"Средний accept p: {accept_p.mean() * 100:.1f}%\n"
    )

    log_text.insert(
        tk.END,
        "\nДИ считаются как  mean ± k·σ "
        "(по ансамблю цепей × сэмплов).\n"
        "Основной ДИ = mean ± 3σ.\n"
    )

    log_text.see(tk.END)

    compute_stats()
    run_diagnostics()

    plot_mcmc_results()
    plot_diagnostics()


def compute_stats():
    """
    Считает среднее и σ по ансамблю (все цепи × все сэмплы) для каждого дня.

    ДИ считаются как  mean ± k·σ  (k = 1, 2, 3).

    Основной ДИ (R_lo / R_hi / p_lo / p_hi) = mean ± 3σ.
    Дополнительно сохраняются ±1σ и ±2σ для графиков/анализа.
    """

    global mcmc_stats

    flat_R = mcmc_samples_R.reshape(-1, days)
    flat_p = mcmc_samples_p.reshape(-1, days)

    R_mean = np.mean(flat_R, axis=0)
    R_std  = np.std(flat_R, axis=0, ddof=1)

    p_mean = np.mean(flat_p, axis=0)
    p_std  = np.std(flat_p, axis=0, ddof=1)

    R_lo1 = R_mean - 1.0 * R_std
    R_hi1 = R_mean + 1.0 * R_std

    R_lo2 = R_mean - 2.0 * R_std
    R_hi2 = R_mean + 2.0 * R_std

    R_lo3 = R_mean - 3.0 * R_std
    R_hi3 = R_mean + 3.0 * R_std

    p_lo1 = p_mean - 1.0 * p_std
    p_hi1 = p_mean + 1.0 * p_std

    p_lo2 = p_mean - 2.0 * p_std
    p_hi2 = p_mean + 2.0 * p_std

    p_lo3 = p_mean - 3.0 * p_std
    p_hi3 = p_mean + 3.0 * p_std

    R_lo1 = np.clip(R_lo1, 0.05, 20.0)
    R_hi1 = np.clip(R_hi1, 0.05, 20.0)
    R_lo2 = np.clip(R_lo2, 0.05, 20.0)
    R_hi2 = np.clip(R_hi2, 0.05, 20.0)
    R_lo3 = np.clip(R_lo3, 0.05, 20.0)
    R_hi3 = np.clip(R_hi3, 0.05, 20.0)

    p_lo1 = np.clip(p_lo1, 0.0, 0.5)
    p_hi1 = np.clip(p_hi1, 0.0, 0.5)
    p_lo2 = np.clip(p_lo2, 0.0, 0.5)
    p_hi2 = np.clip(p_hi2, 0.0, 0.5)
    p_lo3 = np.clip(p_lo3, 0.0, 0.5)
    p_hi3 = np.clip(p_hi3, 0.0, 0.5)

    mcmc_stats = {

        "R_mean":   R_mean,
        "R_median": np.median(flat_R, axis=0),
        "R_std":    R_std,

        "R_lo":     R_lo3,
        "R_hi":     R_hi3,

        "R_lo1":    R_lo1, "R_hi1": R_hi1,
        "R_lo2":    R_lo2, "R_hi2": R_hi2,
        "R_lo3":    R_lo3, "R_hi3": R_hi3,

        "p_mean":   p_mean,
        "p_median": np.median(flat_p, axis=0),
        "p_std":    p_std,

        "p_lo":     p_lo3,
        "p_hi":     p_hi3,

        "p_lo1":    p_lo1, "p_hi1": p_hi1,
        "p_lo2":    p_lo2, "p_hi2": p_hi2,
        "p_lo3":    p_lo3, "p_hi3": p_hi3,
    }

    for key in [
        "p_mean", "p_median", "p_std",
        "p_lo", "p_hi",
        "p_lo1", "p_hi1",
        "p_lo2", "p_hi2",
        "p_lo3", "p_hi3",
    ]:
        mcmc_stats[key] = mcmc_stats[key].copy()
        mcmc_stats[key][~p_active_mask] = 0.0


def gelman_rubin(chains):

    m, n = chains.shape

    if m < 2 or n < 2:
        return np.nan

    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)

    W = chain_vars.mean()

    if W <= 0:
        return np.nan

    B = n * chain_means.var(ddof=1)

    var_hat = (n - 1) / n * W + B / n

    return np.sqrt(var_hat / W)


def global_rhat(samples):

    n_chains, n_samples, days_used = samples.shape

    if n_chains < 2 or n_samples < 2 or days_used < 1:
        return np.nan

    chains_flat = samples.reshape(
        n_chains,
        n_samples * days_used
    )

    m, n = chains_flat.shape

    if m < 2 or n < 2:
        return np.nan

    chain_means = chains_flat.mean(axis=1)
    chain_vars = chains_flat.var(axis=1, ddof=1)

    W = chain_vars.mean()

    if W <= 0:
        return np.nan

    B = n * chain_means.var(ddof=1)

    var_hat = (n - 1) / n * W + B / n

    return float(np.sqrt(var_hat / W))


def effective_sample_size(chains):

    m, n = chains.shape

    if n < 10:
        return np.nan

    x = chains.reshape(-1)
    x = x - x.mean()

    N = len(x)

    if N < 10:
        return np.nan

    var_x = np.sum(x * x)

    if var_x <= 0:
        return np.nan

    max_lag = min(N // 2, 1000)

    acf = np.zeros(max_lag)

    for lag in range(1, max_lag):

        acf[lag] = (
            np.sum(x[:-lag] * x[lag:])
            / var_x
        )

    s = 0.0

    for lag in range(0, max_lag - 1, 2):

        pair = acf[lag] + acf[lag + 1]

        if pair < 0:
            break

        s += pair

    ess = N / (1.0 + 2.0 * s)

    return max(ess, 1.0)


def run_diagnostics():

    global diag_R
    global diag_p

    if mcmc_samples_R is None:
        return

    rhat_R = np.full(days, np.nan)
    rhat_p = np.full(days, np.nan)

    ess_R = np.full(days, np.nan)
    ess_p = np.full(days, np.nan)

    for d in range(days):

        chains_R = mcmc_samples_R[:, :, d]
        rhat_R[d] = gelman_rubin(chains_R)
        ess_R[d] = effective_sample_size(chains_R)

        if p_active_mask[d]:

            chains_p = mcmc_samples_p[:, :, d]
            rhat_p[d] = gelman_rubin(chains_p)
            ess_p[d] = effective_sample_size(chains_p)

    diag_R = {"rhat": rhat_R, "ess": ess_R}
    diag_p = {"rhat": rhat_p, "ess": ess_p}

    log_text.insert(
        tk.END,
        "\n" + "=" * 70 + "\n"
        "ДИАГНОСТИКА\n"
        + "=" * 70 + "\n"
    )

    r = rhat_R[np.isfinite(rhat_R)]
    e = ess_R[np.isfinite(ess_R)]

    if len(r):

        log_text.insert(
            tk.END,
            f"R(t): R-hat "
            f"min={r.min():.4f}, "
            f"mean={r.mean():.4f}, "
            f"max={r.max():.4f}\n"
        )

        log_text.insert(
            tk.END,
            f"R(t): ESS "
            f"min={e.min():.1f}, "
            f"median={np.median(e):.1f}, "
            f"max={e.max():.1f}\n"
        )

    r = rhat_p[np.isfinite(rhat_p)]
    e = ess_p[np.isfinite(ess_p)]

    if len(r):

        log_text.insert(
            tk.END,
            f"p(t): R-hat "
            f"min={r.min():.4f}, "
            f"mean={r.mean():.4f}, "
            f"max={r.max():.4f}\n"
        )

        log_text.insert(
            tk.END,
            f"p(t): ESS "
            f"min={e.min():.1f}, "
            f"median={np.median(e):.1f}, "
            f"max={e.max():.1f}\n"
        )

    else:

        log_text.insert(
            tk.END,
            "p(t): нет активных дней.\n"
        )

    rhat_R_all = rhat_R[np.isfinite(rhat_R)]
    rhat_p_all = rhat_p[np.isfinite(rhat_p)]

    log_text.insert(
        tk.END,
        "\n" + "-" * 70 + "\n"
        "R-HAT ЗА ВЕСЬ ПЕРИОД НАБЛЮДЕНИЯ (по дням)\n"
        + "-" * 70 + "\n"
    )

    if len(rhat_R_all):

        log_text.insert(
            tk.END,
            f"R(t): overall R-hat "
            f"(max по всем дням) = "
            f"{rhat_R_all.max():.4f}\n"
        )
        log_text.insert(
            tk.END,
            f"R(t): overall R-hat "
            f"(mean по всем дням) = "
            f"{rhat_R_all.mean():.4f}\n"
        )
        log_text.insert(
            tk.END,
            f"R(t): overall R-hat "
            f"(min по всем дням) = "
            f"{rhat_R_all.min():.4f}\n"
        )

    if len(rhat_p_all):

        log_text.insert(
            tk.END,
            f"p(t): overall R-hat "
            f"(max по активным дням) = "
            f"{rhat_p_all.max():.4f}\n"
        )
        log_text.insert(
            tk.END,
            f"p(t): overall R-hat "
            f"(mean по активным дням) = "
            f"{rhat_p_all.mean():.4f}\n"
        )
        log_text.insert(
            tk.END,
            f"p(t): overall R-hat "
            f"(min по активным дням) = "
            f"{rhat_p_all.min():.4f}\n"
        )

    else:

        log_text.insert(
            tk.END,
            "p(t): нет активных дней "
            "для расчёта R-hat.\n"
        )

    rhat_R_global = global_rhat(mcmc_samples_R)

    if np.any(p_active_mask):

        samples_p_active = mcmc_samples_p[
            :,
            :,
            p_active_mask
        ]
        rhat_p_global = global_rhat(samples_p_active)

    else:

        rhat_p_global = np.nan

    log_text.insert(
        tk.END,
        "\n" + "=" * 70 + "\n"
        "ГЛОБАЛЬНЫЙ R-HAT "
        "(объединённая выборка: все цепи × все дни)\n"
        + "=" * 70 + "\n"
    )

    if np.isfinite(rhat_R_global):

        log_text.insert(
            tk.END,
            f"GLOBAL R-hat R(t) = "
            f"{rhat_R_global:.4f}\n"
        )

    else:

        log_text.insert(
            tk.END,
            "GLOBAL R-hat R(t): недостаточно данных.\n"
        )

    if np.isfinite(rhat_p_global):

        log_text.insert(
            tk.END,
            f"GLOBAL R-hat p(t) = "
            f"{rhat_p_global:.4f}\n"
        )

    else:

        log_text.insert(
            tk.END,
            "GLOBAL R-hat p(t): нет активных дней "
            "или недостаточно данных.\n"
        )

    global_candidates = [
        v for v in [rhat_R_global, rhat_p_global]
        if np.isfinite(v)
    ]

    if global_candidates:

        global_overall = max(global_candidates)

        log_text.insert(
            tk.END,
            "\n"
            f"ОБЩИЙ GLOBAL R-hat (max по R и p) = "
            f"{global_overall:.4f}\n"
        )

        if global_overall < 1.01:

            verdict = "ОТЛИЧНАЯ сходимость (< 1.01)"

        elif global_overall < 1.05:

            verdict = "ХОРОШАЯ сходимость (< 1.05)"

        elif global_overall < 1.10:

            verdict = (
                "ПРИЕМЛЕМАЯ, но желательно "
                "увеличить сэмплы (< 1.10)"
            )

        else:

            verdict = (
                "ПЛОХАЯ сходимость (>= 1.10) — "
                "увеличьте burn-in и число сэмплов"
            )

        log_text.insert(
            tk.END,
            f"Вердикт GLOBAL: {verdict}\n"
        )

    if len(rhat_p_all):

        combined = np.concatenate(
            [rhat_R_all, rhat_p_all]
        )

    else:

        combined = rhat_R_all

    if len(combined):

        overall = combined.max()

        log_text.insert(
            tk.END,
            "\n"
            f"ОБЩИЙ R-hat по дням "
            f"(max по R и p, весь период) = "
            f"{overall:.4f}\n"
        )

        if overall < 1.01:

            verdict = "ОТЛИЧНАЯ сходимость (< 1.01)"

        elif overall < 1.05:

            verdict = "ХОРОШАЯ сходимость (< 1.05)"

        elif overall < 1.10:

            verdict = (
                "ПРИЕМЛЕМАЯ, но желательно "
                "увеличить сэмплы (< 1.10)"
            )

        else:

            verdict = (
                "ПЛОХАЯ сходимость (>= 1.10) — "
                "увеличьте burn-in и число сэмплов"
            )

        log_text.insert(
            tk.END,
            f"Вердикт по дням: {verdict}\n"
        )

    log_text.see(tk.END)


def plot_diagnostics():

    if diag_R is None:
        return

    t = np.arange(1, days + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    axs[0, 0].plot(
        t, diag_R["rhat"],
        lw=1.2, label="R-hat R(t)"
    )
    axs[0, 0].axhline(1.01, ls="--", label="1.01")
    axs[0, 0].axhline(1.10, ls="--", label="1.10")
    axs[0, 0].set_title("R-hat R(t)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    mask = np.isfinite(diag_p["rhat"])

    if np.any(mask):

        axs[0, 1].plot(
            t[mask],
            diag_p["rhat"][mask],
            lw=1.2,
            label="R-hat p(t)"
        )

    axs[0, 1].axhline(1.01, ls="--", label="1.01")
    axs[0, 1].axhline(1.10, ls="--", label="1.10")
    axs[0, 1].set_title("R-hat p(t), p_init > 0")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].plot(
        t, diag_R["ess"],
        lw=1.2, label="ESS R(t)"
    )
    axs[1, 0].axhline(400, ls="--", label="400")
    axs[1, 0].axhline(100, ls="--", label="100")
    axs[1, 0].set_title("ESS R(t)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    mask = np.isfinite(diag_p["ess"])

    if np.any(mask):

        axs[1, 1].plot(
            t[mask],
            diag_p["ess"][mask],
            lw=1.2,
            label="ESS p(t)"
        )

    axs[1, 1].axhline(400, ls="--", label="400")
    axs[1, 1].axhline(100, ls="--", label="100")
    axs[1, 1].set_title("ESS p(t), p_init > 0")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_xlabel("День")

    plt.suptitle("Диагностика MCMC")
    plt.tight_layout()
    plt.show()


def plot_mcmc_results():

    if mcmc_stats is None:
        return

    st = mcmc_stats
    t = np.arange(1, days + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    axs[0, 0].plot(t, R_init, "--", lw=1.2, label="R_init")
    axs[0, 0].plot(t, st["R_mean"], lw=2, label="R_mean MCMC")

    axs[0, 0].fill_between(
        t, st["R_lo3"], st["R_hi3"],
        alpha=0.15, label="±3σ"
    )
    axs[0, 0].fill_between(
        t, st["R_lo2"], st["R_hi2"],
        alpha=0.25, label="±2σ"
    )
    axs[0, 0].fill_between(
        t, st["R_lo1"], st["R_hi1"],
        alpha=0.35, label="±1σ"
    )

    axs[0, 0].axhline(1, ls=":", label="R=1")
    axs[0, 0].set_title("R(t): начальное vs MCMC (ДИ = mean ± kσ)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(t, st["R_std"], lw=1.5)
    axs[0, 1].set_title("std R(t)")
    axs[0, 1].grid(True, alpha=0.3)

    for c in range(mcmc_samples_R.shape[0]):

        axs[1, 0].plot(
            mcmc_samples_R[c, :, 0],
            lw=0.7, alpha=0.5,
            label=f"Цепь {c + 1}"
        )

    axs[1, 0].set_title("Trace R(1)")
    axs[1, 0].set_xlabel("Сэмпл")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=7)

    flat_R = mcmc_samples_R.reshape(-1, days)

    axs[1, 1].hist(
        flat_R[:, 0],
        bins=40, density=True, alpha=0.7
    )

    axs[1, 1].set_title("Распределение R(1)")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    axs[0, 0].plot(t, p_init, "--", lw=1.2, label="p_init")
    axs[0, 0].plot(t, st["p_mean"], lw=2, label="p_mean MCMC")

    axs[0, 0].fill_between(
        t, st["p_lo3"], st["p_hi3"],
        alpha=0.15, label="±3σ"
    )
    axs[0, 0].fill_between(
        t, st["p_lo2"], st["p_hi2"],
        alpha=0.25, label="±2σ"
    )
    axs[0, 0].fill_between(
        t, st["p_lo1"], st["p_hi1"],
        alpha=0.35, label="±1σ"
    )

    axs[0, 0].set_title("p(t): начальное vs MCMC (ДИ = mean ± kσ)")
    axs[0, 0].set_xlabel("День")
    axs[0, 0].set_ylabel("p(t)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].plot(t, st["p_std"], lw=1.5)
    axs[0, 1].set_title("std p(t)")
    axs[0, 1].grid(True, alpha=0.3)

    active_idx = np.flatnonzero(p_active_mask)

    if len(active_idx):

        idx = active_idx[0]

        for c in range(mcmc_samples_p.shape[0]):

            axs[1, 0].plot(
                mcmc_samples_p[c, :, idx],
                lw=0.7, alpha=0.5,
                label=f"Цепь {c + 1}"
            )

        axs[1, 0].set_title(f"Trace p({idx + 1})")
        axs[1, 0].set_xlabel("Сэмпл")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend(fontsize=7)

        flat_p = mcmc_samples_p.reshape(-1, days)

        active_values = (
            flat_p[:, p_active_mask].ravel()
        )

        axs[1, 1].hist(
            active_values,
            bins=40, density=True, alpha=0.7
        )

        axs[1, 1].set_title(
            "Распределение p на активных днях"
        )
        axs[1, 1].set_xlabel("p")
        axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def sir_rhs(S, I, R_t, p_t, gamma, mu, k):
    """
    Правые части SIR-модели.

    beta(t) = R(t) * gamma
    """

    beta = R_t * gamma

    dS = (
        -beta * S * I
        + mu * (1.0 - S)
        + (k + p_t) * (1.0 - S - I)
    )

    dI = (
        beta * S * I
        - gamma * I
        - mu * I
    )

    return dS, dI


def solve_ode_from_mcmc_means(
    gamma, mu, k, N, S0=None, I0=None
):
    """
    Решает ОДУ по средним R_mean(t) и p_mean(t) из MCMC.

    Параметр передачи beta(t) = R(t) * gamma.
    """

    if mcmc_stats is None:
        raise ValueError("Сначала запустите MCMC.")

    R_t = np.asarray(mcmc_stats["R_mean"], dtype=float)
    p_t = np.asarray(mcmc_stats["p_mean"], dtype=float)

    n = len(R_t)

    if I_obs is not None:
        obs = np.asarray(I_obs, dtype=float)[:n]
    else:
        obs = None

    if obs is not None and len(obs) > 0:
        obs_is_fraction = (
            np.nanmax(np.abs(obs)) <= 1.0 + 1e-12
        )
    else:
        obs_is_fraction = False

    if I0 is None:

        if obs is not None and len(obs) > 0:

            if obs_is_fraction:
                I0 = max(float(obs[0]), 1e-12)
            else:
                if N <= 0:
                    raise ValueError("N должно быть > 0.")
                I0 = max(float(obs[0]) / N, 1e-12)

        else:
            I0 = 1e-6

    if S0 is None:
        S0 = max(1.0 - I0, 0.0)

    if S0 < 0 or I0 < 0:
        raise ValueError("S0 и I0 >= 0.")

    if S0 + I0 > 1.0 + 1e-10:
        raise ValueError("S0 + I0 > 1.")

    S = np.zeros(n)
    I = np.zeros(n)

    transmission_frac = np.zeros(n)

    S[0] = S0
    I[0] = I0

    for j in range(n):

        transmission_frac[j] = max(
            0.0,
            R_t[j] * gamma * S[j] * I[j]
        )

        if j == n - 1:
            break

        R0 = R_t[j]
        R1 = R_t[j + 1]
        p0 = p_t[j]
        p1 = p_t[j + 1]

        def rhs_at(Sx, Ix, q):

            Rq = R0 + q * (R1 - R0)
            pq = p0 + q * (p1 - p0)

            return sir_rhs(
                Sx, Ix, Rq, pq,
                gamma, mu, k
            )

        h = 1.0

        k1S, k1I = rhs_at(S[j], I[j], 0.0)

        k2S, k2I = rhs_at(
            S[j] + 0.5 * h * k1S,
            I[j] + 0.5 * h * k1I,
            0.5
        )

        k3S, k3I = rhs_at(
            S[j] + 0.5 * h * k2S,
            I[j] + 0.5 * h * k2I,
            0.5
        )

        k4S, k4I = rhs_at(
            S[j] + h * k3S,
            I[j] + h * k3I,
            1.0
        )

        S[j + 1] = S[j] + h * (
            k1S + 2*k2S + 2*k3S + k4S
        ) / 6.0

        I[j + 1] = I[j] + h * (
            k1I + 2*k2I + 2*k3I + k4I
        ) / 6.0

        S[j + 1] = max(S[j + 1], 0.0)
        I[j + 1] = max(I[j + 1], 0.0)

        total = S[j + 1] + I[j + 1]

        if total > 1.0 + 1e-8:
            S[j + 1] /= total
            I[j + 1] /= total

    transmission_frac = np.maximum(
        0.0,
        R_t * gamma * S * I
    )

    if obs is not None:

        if obs_is_fraction:

            transmission_model = transmission_frac
            I_model = I

        else:

            if N <= 0:
                raise ValueError("N должно быть > 0.")

            transmission_model = transmission_frac * N
            I_model = I * N

        r2_transmission = r2_score_manual(
            obs, transmission_model
        )

        r2_I = r2_score_manual(obs, I_model)

    else:

        transmission_model = (
            transmission_frac * N
            if N > 0 else transmission_frac
        )

        I_model = I * N if N > 0 else I

        r2_transmission = np.nan
        r2_I = np.nan

    return {

        "S": S,
        "I": I,

        "transmission_frac": transmission_frac,
        "transmission_model": transmission_model,
        "I_model": I_model,

        "R": R_t,
        "p": p_t,

        "r2_transmission": r2_transmission,
        "r2_I": r2_I,

        "N": N,

        "gamma": gamma,
        "mu": mu,
        "k": k,

        "S0": S0,
        "I0": I0,

        "obs": obs,
        "obs_is_fraction": obs_is_fraction
    }


def run_ode_from_mcmc():

    global ode_results

    if mcmc_stats is None:

        messagebox.showwarning(
            "Нет результатов",
            "Сначала запустите MCMC."
        )

        return

    try:

        gamma = get_float(entry_gamma.get())
        mu = get_float(entry_mu.get())
        k = get_float(entry_k.get())
        N = get_float(entry_N.get())

        S0_text = entry_S0.get().strip()
        I0_text = entry_I0.get().strip()

        S0 = get_float(S0_text) if S0_text else None
        I0 = get_float(I0_text) if I0_text else None

        if gamma <= 0:
            raise ValueError("gamma > 0.")
        if mu < 0:
            raise ValueError("mu >= 0.")
        if k < 0:
            raise ValueError("k >= 0.")
        if N <= 0:
            raise ValueError("N > 0.")

        ode_results = solve_ode_from_mcmc_means(
            gamma, mu, k, N, S0, I0
        )

        res = ode_results

        log_text.insert(
            tk.END,
            "\n" + "=" * 70 + "\n"
            "РЕШЕНИЕ ОДУ ПО MCMC\n"
            + "=" * 70 + "\n"
        )

        log_text.insert(
            tk.END,
            "Используются R_mean(t) и p_mean(t).\n"
        )

        log_text.insert(
            tk.END,
            "beta(t) = R(t) * gamma.\n"
        )

        log_text.insert(
            tk.END,
            "p_mean(t) = 0 там, "
            "где исходный p_init(t) = 0.\n"
        )

        log_text.insert(
            tk.END,
            f"R² transmission = "
            f"{res['r2_transmission']:.8f}\n"
        )

        log_text.insert(
            tk.END,
            f"R² I_model = "
            f"{res['r2_I']:.8f}\n"
        )

        log_text.see(tk.END)

        plot_ode_results()

    except Exception as e:

        import traceback

        messagebox.showerror(
            "Ошибка ОДУ",
            f"{e}\n\n"
            f"{traceback.format_exc()}"
        )


def plot_ode_results():

    if ode_results is None:
        return

    res = ode_results
    t = np.arange(1, days + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(t, res["S"], lw=2, label="S(t)")
    ax.plot(t, res["I"], lw=2, label="I(t)")
    ax.plot(
        t, 1.0 - res["S"] - res["I"],
        lw=2, label="1-S-I"
    )

    ax.set_xlabel("День")
    ax.set_ylabel("Доля")
    ax.set_title("Решение ОДУ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))

    if res["obs"] is not None:

        nobs = len(res["obs"])

        ax.plot(
            t[:nobs], res["obs"],
            lw=1.8, label="Факт: новые случаи"
        )

    ax.plot(
        t, res["transmission_model"],
        lw=2, label="Модель: transmission"
    )

    title = "Факт vs модельное transmission"

    if np.isfinite(res["r2_transmission"]):

        title += (
            f"    R² = "
            f"{res['r2_transmission']:.6f}"
        )

    ax.set_title(title)
    ax.set_xlabel("День")
    ax.set_ylabel("Новые случаи")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True
    )

    axs[0].plot(t, res["R"], lw=2, label="R_mean(t)")
    axs[0].axhline(1.0, ls="--", lw=1, label="R=1")
    axs[0].set_ylabel("R(t)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(t, res["p"], lw=2, label="p_mean(t)")
    axs[1].set_xlabel("День")
    axs[1].set_ylabel("p(t)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def export_results():

    if mcmc_samples_R is None:

        messagebox.showwarning(
            "Нет результатов",
            "Сначала запустите MCMC."
        )

        return

    path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        initialfile=(
            "mcmc_results_p_active.xlsx"
        ),
        filetypes=[
            ("Excel files", "*.xlsx"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return

    try:

        st = mcmc_stats
        t = np.arange(1, days + 1)

        data = {

            "Day": t,

            "R_init": R_init,
            "p_init": p_init,

            "p_active": p_active_mask.astype(int),

            "R_mean":   st["R_mean"],
            "R_median": st["R_median"],
            "R_std":    st["R_std"],

            "R_lo3s":   st["R_lo3"],
            "R_hi3s":   st["R_hi3"],
            "R_lo2s":   st["R_lo2"],
            "R_hi2s":   st["R_hi2"],
            "R_lo1s":   st["R_lo1"],
            "R_hi1s":   st["R_hi1"],

            "p_mean":   st["p_mean"],
            "p_median": st["p_median"],
            "p_std":    st["p_std"],

            "p_lo3s":   st["p_lo3"],
            "p_hi3s":   st["p_hi3"],
            "p_lo2s":   st["p_lo2"],
            "p_hi2s":   st["p_hi2"],
            "p_lo1s":   st["p_lo1"],
            "p_hi1s":   st["p_hi1"],

            "Rhat_R": diag_R["rhat"],
            "ESS_R":  diag_R["ess"],

            "Rhat_p": diag_p["rhat"],
            "ESS_p":  diag_p["ess"]
        }

        rhat_R_global = global_rhat(mcmc_samples_R)

        if np.any(p_active_mask):

            rhat_p_global = global_rhat(
                mcmc_samples_p[:, :, p_active_mask]
            )

        else:

            rhat_p_global = np.nan

        data["Global_Rhat_R"] = np.full(
            days, rhat_R_global
        )

        data["Global_Rhat_p"] = np.full(
            days, rhat_p_global
        )

        if I_obs is not None:

            data["I_obs"] = np.asarray(
                I_obs[:days], dtype=float
            )

        if ode_results is not None:

            ode = ode_results

            data.update({

                "S_model": ode["S"],
                "I_model_fraction": ode["I"],
                "I_model": ode["I_model"],

                "transmission_fraction":
                    ode["transmission_frac"],

                "transmission_model":
                    ode["transmission_model"],

                "R2_transmission": np.full(
                    days, ode["r2_transmission"]
                ),

                "R2_I_model": np.full(
                    days, ode["r2_I"]
                )
            })

        df = pd.DataFrame(data)

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":

            df.to_csv(
                path,
                index=False,
                encoding="utf-8-sig"
            )

        else:

            with pd.ExcelWriter(
                path, engine="openpyxl"
            ) as writer:

                df.to_excel(
                    writer,
                    sheet_name="По дням",
                    index=False
                )

                accept_df = pd.DataFrame({

                    "Chain": np.arange(
                        1, len(mcmc_accept_R) + 1
                    ),

                    "Accept_R": mcmc_accept_R,
                    "Accept_p": mcmc_accept_p
                })

                accept_df.to_excel(
                    writer,
                    sheet_name="Accept rate",
                    index=False
                )

                diag_df = pd.DataFrame({

                    "Metric": [
                        "Global_Rhat_R",
                        "Global_Rhat_p"
                    ],

                    "Value": [
                        rhat_R_global,
                        rhat_p_global
                    ]
                })

                diag_df.to_excel(
                    writer,
                    sheet_name="Global R-hat",
                    index=False
                )

        messagebox.showinfo(
            "Успех",
            f"Сохранено:\n{path}"
        )

    except Exception as e:

        import traceback

        messagebox.showerror(
            "Ошибка экспорта",
            f"{e}\n\n"
            f"{traceback.format_exc()}"
        )


def export_samples_npz():

    if mcmc_samples_R is None:

        messagebox.showwarning(
            "Нет результатов",
            "Сначала запустите MCMC."
        )

        return

    path = filedialog.asksaveasfilename(
        defaultextension=".npz",
        initialfile=(
            "mcmc_samples_p_active.npz"
        ),
        filetypes=[
            ("NumPy files", "*.npz"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return

    try:

        np.savez_compressed(

            path,

            samples_R=mcmc_samples_R,
            samples_p=mcmc_samples_p,

            R_init=R_init,
            p_init=p_init,

            p_active_mask=p_active_mask
        )

        messagebox.showinfo(
            "Успех",
            f"Сэмплы сохранены:\n{path}"
        )

    except Exception as e:

        messagebox.showerror("Ошибка", str(e))


root = tk.Tk()

root.title(
    "MCMC: R(t) и p(t) — ДИ = mean ± 3σ"
)

root.geometry("1400x950")


title_frame = tk.Frame(root)
title_frame.pack(fill=tk.X)

tk.Label(
    title_frame,
    text="Байесовский MCMC для R(t) и p(t)",
    font=("Segoe UI", 16, "bold")
).pack(pady=8)

tk.Label(
    title_frame,
    text=(
        "p(t) адаптируется ТОЛЬКО "
        "в дни, где загруженное p_init(t) > 0. "
        "ДИ = mean ± 3σ"
    ),
    font=("Segoe UI", 10)
).pack(pady=2)


param_frame = ttk.LabelFrame(
    root,
    text=" Параметры MCMC ",
    padding=12
)

param_frame.pack(fill=tk.X, padx=15, pady=8)

row = 0

tk.Label(
    param_frame, text="Число цепей:"
).grid(row=row, column=0, sticky="e", padx=5, pady=4)

entry_n_chains = ttk.Entry(param_frame, width=10)
entry_n_chains.insert(0, "4")
entry_n_chains.grid(row=row, column=1, padx=5, pady=4)

tk.Label(
    param_frame, text="Burn-in:"
).grid(row=row, column=2, sticky="e", padx=5, pady=4)

entry_burnin = ttk.Entry(param_frame, width=10)
entry_burnin.insert(0, "1000")
entry_burnin.grid(row=row, column=3, padx=5, pady=4)

tk.Label(
    param_frame, text="Сэмплов на цепь:"
).grid(row=row, column=4, sticky="e", padx=5, pady=4)

entry_n_samples = ttk.Entry(param_frame, width=10)
entry_n_samples.insert(0, "2000")
entry_n_samples.grid(row=row, column=5, padx=5, pady=4)

row += 1

tk.Label(
    param_frame, text="Шаг R:"
).grid(row=row, column=0, sticky="e", padx=5, pady=4)

entry_step_R = ttk.Entry(param_frame, width=10)
entry_step_R.insert(0, "0.05")
entry_step_R.grid(row=row, column=1, padx=5, pady=4)

tk.Label(
    param_frame, text="Шаг p:"
).grid(row=row, column=2, sticky="e", padx=5, pady=4)

entry_step_p = ttk.Entry(param_frame, width=10)
entry_step_p.insert(0, "0.002")
entry_step_p.grid(row=row, column=3, padx=5, pady=4)

tk.Label(
    param_frame, text="Smooth weight:"
).grid(row=row, column=4, sticky="e", padx=5, pady=4)

entry_smooth = ttk.Entry(param_frame, width=10)
entry_smooth.insert(0, "1.0")
entry_smooth.grid(row=row, column=5, padx=5, pady=4)

row += 1

tk.Label(
    param_frame, text="σ_R:"
).grid(row=row, column=0, sticky="e", padx=5, pady=4)

entry_sigma_R = ttk.Entry(param_frame, width=10)
entry_sigma_R.insert(0, "0.5")
entry_sigma_R.grid(row=row, column=1, padx=5, pady=4)

tk.Label(
    param_frame, text="σ_p:"
).grid(row=row, column=2, sticky="e", padx=5, pady=4)

entry_sigma_p = ttk.Entry(param_frame, width=10)
entry_sigma_p.insert(0, "0.02")
entry_sigma_p.grid(row=row, column=3, padx=5, pady=4)


ode_frame = ttk.LabelFrame(
    root,
    text=" Параметры ОДУ ",
    padding=10
)

ode_frame.pack(fill=tk.X, padx=15, pady=4)

ode_defaults = [

    ("gamma:", "0.074"),
    ("mu:", "0.000138"),
    ("k:", "0.0055"),
    ("N:", "1.0"),
    ("S0:", ""),
    ("I0:", "")
]

ode_entries = {}

for col, (label, value) in enumerate(ode_defaults):

    tk.Label(
        ode_frame, text=label
    ).grid(row=0, column=col * 2, sticky="e", padx=3, pady=3)

    ent = ttk.Entry(ode_frame, width=12)

    if value:
        ent.insert(0, value)

    ent.grid(row=0, column=col * 2 + 1, padx=3, pady=3)

    ode_entries[label] = ent


entry_gamma = ode_entries["gamma:"]
entry_mu = ode_entries["mu:"]
entry_k = ode_entries["k:"]
entry_N = ode_entries["N:"]
entry_S0 = ode_entries["S0:"]
entry_I0 = ode_entries["I0:"]

tk.Label(
    ode_frame,
    text=(
        "S и I в ОДУ — доли. "
        "N используется только "
        "для перевода transmission "
        "в количество случаев. "
        "beta(t) = R(t) * gamma."
    )
).grid(
    row=1, column=0, columnspan=14,
    sticky="w", padx=3, pady=2
)


btn_frame = tk.Frame(root)
btn_frame.pack(fill=tk.X, padx=15, pady=8)

ttk.Button(
    btn_frame,
    text="📂 Загрузить Excel/CSV",
    command=load_initial
).pack(side=tk.LEFT, padx=4)

ttk.Button(
    btn_frame,
    text="🚀 Запустить MCMC",
    command=run_mcmc
).pack(side=tk.LEFT, padx=4)

ttk.Button(
    btn_frame,
    text="📊 Диагностика",
    command=plot_diagnostics
).pack(side=tk.LEFT, padx=4)

ttk.Button(
    btn_frame,
    text="🧮 Решить ОДУ по MCMC",
    command=run_ode_from_mcmc
).pack(side=tk.LEFT, padx=4)

ttk.Button(
    btn_frame,
    text="💾 Экспорт результатов",
    command=export_results
).pack(side=tk.LEFT, padx=4)

ttk.Button(
    btn_frame,
    text="💾 Экспорт MCMC (.npz)",
    command=export_samples_npz
).pack(side=tk.LEFT, padx=4)

ttk.Button(
    btn_frame,
    text="❌ Выход",
    command=root.destroy
).pack(side=tk.RIGHT, padx=4)


log_frame = ttk.LabelFrame(
    root, text=" Лог ", padding=8
)

log_frame.pack(
    fill=tk.BOTH, expand=True,
    padx=15, pady=10
)

log_text = scrolledtext.ScrolledText(
    log_frame,
    font=("Consolas", 9),
    height=24,
    wrap=tk.WORD
)

log_text.pack(fill=tk.BOTH, expand=True)

log_text.insert(
    tk.END,

    "ИНСТРУКЦИЯ\n"
    "==================================================\n\n"

    "1. Загрузите Excel/CSV.\n\n"

    "2. В файле должны быть R(t) и p(t).\n\n"

    "3. Активность p определяется ТОЛЬКО "
    "исходным значением p:\n\n"

    "       p_init > 0  -> адаптация MCMC\n"
    "       p_init = 0  -> p навсегда равно 0\n\n"

    "4. Начальное состояние каждой цепи p:\n\n"

    "       p_cur = p_init\n\n"

    "5. S0 и I0 (а также gamma, mu, k, EPOP) "
    "автоматически подгружаются "
    "со 2-го листа Excel.\n\n"

    "6. После MCMC можно решить ОДУ "
    "по R_mean(t) и p_mean(t).\n"
    "   beta(t) = R(t) * gamma.\n\n"

    "7. R² считается между фактом "
    "новых случаев и model transmission.\n\n"

    "8. ДИ считаются как  mean ± k·σ  (k = 1, 2, 3).\n"
    "   Основной ДИ = mean ± 3σ.\n\n"

    "9. Диагностика выводит:\n"
    "   - R-hat по каждому дню\n"
    "   - R-hat за весь период (min/mean/max)\n"
    "   - ГЛОБАЛЬНЫЙ R-hat (все цепи × все дни)\n"
    "   - ОБЩИЙ GLOBAL R-hat (max по R и p)\n"
)


root.mainloop()