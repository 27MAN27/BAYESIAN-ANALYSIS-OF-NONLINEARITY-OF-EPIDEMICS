import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm, uniform
import warnings
warnings.filterwarnings('ignore')

WEIGHT_EXPONENT = 0.01

def get_weight(step):
    return 1.0 / (step + 1) ** WEIGHT_EXPONENT

adapt_p_var = None

def get_float(val):
    if val is None:
        return None
    if isinstance(val, str):
        val = val.replace(',', '.')
    try:
        return float(val)
    except:
        return None

def load_data_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        p = data.get('params', {})

        N = get_float(p.get('EPOP', {}).get('curr'))

        params = {
            'N':      N,
            'R0':     get_float(p.get('R', {}).get('curr')),
            'gamma':  get_float(p.get('ALPHA', {}).get('curr')),
            'mu':     get_float(p.get('MU', {}).get('curr')),
            'S0':     get_float(p.get('S0', {}).get('curr')),
            'I0':     get_float(p.get('Y0', {}).get('curr')),
            'k':      get_float(p.get('K', {}).get('curr')),
            'p':      get_float(p.get('RPOW', {}).get('curr'))
        }

        days_data = data.get('days', [])
        I_fact_full = []
        for day_entry in days_data:
            total = day_entry.get('total', [])
            if len(total) > 1:
                real_val = get_float(total[1].get('real'))
                if real_val is not None:
                    I_fact_full.append(real_val)

        if len(I_fact_full) <= 1:
            messagebox.showerror("Ошибка", "Недостаточно фактических данных.")
            return None, None, None

        I_fact_abs = np.array(I_fact_full)
        I_fact = I_fact_abs / N if N and N > 0 else I_fact_abs

        missing = [key for key, val in params.items() if val is None]
        if missing:
            messagebox.showerror("Ошибка", f"Не найдены параметры: {missing}")
            return None, None, None

        return params, I_fact, I_fact_abs

    except Exception as e:
        messagebox.showerror("Ошибка чтения JSON", str(e))
        return None, None, None

def sir_euler_grid(params, I_fact, window, p_min=0.0, p_max=0.1, p_step=0.001, 
                   k_val=0.055, r0_max=10.0, fixed_p=None, adapt_p=True,
                   mu_fixed=None, use_grid_search=False):                                   
    days = len(I_fact)

    if mu_fixed is not None:
        mu = mu_fixed
    else:
        mu = params.get('mu', 0.00013)                         
    
    k = k_val
    gamma_fixed = params['gamma']

    if not adapt_p or fixed_p is not None:
        p_fixed = fixed_p if fixed_p is not None else params.get('p', 0.0)
        use_adaptive_p = False
    else:
        p_fixed = 0.0
        use_adaptive_p = True

    print(f"\n   🔧 Используем фиксированное mu = {mu:.8f}")
    print(f"   mu НЕ оптимизируется, используется введённое значение")

    S, I, trans, R0, p = run_sir_with_fixed_mu(
        params, I_fact, mu, window, p_min, p_max, p_step,
        k_val, r0_max, p_fixed, use_adaptive_p
    )

    return S, I, trans, R0, p, np.ones(days) * mu

def run_sir_with_fixed_mu(params, I_fact, mu, window, p_min, p_max, p_step,
                          k_val, r0_max, p_fixed, use_adaptive_p):
    days = len(I_fact)
    k = k_val
    gamma_fixed = params['gamma']

    S = np.zeros(days)
    I = np.zeros(days)
    transmission = np.zeros(days)
    best_R0 = np.zeros(days)
    best_p = np.zeros(days)

    S[0] = params['S0']
    I[0] = params['I0']
    transmission[0] = I_fact[0]
    best_R0[0] = params['R0']
    best_p[0] = p_fixed

    R0_MIN = 0.3
    R0_MAX = r0_max  
    P_MIN = p_min
    P_MAX = p_max
    P_START_DAY = 430
    START_DAY = 5

    for t in range(1, days):
        t_prev = t - 1
        prev_r0 = best_R0[t_prev]
        prev_p = best_p[t_prev]

        is_growing = I_fact[t] > I_fact[t_prev]
        is_falling = I_fact[t] < I_fact[t_prev]

        best_error = np.inf
        best_r0 = prev_r0
        
        if use_adaptive_p:
            best_p_val = prev_p
        else:
            best_p_val = p_fixed

        if t >= START_DAY:
            r0_low = max(R0_MIN, prev_r0 * 0.8)
            r0_high = min(R0_MAX, prev_r0 * 2.0)
            r0_grid = np.arange(r0_low, r0_high + 0.0005, 0.02)          
        else:
            r0_grid = [prev_r0]

        if use_adaptive_p and t >= P_START_DAY:
            if is_growing:
                                                     
                p_low = max(P_MIN, prev_p - 0.005)
                p_high = min(P_MAX, prev_p + 0.005)
                p_grid = np.arange(p_low, p_high + 0.0005, 0.0005)
            else:
                                                   
                p_grid = [0.0]
        else:
            p_grid = [p_fixed]

        for R0 in r0_grid:
            for p_candidate in p_grid:
                beta = R0 * gamma_fixed

                dS = -beta * S[t_prev] * I[t_prev] + mu * (1 - S[t_prev]) + (k + p_candidate) * (1 - S[t_prev] - I[t_prev])
                dI = beta * S[t_prev] * I[t_prev] - gamma_fixed * I[t_prev] - mu * I[t_prev]
                
                S_test = max(S[t_prev] + dS, 0)
                I_test = max(I[t_prev] + dI, 0)
                
                error = 0.0
                S_tmp, I_tmp = S_test, I_test
                
                for step in range(window):
                    idx = t + step
                    if idx >= days:
                        break
                        
                    trans_tmp = beta * S_tmp * I_tmp
                    weight = get_weight(step)
                    error += weight * (trans_tmp - I_fact[idx]) ** 2
                    
                    dS_tmp = -beta * S_tmp * I_tmp + mu * (1 - S_tmp) + (k + p_candidate) * (1 - S_tmp - I_tmp)
                    dI_tmp = beta * S_tmp * I_tmp - gamma_fixed * I_tmp - mu * I_tmp
                    
                    S_tmp = max(S_tmp + dS_tmp, 0)
                    I_tmp = max(I_tmp + dI_tmp, 1e-6)
                
                if error < best_error:
                    best_error = error
                    best_r0 = R0
                    best_p_val = p_candidate
        
        best_r0 = np.clip(best_r0, R0_MIN, R0_MAX)
        best_p_val = np.clip(best_p_val, P_MIN, P_MAX)

        if not is_growing and t >= P_START_DAY:
            best_p_val = 0.0

        best_R0[t] = best_r0
        best_p[t] = best_p_val
        
        beta_best = best_r0 * gamma_fixed

        dS = -beta_best * S[t_prev] * I[t_prev] + mu * (1 - S[t_prev]) + (k + best_p_val) * (1 - S[t_prev] - I[t_prev])
        dI = beta_best * S[t_prev] * I[t_prev] - gamma_fixed * I[t_prev] - mu * I[t_prev]
        
        S[t] = max(S[t_prev] + dS, 0)
        I[t] = max(I[t_prev] + dI, 0)
        transmission[t] = beta_best * S[t] * I[t]
    
    return S, I, transmission, best_R0, best_p

def sir_euler_grid_bayesian(params, I_fact, window=5, p_min=0.0, p_max=0.1, p_step=0.001, 
                            k_val=0.055, r0_max=10.0, fixed_p=None, adapt_p=True,
                            mu_fixed=None):                          
    days = len(I_fact)

    if mu_fixed is not None:
        mu = mu_fixed
    else:
        mu = params.get('mu', 0.001)                         
    
    k = k_val
    gamma_fixed = params['gamma']

    if not adapt_p or fixed_p is not None:
        p_fixed = fixed_p if fixed_p is not None else params.get('p', 0.0)
        use_adaptive_p = False
    else:
        p_fixed = 0.0
        use_adaptive_p = True

    S = np.zeros(days)
    I = np.zeros(days)
    transmission = np.zeros(days)
    best_R0 = np.zeros(days)
    best_p = np.zeros(days)

    S[0] = params['S0']
    I[0] = params['I0']
    transmission[0] = I_fact[0]
    best_R0[0] = params['R0']
    best_p[0] = p_fixed

    R0_MIN = 0.3
    R0_MAX = r0_max  
    P_MIN = p_min
    P_MAX = p_max
    P_START_DAY = 430
    START_DAY = 5

    print(f"\n   🔧 Байесовская модель с фиксированным mu = {mu:.8f}")

    for t in range(1, days):
        t_prev = t - 1
        prev_r0 = best_R0[t_prev]
        prev_p = best_p[t_prev]
        
        is_growing = I_fact[t] > I_fact[t_prev]
        is_falling = I_fact[t] < I_fact[t_prev]

        best_error = np.inf
        best_r0 = prev_r0
        
        if use_adaptive_p:
            best_p_val = prev_p
        else:
            best_p_val = p_fixed

        if t >= START_DAY:
            r0_low = max(R0_MIN, prev_r0 * 0.8)
            r0_high = min(R0_MAX, prev_r0 * 2.0)
            r0_grid = np.arange(r0_low, r0_high + 0.0005, 0.02)
        else:
            r0_grid = [prev_r0]

        if use_adaptive_p and t >= P_START_DAY and is_growing:
            p_low = max(P_MIN, prev_p - 0.005)
            p_high = min(P_MAX, prev_p + 0.005)
            p_grid = np.arange(p_low, p_high + 0.0005, 0.0005)
        else:
            p_grid = [p_fixed]
        
        for R0 in r0_grid:
            for p_candidate in p_grid:
                beta = R0 * gamma_fixed

                dS = -beta * S[t_prev] * I[t_prev] + mu * (1 - S[t_prev]) + (k + p_candidate) * (1 - S[t_prev] - I[t_prev])
                dI = beta * S[t_prev] * I[t_prev] - gamma_fixed * I[t_prev] - mu * I[t_prev]
                
                S_test = max(S[t_prev] + dS, 0)
                I_test = max(I[t_prev] + dI, 0)
                
                error = 0.0
                S_tmp, I_tmp = S_test, I_test
                
                for step in range(window):
                    idx = t + step
                    if idx >= days:
                        break
                            
                    trans_tmp = beta * S_tmp * I_tmp
                    weight = get_weight(step)
                    error += weight * (trans_tmp - I_fact[idx]) ** 2
                    
                    dS_tmp = -beta * S_tmp * I_tmp + mu * (1 - S_tmp) + (k + p_candidate) * (1 - S_tmp - I_tmp)
                    dI_tmp = beta * S_tmp * I_tmp - gamma_fixed * I_tmp - mu * I_tmp
                    
                    S_tmp = max(S_tmp + dS_tmp, 0)
                    I_tmp = max(I_tmp + dI_tmp, 1e-6)
                
                if error < best_error:
                    best_error = error
                    best_r0 = R0
                    best_p_val = p_candidate
        
        best_r0 = np.clip(best_r0, R0_MIN, R0_MAX)
        best_p_val = np.clip(best_p_val, P_MIN, P_MAX)
        
        best_R0[t] = best_r0
        best_p[t] = best_p_val
        
        beta_best = best_r0 * gamma_fixed

        dS = -beta_best * S[t_prev] * I[t_prev] + mu * (1 - S[t_prev]) + (k + best_p_val) * (1 - S[t_prev] - I[t_prev])
        dI = beta_best * S[t_prev] * I[t_prev] - gamma_fixed * I[t_prev] - mu * I[t_prev]
        
        S[t] = max(S[t_prev] + dS, 0)
        I[t] = max(I[t_prev] + dI, 0)
        transmission[t] = beta_best * S[t] * I[t]
    
    print(f"   R0: адаптируется с {START_DAY}-го дня")
    print(f"   p: {'адаптивное с 430 дня' if use_adaptive_p else f'фиксированное (={p_fixed:.6f})'}")
    print(f"   mu: ФИКСИРОВАННОЕ = {mu:.8f}")
    print(f"   γ фиксирована: {gamma_fixed:.6f}")

    return S, I, transmission, best_R0, best_p, np.ones(days) * mu, np.ones(days) * gamma_fixed

def save_optimized_params_to_excel(params_opt, filepath=None):
    import pandas as pd
    from datetime import datetime
    
    if filepath is None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"optimized_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        if not filepath:
            return
    
    data = {
        'Параметр': [
            'S0 (начальная доля восприимчивых)',
            'I0 (начальная доля инфицированных)',
            'γ (скорость выздоровления)',
            'μ (скорость смертности/рождаемости)',
            'k (скорость потери иммунитета)',
            'p (параметр преодоления иммунитета)',
            'R₀ (эффективное репродуктивное число)',
            'EPOP (численность населения)',
            'Window (размер окна)',
            'R0_max (максимальное R)',
            'S0_abs (абсолютное)',
            'I0_abs (абсолютное)',
            'R² (качество модели)',
            'MAE (средняя абсолютная ошибка)',
            'RMSE (среднеквадратичная ошибка)'
        ],
        'Значение': [
            params_opt.get('S0', 0),
            params_opt.get('I0', 0),
            params_opt.get('gamma', 0),
            params_opt.get('mu', 0),
            params_opt.get('k', 0),
            params_opt.get('p', 0),
            params_opt.get('R0', 0),
            params_opt.get('EPOP', 0),
            params_opt.get('window', 0),
            params_opt.get('r0_max', 0),
            params_opt.get('S0_abs', 0),
            params_opt.get('I0_abs', 0),
            params_opt.get('r2', 0),
            params_opt.get('MAE', 0),
            params_opt.get('RMSE', 0)
        ]
    }
    
    df = pd.DataFrame(data)
    
    if 'R_stats' in params_opt:
        stats_data = {
            'Параметр': list(params_opt['R_stats'].keys()),
            'Значение': list(params_opt['R_stats'].values())
        }
        df_stats = pd.DataFrame(stats_data)
    else:
        df_stats = None
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Оптимизированные параметры', index=False)
        if df_stats is not None:
            df_stats.to_excel(writer, sheet_name='Статистика R(t) и p(t)', index=False)
    
    messagebox.showinfo("Успешно", f"Параметры сохранены в:\n{filepath}")
    return filepath

def generate_bayesian_trajectories(
    mcmc_samples_R0,
    mcmc_samples_p,
    I_fact,
    params,
    n_sim=10,
    window=5,
    k_val=0.055,
    r0_max=10.0,
    p_max=0.4,
    add_noise=True,
    noise_level=0.10,
    return_p=False,
    return_S=False,
    return_I=False,
    return_mu=False,
    verbose=True,
    adapt_p=True,
    mu_fixed=None                          
):
    days = len(I_fact)

    if mu_fixed is None:
        mu_fixed = params.get('mu', 0.001)
    
    trajectories_R0 = []
    trajectories_p = [] if return_p else None
    trajectories_S = [] if return_S else None
    trajectories_I = [] if return_I else None
    trajectories_mu = [] if return_mu else None
    trajectories_gamma = []
    
    if verbose:
        print(f"🔄 Генерация {n_sim} байесовских траекторий...")
        print(f"   Шум: {'включен' if add_noise else 'выключен'}, уровень: {noise_level*100:.0f}%")
        print(f"   Адаптация p: {'включена' if adapt_p else 'выключена'}")
        print(f"   mu: ФИКСИРОВАННОЕ = {mu_fixed:.8f}")
    
    for i in range(n_sim):
        idx = np.random.choice(len(mcmc_samples_R0))
        R0_sample = mcmc_samples_R0[idx]
        p_sample = mcmc_samples_p[idx]
        
        params_copy = params.copy()
        params_copy['R0'] = R0_sample
        params_copy['p'] = p_sample
        
        S_traj, I_traj, _, r_traj, p_traj, mu_traj, gamma_traj = sir_euler_grid_bayesian(
            params_copy,
            I_fact.copy(),
            window=window,
            p_min=0.0,
            p_max=p_max,
            p_step=0.001,
            k_val=k_val,
            r0_max=r0_max,
            fixed_p=None if adapt_p else p_sample,
            adapt_p=adapt_p,
            mu_fixed=mu_fixed                                 
        )
        
        if add_noise:
            noise = np.random.normal(0, noise_level, len(r_traj))
            r_traj = np.clip(r_traj + noise, 0.3, r0_max)
        
        trajectories_R0.append(r_traj)
        trajectories_gamma.append(gamma_traj)
        
        if return_p:
            trajectories_p.append(p_traj)
        if return_S:
            trajectories_S.append(S_traj)
        if return_I:
            trajectories_I.append(I_traj)
        if return_mu:
            trajectories_mu.append(mu_traj)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Траектория {i+1}/{n_sim}")
    
    trajectories_R0 = np.array(trajectories_R0)
    trajectories_gamma = np.array(trajectories_gamma)
    
    result = {
        'R0': trajectories_R0,
        'mean_R0': np.mean(trajectories_R0, axis=0),
        'median_R0': np.median(trajectories_R0, axis=0),
        'std_R0': np.std(trajectories_R0, axis=0),
        'lower_ci_R0': np.percentile(trajectories_R0, 2.5, axis=0),
        'upper_ci_R0': np.percentile(trajectories_R0, 97.5, axis=0),
        'gamma': trajectories_gamma,
        'mean_gamma': np.mean(trajectories_gamma, axis=0),
        'median_gamma': np.median(trajectories_gamma, axis=0),
        'std_gamma': np.std(trajectories_gamma, axis=0),
        'lower_ci_gamma': np.percentile(trajectories_gamma, 2.5, axis=0),
        'upper_ci_gamma': np.percentile(trajectories_gamma, 97.5, axis=0)
    }
    
    if return_p and trajectories_p is not None:
        trajectories_p = np.array(trajectories_p)
        result['p'] = trajectories_p
        result['mean_p'] = np.mean(trajectories_p, axis=0)
        result['median_p'] = np.median(trajectories_p, axis=0)
        result['std_p'] = np.std(trajectories_p, axis=0)
        result['lower_ci_p'] = np.percentile(trajectories_p, 2.5, axis=0)
        result['upper_ci_p'] = np.percentile(trajectories_p, 97.5, axis=0)
    
    if return_mu and trajectories_mu is not None:
        trajectories_mu = np.array(trajectories_mu)
        result['mu'] = trajectories_mu
        result['mean_mu'] = np.mean(trajectories_mu, axis=0)
        result['median_mu'] = np.median(trajectories_mu, axis=0)
        result['std_mu'] = np.std(trajectories_mu, axis=0)
        result['lower_ci_mu'] = np.percentile(trajectories_mu, 2.5, axis=0)
        result['upper_ci_mu'] = np.percentile(trajectories_mu, 97.5, axis=0)
    
    if return_S and trajectories_S is not None:
        trajectories_S = np.array(trajectories_S)
        result['S'] = trajectories_S
        result['mean_S'] = np.mean(trajectories_S, axis=0)
    
    if return_I and trajectories_I is not None:
        trajectories_I = np.array(trajectories_I)
        result['I'] = trajectories_I
        result['mean_I'] = np.mean(trajectories_I, axis=0)
    
    if verbose:
        print(f"✅ Сгенерировано {n_sim} траекторий")
        print(f"   Средняя R(t) = {np.mean(result['mean_R0']):.4f}")
        print(f"   mu фиксированное = {mu_fixed:.8f}")
        print(f"   Средняя γ(t) = {np.mean(result['mean_gamma']):.6f}")
    
    return result


def export_S_I_from_grid():
    """
    Выгружает S(t) и I(t) из сеточной модели за каждый день в Excel/CSV.
    """
    global last_S, last_I, last_I_fact, last_I_fact_abs, current_best_R0_arr
    global current_best_p, current_best_gamma, current_best_mu, last_params

    if last_S is None or last_I is None:
        messagebox.showwarning(
            "Нет данных",
            "Сначала загрузите JSON и запустите симуляцию (▶ Запустить симуляцию)."
        )
        return

    try:
        days = len(last_S)

        data = {
            'День': np.arange(1, days + 1),
            'S(t)': np.asarray(last_S, dtype=float),
            'I(t)': np.asarray(last_I, dtype=float),
        }

        if current_best_R0_arr is not None and len(current_best_R0_arr) >= days:
            data['R(t)'] = np.asarray(current_best_R0_arr[:days], dtype=float)
        if current_best_p is not None and len(current_best_p) >= days:
            data['p(t)'] = np.asarray(current_best_p[:days], dtype=float)

        if current_best_gamma is not None and len(current_best_gamma) >= days:
            data['gamma(t)'] = np.asarray(current_best_gamma[:days], dtype=float)
        if current_best_mu is not None and len(current_best_mu) >= days:
            data['mu(t)'] = np.asarray(current_best_mu[:days], dtype=float)

        if last_I_fact_abs is not None and len(last_I_fact_abs) >= days:
            data['I_fact_abs'] = np.asarray(last_I_fact_abs[:days], dtype=float)
        if last_I_fact is not None and len(last_I_fact) >= days:
            data['I_fact_norm'] = np.asarray(last_I_fact[:days], dtype=float)

        N = None
        if last_params is not None:
            N = last_params.get('N', None)
        if N is not None and N > 0:
            data['S_abs'] = data['S(t)'] * N
            data['I_abs'] = data['I(t)'] * N

        df = pd.DataFrame(data)

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile="S_I_grid.xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            stats = {
                'Параметр': [
                    'Дней',
                    'S(0)', 'S(min)', 'S(max)', 'S(конец)',
                    'I(0)', 'I(min)', 'I(max)', 'I(конец)',
                    'День min S', 'День max I',
                    'N (EPOP)',
                    'Среднее R(t)', 'Среднее p(t) (активные)',
                ],
                'Значение': [
                    days,
                    float(data['S(t)'][0]),
                    float(np.min(data['S(t)'])),
                    float(np.max(data['S(t)'])),
                    float(data['S(t)'][-1]),
                    float(data['I(t)'][0]),
                    float(np.min(data['I(t)'])),
                    float(np.max(data['I(t)'])),
                    float(data['I(t)'][-1]),
                    int(np.argmin(data['S(t)'])) + 1,
                    int(np.argmax(data['I(t)'])) + 1,
                    N if N is not None else '—',
                    float(np.mean(data['R(t)'])) if 'R(t)' in data else '—',
                    float(np.mean(data['p(t)'][data['p(t)'] > 0.0001]))
                        if 'p(t)' in data and np.any(data['p(t)'] > 0.0001) else 0.0,
                ]
            }
            df_stats = pd.DataFrame(stats)

            if last_params is not None:
                params_df = pd.DataFrame({
                    'Параметр': ['S0', 'I0', 'R0', 'gamma', 'mu', 'k', 'N', 'Window'],
                    'Значение': [
                        last_params.get('S0', '—'),
                        last_params.get('I0', '—'),
                        last_params.get('R0', '—'),
                        last_params.get('gamma', '—'),
                        last_params.get('mu', '—'),
                        last_params.get('k', '—'),
                        last_params.get('N', '—'),
                        int(window_entry.get() or 15),
                    ]
                })
            else:
                params_df = None

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='S_I_по_дням', index=False)
                df_stats.to_excel(writer, sheet_name='Статистика', index=False)
                if params_df is not None:
                    params_df.to_excel(writer, sheet_name='Параметры', index=False)

        text_area.insert(tk.END, f"\n💾 S(t) и I(t) сохранены: {file_path}\n")
        text_area.insert(tk.END, f"   Дней: {days}\n")
        text_area.insert(tk.END, f"   S(0) = {data['S(t)'][0]:.6f}, "
                                 f"S(конец) = {data['S(t)'][-1]:.6f}, "
                                 f"S(min) = {np.min(data['S(t)']):.6f} "
                                 f"(день {np.argmin(data['S(t)'])+1})\n")
        text_area.insert(tk.END, f"   I(0) = {data['I(t)'][0]:.6f}, "
                                 f"I(конец) = {data['I(t)'][-1]:.6f}, "
                                 f"I(max) = {np.max(data['I(t)']):.6f} "
                                 f"(день {np.argmax(data['I(t)'])+1})\n")
        text_area.see(tk.END)

        messagebox.showinfo("Успешно", f"S(t) и I(t) сохранены в:\n{file_path}")

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка выгрузки S/I: {error_detail}")
        messagebox.showerror("Ошибка экспорта",
                             f"{e}\n\n{error_detail}")
        
def plot_bayesian_results(
    result,
    current_best_R0_arr=None,
    current_best_p=None,
    current_S=None,
    current_I_fact=None,
    plot_type="all",
    show_ci=True,
    title="Байесовские оценки",
    p_start_day=430
):
    
    t = np.arange(len(result['mean_R0']))
    
    plots = []
    if plot_type == "all" or plot_type == "r":
        plots.append(('R(t)', result['mean_R0'], 'blue', result['lower_ci_R0'], result['upper_ci_R0'], None))
    if plot_type == "all" or plot_type == "gamma":
        if 'mean_gamma' in result:
            plots.append(('γ(t)', result['mean_gamma'], 'red', result.get('lower_ci_gamma'), result.get('upper_ci_gamma'), None))
    if plot_type == "all" or plot_type == "p":
        if 'mean_p' in result:
            plots.append(('p(t)', result['mean_p'], 'green', result.get('lower_ci_p'), result.get('upper_ci_p'), p_start_day))
    if plot_type == "all" or plot_type == "s":
        if 'mean_S' in result:
            plots.append(('S(t)', result['mean_S'], 'magenta', None, None, None))
    if plot_type == "all" or plot_type == "a":
        if 'mean_I' in result:
            plots.append(('A(t)', result['mean_I'], 'orange', None, None, None))
    
    n_plots = len(plots)
    fig, axs = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axs = [axs]
    
    for ax, (label, data, color, lower, upper, start_day) in zip(axs, plots):
        ax.plot(t, data, '-', lw=2.5, color=color, label=f'Среднее {label}')
        
        if show_ci and lower is not None and upper is not None:
            if label == 'p(t)' and start_day is not None:
                mask = t >= start_day
                ax.fill_between(t[mask], lower[mask], upper[mask], 
                               color=color, alpha=0.3, label=f'95% ДИ (с {start_day} дня)')
                ax.axvline(x=start_day, color='purple', linestyle='--', alpha=0.7, 
                          label=f'Начало адаптации p')
            else:
                ax.fill_between(t, lower, upper, color=color, alpha=0.3, label='95% ДИ')
        
        if label == 'R(t)' and current_best_R0_arr is not None:
            ax.plot(t, current_best_R0_arr, 'r--', lw=1.5, label='Текущая R(t)')
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R=1')
        
        if label == 'p(t)' and current_best_p is not None:
            ax.plot(t, current_best_p, 'r--', lw=1.5, label='Текущая p(t)')
            if start_day is not None:
                ax.axvline(x=start_day, color='purple', linestyle='--', alpha=0.7, 
                          label='Начало адаптации p')
        
        if label == 'S(t)' and current_S is not None:
            ax.plot(t, current_S, 'r--', lw=1.5, label='Текущая S(t)')
        
        if label == 'A(t)' and current_I_fact is not None:
            ax.plot(t, current_I_fact, 'r--', lw=1.5, label='Фактические данные')
        
        ax.set_xlabel('День', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        
        if label == 'p(t)' and start_day is not None:
            ax.set_title(f'{label} с 95% ДИ (с {start_day} дня)', fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'{label} с 95% ДИ' if show_ci else label, fontsize=13, fontweight='bold')
        
        ax.legend()
        ax.grid(True, alpha=0.4)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_comprehensive_bayesian():
    global mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params, current_best_R0_arr, current_best_p, current_S, current_I_fact, adapt_p_var
    
    if mcmc_samples_R0 is None:
        messagebox.showwarning("Предупреждение", "Сначала запустите MCMC.")
        return
    
    if last_I_fact is None or last_params is None:
        messagebox.showwarning("Предупреждение", "Сначала загрузите JSON-файл.")
        return
    
    window_val = int(window_entry.get() or 5)
    k_val = float(k_entry.get() or 0.055)
    r0_max_val = float(r0_max_entry.get() or 20.0)
    
    use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True
    
    result = generate_bayesian_trajectories(
        mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params,
        n_sim=10,
        window=window_val,
        k_val=k_val,
        r0_max=r0_max_val,
        return_p=True,
        return_S=True,
        return_I=True,
        add_noise=True,
        noise_level=0.08,
        adapt_p=use_adapt_p
    )
    
    plot_bayesian_results(
        result,
        current_best_R0_arr=current_best_R0_arr,
        current_best_p=current_best_p,
        current_S=current_S,
        current_I_fact=current_I_fact,
        plot_type="all",
        title="Полный байесовский анализ (R, p, S, A)",
        p_start_day=430
    )

def calculate_lambda_regression(R_data, Y_data):
    R_data = np.array(R_data)
    Y_data = np.array(Y_data)

    mask = (Y_data > 1e-9) & (R_data > 0)
    R_data = R_data[mask]
    Y_data = Y_data[mask]
    
    if len(R_data) < 3:
        print("❌ Недостаточно данных для регрессии (меньше 3 точек)")
        return None

    log_R = np.log(R_data)
    log_Y = np.log(Y_data)

    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(log_Y, log_R)

    p = slope              
    R0 = np.exp(intercept)                        

    log_R_pred = intercept + slope * log_Y
    R_pred = np.exp(log_R_pred)

    r2 = r_value ** 2
    mae = np.mean(np.abs(R_data - R_pred))
    rmse = np.sqrt(np.mean((R_data - R_pred) ** 2))
    
    results = {
        'p': p,                                                           
        'R0': R0,                                  
        'R2': r2,                                            
        'MAE': mae,                                           
        'RMSE': rmse,                                         
        'std_err': std_err,                              
        'p_value': p_value,                          
        'n_points': len(R_data),                     
        'intercept': intercept,            
        'slope': slope,               
        'R_pred': R_pred,                                  
        'log_Y': log_Y,                                           
        'log_R': log_R,                                           
        'log_R_pred': log_R_pred                         
    }
    
    return results

def print_regression_results(results):
    if results is None:
        return
    
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ РЕГРЕССИИ: R(t) = R0 * Y(t)^p")
    print("="*60)
    print(f"   Формула: R(t) = {results['R0']:.6f} * Y(t)^{results['p']:.6f}")
    print("-"*60)
    print(f"\n📈 ПАРАМЕТРЫ:")
    print(f"   R0 = {results['R0']:.6f}  (начальное репродуктивное число)")
    print(f"   p  = {results['p']:.6f}  (коэффициент нелинейности)")
    print(f"   ln(R0) = {results['intercept']:.6f}")
    print("-"*60)
    print(f"\n📊 КАЧЕСТВО МОДЕЛИ:")
    print(f"   R²  = {results['R2']:.6f}")
    print(f"   MAE = {results['MAE']:.6f}")
    print(f"   RMSE = {results['RMSE']:.6f}")
    print(f"   Std Error = {results['std_err']:.6f}")
    print(f"   p-value = {results['p_value']:.6f}")
    print(f"   Точек = {results['n_points']}")
    print("-"*60)

    print(f"\n📖 ИНТЕРПРЕТАЦИЯ:")
    if results['R2'] > 0.9:
        print("   ✅ Отличное качество модели (R² > 0.9)")
    elif results['R2'] > 0.7:
        print("   ✅ Хорошее качество модели (0.7 < R² ≤ 0.9)")
    elif results['R2'] > 0.5:
        print("   ⚠️ Удовлетворительное качество (0.5 < R² ≤ 0.7)")
    else:
        print("   ❌ Плохое качество модели (R² ≤ 0.5)")
    
    if results['p'] > 1:
        print(f"   📈 Суперлинейный рост (p = {results['p']:.4f} > 1) — ускорение")
    elif results['p'] == 1:
        print(f"   ➖ Линейная зависимость (p = {results['p']:.4f} = 1)")
    elif results['p'] > 0:
        print(f"   📉 Сублинейный рост (0 < p = {results['p']:.4f} < 1) — замедление")
    else:
        print(f"   🔄 Отрицательная зависимость (p = {results['p']:.4f} < 0)")
    
    if results['R0'] > 1:
        print(f"   ⚠️ R0 = {results['R0']:.4f} > 1 — эпидемия может расти")
    else:
        print(f"   ✅ R0 = {results['R0']:.4f} ≤ 1 — эпидемия затухает")
    print("="*60)

def plot_regression_results(results, R_data, Y_data, title="Регрессия R(t) = R0 * Y(t)^p"):
    if results is None:
        return
    
    t = np.arange(len(R_data))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.scatter(results['log_Y'], results['log_R'], alpha=0.6, s=30, label='Данные')
    ax1.plot(results['log_Y'], results['log_R_pred'], 'r-', lw=2, 
             label=f'R² = {results["R2"]:.4f}, p = {results["p"]:.4f}')
    ax1.set_xlabel('ln(Y(t))', fontsize=12)
    ax1.set_ylabel('ln(R(t))', fontsize=12)
    ax1.set_title('log-log зависимость', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.scatter(t, R_data, alpha=0.6, s=30, label='Фактические данные')
    ax2.plot(t, results['R_pred'], 'r-', lw=2, label=f'R0 * Y^p')
    ax2.set_xlabel('Индекс', fontsize=12)
    ax2.set_ylabel('R(t)', fontsize=12)
    ax2.set_title('Сравнение модели с данными', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    error = np.abs(R_data - results['R_pred'])
    ax3.plot(t, error, 'g-', lw=2, label='Ошибка')
    ax3.axhline(np.mean(error), color='red', linestyle='--', alpha=0.7,
                label=f'Средняя ошибка = {np.mean(error):.4f}')
    ax3.set_xlabel('Индекс', fontsize=12)
    ax3.set_ylabel('|R_факт - R_пред|', fontsize=12)
    ax3.set_title('Ошибка предсказания', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.4)

    stats_text = f"""
    Параметры:
    R0 = {results['R0']:.6f}
    p  = {results['p']:.6f}
    
    Качество:
    R²  = {results['R2']:.6f}
    MAE = {results['MAE']:.6f}
    RMSE = {results['RMSE']:.6f}
    
    Интерпретация:
    {'Суперлинейный рост' if results['p'] > 1 else 
     'Линейная зависимость' if results['p'] == 1 else
     'Сублинейный рост' if results['p'] > 0 else
     'Отрицательная зависимость'}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax4.axis('off')
    ax4.set_title('Статистика', fontsize=14)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_lambda_with_regression(R0_data, I_total_data, title="Анализ λ с регрессией"):
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")
    
    results = calculate_lambda_regression(R0_data, I_total_data)
    
    if results is not None:
        print_regression_results(results)
        plot_regression_results(results, R0_data, I_total_data, title)
    
    return results

def analyze_lambda_with_regression_button():
    global current_best_R0_arr, current_I_fact, current_best_gamma, current_N
    
    try:
        if current_best_R0_arr is None or current_I_fact is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите JSON-файл и выполните симуляцию.")
            return
        
        if current_best_gamma is not None:
            gamma = np.mean(current_best_gamma)
        else:
            try:
                gamma = float(gamma_entry.get() or 0.074)
            except:
                gamma = 0.074

        R0_data = current_best_R0_arr
        I_new_data = current_I_fact

        I_total = calculate_total_from_new(I_new_data, gamma)

        R0_mean = np.mean(R0_data)
        R0_norm = R0_data / R0_mean
        
        print(f"\n📊 АНАЛИЗ λ С РЕГРЕССИЕЙ (Сеточная модель)")
        print(f"   γ = {gamma:.6f}")
        print(f"   R0_mean = {R0_mean:.4f}")
        print(f"   Точек = {len(R0_data)}")
        print("-"*70)

        results = analyze_lambda_with_regression(
            R0_norm, 
            I_total, 
            title="Сеточный анализ λ с регрессией R(t) = R0 * I_total(t)^p"
        )
        
        if results is not None:
            global last_regression_results
            last_regression_results = results
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

def analyze_lambda_unified(
    R0_data,
    Y_data,
    title="Анализ λ",
    source="данные",
    show_plot=True,
    return_stats=False
):
    
    if R0_data is None or Y_data is None:
        print("❌ Нет данных для анализа")
        return None, None, None
    
    mask = Y_data > 1e-9
    Y_log = np.log(Y_data[mask])
    R0_log = np.log(R0_data[mask])
    
    if len(Y_log) < 3:
        print("❌ Недостаточно данных для анализа λ (меньше 3 точек)")
        return None, None, None
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(Y_log, R0_log)
    
    lambda_est = slope
    R0_start = np.exp(intercept)
    R0_pred = np.exp(intercept + lambda_est * Y_log)
    r2 = r_value ** 2
    
    stats = {
        'lambda': lambda_est,
        'R0_start': R0_start,
        'r2': r2,
        'std_err': std_err,
        'n_points': len(Y_log),
        'p_value': p_value
    }
    
    print(f"\n📊 АНАЛИЗ λ ({source})")
    print(f"   λ = {lambda_est:.6f}")
    print(f"   R(0) = {R0_start:.6f}")
    print(f"   R² = {r2:.6f}")
    print(f"   Стандартная ошибка = {std_err:.6f}")
    print(f"   Точек = {len(Y_log)}")
    
    if show_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        ax1.scatter(Y_log, R0_log, alpha=0.6, s=20, label=source)
        ax1.plot(Y_log, intercept + lambda_est * Y_log, 'r-', lw=2, 
                 label=f'λ = {lambda_est:.4f}\nR(0) = {R0_start:.4f}\nR² = {r2:.4f}')
        ax1.set_xlabel('log(Y(t))', fontsize=12)
        ax1.set_ylabel('log(R0(t))', fontsize=12)
        ax1.set_title(f'{title}: log(R0) vs log(Y)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.5)
        
        Y_lambda = Y_data ** lambda_est
        ax2.scatter(Y_lambda[mask], R0_data[mask], alpha=0.6, s=20, label='Данные')
        ax2.plot(Y_lambda[mask], R0_start * Y_lambda[mask], 'r-', lw=2, 
                 label=f'R(t) = {R0_start:.4f} * Y(t)^{lambda_est:.4f}')
        ax2.set_xlabel(f'Y(t)^{lambda_est:.4f}', fontsize=12)
        ax2.set_ylabel('R0(t)', fontsize=12)
        ax2.set_title(f'{title}: R0 vs Y^λ', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.5)
        
        t = np.arange(len(R0_data))
        ax3.plot(t[mask], R0_data[mask], 'b-', lw=2, label='Фактический R0')
        ax3.plot(t[mask], R0_pred, 'r--', lw=2, label=f'Предсказанный R0 (λ={lambda_est:.4f})')
        ax3.axhline(R0_start, color='green', linestyle=':', alpha=0.7, label=f'R(0) = {R0_start:.4f}')
        ax3.set_xlabel('День', fontsize=12)
        ax3.set_ylabel('R0', fontsize=12)
        ax3.set_title(f'{title}: сравнение', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.5)
        
        error = np.abs(R0_data[mask] - R0_pred)
        ax4.plot(t[mask], error, 'g-', lw=2, label='Ошибка')
        ax4.axhline(np.mean(error), color='red', linestyle='--', alpha=0.7, 
                    label=f'Средняя ошибка = {np.mean(error):.4f}')
        ax4.set_xlabel('День', fontsize=12)
        ax4.set_ylabel('|R0_факт - R0_пред|', fontsize=12)
        ax4.set_title(f'{title}: ошибка предсказания', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.5)
        
        plt.suptitle(f'{title}: R(t) = R(0) * Y(t)^λ\nλ = {lambda_est:.4f}, R(0) = {R0_start:.4f}, R² = {r2:.4f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    if return_stats:
        return lambda_est, R0_start, r2, stats
    else:
        return lambda_est, R0_start, r2

def analyze_r0_normalized_unified(
    R0_data,
    Y_data,
    norm_by=None,
    title="R0 Normalization",
    source="data",
    show_plot=True,
    return_stats=False
):
    
    if R0_data is None or Y_data is None:
        print("❌ No data for analysis")
        return
    
    if norm_by is None:
        R0_norm_factor = np.mean(R0_data)
    else:
        R0_norm_factor = norm_by
    
    R0_norm = R0_data / R0_norm_factor
    
    print(f"\n📊 R0 NORMALIZATION ({source})")
    print(f"   Normalization factor: {R0_norm_factor:.4f}")
    print(f"   Mean R0 = {np.mean(R0_data):.6f}")
    print(f"   Mean normalized R0 = {np.mean(R0_norm):.6f}")
    
    mask = Y_data > 1e-9
    Y_log = np.log(Y_data[mask])
    R0_norm_log = np.log(R0_norm[mask])
    
    if len(Y_log) >= 3:
        from scipy.stats import linregress
        slope, intercept, r_value, _, std_err = linregress(Y_log, R0_norm_log)
        lambda_est = slope
        R0_start_norm = np.exp(intercept)
        R0_norm_pred = np.exp(intercept + lambda_est * Y_log)
        r2 = r_value ** 2
    else:
        print("⚠️ Insufficient data for nonlinearity analysis")
        lambda_est = None
        R0_start_norm = None
        r2 = None
    
    if show_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        t = np.arange(len(R0_data))
        
        ax1.plot(t, R0_data, 'b-', lw=2, label='R(t)')
        ax1.axhline(R0_norm_factor, color='red', linestyle='--', alpha=0.7, 
                   label=f'Normalization factor = {R0_norm_factor:.4f}')
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('R(t)', fontsize=12)
        ax1.set_title('R(t) dynamics', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.5)
        
        ax2.plot(t, R0_norm, 'g-', lw=2, label='R_norm(t) = R(t)/R_mean')
        if lambda_est is not None:
            ax2.plot(t[mask], R0_norm_pred, 'r--', lw=2, label=f'Predicted (λ={lambda_est:.4f})')
        ax2.axhline(1.0, color='red', linestyle=':', alpha=0.7, label='Normalized mean = 1')
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Normalized R(t)', fontsize=12)
        ax2.set_title('Normalized R(t) and predicted', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.5)
        
        if lambda_est is not None:
            ax3.scatter(Y_log, R0_norm_log, alpha=0.6, s=20, label='Data')
            ax3.plot(Y_log, intercept + lambda_est * Y_log, 'r-', lw=2, 
                    label=f'λ = {lambda_est:.4f}\nR(0)_norm = {R0_start_norm:.4f}\nR² = {r2:.4f}')
            ax3.set_xlabel('log(Y(t))', fontsize=12)
            ax3.set_ylabel('log(R_norm(t))', fontsize=12)
            ax3.set_title('Dependence of log(R_norm(t)) on log(Y)', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.5)
        
        if lambda_est is not None:
            error = np.abs(R0_norm[mask] - R0_norm_pred)
            ax4.plot(t[mask], error, 'orange', lw=2, label='Error')
            ax4.axhline(np.mean(error), color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean error = {np.mean(error):.4f}')
            ax4.set_xlabel('Day', fontsize=12)
            ax4.set_ylabel('Error', fontsize=12)
            ax4.set_title('Error of prediction of R_norm(t)', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.5)
        
        plt.suptitle(f'Identification of exponent in the incidence rate function\nλ = {lambda_est:.4f}, R(0)_norm = {R0_start_norm:.4f}, R² = {r2:.4f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    stats = {
        'R0_mean': np.mean(R0_data),
        'R0_std': np.std(R0_data),
        'R0_norm_mean': np.mean(R0_norm),
        'R0_norm_std': np.std(R0_norm),
        'R0_norm_factor': R0_norm_factor,
        'lambda': lambda_est,
        'R0_start_norm': R0_start_norm,
        'r2': r2
    }
    
    if return_stats:
        return stats
    
    return R0_norm

def calculate_total_from_new(new_cases, gamma):
    days = len(new_cases)
    total_cases = np.zeros(days)
    decay = np.exp(-gamma)
    
    total_cases[0] = new_cases[0]
    for t in range(1, days):
        total_cases[t] = total_cases[t-1] * decay + new_cases[t]
    
    return total_cases

def analyze_lambda_with_total(
    R0_data,
    I_new_data,
    gamma,
    title="Анализ λ с общей пораженностью",
    source="данные",
    show_plot=True,
    return_stats=False
):
    
    if R0_data is None or I_new_data is None:
        print("❌ Нет данных для анализа")
        return None, None, None
    
    I_total = calculate_total_from_new(I_new_data, gamma)
    
    mask = I_total > 1e-9
    I_log = np.log(I_total[mask])
    R0_log = np.log(R0_data[mask])
    
    if len(I_log) < 3:
        print("❌ Недостаточно данных для анализа λ (меньше 3 точек)")
        return None, None, None
    
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(I_log, R0_log)
    
    lambda_est = slope
    R0_start = np.exp(intercept)
    R0_pred = np.exp(intercept + lambda_est * I_log)
    r2 = r_value ** 2
    
    stats = {
        'lambda': lambda_est,
        'R0_start': R0_start,
        'r2': r2,
        'std_err': std_err,
        'n_points': len(I_log),
        'p_value': p_value,
        'gamma': gamma
    }
    
    print(f"\n📊 АНАЛИЗ λ С ОБЩЕЙ ПОРАЖЕННОСТЬЮ ({source})")
    print(f"   γ = {gamma:.6f}")
    print(f"   λ = {lambda_est:.6f}")
    print(f"   R(0) = {R0_start:.6f}")
    print(f"   R² = {r2:.6f}")
    print(f"   Стандартная ошибка = {std_err:.6f}")
    print(f"   Точек = {len(I_log)}")
    
    if show_plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        t = np.arange(len(R0_data))
        
        ax1.scatter(I_log, R0_log, alpha=0.6, s=20, label=source)
        ax1.plot(I_log, intercept + lambda_est * I_log, 'r-', lw=2, 
                 label=f'λ = {lambda_est:.4f}\nR(0) = {R0_start:.4f}\nR² = {r2:.4f}')
        ax1.set_xlabel('log(I_total(t))', fontsize=12)
        ax1.set_ylabel('log(R0(t))', fontsize=12)
        ax1.set_title(f'{title}: log(R0) vs log(I_total)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.5)
        
        ax2.plot(t[mask], R0_data[mask], 'b-', lw=2, label='Фактический R0')
        ax2.plot(t[mask], R0_pred, 'r--', lw=2, label=f'Предсказанный R0 (λ={lambda_est:.4f})')
        ax2.axhline(R0_start, color='green', linestyle=':', alpha=0.7, label=f'R(0) = {R0_start:.4f}')
        ax2.set_xlabel('День', fontsize=12)
        ax2.set_ylabel('R0', fontsize=12)
        ax2.set_title(f'{title}: сравнение', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.5)
        
        ax3.plot(t, I_new_data, 'b-', lw=1.5, label='Новые случаи')
        ax3.plot(t, I_total, 'r-', lw=2, label='Общая пораженность')
        ax3.set_xlabel('День', fontsize=12)
        ax3.set_ylabel('Количество', fontsize=12)
        ax3.set_title(f'Сравнение новых случаев и общей пораженности (γ={gamma:.4f})', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.5)
        
        error = np.abs(R0_data[mask] - R0_pred)
        ax4.plot(t[mask], error, 'g-', lw=2, label='Ошибка')
        ax4.axhline(np.mean(error), color='red', linestyle='--', alpha=0.7, 
                    label=f'Средняя ошибка = {np.mean(error):.4f}')
        ax4.set_xlabel('День', fontsize=12)
        ax4.set_ylabel('|R0_факт - R0_пред|', fontsize=12)
        ax4.set_title(f'{title}: ошибка предсказания', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.5)
        
        plt.suptitle(f'{title}: R(t) = R(0) * I_total(t)^λ\nγ = {gamma:.4f}, λ = {lambda_est:.4f}, R(0) = {R0_start:.4f}, R² = {r2:.4f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    if return_stats:
        return lambda_est, R0_start, r2, stats
    else:
        return lambda_est, R0_start, r2

def show_table_unified(
    data,
    headers,
    title="Таблица",
    width=1100,
    height=750,
    font_size=9,
    col_widths=None,
    copyable=True,
    exportable=True
):
    window = tk.Toplevel(root)
    window.title(title)
    window.geometry(f"{width}x{height}")
    
    text_widget = scrolledtext.ScrolledText(window, font=("Consolas", font_size))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    text = ""
    
    for i, header in enumerate(headers):
        col_width = col_widths.get(i, 15) if col_widths else 15
        text += f"{header:<{col_width}}"
    text += "\n"
    text += "-" * 85 + "\n"
    
    for row in data:
        for i, cell in enumerate(row):
            col_width = col_widths.get(i, 15) if col_widths else 15
            text += f"{cell:<{col_width}}"
        text += "\n"
    
    text_widget.insert(tk.END, text)
    text_widget.config(state=tk.DISABLED)
    
    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=5)
    
    if copyable:
        def copy_to_clipboard():
            window.clipboard_clear()
            window.clipboard_append(text)
            messagebox.showinfo("Скопировано", "Таблица скопирована в буфер обмена")
        
        tk.Button(btn_frame, text="📋 Копировать", font=("Arial", 10),
                  command=copy_to_clipboard).pack(side=tk.LEFT, padx=5)
    
    if exportable:
        def export_to_excel():
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
                if file_path:
                    df = pd.DataFrame(data, columns=headers)
                    df.to_excel(file_path, index=False)
                    messagebox.showinfo("Успешно", f"Таблица сохранена в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
        
        tk.Button(btn_frame, text="📊 Экспорт в Excel", font=("Arial", 10),
                  command=export_to_excel).pack(side=tk.LEFT, padx=5)
    
    tk.Button(btn_frame, text="❌ Закрыть", font=("Arial", 10),
              command=window.destroy).pack(side=tk.LEFT, padx=5)
    
    return window

def analyze_lambda_split(
    R0_data,
    Y_data,
    title="Анализ λ (раздельный)",
    source="данные",
    show_plot=True,
    return_stats=False
):
    
    if R0_data is None or Y_data is None:
        print("❌ Нет данных для анализа")
        return None, None, None
    
    R0_data = np.array(R0_data)
    Y_data = np.array(Y_data)
    
    min_len = min(len(R0_data), len(Y_data))
    
    if len(R0_data) != len(Y_data):
        print(f"⚠️ ВНИМАНИЕ: R0_data ({len(R0_data)}) и Y_data ({len(Y_data)}) имеют разную длину!")
        print(f"   Обрезаем до {min_len} точек")
        R0_data = R0_data[:min_len]
        Y_data = Y_data[:min_len]
    
    mask_up = np.zeros(len(Y_data), dtype=bool)
    mask_down = np.zeros(len(Y_data), dtype=bool)
    
    for t in range(1, len(Y_data)):
        if Y_data[t] > Y_data[t-1]:
            mask_up[t] = True
        elif Y_data[t] < Y_data[t-1]:
            mask_down[t] = True
    
    mask_up = mask_up & (Y_data > 1e-9)
    mask_down = mask_down & (Y_data > 1e-9)
    
    Y_up = Y_data[mask_up]
    R0_up = R0_data[mask_up]
    Y_down = Y_data[mask_down]
    R0_down = R0_data[mask_down]
    
    mask_all = Y_data > 1e-9
    Y_all = Y_data[mask_all]
    R0_all = R0_data[mask_all]
    
    R0_prime = np.zeros(len(R0_data))
    for t in range(1, len(R0_data)):
        dt = 1.0
        R0_prime[t] = (R0_data[t] - R0_data[t-1]) / dt
    R0_prime[0] = R0_prime[1] if len(R0_prime) > 1 else 0
    
    R0_prime_up = R0_prime[mask_up]
    R0_prime_down = R0_prime[mask_down]
    R0_prime_all = R0_prime[mask_all]
    
    R0_prime_positive = R0_prime_all[R0_prime_all > 0]
    R0_prime_negative = R0_prime_all[R0_prime_all < 0]
    R0_prime_zero = R0_prime_all[np.abs(R0_prime_all) < 1e-10]
    
    mean_positive_all = np.mean(R0_prime_positive) if len(R0_prime_positive) > 0 else 0
    mean_negative_all = np.mean(R0_prime_negative) if len(R0_prime_negative) > 0 else 0
    std_positive_all = np.std(R0_prime_positive) if len(R0_prime_positive) > 0 else 0
    std_negative_all = np.std(R0_prime_negative) if len(R0_prime_negative) > 0 else 0
    max_positive_all = np.max(R0_prime_positive) if len(R0_prime_positive) > 0 else 0
    min_negative_all = np.min(R0_prime_negative) if len(R0_prime_negative) > 0 else 0
    
    R0_prime_up_positive = R0_prime_up[R0_prime_up > 0]
    R0_prime_up_negative = R0_prime_up[R0_prime_up < 0]
    
    mean_positive_up = np.mean(R0_prime_up_positive) if len(R0_prime_up_positive) > 0 else 0
    mean_negative_up = np.mean(R0_prime_up_negative) if len(R0_prime_up_negative) > 0 else 0
    std_positive_up = np.std(R0_prime_up_positive) if len(R0_prime_up_positive) > 0 else 0
    std_negative_up = np.std(R0_prime_up_negative) if len(R0_prime_up_negative) > 0 else 0
    
    R0_prime_down_positive = R0_prime_down[R0_prime_down > 0]
    R0_prime_down_negative = R0_prime_down[R0_prime_down < 0]
    
    mean_positive_down = np.mean(R0_prime_down_positive) if len(R0_prime_down_positive) > 0 else 0
    mean_negative_down = np.mean(R0_prime_down_negative) if len(R0_prime_down_negative) > 0 else 0
    std_positive_down = np.std(R0_prime_down_positive) if len(R0_prime_down_positive) > 0 else 0
    std_negative_down = np.std(R0_prime_down_negative) if len(R0_prime_down_negative) > 0 else 0
    
    print(f"\n📊 АНАЛИЗ λ С РАЗДЕЛЕНИЕМ ПО ТРЕНДАМ")
    print("="*60)
    print(f"   Длина R0_data: {len(R0_data)}")
    print(f"   Длина Y_data:  {len(Y_data)}")
    print(f"   Всего точек: {len(Y_all)}")
    print(f"   Восходящий тренд (Y↑): {len(Y_up)} точек ({len(Y_up)/len(Y_all)*100:.1f}%)")
    print(f"   Нисходящий тренд (Y↓): {len(Y_down)} точек ({len(Y_down)/len(Y_all)*100:.1f}%)")
    print("="*60)
    
    print(f"\n📊 СТАТИСТИКА ПРОИЗВОДНОЙ R'(t):")
    print("-"*40)
    print(f"\n🔺 ВСЕ ДАННЫЕ:")
    print(f"   Положительные (R' > 0): {len(R0_prime_positive)} точек")
    print(f"     Среднее = {mean_positive_all:.6f} ± {std_positive_all:.6f}")
    print(f"     Максимум = {max_positive_all:.6f}")
    print(f"   Отрицательные (R' < 0): {len(R0_prime_negative)} точек")
    print(f"     Среднее = {mean_negative_all:.6f} ± {std_negative_all:.6f}")
    print(f"     Минимум = {min_negative_all:.6f}")
    print(f"   Нулевые (R' ≈ 0): {len(R0_prime_zero)} точек")
    
    print(f"\n🔺 ВОСХОДЯЩИЙ ТРЕНД (Y↑):")
    print(f"   Положительные (R' > 0): {len(R0_prime_up_positive)} точек")
    print(f"     Среднее = {mean_positive_up:.6f} ± {std_positive_up:.6f}")
    print(f"   Отрицательные (R' < 0): {len(R0_prime_up_negative)} точек")
    print(f"     Среднее = {mean_negative_up:.6f} ± {std_negative_up:.6f}")
    
    print(f"\n🔺 НИСХОДЯЩИЙ ТРЕНД (Y↓):")
    print(f"   Положительные (R' > 0): {len(R0_prime_down_positive)} точек")
    print(f"     Среднее = {mean_positive_down:.6f} ± {std_positive_down:.6f}")
    print(f"   Отрицательные (R' < 0): {len(R0_prime_down_negative)} точек")
    print(f"     Среднее = {mean_negative_down:.6f} ± {std_negative_down:.6f}")
    print("="*60)
    
    from scipy.stats import linregress
    
    results = {}
    
    if len(Y_all) >= 3:
        Y_log_all = np.log(Y_all)
        R0_log_all = np.log(R0_all)
        slope_all, intercept_all, r_value_all, p_value_all, std_err_all = linregress(Y_log_all, R0_log_all)
        results['all'] = {
            'lambda': slope_all,
            'R0_start': np.exp(intercept_all),
            'r2': r_value_all ** 2,
            'std_err': std_err_all,
            'p_value': p_value_all,
            'n_points': len(Y_all),
            'slope': slope_all,
            'intercept': intercept_all
        }
        print(f"\n📈 ВСЕ ДАННЫЕ:")
        print(f"   λ = {slope_all:.6f}")
        print(f"   R(0) = {np.exp(intercept_all):.6f}")
        print(f"   R² = {r_value_all**2:.6f}")
    else:
        results['all'] = None
        print("\n⚠️ Недостаточно данных для анализа (все)")
    
    if len(Y_up) >= 3:
        Y_log_up = np.log(Y_up)
        R0_log_up = np.log(R0_up)
        slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(Y_log_up, R0_log_up)
        results['up'] = {
            'lambda': slope_up,
            'R0_start': np.exp(intercept_up),
            'r2': r_value_up ** 2,
            'std_err': std_err_up,
            'p_value': p_value_up,
            'n_points': len(Y_up),
            'slope': slope_up,
            'intercept': intercept_up
        }
        print(f"\n📈 ВОСХОДЯЩИЙ ТРЕНД (Y↑):")
        print(f"   λ_up = {slope_up:.6f}")
        print(f"   R(0)_up = {np.exp(intercept_up):.6f}")
        print(f"   R²_up = {r_value_up**2:.6f}")
    else:
        results['up'] = None
        print("\n⚠️ Недостаточно данных для анализа (восходящий тренд)")
    
    if len(Y_down) >= 3:
        Y_log_down = np.log(Y_down)
        R0_log_down = np.log(R0_down)
        slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(Y_log_down, R0_log_down)
        results['down'] = {
            'lambda': slope_down,
            'R0_start': np.exp(intercept_down),
            'r2': r_value_down ** 2,
            'std_err': std_err_down,
            'p_value': p_value_down,
            'n_points': len(Y_down),
            'slope': slope_down,
            'intercept': intercept_down
        }
        print(f"\n📈 НИСХОДЯЩИЙ ТРЕНД (Y↓):")
        print(f"   λ_down = {slope_down:.6f}")
        print(f"   R(0)_down = {np.exp(intercept_down):.6f}")
        print(f"   R²_down = {r_value_down**2:.6f}")
    else:
        results['down'] = None
        print("\n⚠️ Недостаточно данных для анализа (нисходящий тренд)")
    
    if show_plot:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        t = np.arange(len(R0_data))
        
        ax1 = axs[0, 0]
        ax1.plot(t, R0_data, 'k-', lw=1.5, alpha=0.5, label='R(t)')
        if len(Y_up) > 0:
            t_up = t[mask_up]
            ax1.scatter(t_up, R0_up, c='green', s=30, alpha=0.7, label=f'Y↑ ({len(Y_up)} точек)')
        if len(Y_down) > 0:
            t_down = t[mask_down]
            ax1.scatter(t_down, R0_down, c='red', s=30, alpha=0.7, label=f'Y↓ ({len(Y_down)} точек)')
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R0 = 1')
        ax1.set_xlabel('День', fontsize=12)
        ax1.set_ylabel('R(t)', fontsize=12)
        ax1.set_title('R(t) с выделением трендов Y(t)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        
        ax2 = axs[0, 1]
        ax2.plot(t, R0_prime, 'b-', lw=2, label="R'(t)")
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label="R'(t) = 0")
        if len(R0_prime_positive) > 0:
            ax2.axhline(mean_positive_all, color='green', linestyle=':', alpha=0.8, 
                       linewidth=2, label=f'Среднее R\' > 0 = {mean_positive_all:.5f}')
        if len(R0_prime_negative) > 0:
            ax2.axhline(mean_negative_all, color='orange', linestyle=':', alpha=0.8, 
                       linewidth=2, label=f'Среднее R\' < 0 = {mean_negative_all:.5f}')
        if len(Y_up) > 0:
            ax2.scatter(t_up, R0_prime_up, c='green', s=20, alpha=0.6, label='Y↑')
        if len(Y_down) > 0:
            ax2.scatter(t_down, R0_prime_down, c='red', s=20, alpha=0.6, label='Y↓')
        ax2.set_xlabel('День', fontsize=12)
        ax2.set_ylabel("R'(t)", fontsize=12)
        ax2.set_title("Производная R'(t) со средними значениями", fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.4)
        
        ax3 = axs[0, 2]
        if len(Y_up) >= 3:
            ax3.scatter(Y_log_up, R0_log_up, c='green', s=30, alpha=0.7, label='Y↑')
            ax3.plot(Y_log_up, intercept_up + slope_up * Y_log_up, 'g-', lw=2,
                    label=f'λ_up = {slope_up:.4f}, R²_up = {r_value_up**2:.4f}')
        if len(Y_down) >= 3:
            ax3.scatter(Y_log_down, R0_log_down, c='red', s=30, alpha=0.7, label='Y↓')
            ax3.plot(Y_log_down, intercept_down + slope_down * Y_log_down, 'r-', lw=2,
                    label=f'λ_down = {slope_down:.4f}, R²_down = {r_value_down**2:.4f}')
        if len(Y_all) >= 3:
            ax3.scatter(Y_log_all, R0_log_all, c='blue', s=10, alpha=0.3, label='Все данные')
            ax3.plot(Y_log_all, intercept_all + slope_all * Y_log_all, 'b--', lw=1.5,
                    label=f'λ_all = {slope_all:.4f}, R²_all = {r_value_all**2:.4f}')
        ax3.set_xlabel('log(Y(t))', fontsize=12)
        ax3.set_ylabel('log(R(t))', fontsize=12)
        ax3.set_title('log-log зависимость с разделением', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.4)
        
        ax4 = axs[1, 0]
        labels = []
        lambdas = []
        r2s = []
        colors = []
        if results['all'] is not None:
            labels.append('Все')
            lambdas.append(results['all']['lambda'])
            r2s.append(results['all']['r2'])
            colors.append('blue')
        if results['up'] is not None:
            labels.append('Y↑')
            lambdas.append(results['up']['lambda'])
            r2s.append(results['up']['r2'])
            colors.append('green')
        if results['down'] is not None:
            labels.append('Y↓')
            lambdas.append(results['down']['lambda'])
            r2s.append(results['down']['r2'])
            colors.append('red')
        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax4.bar(x - width/2, lambdas, width, color=colors, alpha=0.7, edgecolor='black', label='λ')
        for bar, val, r2 in zip(bars1, lambdas, r2s):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'λ={val:.4f}\nR²={r2:.4f}', ha='center', va='bottom', fontsize=9)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('λ', fontsize=12)
        ax4.set_title('Сравнение λ для разных трендов', fontsize=14)
        ax4.grid(True, alpha=0.4, axis='y')
        
        ax5 = axs[1, 1]
        ax5.plot(t, Y_data, 'b-', lw=1.5, alpha=0.5, label='Y(t)')
        if len(Y_up) > 0:
            ax5.scatter(t_up, Y_up, c='green', s=20, alpha=0.7, label='Y↑')
        if len(Y_down) > 0:
            ax5.scatter(t_down, Y_down, c='red', s=20, alpha=0.7, label='Y↓')
        ax5.set_xlabel('День', fontsize=12)
        ax5.set_ylabel('Y(t)', fontsize=12)
        ax5.set_title('Динамика Y(t) с выделением трендов', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.4)
        
        ax6 = axs[1, 2]
        mask_all_indices = np.where(mask_all)[0]
        if len(R0_prime_all) > 0:
            valid_mask = np.abs(R0_prime_all) < np.percentile(np.abs(R0_prime_all), 99)
            valid_indices = mask_all_indices[valid_mask]
            ax6.scatter(Y_data[valid_indices], R0_prime[valid_indices], 
                       c='blue', s=10, alpha=0.3, label='Все точки')
        if len(Y_up) > 0 and len(R0_prime_up) > 0:
            mask_up_indices = np.where(mask_up)[0]
            valid_mask_up = np.abs(R0_prime_up) < np.percentile(np.abs(R0_prime_up), 99)
            valid_indices_up = mask_up_indices[valid_mask_up]
            ax6.scatter(Y_data[valid_indices_up], R0_prime[valid_indices_up], 
                       c='green', s=20, alpha=0.6, label='Y↑')
        if len(Y_down) > 0 and len(R0_prime_down) > 0:
            mask_down_indices = np.where(mask_down)[0]
            valid_mask_down = np.abs(R0_prime_down) < np.percentile(np.abs(R0_prime_down), 99)
            valid_indices_down = mask_down_indices[valid_mask_down]
            ax6.scatter(Y_data[valid_indices_down], R0_prime[valid_indices_down], 
                       c='red', s=20, alpha=0.6, label='Y↓')
        ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Y(t)', fontsize=12)
        ax6.set_ylabel("R'(t)", fontsize=12)
        ax6.set_title("Зависимость R'(t) от Y(t)", fontsize=14)
        ax6.legend()
        ax6.grid(True, alpha=0.4)
        
        plt.suptitle(f'{title}\nλ_all = {results["all"]["lambda"] if results["all"] is not None else "—"} | λ_up = {results["up"]["lambda"] if results["up"] is not None else "—"} | λ_down = {results["down"]["lambda"] if results["down"] is not None else "—"}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    if return_stats:
        return results
    else:
        return results['all']['lambda'] if results['all'] is not None else None,               results['all']['R0_start'] if results['all'] is not None else None,               results['all']['r2'] if results['all'] is not None else None, results

def analyze_lambda_grid_split_with_total():
    global current_best_R0_arr, current_I_fact, current_best_gamma
    
    try:
        if current_best_R0_arr is None or current_I_fact is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите JSON-файл и выполните симуляцию.")
            return
        
        if current_best_gamma is not None:
            gamma = np.mean(current_best_gamma)
        else:
            try:
                gamma = float(gamma_entry.get() or 0.074)
            except:
                gamma = 0.074
        
        min_len = min(len(current_best_R0_arr), len(current_I_fact))
        
        if len(current_best_R0_arr) != len(current_I_fact):
            print(f"⚠️ ВНИМАНИЕ: R0 ({len(current_best_R0_arr)}) и I_fact ({len(current_I_fact)}) имеют разную длину!")
            print(f"   Обрезаем до {min_len} точек")
            R0_trimmed = current_best_R0_arr[:min_len]
            I_fact_trimmed = current_I_fact[:min_len]
        else:
            R0_trimmed = current_best_R0_arr
            I_fact_trimmed = current_I_fact
        
        R0_mean = np.mean(R0_trimmed)
        R0_norm = R0_trimmed / R0_mean
        
        analyze_lambda_split_with_total(
            R0_norm,
            I_fact_trimmed,
            gamma=gamma,
            title="Сеточный анализ λ (раздельный по трендам, общая пораженность)",
            source="Grid модель"
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

def analyze_lambda_split_with_total(
    R0_data,
    I_new_data,
    gamma,
    title="Анализ λ с разделением по трендам (общая пораженность)",
    source="данные",
    show_plot=True,
    return_stats=False
):
    
    if R0_data is None or I_new_data is None:
        print("❌ Нет данных для анализа")
        return None, None, None
    
    I_total = calculate_total_from_new(I_new_data, gamma)
    Y_data = I_total
    
    R0_data = np.array(R0_data)
    Y_data = np.array(Y_data)
    
    min_len = min(len(R0_data), len(Y_data))
    
    if len(R0_data) != len(Y_data):
        print(f"⚠️ ВНИМАНИЕ: R0_data ({len(R0_data)}) и Y_data ({len(Y_data)}) имеют разную длину!")
        print(f"   Обрезаем до {min_len} точек")
        R0_data = R0_data[:min_len]
        Y_data = Y_data[:min_len]
    
    mask_up = np.zeros(len(Y_data), dtype=bool)
    mask_down = np.zeros(len(Y_data), dtype=bool)
    
    for t in range(1, len(Y_data)):
        if Y_data[t] > Y_data[t-1]:
            mask_up[t] = True
        elif Y_data[t] < Y_data[t-1]:
            mask_down[t] = True
    
    mask_up = mask_up & (Y_data > 1e-9)
    mask_down = mask_down & (Y_data > 1e-9)
    
    Y_up = Y_data[mask_up]
    R0_up = R0_data[mask_up]
    Y_down = Y_data[mask_down]
    R0_down = R0_data[mask_down]
    
    mask_all = Y_data > 1e-9
    Y_all = Y_data[mask_all]
    R0_all = R0_data[mask_all]
    
    R0_prime = np.zeros(len(R0_data))
    for t in range(1, len(R0_data)):
        dt = 1.0
        R0_prime[t] = (R0_data[t] - R0_data[t-1]) / dt
    R0_prime[0] = R0_prime[1] if len(R0_prime) > 1 else 0
    
    R0_prime_up = R0_prime[mask_up]
    R0_prime_down = R0_prime[mask_down]
    R0_prime_all = R0_prime[mask_all]
    
    R0_prime_positive = R0_prime_all[R0_prime_all > 0]
    R0_prime_negative = R0_prime_all[R0_prime_all < 0]
    R0_prime_zero = R0_prime_all[np.abs(R0_prime_all) < 1e-10]
    
    mean_positive_all = np.mean(R0_prime_positive) if len(R0_prime_positive) > 0 else 0
    mean_negative_all = np.mean(R0_prime_negative) if len(R0_prime_negative) > 0 else 0
    std_positive_all = np.std(R0_prime_positive) if len(R0_prime_positive) > 0 else 0
    std_negative_all = np.std(R0_prime_negative) if len(R0_prime_negative) > 0 else 0
    max_positive_all = np.max(R0_prime_positive) if len(R0_prime_positive) > 0 else 0
    min_negative_all = np.min(R0_prime_negative) if len(R0_prime_negative) > 0 else 0
    
    R0_prime_up_positive = R0_prime_up[R0_prime_up > 0]
    R0_prime_up_negative = R0_prime_up[R0_prime_up < 0]
    
    mean_positive_up = np.mean(R0_prime_up_positive) if len(R0_prime_up_positive) > 0 else 0
    mean_negative_up = np.mean(R0_prime_up_negative) if len(R0_prime_up_negative) > 0 else 0
    std_positive_up = np.std(R0_prime_up_positive) if len(R0_prime_up_positive) > 0 else 0
    std_negative_up = np.std(R0_prime_up_negative) if len(R0_prime_up_negative) > 0 else 0
    
    R0_prime_down_positive = R0_prime_down[R0_prime_down > 0]
    R0_prime_down_negative = R0_prime_down[R0_prime_down < 0]
    
    mean_positive_down = np.mean(R0_prime_down_positive) if len(R0_prime_down_positive) > 0 else 0
    mean_negative_down = np.mean(R0_prime_down_negative) if len(R0_prime_down_negative) > 0 else 0
    std_positive_down = np.std(R0_prime_down_positive) if len(R0_prime_down_positive) > 0 else 0
    std_negative_down = np.std(R0_prime_down_negative) if len(R0_prime_down_negative) > 0 else 0
    
    print(f"\n📊 АНАЛИЗ λ С РАЗДЕЛЕНИЕМ ПО ТРЕНДАМ (ОБЩАЯ ПОРАЖЕННОСТЬ)")
    print("="*60)
    print(f"   γ = {gamma:.6f}")
    print(f"   Длина R0_data: {len(R0_data)}")
    print(f"   Длина Y_data:  {len(Y_data)}")
    print(f"   Всего точек: {len(Y_all)}")
    print(f"   Восходящий тренд (Y↑): {len(Y_up)} точек ({len(Y_up)/len(Y_all)*100:.1f}%)")
    print(f"   Нисходящий тренд (Y↓): {len(Y_down)} точек ({len(Y_down)/len(Y_all)*100:.1f}%)")
    print("="*60)
    
    print(f"\n📊 СТАТИСТИКА ПРОИЗВОДНОЙ R'(t):")
    print("-"*40)
    print(f"\n🔺 ВСЕ ДАННЫЕ:")
    print(f"   Положительные (R' > 0): {len(R0_prime_positive)} точек")
    print(f"     Среднее = {mean_positive_all:.6f} ± {std_positive_all:.6f}")
    print(f"     Максимум = {max_positive_all:.6f}")
    print(f"   Отрицательные (R' < 0): {len(R0_prime_negative)} точек")
    print(f"     Среднее = {mean_negative_all:.6f} ± {std_negative_all:.6f}")
    print(f"     Минимум = {min_negative_all:.6f}")
    print(f"   Нулевые (R' ≈ 0): {len(R0_prime_zero)} точек")
    
    print(f"\n🔺 ВОСХОДЯЩИЙ ТРЕНД (Y↑):")
    print(f"   Положительные (R' > 0): {len(R0_prime_up_positive)} точек")
    print(f"     Среднее = {mean_positive_up:.6f} ± {std_positive_up:.6f}")
    print(f"   Отрицательные (R' < 0): {len(R0_prime_up_negative)} точек")
    print(f"     Среднее = {mean_negative_up:.6f} ± {std_negative_up:.6f}")
    
    print(f"\n🔺 НИСХОДЯЩИЙ ТРЕНД (Y↓):")
    print(f"   Положительные (R' > 0): {len(R0_prime_down_positive)} точек")
    print(f"     Среднее = {mean_positive_down:.6f} ± {std_positive_down:.6f}")
    print(f"   Отрицательные (R' < 0): {len(R0_prime_down_negative)} точек")
    print(f"     Среднее = {mean_negative_down:.6f} ± {std_negative_down:.6f}")
    print("="*60)
    
    from scipy.stats import linregress
    
    results = {}
    
    if len(Y_all) >= 3:
        Y_log_all = np.log(Y_all)
        R0_log_all = np.log(R0_all)
        slope_all, intercept_all, r_value_all, p_value_all, std_err_all = linregress(Y_log_all, R0_log_all)
        results['all'] = {
            'lambda': slope_all,
            'R0_start': np.exp(intercept_all),
            'r2': r_value_all ** 2,
            'std_err': std_err_all,
            'p_value': p_value_all,
            'n_points': len(Y_all),
            'slope': slope_all,
            'intercept': intercept_all
        }
        print(f"\n📈 ВСЕ ДАННЫЕ:")
        print(f"   λ = {slope_all:.6f}")
        print(f"   R(0) = {np.exp(intercept_all):.6f}")
        print(f"   R² = {r_value_all**2:.6f}")
    else:
        results['all'] = None
        print("\n⚠️ Недостаточно данных для анализа (все)")
    
    if len(Y_up) >= 3:
        Y_log_up = np.log(Y_up)
        R0_log_up = np.log(R0_up)
        slope_up, intercept_up, r_value_up, p_value_up, std_err_up = linregress(Y_log_up, R0_log_up)
        results['up'] = {
            'lambda': slope_up,
            'R0_start': np.exp(intercept_up),
            'r2': r_value_up ** 2,
            'std_err': std_err_up,
            'p_value': p_value_up,
            'n_points': len(Y_up),
            'slope': slope_up,
            'intercept': intercept_up
        }
        print(f"\n📈 ВОСХОДЯЩИЙ ТРЕНД (Y↑):")
        print(f"   λ_up = {slope_up:.6f}")
        print(f"   R(0)_up = {np.exp(intercept_up):.6f}")
        print(f"   R²_up = {r_value_up**2:.6f}")
    else:
        results['up'] = None
        print("\n⚠️ Недостаточно данных для анализа (восходящий тренд)")
    
    if len(Y_down) >= 3:
        Y_log_down = np.log(Y_down)
        R0_log_down = np.log(R0_down)
        slope_down, intercept_down, r_value_down, p_value_down, std_err_down = linregress(Y_log_down, R0_log_down)
        results['down'] = {
            'lambda': slope_down,
            'R0_start': np.exp(intercept_down),
            'r2': r_value_down ** 2,
            'std_err': std_err_down,
            'p_value': p_value_down,
            'n_points': len(Y_down),
            'slope': slope_down,
            'intercept': intercept_down
        }
        print(f"\n📈 НИСХОДЯЩИЙ ТРЕНД (Y↓):")
        print(f"   λ_down = {slope_down:.6f}")
        print(f"   R(0)_down = {np.exp(intercept_down):.6f}")
        print(f"   R²_down = {r_value_down**2:.6f}")
    else:
        results['down'] = None
        print("\n⚠️ Недостаточно данных для анализа (нисходящий тренд)")
    
    if show_plot:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        t = np.arange(len(R0_data))
        
        ax1 = axs[0, 0]
        ax1.plot(t, R0_data, 'k-', lw=1.5, alpha=0.5, label='R(t)')
        if len(Y_up) > 0:
            t_up = t[mask_up]
            ax1.scatter(t_up, R0_up, c='green', s=30, alpha=0.7, label=f'I_total↑ ({len(Y_up)} точек)')
        if len(Y_down) > 0:
            t_down = t[mask_down]
            ax1.scatter(t_down, R0_down, c='red', s=30, alpha=0.7, label=f'I_total↓ ({len(Y_down)} точек)')
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R0 = 1')
        ax1.set_xlabel('День', fontsize=12)
        ax1.set_ylabel('R(t)', fontsize=12)
        ax1.set_title('R(t) с выделением трендов I_total(t)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        
        ax2 = axs[0, 1]
        ax2.plot(t, R0_prime, 'b-', lw=2, label="R'(t)")
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label="R'(t) = 0")
        if len(R0_prime_positive) > 0:
            ax2.axhline(mean_positive_all, color='green', linestyle=':', alpha=0.8, 
                       linewidth=2, label=f'Среднее R\' > 0 = {mean_positive_all:.5f}')
        if len(R0_prime_negative) > 0:
            ax2.axhline(mean_negative_all, color='orange', linestyle=':', alpha=0.8, 
                       linewidth=2, label=f'Среднее R\' < 0 = {mean_negative_all:.5f}')
        if len(Y_up) > 0:
            ax2.scatter(t_up, R0_prime_up, c='green', s=20, alpha=0.6, label='I_total↑')
        if len(Y_down) > 0:
            ax2.scatter(t_down, R0_prime_down, c='red', s=20, alpha=0.6, label='I_total↓')
        ax2.set_xlabel('День', fontsize=12)
        ax2.set_ylabel("R'(t)", fontsize=12)
        ax2.set_title("Производная R'(t) со средними значениями", fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.4)
        
        ax3 = axs[0, 2]
        if len(Y_up) >= 3:
            ax3.scatter(Y_log_up, R0_log_up, c='green', s=30, alpha=0.7, label='I_total↑')
            ax3.plot(Y_log_up, intercept_up + slope_up * Y_log_up, 'g-', lw=2,
                    label=f'λ_up = {slope_up:.4f}, R²_up = {r_value_up**2:.4f}')
        if len(Y_down) >= 3:
            ax3.scatter(Y_log_down, R0_log_down, c='red', s=30, alpha=0.7, label='I_total↓')
            ax3.plot(Y_log_down, intercept_down + slope_down * Y_log_down, 'r-', lw=2,
                    label=f'λ_down = {slope_down:.4f}, R²_down = {r_value_down**2:.4f}')
        if len(Y_all) >= 3:
            ax3.scatter(Y_log_all, R0_log_all, c='blue', s=10, alpha=0.3, label='Все данные')
            ax3.plot(Y_log_all, intercept_all + slope_all * Y_log_all, 'b--', lw=1.5,
                    label=f'λ_all = {slope_all:.4f}, R²_all = {r_value_all**2:.4f}')
        ax3.set_xlabel('log(I_total(t))', fontsize=12)
        ax3.set_ylabel('log(R(t))', fontsize=12)
        ax3.set_title('log-log зависимость I_total с разделением', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.4)
        
        ax4 = axs[1, 0]
        labels = []
        lambdas = []
        r2s = []
        colors = []
        if results['all'] is not None:
            labels.append('Все')
            lambdas.append(results['all']['lambda'])
            r2s.append(results['all']['r2'])
            colors.append('blue')
        if results['up'] is not None:
            labels.append('I_total↑')
            lambdas.append(results['up']['lambda'])
            r2s.append(results['up']['r2'])
            colors.append('green')
        if results['down'] is not None:
            labels.append('I_total↓')
            lambdas.append(results['down']['lambda'])
            r2s.append(results['down']['r2'])
            colors.append('red')
        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax4.bar(x - width/2, lambdas, width, color=colors, alpha=0.7, edgecolor='black', label='λ')
        for bar, val, r2 in zip(bars1, lambdas, r2s):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'λ={val:.4f}\nR²={r2:.4f}', ha='center', va='bottom', fontsize=9)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('λ', fontsize=12)
        ax4.set_title('Сравнение λ для разных трендов (I_total)', fontsize=14)
        ax4.grid(True, alpha=0.4, axis='y')
        
        ax5 = axs[1, 1]
        ax5.plot(t, Y_data, 'b-', lw=1.5, alpha=0.5, label='I_total(t)')
        if len(Y_up) > 0:
            ax5.scatter(t_up, Y_up, c='green', s=20, alpha=0.7, label='I_total↑')
        if len(Y_down) > 0:
            ax5.scatter(t_down, Y_down, c='red', s=20, alpha=0.7, label='I_total↓')
        ax5.set_xlabel('День', fontsize=12)
        ax5.set_ylabel('I_total(t)', fontsize=12)
        ax5.set_title('Динамика I_total(t) с выделением трендов', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.4)
        
        ax6 = axs[1, 2]
        mask_all_indices = np.where(mask_all)[0]
        if len(R0_prime_all) > 0:
            valid_mask = np.abs(R0_prime_all) < np.percentile(np.abs(R0_prime_all), 99)
            valid_indices = mask_all_indices[valid_mask]
            ax6.scatter(Y_data[valid_indices], R0_prime[valid_indices], 
                       c='blue', s=10, alpha=0.3, label='Все точки')
        if len(Y_up) > 0 and len(R0_prime_up) > 0:
            mask_up_indices = np.where(mask_up)[0]
            valid_mask_up = np.abs(R0_prime_up) < np.percentile(np.abs(R0_prime_up), 99)
            valid_indices_up = mask_up_indices[valid_mask_up]
            ax6.scatter(Y_data[valid_indices_up], R0_prime[valid_indices_up], 
                       c='green', s=20, alpha=0.6, label='I_total↑')
        if len(Y_down) > 0 and len(R0_prime_down) > 0:
            mask_down_indices = np.where(mask_down)[0]
            valid_mask_down = np.abs(R0_prime_down) < np.percentile(np.abs(R0_prime_down), 99)
            valid_indices_down = mask_down_indices[valid_mask_down]
            ax6.scatter(Y_data[valid_indices_down], R0_prime[valid_indices_down], 
                       c='red', s=20, alpha=0.6, label='I_total↓')
        ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax6.set_xlabel('I_total(t)', fontsize=12)
        ax6.set_ylabel("R'(t)", fontsize=12)
        ax6.set_title("Зависимость R'(t) от I_total(t)", fontsize=14)
        ax6.legend()
        ax6.grid(True, alpha=0.4)
        
        plt.suptitle(f'{title}\nγ = {gamma:.6f} | λ_all = {results["all"]["lambda"] if results["all"] is not None else "—"} | λ_up = {results["up"]["lambda"] if results["up"] is not None else "—"} | λ_down = {results["down"]["lambda"] if results["down"] is not None else "—"}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    if return_stats:
        return results
    else:
        return results['all']['lambda'] if results['all'] is not None else None,               results['all']['R0_start'] if results['all'] is not None else None,               results['all']['r2'] if results['all'] is not None else None, results

def calculate_y_per_epop(I_new_data, gamma, N, normalize=False):
    Y_total = calculate_total_from_new(I_new_data, gamma)

    if normalize:
        Y_norm = Y_total
        Y_abs = Y_total * N if N is not None and N > 0 else Y_total
    else:
                                
        Y_abs = Y_total
        Y_norm = Y_total / N if N is not None and N > 0 else Y_total

    if N is not None and N > 0:
        Y_epop = Y_abs / N * 100000
    else:
        Y_epop = Y_abs
    
    return Y_abs, Y_norm, Y_epop

def get_epop_value():
    global current_N, last_params
    
    if current_N is not None:
        return current_N
    elif last_params is not None:
        return last_params.get('N', None)
    return None

def format_number(num):
    if num is None:
        return "Н/Д"
    if isinstance(num, (int, float)):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
    return str(num)

def show_r_y_tables():
    global current_best_R0_arr, current_I_fact, current_I_fact_abs, current_best_gamma, mcmc_samples_R0, last_I_fact, last_params, adapt_p_var, current_N
    
    try:
                                            
        has_grid = current_best_R0_arr is not None and current_I_fact is not None

        has_bayes = mcmc_samples_R0 is not None and last_I_fact is not None
        
        if not has_grid and not has_bayes:
            messagebox.showwarning("Предупреждение", 
                "Нет данных. Сначала загрузите JSON и выполните симуляцию (для сетки) или MCMC (для байеса).")
            return

        N = get_epop_value()

        format_window = tk.Toplevel(root)
        format_window.title("Выбор формата Y(t)")
        format_window.geometry("500x250")
        format_window.transient(root)
        format_window.grab_set()

        format_window.update_idletasks()
        width = 500
        height = 250
        x = (format_window.winfo_screenwidth() // 2) - (width // 2)
        y = (format_window.winfo_screenheight() // 2) - (height // 2)
        format_window.geometry(f'{width}x{height}+{x}+{y}')
        
        format_var = tk.StringVar(value="norm")
        
        tk.Label(format_window, text="Выберите формат отображения Y(t):", 
                 font=("Arial", 12, "bold")).pack(pady=15)
        
        tk.Radiobutton(format_window, text="Доли (нормированные) - Y_abs / EPOP", 
                       variable=format_var, value="norm", font=("Arial", 10)).pack(anchor=tk.W, padx=40, pady=3)
        tk.Radiobutton(format_window, text="Абсолютные значения", 
                       variable=format_var, value="abs", font=("Arial", 10)).pack(anchor=tk.W, padx=40, pady=3)
        tk.Radiobutton(format_window, text="На 100 000 населения (EPOP)", 
                       variable=format_var, value="epop", font=("Arial", 10)).pack(anchor=tk.W, padx=40, pady=3)
        
        if N is not None and N > 0:
            tk.Label(format_window, text=f"EPOP = {format_number(N)}", 
                     font=("Arial", 9), fg="gray").pack(pady=5)
        else:
            tk.Label(format_window, text="⚠️ EPOP не найден!", 
                     font=("Arial", 9), fg="red").pack(pady=5)
        
        def confirm_format():
            format_window.destroy()
            show_table_with_format(format_var.get())
        
        tk.Button(format_window, text="Показать таблицу", font=("Arial", 10, "bold"),
                  command=confirm_format, bg="#4CAF50", fg="white", padx=20, pady=5).pack(pady=10)
        
        format_window.wait_window()
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

def show_table_with_format(format_type):
    global current_best_R0_arr, current_I_fact, current_I_fact_abs, current_best_gamma, mcmc_samples_R0, last_I_fact, last_params, adapt_p_var, current_N
    
    try:
        has_grid = current_best_R0_arr is not None and current_I_fact is not None
        has_bayes = mcmc_samples_R0 is not None and last_I_fact is not None
        
        N = get_epop_value()

        table_window = tk.Toplevel(root)
        table_window.title(f"Таблицы R(t) и Y(t) - {format_type.upper()}")
        table_window.geometry("1600x850")

        table_window.update_idletasks()
        width = 1600
        height = 850
        x = (table_window.winfo_screenwidth() // 2) - (width // 2)
        y = (table_window.winfo_screenheight() // 2) - (height // 2)
        table_window.geometry(f'{width}x{height}+{x}+{y}')

        text_frame = tk.Frame(table_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = scrolledtext.ScrolledText(
            text_frame, 
            font=("Consolas", 8), 
            wrap=tk.NONE,
            width=180,
            height=40
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        h_scroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
        h_scroll.pack(fill=tk.X)
        text_widget.config(xscrollcommand=h_scroll.set)
        
        text = ""

        text += "="*150 + "\n"
        if format_type == "norm":
            text += "📊 Y(t) в ДОЛЯХ (нормированные значения)\n"
            text += f"   Y_norm = Y_abs / EPOP\n"
            if N is not None and N > 0:
                text += f"   EPOP = {format_number(N)}\n"
            else:
                text += "   ⚠️ EPOP не найден! Показаны абсолютные значения.\n"
        elif format_type == "abs":
            text += "📊 Y(t) в АБСОЛЮТНЫХ значениях\n"
        else:
            text += "📊 Y(t) на 100 000 населения\n"
            if N is not None and N > 0:
                text += f"   EPOP = {format_number(N)}\n"
                text += f"   Y_epop = Y_abs / EPOP * 100000\n"
            else:
                text += "   ⚠️ EPOP не найден!\n"
        text += "="*150 + "\n\n"

        if has_grid:
            text += "\n" + "="*150 + "\n"
            text += "📊 СЕТОЧНАЯ МОДЕЛЬ: R(t) и Y(t)\n"
            text += "="*150 + "\n\n"
            
            R_grid = current_best_R0_arr

            if current_I_fact_abs is not None:
                I_new = current_I_fact_abs
            else:
                I_new = current_I_fact
            
            if current_best_gamma is not None:
                gamma_grid = np.mean(current_best_gamma)
            else:
                try:
                    gamma_grid = float(gamma_entry.get() or 0.074)
                except:
                    gamma_grid = 0.074

            Y_abs = calculate_total_from_new(I_new, gamma_grid)

            if N is not None and N > 0:
                Y_norm = Y_abs / N
                Y_epop = Y_abs / N * 100000
            else:
                Y_norm = Y_abs
                Y_epop = Y_abs
            
            if format_type == "norm":
                Y_show = Y_norm
                col_label = "Y(t)_доли"
                fmt = ".10f"
            elif format_type == "abs":
                Y_show = Y_abs
                col_label = "Y(t)_абс"
                fmt = ".4f"
            else:
                Y_show = Y_epop
                col_label = "Y(t)_на_100K"
                fmt = ".4f"

            headers = ["День", "R(t)_сетка", col_label]
            col_widths = [8, 18, 20]
            
            for h, w in zip(headers, col_widths):
                text += f"{h:<{w}}"
            text += "\n"
            text += "-"*150 + "\n"

            for i in range(len(R_grid)):
                day = i + 1
                r_val = R_grid[i]
                y_val = Y_show[i]
                if format_type == "norm":
                    text += f"{day:<{col_widths[0]}} {r_val:<{col_widths[1]}.10f} {y_val:<{col_widths[2]}.10f}\n"
                else:
                    text += f"{day:<{col_widths[0]}} {r_val:<{col_widths[1]}.10f} {y_val:<{col_widths[2]}.4f}\n"

            text += "\n" + "-"*150 + "\n"
            text += "📈 СТАТИСТИКА СЕТОЧНОЙ МОДЕЛИ:\n"
            text += f"  Дней: {len(R_grid)}\n"
            text += f"  R(t) среднее: {np.mean(R_grid):.10f}\n"
            text += f"  R(t) std: {np.std(R_grid):.10f}\n"
            text += f"  R(t) мин: {np.min(R_grid):.10f} (день {np.argmin(R_grid)+1})\n"
            text += f"  R(t) макс: {np.max(R_grid):.10f} (день {np.argmax(R_grid)+1})\n"
            
            if format_type == "norm":
                text += f"  Y(t) среднее (доли): {np.mean(Y_show):.10f}\n"
                text += f"  Y(t) макс (доли): {np.max(Y_show):.10f} (день {np.argmax(Y_show)+1})\n"
                text += f"  Y(t) суммарная (доли): {np.sum(Y_show):.10f}\n"
                text += f"  Максимальная доля: {np.max(Y_show)*100:.6f}%\n"
            elif format_type == "abs":
                text += f"  Y(t) среднее (абс): {np.mean(Y_show):.4f}\n"
                text += f"  Y(t) макс (абс): {np.max(Y_show):.4f} (день {np.argmax(Y_show)+1})\n"
                text += f"  Y(t) суммарная (абс): {np.sum(Y_show):.4f}\n"
            else:
                text += f"  Y(t) среднее (на 100K): {np.mean(Y_show):.4f}\n"
                text += f"  Y(t) макс (на 100K): {np.max(Y_show):.4f} (день {np.argmax(Y_show)+1})\n"
            text += f"  gamma = {gamma_grid:.10f}\n"
            if N is not None and N > 0:
                text += f"  EPOP = {format_number(N)}\n"

        if has_bayes:
            text += "\n" + "="*150 + "\n"
            text += "📊 БАЙЕСОВСКАЯ МОДЕЛЬ: R(t) и Y(t)\n"
            text += "="*150 + "\n\n"
            
            try:
                mu_val = float(mu_entry.get())
            except:
                mu_val = last_params.get('mu', 0.001)
            
            use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True
            
            result = generate_bayesian_trajectories(
                mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params,
                n_sim=10,
                window=int(window_entry.get() or 5),
                k_val=float(k_entry.get() or 0.055),
                r0_max=float(r0_max_entry.get() or 20.0),
                adapt_p=use_adapt_p,
                mu_fixed=mu_val,
                verbose=False
            )
            
            R_bayes = result['mean_R0']
            
            if current_I_fact_abs is not None:
                I_new_bayes = current_I_fact_abs
            else:
                I_new_bayes = last_I_fact
            
            if current_best_gamma is not None:
                gamma_bayes = np.mean(current_best_gamma)
            else:
                try:
                    gamma_bayes = float(gamma_entry.get() or 0.074)
                except:
                    gamma_bayes = 0.074
            
            Y_abs_bayes = calculate_total_from_new(I_new_bayes, gamma_bayes)
            
            if N is not None and N > 0:
                Y_norm_bayes = Y_abs_bayes / N
                Y_epop_bayes = Y_abs_bayes / N * 100000
            else:
                Y_norm_bayes = Y_abs_bayes
                Y_epop_bayes = Y_abs_bayes
            
            if format_type == "norm":
                Y_show_bayes = Y_norm_bayes
                col_label = "Y(t)_доли"
                fmt = ".10f"
            elif format_type == "abs":
                Y_show_bayes = Y_abs_bayes
                col_label = "Y(t)_абс"
                fmt = ".4f"
            else:
                Y_show_bayes = Y_epop_bayes
                col_label = "Y(t)_на_100K"
                fmt = ".4f"
            
            headers = ["День", "R(t)_байес", col_label]
            col_widths = [8, 18, 20]
            
            for h, w in zip(headers, col_widths):
                text += f"{h:<{w}}"
            text += "\n"
            text += "-"*150 + "\n"

            for i in range(len(R_bayes)):
                day = i + 1
                r_val = R_bayes[i]
                y_val = Y_show_bayes[i]
                if format_type == "norm":
                    text += f"{day:<{col_widths[0]}} {r_val:<{col_widths[1]}.10f} {y_val:<{col_widths[2]}.10f}\n"
                else:
                    text += f"{day:<{col_widths[0]}} {r_val:<{col_widths[1]}.10f} {y_val:<{col_widths[2]}.4f}\n"
            
            text += "\n" + "-"*150 + "\n"
            text += "📈 СТАТИСТИКА БАЙЕСОВСКОЙ МОДЕЛИ:\n"
            text += f"  Дней: {len(R_bayes)}\n"
            text += f"  R(t) среднее: {np.mean(R_bayes):.10f}\n"
            text += f"  R(t) std: {np.std(R_bayes):.10f}\n"
            text += f"  R(t) мин: {np.min(R_bayes):.10f} (день {np.argmin(R_bayes)+1})\n"
            text += f"  R(t) макс: {np.max(R_bayes):.10f} (день {np.argmax(R_bayes)+1})\n"
            
            if format_type == "norm":
                text += f"  Y(t) среднее (доли): {np.mean(Y_show_bayes):.10f}\n"
                text += f"  Y(t) макс (доли): {np.max(Y_show_bayes):.10f} (день {np.argmax(Y_show_bayes)+1})\n"
                text += f"  Y(t) суммарная (доли): {np.sum(Y_show_bayes):.10f}\n"
                text += f"  Максимальная доля: {np.max(Y_show_bayes)*100:.6f}%\n"
            elif format_type == "abs":
                text += f"  Y(t) среднее (абс): {np.mean(Y_show_bayes):.4f}\n"
                text += f"  Y(t) макс (абс): {np.max(Y_show_bayes):.4f} (день {np.argmax(Y_show_bayes)+1})\n"
            else:
                text += f"  Y(t) среднее (на 100K): {np.mean(Y_show_bayes):.4f}\n"
                text += f"  Y(t) макс (на 100K): {np.max(Y_show_bayes):.4f} (день {np.argmax(Y_show_bayes)+1})\n"
            text += f"  mu = {mu_val:.10f}\n"
            text += f"  gamma = {gamma_bayes:.10f}\n"
            if N is not None and N > 0:
                text += f"  EPOP = {format_number(N)}\n"
        
        text += "\n" + "="*150 + "\n"
        
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)

        btn_frame = tk.Frame(table_window)
        btn_frame.pack(pady=5)
        
        def copy_table():
            table_window.clipboard_clear()
            table_window.clipboard_append(text)
            messagebox.showinfo("Скопировано", "Таблица скопирована в буфер обмена")
        
        tk.Button(btn_frame, text="📋 Копировать", font=("Arial", 10),
                  command=copy_table, padx=10).pack(side=tk.LEFT, padx=5)
        
        def export_to_excel():
            try:

                has_grid = current_best_R0_arr is not None and current_I_fact is not None
                has_bayes = mcmc_samples_R0 is not None and last_I_fact is not None

                max_len = 0
                if has_grid:
                    max_len = max(max_len, len(R_grid))
                if has_bayes:
                    max_len = max(max_len, len(R_bayes))
                
                if max_len == 0:
                    messagebox.showwarning("Предупреждение", "Нет данных для экспорта.")
                    return

                columns = ["День"]

                if has_grid:
                    columns.append("R(t)_сетка")
                    columns.append(col_label + "_сетка")
                
                if has_bayes:
                    columns.append("R(t)_байес")
                    columns.append(col_label + "_байес")

                data = []
                for i in range(max_len):
                    row = [i + 1]        
                    
                    if has_grid and i < len(R_grid):
                        row.append(R_grid[i])
                        row.append(Y_show[i])
                    elif has_grid:
                        row.append("")
                        row.append("")
                    
                    if has_bayes and i < len(R_bayes):
                        row.append(R_bayes[i])
                        row.append(Y_show_bayes[i])
                    elif has_bayes:
                        row.append("")
                        row.append("")
                    
                    data.append(row)

                expected_len = len(columns)
                for i, row in enumerate(data):
                    if len(row) != expected_len:
                        print(f"⚠️ Строка {i} имеет {len(row)} столбцов, ожидается {expected_len}")
                                                       
                        while len(row) < expected_len:
                            row.append("")
                                         
                        if len(row) > expected_len:
                            row = row[:expected_len]
                        data[i] = row

                df = pd.DataFrame(data, columns=columns)

                file_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    initialfile=f"R_Y_tables_{format_type}.xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
                if file_path:
                    df.to_excel(file_path, index=False)
                    messagebox.showinfo("Успешно", f"Таблица сохранена в:\n{file_path}")
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"❌ Ошибка экспорта: {error_detail}")
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
        
        tk.Button(btn_frame, text="📊 Экспорт в Excel", font=("Arial", 10),
                  command=export_to_excel, padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="❌ Закрыть", font=("Arial", 10),
                  command=table_window.destroy, padx=10).pack(side=tk.LEFT, padx=5)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

def analyze_regression_from_table():
    global current_best_R0_arr, current_I_fact, current_I_fact_abs, current_best_gamma, current_N
    
    try:
                                  
        if current_best_R0_arr is None or current_I_fact is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала загрузите JSON и выполните симуляцию.")
            return

        N = get_epop_value()
        if N is None or N <= 0:
            messagebox.showwarning("Предупреждение", 
                "EPOP не найден. Невозможно рассчитать доли Y(t).")
            return

        if current_best_gamma is not None:
            gamma = np.mean(current_best_gamma)
        else:
            try:
                gamma = float(gamma_entry.get() or 0.074)
            except:
                gamma = 0.074

        R_data = current_best_R0_arr

        if current_I_fact_abs is not None:
            I_new = current_I_fact_abs
        else:
            I_new = current_I_fact

        Y_abs = calculate_total_from_new(I_new, gamma)

        Y_norm = Y_abs / N

        mask = (Y_norm > 1e-9) & (R_data > 0)
        R_filtered = R_data[mask]
        Y_filtered = Y_norm[mask]
        
        if len(R_filtered) < 3:
            messagebox.showwarning("Предупреждение", 
                f"Недостаточно данных для регрессии (нужно минимум 3 точки, получено {len(R_filtered)})")
            return

        log_Y = np.log(Y_filtered)
        log_R = np.log(R_filtered)
        
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(log_Y, log_R)
        
        p_exp = slope                
        R0 = np.exp(intercept)                
        r2 = r_value ** 2

        R_pred = R0 * (Y_filtered ** p_exp)

        mae = np.mean(np.abs(R_filtered - R_pred))
        rmse = np.sqrt(np.mean((R_filtered - R_pred) ** 2))

        results = {
            'R0': R0,
            'p': p_exp,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'std_err': std_err,
            'p_value': p_value,
            'n_points': len(R_filtered),
            'R_filtered': R_filtered,
            'Y_filtered': Y_filtered,
            'R_pred': R_pred,
            'log_Y': log_Y,
            'log_R': log_R
        }

        text = "\n" + "="*70 + "\n"
        text += "📊 РЕГРЕССИОННЫЙ АНАЛИЗ: R(t) = R0 * Y(t)^p\n"
        text += "="*70 + "\n"
        text += f"   Формула: R(t) = {R0:.6f} * Y(t)^{p_exp:.6f}\n"
        text += "-"*70 + "\n"
        text += f"\n📈 ПАРАМЕТРЫ:\n"
        text += f"   R0 = {R0:.6f}  (начальное репродуктивное число)\n"
        text += f"   p  = {p_exp:.6f}  (коэффициент нелинейности)\n"
        text += f"   ln(R0) = {intercept:.6f}\n"
        text += "-"*70 + "\n"
        text += f"\n📊 КАЧЕСТВО МОДЕЛИ:\n"
        text += f"   R²  = {r2:.6f}\n"
        text += f"   MAE = {mae:.6f}\n"
        text += f"   RMSE = {rmse:.6f}\n"
        text += f"   Std Error = {std_err:.6f}\n"
        text += f"   p-value = {p_value:.6f}\n"
        text += f"   Точек = {len(R_filtered)}\n"
        text += "-"*70 + "\n"

        text += f"\n📖 ИНТЕРПРЕТАЦИЯ:\n"
        if r2 > 0.9:
            text += "   ✅ Отличное качество модели (R² > 0.9)\n"
        elif r2 > 0.7:
            text += "   ✅ Хорошее качество модели (0.7 < R² ≤ 0.9)\n"
        elif r2 > 0.5:
            text += "   ⚠️ Удовлетворительное качество (0.5 < R² ≤ 0.7)\n"
        else:
            text += "   ❌ Плохое качество модели (R² ≤ 0.5)\n"
        
        if p_exp > 1:
            text += f"   📈 Суперлинейный рост (p = {p_exp:.4f} > 1) — ускорение\n"
        elif p_exp == 1:
            text += f"   ➖ Линейная зависимость (p = {p_exp:.4f} = 1)\n"
        elif p_exp > 0:
            text += f"   📉 Сублинейный рост (0 < p = {p_exp:.4f} < 1) — замедление\n"
        else:
            text += f"   🔄 Отрицательная зависимость (p = {p_exp:.4f} < 0)\n"
        
        if R0 > 1:
            text += f"   ⚠️ R0 = {R0:.4f} > 1 — эпидемия может расти\n"
        else:
            text += f"   ✅ R0 = {R0:.4f} ≤ 1 — эпидемия затухает\n"
        text += "="*70 + "\n\n"

        text_area.insert(tk.END, text)
        text_area.see(tk.END)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        ax1.scatter(log_Y, log_R, alpha=0.6, s=30, label='Данные')
        ax1.plot(log_Y, intercept + slope * log_Y, 'r-', lw=2, 
                 label=f'ln(R) = {intercept:.4f} + {slope:.4f}·ln(Y)\nR² = {r2:.4f}')
        ax1.set_xlabel('ln(Y(t))', fontsize=12)
        ax1.set_ylabel('ln(R(t))', fontsize=12)
        ax1.set_title('log-log зависимость', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        Y_p = Y_filtered ** p_exp
        ax2.scatter(Y_p, R_filtered, alpha=0.6, s=30, label='Данные')
        ax2.plot(Y_p, R0 * Y_p, 'r-', lw=2, label=f'R = {R0:.4f}·Y^{p_exp:.4f}')
        ax2.set_xlabel(f'Y(t)^{p_exp:.4f}', fontsize=12)
        ax2.set_ylabel('R(t)', fontsize=12)
        ax2.set_title('Зависимость R от Y^p', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.4)

        t = np.arange(len(R_filtered))
        ax3.plot(t, R_filtered, 'b-', lw=2, label='Фактические данные')
        ax3.plot(t, R_pred, 'r--', lw=2, label='Предсказанные значения')
        ax3.set_xlabel('Индекс', fontsize=12)
        ax3.set_ylabel('R(t)', fontsize=12)
        ax3.set_title('Сравнение модели с данными', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.4)

        error = np.abs(R_filtered - R_pred)
        ax4.plot(t, error, 'g-', lw=2, label='Ошибка')
        ax4.axhline(np.mean(error), color='red', linestyle='--', alpha=0.7,
                    label=f'Средняя ошибка = {np.mean(error):.4f}')
        ax4.set_xlabel('Индекс', fontsize=12)
        ax4.set_ylabel('|R_факт - R_пред|', fontsize=12)
        ax4.set_title('Ошибка предсказания', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.4)
        
        plt.suptitle(f'Регрессионный анализ: R(t) = R0·Y(t)^p\n'
                     f'R0 = {R0:.6f}, p = {p_exp:.6f}, R² = {r2:.6f}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        global last_regression_results
        last_regression_results = results
        
        return results
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при регрессионном анализе: {str(e)}")

def analyze_lambda_split_button():
    global mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params, adapt_p_var
    
    try:
        if mcmc_samples_R0 is None:
            messagebox.showwarning("Предупреждение", "Сначала запустите MCMC.")
            return
        
        if last_I_fact is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите JSON-файл.")
            return

        try:
            mu_val = float(mu_entry.get())
        except:
            mu_val = last_params.get('mu', 0.001)
        
        use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True
        
        result = generate_bayesian_trajectories(
            mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params,
            n_sim=10,
            window=int(window_entry.get() or 5),
            k_val=float(k_entry.get() or 0.055),
            r0_max=float(r0_max_entry.get() or 10.0),
            adapt_p=use_adapt_p,
            mu_fixed=mu_val                 
        )
        
        R0_bayesian = result['mean_R0']
        Y = last_I_fact
        
        min_len = min(len(R0_bayesian), len(Y))
        if len(R0_bayesian) != len(Y):
            print(f"⚠️ ВНИМАНИЕ: R0 ({len(R0_bayesian)}) и Y ({len(Y)}) имеют разную длину!")
            print(f"   Обрезаем до {min_len} точек")
            R0_bayesian = R0_bayesian[:min_len]
            Y = Y[:min_len]
        
        analyze_lambda_split(
            R0_bayesian,
            Y,
            title="Байесовский анализ λ (раздельный по трендам)",
            source="MCMC"
        )
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

class MCMC_Bayesian:
    def __init__(self, params, I_fact, S, I, window=5, k=0.055):
        self.params = params
        self.I_fact = I_fact
        self.S = S
        self.I = I
        self.window = window
        self.k = k
        self.gamma = params['gamma']
        self.mu = params['mu']
        self.chains_R0 = []
        self.chains_p = []
        
        self.n_samples = 5000
        self.burn_in = 1000
        self.n_chains = 4
        
        self.samples_R0 = None
        self.samples_p = None
        self.acceptance_rate = 0
    
    def calculate_log_likelihood(self, R0, p_val=0.0):
        beta = R0 * self.gamma
        t_prev = len(self.S) - 1
        S_prev = self.S[t_prev]
        I_prev = self.I[t_prev]
        
        dS = -beta * S_prev * I_prev + self.mu * (1 - S_prev) + (self.k + p_val) * (1 - S_prev - I_prev)
        dI = beta * S_prev * I_prev - self.gamma * I_prev - self.mu * I_prev
        
        S_test = max(S_prev + dS, 0)
        I_test = max(I_prev + dI, 0)
        
        error = 0.0
        S_tmp, I_tmp = S_test, I_test
        
        for step in range(self.window):
            idx = t_prev + 1 + step
            if idx >= len(self.I_fact):
                break
            trans_tmp = beta * S_tmp * I_tmp
            weight = get_weight(step)
            error += weight * (trans_tmp - self.I_fact[idx]) ** 2
            error += 0.001
            
            dS_tmp = -beta * S_tmp * I_tmp + self.mu * (1 - S_tmp) + (self.k + p_val) * (1 - S_tmp - I_tmp)
            dI_tmp = beta * S_tmp * I_tmp - self.gamma * I_tmp - self.mu * I_tmp
            S_tmp = max(S_tmp + dS_tmp, 0)
            I_tmp = max(I_tmp + dI_tmp, 1e-6)
        
        log_likelihood = -0.5 * error
        log_prior_R0 = -0.5 * ((R0 - 2.5) / 2.0) ** 2
        log_prior_p = -0.5 * ((p_val - 0.1) / 0.1) ** 2
        
        return log_likelihood + log_prior_R0 + log_prior_p
    
    def metropolis_hastings(self, R0_current, p_current=0.0):
        R0_proposed = R0_current + np.random.normal(0, 0.2)
        p_proposed = p_current + np.random.normal(0, 0.005)
        
        R0_proposed = np.clip(R0_proposed, 0.3, 10.0)
        p_proposed = np.clip(p_proposed, 0.0, 0.4)
        
        log_lik_current = self.calculate_log_likelihood(R0_current, p_current)
        log_lik_proposed = self.calculate_log_likelihood(R0_proposed, p_proposed)
        
        if log_lik_proposed > log_lik_current:
            return R0_proposed, p_proposed, True
        else:
            accept_prob = np.exp(log_lik_proposed - log_lik_current)
            if np.random.random() < accept_prob:
                return R0_proposed, p_proposed, True
            else:
                return R0_current, p_current, False
    
    def run_mcmc(self, n_samples=5000, burn_in=1000):
        self.n_samples = n_samples
        self.burn_in = burn_in
        
        print(f"🔄 Запуск MCMC (сэмплов: {n_samples}, прогревочных: {burn_in})...")
        print("   Сэмплируем R0 и p одновременно!")
        
        chains_R0 = []
        chains_p = []
        chains_accept = []
        self.chains_R0 = chains_R0
        self.chains_p = chains_p
        
        for chain in range(self.n_chains):
            R0_current = np.random.uniform(0.3, 10.0)
            p_current = np.random.uniform(0.0, 0.4)
            
            chain_R0 = []
            chain_p = []
            accepted = 0
            
            for _ in range(burn_in):
                R0_current, p_current, acc = self.metropolis_hastings(R0_current, p_current)
                if acc:
                    accepted += 1
            
            for i in range(n_samples):
                R0_current, p_current, acc = self.metropolis_hastings(R0_current, p_current)
                if acc:
                    accepted += 1
                
                chain_R0.append(R0_current)
                chain_p.append(p_current)
            
            chains_R0.append(np.array(chain_R0))
            chains_p.append(np.array(chain_p))
            chains_accept.append(accepted / (burn_in + n_samples))
            
            print(f"  Цепь {chain+1}: принято {accepted/(burn_in+n_samples)*100:.1f}%")
        
        self.samples_R0 = np.concatenate(chains_R0)
        self.samples_p = np.concatenate(chains_p)
        self.acceptance_rate = np.mean(chains_accept)
        
        self.R0_mean = np.mean(self.samples_R0)
        self.R0_std = np.std(self.samples_R0)
        self.R0_ci_lower = np.percentile(self.samples_R0, 2.5)
        self.R0_ci_upper = np.percentile(self.samples_R0, 97.5)
        
        self.p_mean = np.mean(self.samples_p)
        self.p_std = np.std(self.samples_p)
        self.p_ci_lower = np.percentile(self.samples_p, 2.5)
        self.p_ci_upper = np.percentile(self.samples_p, 97.5)
        
        print(f"✅ MCMC завершен!")
        print(f"   R0: среднее = {self.R0_mean:.6f}, 95% CI = [{self.R0_ci_lower:.6f}, {self.R0_ci_upper:.6f}]")
        print(f"   p:  среднее = {self.p_mean:.6f}, 95% CI = [{self.p_ci_lower:.6f}, {self.p_ci_upper:.6f}]")
        
        return self.samples_R0, self.samples_p

def plot_weights(window=5):
    steps = np.arange(1, window + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    exponents = [0.3, WEIGHT_EXPONENT, 0.7, 0.85, 1.0]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for exp, color in zip(exponents, colors):
        weights = 1.0 / (steps ** exp)
        label = f'1/(step)^{exp}'
        if exp == WEIGHT_EXPONENT:
            label += ' ← ИСПОЛЬЗУЕТСЯ'
        ax1.plot(steps, weights, 'o-', color=color, 
                label=label, linewidth=2, markersize=8)
    
    ax1.axvline(x=window, color='red', linestyle='--', alpha=0.7, 
               label=f'window = {window}', linewidth=2)
    
    ax1.set_xlabel('Шаг (день)', fontsize=12)
    ax1.set_ylabel('Вес', fontsize=12)
    ax1.set_title(f'Веса ошибки (window = {window})', fontsize=14)
    ax1.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 1.1])
    ax1.set_xlim([0.5, window + 0.5])
    
    weights_orig = 1.0 / (steps ** WEIGHT_EXPONENT)
    weights_norm = weights_orig / weights_orig.sum()
    
    ax2.bar(steps, weights_norm, color='red', alpha=0.7, edgecolor='darkred', linewidth=1)
    ax2.set_xlabel('Шаг (день)', fontsize=12)
    ax2.set_ylabel('Нормированный вес', fontsize=12)
    ax2.set_title(f'Нормированные веса (window = {window}, сумма = 1)', fontsize=14)
    ax2.grid(True, alpha=0.5, linestyle='-', linewidth=0.8, axis='y')
    ax2.set_xlim([0.5, window + 0.5])
    
    for i, v in enumerate(weights_norm):
        ax2.text(i + 1, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    
    ax2.text(window * 0.7, 0.8, f'Сумма весов = {weights_norm.sum():.2f}', 
             fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    ax2.text(window * 0.7, 0.7, f'Всего дней: {window}', 
             fontsize=10, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    
    plt.suptitle(f'Анализ весов для взвешенной ошибки (window = {window})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def show_r_and_p():
    global current_best_R0_arr, current_best_p, current_I_fact, current_best_gamma, current_N, current_S, current_I, current_best_mu, last_params, epop_optimized
    
    try:
        if current_best_R0_arr is None or current_best_p is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала загрузите JSON и выполните симуляцию.")
            return

        result_window = tk.Toplevel(root)
        result_window.title("Все параметры модели — R(t), p(t), S0, I0, mu, k, EPOP")
        result_window.geometry("1400x800")

        result_window.update_idletasks()
        width = 1400
        height = 800
        x = (result_window.winfo_screenwidth() // 2) - (width // 2)
        y = (result_window.winfo_screenheight() // 2) - (height // 2)
        result_window.geometry(f'{width}x{height}+{x}+{y}')

        text_frame = tk.Frame(result_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = scrolledtext.ScrolledText(
            text_frame, 
            font=("Consolas", 8), 
            wrap=tk.NONE,
            width=180,
            height=35
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        h_scroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
        h_scroll.pack(fill=tk.X)
        text_widget.config(xscrollcommand=h_scroll.set)

        R_data = current_best_R0_arr
        p_data = current_best_p
        days = len(R_data)

        # --- I(t) ---
        if current_I is not None and len(current_I) >= days:
            I_data = np.asarray(current_I[:days], dtype=float)
        else:
            I_data = None

        if current_S is not None:
            S0_opt = current_S[0]
        elif last_params is not None:
            S0_opt = last_params.get('S0', 0)
        else:
            S0_opt = 0
        
        if current_I is not None:
            I0_opt = current_I[0]
        elif last_params is not None:
            I0_opt = last_params.get('I0', 0)
        else:
            I0_opt = 0
        
        if current_best_gamma is not None:
            gamma = np.mean(current_best_gamma)
        else:
            try:
                gamma = float(gamma_entry.get() or 0.074)
            except:
                gamma = 0.074
        
        if current_best_mu is not None:
            mu = np.mean(current_best_mu)
        else:
            try:
                mu = float(mu_entry.get() or 0.00013)
            except:
                mu = 0.00013
        
        try:
            k = float(k_entry.get() or 0.0055)
        except:
            k = 0.0055
        
        if epop_optimized is not None:
            EPOP = epop_optimized
        elif current_N is not None:
            EPOP = current_N
        elif last_params is not None:
            EPOP = last_params.get('N', 0)
        else:
            EPOP = 0
        
        R0_mean = np.mean(R_data)

        N = EPOP if EPOP else 1

        text = ""
        text += "="*140 + "\n"
        text += "📊 ВСЕ ПАРАМЕТРЫ МОДЕЛИ (7 параметров)\n"
        text += "="*140 + "\n"
        text += f"   Дней: {days}\n"
        text += "="*140 + "\n\n"

        text += "🔧 ОСНОВНЫЕ ПАРАМЕТРЫ МОДЕЛИ:\n"
        text += "-"*80 + "\n"
        text += f"   1. S0  (начальная доля восприимчивых)     = {S0_opt:.10f}  (абс: {S0_opt * N:.2f})\n"
        text += f"   2. I0  (начальная доля инфицированных)    = {I0_opt:.10f}  (абс: {I0_opt * N:.2f})\n"
        text += f"   3. γ   (скорость выздоровления)           = {gamma:.8f}\n"
        text += f"   4. μ   (скорость смертности/рождаемости)  = {mu:.8f}\n"
        text += f"   5. k   (скорость потери иммунитета)       = {k:.8f}\n"
        text += f"   6. p   (параметр преодоления иммунитета)  = (см. таблицу ниже)\n"
        text += f"   7. R₀  (эффективное репродуктивное число) = (см. таблицу ниже)\n"
        text += "-"*80 + "\n"
        text += f"   EPOP (численность населения)              = {N:.0f}\n"
        text += f"   Window (размер окна)                      = {window_entry.get() or 15}\n"
        text += "-"*80 + "\n\n"

        text += "📊 СТАТИСТИКА R(t):\n"
        text += "-"*60 + "\n"
        text += f"  Среднее R(t) = {np.mean(R_data):.6f}\n"
        text += f"  Std R(t) = {np.std(R_data):.6f}\n"
        text += f"  Min R(t) = {np.min(R_data):.6f} (день {np.argmin(R_data)+1})\n"
        text += f"  Max R(t) = {np.max(R_data):.6f} (день {np.argmax(R_data)+1})\n"
        text += f"  Последнее R(t) = {R_data[-1]:.6f}\n"
        days_r_gt_1 = np.sum(R_data > 1.0)
        text += f"  Дней с R > 1: {days_r_gt_1} ({days_r_gt_1/days*100:.1f}%)\n"
        text += "-"*60 + "\n\n"

        text += "📊 СТАТИСТИКА p(t):\n"
        text += "-"*60 + "\n"
        p_mean = np.mean(p_data[p_data > 0.0001]) if np.any(p_data > 0.0001) else 0
        p_std = np.std(p_data[p_data > 0.0001]) if np.any(p_data > 0.0001) else 0
        text += f"  Среднее p(t) = {p_mean:.8f} (только где p > 0.0001)\n"
        text += f"  Std p(t) = {p_std:.8f}\n"
        text += f"  Min p(t) = {np.min(p_data):.8f} (день {np.argmin(p_data)+1})\n"
        text += f"  Max p(t) = {np.max(p_data):.8f} (день {np.argmax(p_data)+1})\n"
        text += f"  Последнее p(t) = {p_data[-1]:.8f}\n"
        days_p_gt_0 = np.sum(p_data > 0.0001)
        text += f"  Дней с p > 0: {days_p_gt_0} ({days_p_gt_0/days*100:.1f}%)\n"
        p_integral = np.sum(p_data)
        text += f"  Суммарная p (интеграл) = {p_integral:.6f}\n"
        text += "-"*60 + "\n\n"

        # --- Статистика I(t) ---
        text += "📊 СТАТИСТИКА I(t):\n"
        text += "-" * 60 + "\n"
        if I_data is not None:
            text += f"  Среднее I(t) = {np.mean(I_data):.10f}\n"
            text += f"  Std I(t) = {np.std(I_data):.10f}\n"
            text += f"  Min I(t) = {np.min(I_data):.10f} (день {np.argmin(I_data)+1})\n"
            text += f"  Max I(t) = {np.max(I_data):.10f} (день {np.argmax(I_data)+1})\n"
            text += f"  Последнее I(t) = {I_data[-1]:.10f}\n"

            if EPOP and EPOP > 0:
                text += f"  Среднее I_abs = {np.mean(I_data) * EPOP:.2f}\n"
                text += f"  Max I_abs = {np.max(I_data) * EPOP:.2f} (день {np.argmax(I_data)+1})\n"

            text += f"  I(0) = {I_data[0]:.10f}"
            if EPOP and EPOP > 0:
                text += f"  (абс: {I_data[0] * EPOP:.2f})"
            text += "\n"
        else:
            text += "  ⚠️ I(t) недоступно (нет данных симуляции)\n"
        text += "-" * 60 + "\n\n"

        text += "="*160 + "\n"
        text += "📊 ТАБЛИЦА ПО ДНЯМ: R(t), p(t), I(t)\n"
        text += "="*160 + "\n"
        text += (
            f"{'День':<8} {'R(t)':<16} {'p(t)':<16} {'I(t)':<20} "
            f"{'R(t) статус':<16} {'p(t) статус':<16}\n"
        )
        text += "-" * 160 + "\n"

        step = max(1, days // 200)
        for i in range(0, days, step):
            day = i + 1
            r_val = R_data[i]
            p_val = p_data[i]
            i_val = I_data[i] if I_data is not None else float("nan")

            if r_val > 1.5:
                r_status = "🔴 ВЫСОКИЙ"
            elif r_val > 1.0:
                r_status = "🟡 РОСТ"
            elif r_val > 0.5:
                r_status = "🟢 ЗАТУХАНИЕ"
            else:
                r_status = "🔵 НИЗКИЙ"

            if p_val > 0.01:
                p_status = f"🟣 АКТИВЕН ({p_val:.4f})"
            elif p_val > 0.001:
                p_status = "🟠 СЛАБЫЙ"
            else:
                p_status = "⚪ НУЛЕВОЙ"

            if I_data is not None:
                i_str = f"{i_val:<20.10f}"
            else:
                i_str = f"{'—':<20}"

            text += (
                f"{day:<8} {r_val:<16.6f} {p_val:<16.6f} {i_str} "
                f"{r_status:<16} {p_status:<16}\n"
            )

        text += "=" * 160 + "\n"

        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)

        btn_frame = tk.Frame(result_window)
        btn_frame.pack(pady=5)
        
        def copy_table():
            result_window.clipboard_clear()
            result_window.clipboard_append(text)
            messagebox.showinfo("Скопировано", "Таблица скопирована в буфер обмена")
        
        tk.Button(btn_frame, text="📋 Копировать", font=("Arial", 10),
                  command=copy_table, padx=10).pack(side=tk.LEFT, padx=5)
        
        def export_to_excel():
            try:
                df_dict = {
                    'День': np.arange(1, days + 1),
                    'R(t)': R_data,
                    'p(t)': p_data,
                }

                if I_data is not None:
                    df_dict['I(t)'] = I_data
                    if EPOP and EPOP > 0:
                        df_dict['I_abs'] = I_data * EPOP

                df = pd.DataFrame(df_dict)
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    initialfile="all_parameters.xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
                if file_path:
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='R_p_I по дням', index=False)
                        
                        params_df = pd.DataFrame({
                            'Параметр': ['S0', 'I0', 'gamma', 'mu', 'k', 'EPOP', 'Window'],
                            'Значение': [S0_opt, I0_opt, gamma, mu, k, N, window_entry.get() or 15]
                        })
                        params_df.to_excel(writer, sheet_name='Параметры модели', index=False)
                    
                    messagebox.showinfo("Успешно", f"Таблица сохранена в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
        
        tk.Button(btn_frame, text="📊 Экспорт в Excel", font=("Arial", 10),
                  command=export_to_excel, padx=10).pack(side=tk.LEFT, padx=5)
        
        def plot_r_and_p():
            if I_data is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 13))
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            t = np.arange(days)

            ax1.plot(t, R_data, 'b-', lw=2, label='R(t)')
            ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R=1')
            ax1.axhline(np.mean(R_data), color='green', linestyle=':', alpha=0.7,
                       label=f'Среднее R = {np.mean(R_data):.4f}')
            ax1.set_xlabel('День', fontsize=12)
            ax1.set_ylabel('R(t)', fontsize=12)
            ax1.set_title('Динамика R(t) после адаптации', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.4)

            ax2.plot(t, p_data, 'g-', lw=2, label='p(t)')
            ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='p=0')
            ax2.axvline(x=430, color='purple', linestyle='--', alpha=0.7,
                       label='Начало адаптации p (день 430)')
            if np.any(p_data > 0.0001):
                ax2.axhline(np.mean(p_data[p_data > 0.0001]), color='blue', linestyle=':', alpha=0.7,
                           label=f'Среднее p (активные) = {np.mean(p_data[p_data > 0.0001]):.8f}')
            ax2.set_xlabel('День', fontsize=12)
            ax2.set_ylabel('p(t)', fontsize=12)
            ax2.set_title('Динамика p(t) после адаптации', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.4)

            if I_data is not None:
                ax3.plot(t, I_data, 'm-', lw=2, label='I(t)')
                ax3.set_xlabel('День', fontsize=12)
                ax3.set_ylabel('I(t)', fontsize=12)
                ax3.set_title('Динамика I(t)', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.4)
                ax3.legend()

            plt.suptitle(f'R(t), p(t), I(t) после адаптации (γ={gamma:.6f}, μ={mu:.8f}, k={k:.6f})', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        tk.Button(btn_frame, text="📈 Построить графики", font=("Arial", 10),
                  command=plot_r_and_p, padx=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="❌ Закрыть", font=("Arial", 10),
                  command=result_window.destroy, padx=10).pack(side=tk.LEFT, padx=5)
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")

def build_r0_day_matrix(params, I_fact, S, I, window=5, k_val=0.055, r0_max=10.0):
    gamma = params['gamma']
    mu = params['mu']
    k = k_val
    
    days = len(I_fact)
    
    R0_MIN = 0.3
    R0_MAX = r0_max
    r0_grid = np.arange(R0_MIN, R0_MAX + 0.1, 0.2)
    
    error_matrix = np.zeros((len(r0_grid), days))
    best_r0_per_day = np.zeros(days)
    min_error_per_day = np.zeros(days)
    
    print(f"🔄 Построение матрицы R0×Дни ({len(r0_grid)} x {days})...")
    
    for t in range(1, days):
        t_prev = t - 1
        S_prev = S[t_prev]
        I_prev = I[t_prev]
        
        best_error = np.inf
        best_r0 = 0.0
        
        for i, R0 in enumerate(r0_grid):
            beta = R0 * gamma
            
            dS = -beta * S_prev * I_prev + mu * (1 - S_prev) + k * (1 - S_prev - I_prev)
            dI = beta * S_prev * I_prev - gamma * I_prev - mu * I_prev
            
            S_test = max(S_prev + dS, 0)
            I_test = max(I_prev + dI, 0)
            
            error = 0.0
            S_tmp, I_tmp = S_test, I_test
            
            for step in range(window):
                idx = t + step
                if idx >= days:
                    break
                trans_tmp = beta * S_tmp * I_tmp
                weight = get_weight(step)
                error += weight * (trans_tmp - I_fact[idx]) ** 2
                
                dS_tmp = -beta * S_tmp * I_tmp + mu * (1 - S_tmp) + k * (1 - S_tmp - I_tmp)
                dI_tmp = beta * S_tmp * I_tmp - gamma * I_tmp - mu * I_tmp
                S_tmp = max(S_tmp + dS_tmp, 0)
                I_tmp = max(I_tmp + dI_tmp, 1e-6)
            
            error_matrix[i, t] = error
            
            if error < best_error:
                best_error = error
                best_r0 = R0
        
        best_r0_per_day[t] = best_r0
        min_error_per_day[t] = best_error
        
        if t % 100 == 0:
            print(f"  День {t}/{days} обработан")
    
    return error_matrix, r0_grid, best_r0_per_day, min_error_per_day

def plot_r0_day_matrix(error_matrix, r0_grid, best_r0_per_day, days, window=5, r0_max=10.0):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    im = ax1.imshow(error_matrix, 
                    extent=[0, days, r0_grid[0], r0_grid[-1]],
                    origin='lower',
                    aspect='auto',
                    cmap='plasma',
                    interpolation='bilinear')
    
    ax1.plot(np.arange(days), best_r0_per_day, 'w-', lw=2, label='Оптимальный R0')
    ax1.set_xlabel('День', fontsize=12)
    ax1.set_ylabel('R0', fontsize=12)
    ax1.set_title('Матрица ошибок R0 × Дни (тепловая карта)', fontsize=14)
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Ошибка')
    
    ax2.plot(np.arange(days), best_r0_per_day, 'b-', lw=2, label='Оптимальный R0')
    ax2.axhline(np.mean(best_r0_per_day[10:]), color='red', linestyle='--', alpha=0.7, 
                label=f'Mean R(t) = {np.mean(best_r0_per_day[10:]):.3f}')
    ax2.set_xlabel('День', fontsize=12)
    ax2.set_ylabel('R0', fontsize=12)
    ax2.set_title('Оптимальный R0 для каждого дня', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    
    min_error = np.min(error_matrix, axis=0)
    ax3.plot(np.arange(days), min_error, 'g-', lw=2, label='Минимальная ошибка')
    ax3.set_xlabel('День', fontsize=12)
    ax3.set_ylabel('Ошибка', fontsize=12)
    ax3.set_title('Минимальная ошибка для каждого дня', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.5)
    
    mid_day = days // 2
    ax4.plot(r0_grid, error_matrix[:, mid_day], 'r-', lw=2, label=f'День {mid_day}')
    ax4.axvline(best_r0_per_day[mid_day], color='blue', linestyle='--', alpha=0.7, 
                label=f'Оптимум R0 = {best_r0_per_day[mid_day]:.2f}')
    ax4.set_xlabel('R0', fontsize=12)
    ax4.set_ylabel('Ошибка', fontsize=12)
    ax4.set_title(f'Профиль ошибок для дня {mid_day}', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.5)
    
    plt.suptitle(f'Матрица ошибок R0 × Дни (window={window}, R0_max={r0_max:.1f})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def show_r0_day_matrix_values(error_matrix, r0_grid, days):
    matrix_window = tk.Toplevel(root)
    matrix_window.title("Матрица ошибок R0 × Дни")
    matrix_window.geometry("1200x700")
    
    header_frame = tk.Frame(matrix_window)
    header_frame.pack(fill=tk.X, padx=10, pady=5)
    
    tk.Label(header_frame, text="Матрица ошибок: R0 (строки) × Дни (столбцы)", 
             font=("Arial", 14, "bold")).pack()
    
    tk.Label(header_frame, text=f"Строки - значения R0, Столбцы - дни (каждый {50}-й день)", 
             font=("Arial", 10)).pack()
    
    text_frame = tk.Frame(matrix_window)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    text_widget = scrolledtext.ScrolledText(text_frame, font=("Consolas", 7), wrap=tk.NONE)
    text_widget.pack(fill=tk.BOTH, expand=True)
    
    h_scroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
    h_scroll.pack(fill=tk.X)
    text_widget.config(xscrollcommand=h_scroll.set)
    
    text = ""
    
    day_step = 50
    day_indices = np.arange(0, days, day_step)
    
    text += "R0\\День"
    for d in day_indices:
        text += f"  {d:>10}"
    text += "\n"
    text += "-" * (len(text) + 20) + "\n"
    
    for i, R0 in enumerate(r0_grid[::2]):
        row = f"{R0:.1f}"
        for d in day_indices:
            row += f"  {error_matrix[i*2, d]:>10.6f}"
        text += row + "\n"
    
    text += "-" * (len(text) + 20) + "\n"
    
    text += f"\n"
    text += f"Размер матрицы: {len(r0_grid)} x {days} = {len(r0_grid)*days} значений\n"
    text += f"Минимальная ошибка: {np.min(error_matrix):.6f}\n"
    text += f"Максимальная ошибка: {np.max(error_matrix):.6f}\n"
    text += f"Средняя ошибка: {np.mean(error_matrix):.6f}\n"
    text += f"Оптимальный R0 (средний): {np.mean(np.min(error_matrix, axis=0)):.3f}\n"
    
    text_widget.insert(tk.END, text)
    text_widget.config(state=tk.DISABLED)
    
    btn_frame = tk.Frame(matrix_window)
    btn_frame.pack(pady=5)
    
    tk.Button(btn_frame, text="Закрыть", font=("Arial", 10),
              command=matrix_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def copy_matrix():
        matrix_window.clipboard_clear()
        matrix_window.clipboard_append(text)
        messagebox.showinfo("Скопировано", "Матрица скопирована в буфер обмена")
    
    tk.Button(btn_frame, text="Копировать в буфер", font=("Arial", 10),
              command=copy_matrix).pack(side=tk.LEFT, padx=5)
    
    def export_to_excel():
        try:
            df = pd.DataFrame(error_matrix, 
                             index=[f"{r:.1f}" for r in r0_grid],
                             columns=[f"{d}" for d in range(days)])
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                initialfile="r0_day_matrix.xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if file_path:
                df.to_excel(file_path)
                messagebox.showinfo("Успешно", f"Матрица сохранена в:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
    
    tk.Button(btn_frame, text="📊 Экспорт в Excel", font=("Arial", 10),
              command=export_to_excel).pack(side=tk.LEFT, padx=5)

def find_optimal_epop(force_recalculate=False):
    global current_best_R0_arr, current_I_fact, current_I_fact_abs, current_best_gamma, current_N, epop_optimized, epop_optimized_params, last_params
    
    try:
                                  
        if current_best_R0_arr is None or current_I_fact is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала загрузите JSON и выполните симуляцию.")
            return None, None

        current_params = {
            'window': int(window_entry.get() or 5),
            'k': float(k_entry.get() or 0.055),
            'r0_max': float(r0_max_entry.get() or 20.0),
            'gamma': float(gamma_entry.get() or 0.074),
            'mu': float(mu_entry.get() or 0.001),
            'adapt_p': adapt_p_var.get() if adapt_p_var is not None else True
        }

        if epop_optimized is not None and not force_recalculate:
            if epop_optimized_params == current_params:
                print(f"✅ Используем оптимизированный EPOP = {epop_optimized:.0f}")
                return epop_optimized, None

        try:
            epop_min = float(epop_min_entry.get() or 10000)
            epop_max = float(epop_max_entry.get() or 100000)
            epop_step = float(epop_step_entry.get() or 5000)
        except:
            epop_min = 10000
            epop_max = 100000
            epop_step = 5000

        epop_values = np.arange(epop_min, epop_max + epop_step/2, epop_step)
        
        print("\n" + "="*70)
        print("🔍 ПОДБОР ОПТИМАЛЬНОГО EPOP")
        print("="*70)
        print(f"   Диапазон: [{epop_min:.0f} - {epop_max:.0f}], шаг: {epop_step:.0f}")
        print(f"   Количество значений: {len(epop_values)}")
        print("-"*70)
        
        best_epop = None
        best_error = np.inf
        results = {}

        gamma = current_best_gamma if current_best_gamma is not None else float(gamma_entry.get() or 0.074)
        window = int(window_entry.get() or 5)
        k_val = float(k_entry.get() or 0.055)
        r0_max_val = float(r0_max_entry.get() or 20.0)
        mu_val = float(mu_entry.get() or 0.001)
        use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True

        I_fact_abs = current_I_fact
        
        for idx, epop_candidate in enumerate(epop_values):
            print(f"\n🔄 EPOP = {epop_candidate:.0f} ({idx+1}/{len(epop_values)})...")

            I_fact_norm = I_fact_abs / epop_candidate

            params_copy = last_params.copy()
            params_copy['N'] = epop_candidate

            S, I, transmission, R0, p, mu_arr = sir_euler_grid(
                params_copy, 
                I_fact_norm,
                window=window,
                p_min=0.0,
                p_max=0.4,
                p_step=0.001,
                k_val=k_val,
                r0_max=r0_max_val,
                adapt_p=use_adapt_p,
                mu_fixed=mu_val
            )

            trans_abs = transmission * epop_candidate

            mae = np.mean(np.abs(trans_abs - I_fact_abs))
            rmse = np.sqrt(np.mean((trans_abs - I_fact_abs) ** 2))
            r2 = 1 - np.sum((I_fact_abs - trans_abs)**2) / (np.sum((I_fact_abs - np.mean(I_fact_abs))**2) + 1e-9)
            
            results[epop_candidate] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'S': S,
                'I': I,
                'R0': R0,
                'p': p,
                'transmission': trans_abs
            }
            
            print(f"   MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
            
            if mae < best_error:
                best_error = mae
                best_epop = epop_candidate
                print(f"   ✅ НОВЫЙ ЛУЧШИЙ! EPOP = {epop_candidate:.0f}, MAE = {mae:.2f}")

        epop_optimized = best_epop
        epop_optimized_params = current_params
        
        print("\n" + "="*70)
        print(f"✅ ПОДБОР EPOP ЗАВЕРШЕН!")
        print(f"   Лучший EPOP = {best_epop:.0f}")
        print(f"   Лучшая MAE = {best_error:.2f}")
        print(f"   R² = {results[best_epop]['R2']:.4f}")
        print("="*70)

        text_area.insert(tk.END, f"\n✅ Оптимальный EPOP: {best_epop:.0f}\n")
        text_area.insert(tk.END, f"   MAE = {best_error:.2f}\n")
        text_area.insert(tk.END, f"   R² = {results[best_epop]['R2']:.4f}\n")
        text_area.see(tk.END)
        
        return best_epop, results
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при подборе EPOP: {str(e)}")
        return None, None

def plot_epop_search_results(results, best_epop):
    if results is None:
        return
    
    epop_values = sorted(results.keys())
    mae_values = [results[epop]['MAE'] for epop in epop_values]
    rmse_values = [results[epop]['RMSE'] for epop in epop_values]
    r2_values = [results[epop]['R2'] for epop in epop_values]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.plot(epop_values, mae_values, 'b-o', lw=2, markersize=8, label='MAE')
    ax1.axvline(best_epop, color='red', linestyle='--', alpha=0.7, 
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax1.axhline(min(mae_values), color='green', linestyle='--', alpha=0.7,
                label=f'Лучшая MAE = {min(mae_values):.2f}')
    ax1.set_xlabel('EPOP', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('Зависимость MAE от EPOP', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(epop_values, rmse_values, 'r-o', lw=2, markersize=8, label='RMSE')
    ax2.axvline(best_epop, color='red', linestyle='--', alpha=0.7,
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax2.set_xlabel('EPOP', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Зависимость RMSE от EPOP', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    ax3.plot(epop_values, r2_values, 'g-o', lw=2, markersize=8, label='R²')
    ax3.axvline(best_epop, color='red', linestyle='--', alpha=0.7,
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax3.axhline(max(r2_values), color='blue', linestyle='--', alpha=0.7,
                label=f'Лучший R² = {max(r2_values):.4f}')
    ax3.set_xlabel('EPOP', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title('Зависимость R² от EPOP', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.4)

    best_idx = np.argmin(mae_values)
    stats_text = f"""
    Лучший EPOP = {best_epop:.0f}
    Лучшая MAE = {min(mae_values):.2f}
    Лучшая RMSE = {rmse_values[best_idx]:.2f}
    Лучший R² = {r2_values[best_idx]:.4f}
    Проверено значений: {len(epop_values)}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=14, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Итоговая статистика', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Результаты подбора оптимального EPOP\nЛучший EPOP = {best_epop:.0f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def get_bayesian_results(force_recalculate=False):
    global saved_bayesian_results, saved_bayesian_params, mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params, adapt_p_var, epop_bayesian_optimized, epop_optimized

    if mcmc_samples_R0 is None:
        print("❌ Нет данных MCMC. Запустите MCMC сначала.")
        return None
    
    if last_I_fact is None or last_params is None:
        print("❌ Нет данных JSON. Загрузите JSON-файл сначала.")
        return None

    try:
        mu_val = float(mu_entry.get())
    except:
        mu_val = last_params.get('mu', 0.001)

    if epop_bayesian_optimized is not None:
        epop_val = epop_bayesian_optimized
        print(f"🔧 Используем оптимизированный EPOP (байес) = {epop_val:.0f}")
    elif epop_optimized is not None:
        epop_val = epop_optimized
        print(f"🔧 Используем оптимизированный EPOP (сетка) = {epop_val:.0f}")
    else:
        epop_val = last_params.get('N', 31010)
        print(f"🔧 Используем EPOP из JSON = {epop_val:.0f}")
    
    print(f"🔧 Используем фиксированное mu = {mu_val:.8f} (из GUI)")

    current_params = {
        'window': int(window_entry.get() or 5),
        'k': float(k_entry.get() or 0.055),
        'r0_max': float(r0_max_entry.get() or 20.0),
        'mu': mu_val,
        'epop': epop_val,
        'adapt_p': adapt_p_var.get() if adapt_p_var is not None else True,
        'n_sim': 10,
        'noise_level': 0.08
    }

    if saved_bayesian_results is not None and not force_recalculate:
        if saved_bayesian_params == current_params:
            print("✅ Используем сохраненные байесовские траектории (кеш)")
            return saved_bayesian_results
        else:
            print("🔄 Параметры изменились, пересчитываем траектории...")

    if last_I_fact_abs is not None:
        I_fact_abs = last_I_fact_abs
    else:
        I_fact_abs = last_I_fact * (last_params.get('N', 1) if last_params is not None else 1)
    
    I_fact_norm = I_fact_abs / epop_val

    params_copy = last_params.copy()
    params_copy['N'] = epop_val

    print("🔄 Генерация байесовских траекторий...")
    
    window_val = current_params['window']
    k_val = current_params['k']
    r0_max_val = current_params['r0_max']
    use_adapt_p = current_params['adapt_p']
    
    result = generate_bayesian_trajectories(
        mcmc_samples_R0, mcmc_samples_p, I_fact_norm, params_copy,
        n_sim=10,
        window=window_val,
        k_val=k_val,
        r0_max=r0_max_val,
        return_p=True,
        return_S=True,
        return_I=True,
        add_noise=True,
        noise_level=0.08,
        adapt_p=use_adapt_p,
        mu_fixed=mu_val,
        verbose=False
    )

    saved_bayesian_results = result
    saved_bayesian_params = current_params
    
    print(f"✅ Байесовские траектории сохранены в кеш")
    print(f"   EPOP = {epop_val:.0f}, mu = {mu_val:.8f}")
    return result

def run_simulation(json_path):
    global last_S, last_I, current_best_R0, last_params, last_I_fact, last_I_fact_abs, current_best_p, current_transmission, current_I_fact, current_json_path, current_N, current_S, current_I, current_params, current_best_R0_arr, current_best_gamma, current_best_mu, adapt_p_var, epop_optimized, last_optimized_params
    try:
        window = int(window_entry.get())
        if window < 1:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "Window должен быть целым числом ≥ 1")
        return

    try:
        p_min = 0.0
        if p_min < 0:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "P min должен быть числом ≥ 0")
        return

    try:
        p_max = 0.4
        if p_max < p_min:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "P max должен быть числом > P min")
        return

    try:
        p_step = 0.001
        if p_step <= 0 or p_step > 0.01:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "P step должен быть числом > 0 и <= 0.01")
        return

    try:
        k_val = float(k_entry.get())
        if k_val < 0:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "k должен быть неотрицательным числом")
        return
   
    try:
        r0_max_val = float(r0_max_entry.get())
        if r0_max_val <= 0:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "R0_max должен быть положительным числом")
        return

    try:
        gamma_val = float(gamma_entry.get())
        if gamma_val <= 0:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "gamma должен быть положительным числом")
        return

    try:
        mu_val = float(mu_entry.get())
        if mu_val < 0:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "mu должен быть неотрицательным числом")
        return

    result = load_data_from_json(json_path)
    if result[0] is None:
        return
    
    params, I_fact, I_fact_abs = result

    if epop_optimized is not None:
        print(f"✅ Используем оптимизированный EPOP: {epop_optimized:.0f}")
        params['N'] = epop_optimized
        I_fact = I_fact_abs / epop_optimized
    else:
        print(f"ℹ️ Используем EPOP из JSON: {params['N']:.0f}")

    params['gamma'] = gamma_val

    use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True

    S, I, transmission, best_R0, best_p, best_mu = sir_euler_grid(
        params, I_fact, window=window, p_min=p_min, p_max=p_max, p_step=p_step, 
        k_val=k_val, r0_max=r0_max_val,
        adapt_p=use_adapt_p,
        mu_fixed=mu_val,
        use_grid_search=False                          
    )
    
    best_gamma = np.ones(len(best_R0)) * gamma_val

    current_best_R0 = best_R0
    last_S = S
    last_I = I
    last_params = params
    last_I_fact = I_fact
    last_I_fact_abs = I_fact_abs
    current_best_p = best_p
    current_best_mu = best_mu                      
    current_transmission = transmission * params.get('N', 1)
    current_I_fact = I_fact_abs
    current_json_path = json_path
    current_N = params.get('N')
    current_S = S
    current_I = I
    current_params = params
    current_best_R0_arr = best_R0
    current_best_gamma = best_gamma

    trans_abs = current_transmission
    mae = np.mean(np.abs(trans_abs - I_fact_abs))
    rmse = np.sqrt(np.mean((trans_abs - I_fact_abs) ** 2))
    r2 = 1 - np.sum((I_fact_abs - trans_abs)**2) / (np.sum((I_fact_abs - np.mean(I_fact_abs))**2) + 1e-9)
    euclidean_dist = np.linalg.norm(trans_abs - I_fact_abs)

    s_min = S.min()
    s_min_day = int(np.argmin(S))
    k_from_json = params.get('k', 0.0)
    p_changes = np.sum(np.diff(best_p) != 0)
    p_nonzero_days = np.sum(best_p > 0.000001)

    S0_opt = S[0]                                                        
    I0_opt = I[0]                                                         
    gamma_opt = np.mean(best_gamma) if best_gamma is not None else gamma_val
    mu_opt = np.mean(best_mu) if best_mu is not None else mu_val
    k_opt = k_val                                  
    p_opt = np.mean(best_p[best_p > 0.0001]) if np.any(best_p > 0.0001) else 0.0
    R0_opt = np.mean(best_R0)                                      
    EPOP_opt = params.get('N', 0)

    last_optimized_params = {
        'S0': S0_opt,
        'I0': I0_opt,
        'gamma': gamma_opt,
        'mu': mu_opt,
        'k': k_opt,
        'p': p_opt,
        'R0': R0_opt,
        'EPOP': EPOP_opt,
        'window': window,
        'r0_max': r0_max_val,
        'S0_abs': S0_opt * EPOP_opt,
        'I0_abs': I0_opt * EPOP_opt,
        'r2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'R_stats': {
            'Среднее R': R0_opt,
            'Мин R': best_R0.min(),
            'Макс R': best_R0.max(),
            'Станд R': best_R0.std(),
            'Дней R>1': np.sum(best_R0 > 1.0),
            'Процент R>1': np.sum(best_R0 > 1.0)/len(best_R0)*100,
            'Среднее p': p_opt,
            'Мин p': best_p.min(),
            'Макс p': best_p.max(),
            'Дней p>0': np.sum(best_p > 0.0001),
            'Интеграл p': best_p.sum()
        }
    }

    param_text = f"""
{'='*70}
🔧 ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ МОДЕЛИ (7 параметров)
{'='*70}

📊 ПАРАМЕТРЫ МОДЕЛИ SIR:

   1. S0  (начальная доля восприимчивых)     = {S0_opt:.10f}
   2. I0  (начальная доля инфицированных)    = {I0_opt:.10f}
   3. γ   (скорость выздоровления)           = {gamma_opt:.8f}
   4. μ   (скорость смертности/рождаемости)  = {mu_opt:.8f}
   5. k   (скорость потери иммунитета)       = {k_opt:.8f}
   6. p   (параметр преодоления иммунитета)  = {p_opt:.8f}
   7. R₀  (эффективное репродуктивное число) = {R0_opt:.6f}

{'─'*70}

📊 ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ:

   EPOP (численность населения)              = {EPOP_opt:.0f}
   Window (размер окна)                      = {window}
   R0_max (максимальное R)                   = {r0_max_val:.1f}

{'─'*70}

📊 АБСОЛЮТНЫЕ ЗНАЧЕНИЯ (с учетом EPOP):

   S0_abs  = {S0_opt * EPOP_opt:.2f}  (восприимчивые)
   I0_abs  = {I0_opt * EPOP_opt:.2f}  (инфицированные)
   EPOP    = {EPOP_opt:.0f}  (общая численность)

{'─'*70}

📊 СТАТИСТИКА ПОЛУЧЕННЫХ ПАРАМЕТРОВ:

   R₀ (среднее)   = {R0_opt:.6f}
   R₀ (минимум)   = {best_R0.min():.6f}
   R₀ (максимум)  = {best_R0.max():.6f}
   R₀ (стандарт)  = {best_R0.std():.6f}
   
   p (среднее)    = {p_opt:.8f}  (только где p > 0.0001)
   p (минимум)    = {best_p.min():.8f}
   p (максимум)   = {best_p.max():.8f}
   
   γ (фиксированное) = {gamma_opt:.8f}
   μ (фиксированное) = {mu_opt:.8f}

{'─'*70}

📊 КАЧЕСТВО МОДЕЛИ:

   R²  = {r2:.4f}
   MAE = {mae:.2f}
   RMSE = {rmse:.2f}

{'='*70}
"""

    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, param_text)

    t = np.arange(len(I_fact))

    save_btn = tk.Button(text_area.master, text="💾 Сохранить параметры в Excel", 
                         font=("Segoe UI", 10, "bold"),
                         command=lambda: save_optimized_params_to_excel(last_optimized_params),
                         bg="#4CAF50", fg="white", padx=10, pady=5)
    save_btn.pack(pady=5)
        
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    
    axs[0, 0].plot(t, trans_abs, 'r-', lw=2.5, label='model data')
    axs[0, 0].plot(t, I_fact_abs, 'g-', lw=1.8, label='real data')
    axs[0, 0].set_ylabel('New cases', fontsize=12)
    axs[0, 0].set_title('Model and real dynamics', fontsize=14)
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    axs[0, 0].set_ylim([0, 500])
    
    axs[0, 1].plot(t, trans_abs, 'r-', lw=2.5, label='model data')
    axs[0, 1].plot(t, I_fact_abs, 'g-', lw=1.8, label='real data')
    axs[0, 1].set_ylabel('New cases', fontsize=12)
    axs[0, 1].set_title('Model and real dynamics', fontsize=14)
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    
    axs[1, 0].plot(t, best_R0, 'b-', lw=2.5, label='R(t)')
    axs[1, 0].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R0 = 1')
    axs[1, 0].axhline(r0_max_val, color='orange', linestyle='--', alpha=0.5, label=f'R0_max = {r0_max_val:.1f}')
    axs[1, 0].set_xlabel('Day', fontsize=12)
    axs[1, 0].set_ylabel('R(t)', fontsize=12)
    axs[1, 0].set_title(f'R(t)', fontsize=14)
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    axs[1, 0].set_ylim([0, min(4, r0_max_val * 1.1)])
    
    axs[1, 1].plot(t, best_R0, 'b-', lw=2.5, label='R(t)')
    axs[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R0 = 1')
    axs[1, 1].axhline(r0_max_val, color='orange', linestyle='--', alpha=0.5, label=f'R0_max = {r0_max_val:.1f}')
    axs[1, 1].set_xlabel('Day', fontsize=12)
    axs[1, 1].set_ylabel('R(t)', fontsize=12)
    axs[1, 1].set_title(f'R(t)', fontsize=14)
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    axs[1, 1].set_ylim([0, r0_max_val * 1.1])
    
    p_threshold = 0.01
    p_masked = np.where(best_p >= p_threshold, best_p, 0)

    axs[2, 0].plot(t, p_masked, 'g-', lw=2.5, label=f'Adaptive Ψ(t)')
    axs[2, 0].axhline(0.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Ψ(t) = 0')
    axs[2, 0].axhline(p_min, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Ψ(t)_min = {p_min}')
    axs[2, 0].axhline(p_max, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Ψ(t)_max = {p_max}')
    axs[2, 0].fill_between(t, p_min, p_max, alpha=0.1, color='gray')
    if use_adapt_p:
        axs[2, 0].axvline(x=430, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='The beginning of adaptation Ψ(t)')

    if np.any(best_p >= p_threshold):
        max_idx = np.argmax(best_p)
        axs[2, 0].plot(max_idx, best_p[max_idx], 'ro', markersize=8, label=f'Max: {best_p[max_idx]:.4f}')
        p_above_threshold = best_p[best_p >= p_threshold]
        if len(p_above_threshold) > 0:
            min_above_idx = np.where(best_p == p_above_threshold.min())[0][0]
            axs[2, 0].plot(min_above_idx, best_p[min_above_idx], 'bo', markersize=8, label=f'Min: {best_p[min_above_idx]:.4f}')

    axs[2, 0].set_xlabel('Day', fontsize=12)
    axs[2, 0].set_ylabel('Ψ(t)', fontsize=12)
    axs[2, 0].set_title(f'Immune overcoming capacity dynamics', fontsize=14)
    axs[2, 0].legend(loc='best')
    axs[2, 0].grid(True, alpha=0.5, linestyle='-', linewidth=0.8)

    p_min_display = 0
    p_max_display = max(best_p.max() * 1.1, 0.05)
    if p_max_display - p_min_display < 0.02:
        p_max_display = p_min_display + 0.05
    axs[2, 0].set_ylim([p_min_display, p_max_display])
    
    axs[2, 1].plot(t, S, 'm-', lw=2.5, label='S(t)')
    axs[2, 1].set_xlabel('Day', fontsize=12)
    axs[2, 1].set_ylabel('S(t)', fontsize=12)
    axs[2, 1].set_title('Dynamic of susceptible', fontsize=14)
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    axs[2, 1].set_ylim([0, 1.05])

    plt.suptitle(f"Modeling of COVID-19 dynamics", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def add_save_button_to_text_area():
    btn_frame = tk.Frame(text_area.master)
    btn_frame.pack(pady=5)
    
    save_btn = tk.Button(
        btn_frame, 
        text="💾 Сохранить параметры в Excel", 
        font=("Segoe UI", 10, "bold"),
        command=lambda: save_optimized_params_to_excel(last_optimized_params),
        bg="#4CAF50", 
        fg="white", 
        padx=15, 
        pady=8,
        relief=tk.RAISED,
        bd=2
    )
    save_btn.pack(side=tk.LEFT, padx=5)
    
    copy_btn = tk.Button(
        btn_frame,
        text="📋 Копировать параметры",
        font=("Segoe UI", 10),
        command=lambda: copy_params_to_clipboard(last_optimized_params),
        bg="#2196F3",
        fg="white",
        padx=15,
        pady=8,
        relief=tk.RAISED,
        bd=2
    )
    copy_btn.pack(side=tk.LEFT, padx=5)
    
    return btn_frame

def copy_params_to_clipboard(params):
    if params is None:
        messagebox.showwarning("Предупреждение", "Сначала выполните симуляцию.")
        return
    
    text = "ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ МОДЕЛИ\n"
    text += "="*50 + "\n"
    text += f"S0  = {params['S0']:.10f}\n"
    text += f"I0  = {params['I0']:.10f}\n"
    text += f"γ   = {params['gamma']:.8f}\n"
    text += f"μ   = {params['mu']:.8f}\n"
    text += f"k   = {params['k']:.8f}\n"
    text += f"p   = {params['p']:.8f}\n"
    text += f"R₀  = {params['R0']:.6f}\n"
    text += f"EPOP = {params['EPOP']:.0f}\n"
    text += f"Window = {params['window']}\n"
    text += f"R² = {params['r2']:.4f}\n"
    text += f"MAE = {params['MAE']:.2f}\n"
    text += f"RMSE = {params['RMSE']:.2f}\n"
    
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Скопировано", "Параметры скопированы в буфер обмена")

def run_simulation_from_gui():
    if current_json_path is None:
        messagebox.showwarning("Предупреждение", "Сначала загрузите JSON-файл.")
        return
    run_simulation(current_json_path)

def find_and_plot_epop():
    best_epop, results = find_optimal_epop(True)
    if best_epop is not None and results is not None:
        plot_epop_search_results(results, best_epop)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        epop_values = sorted(results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(epop_values)))
        
        for i, epop in enumerate(epop_values[:5]):                       
            trans = results[epop]['transmission']
            ax1.plot(trans, color=colors[i], alpha=0.7, label=f'EPOP={epop:.0f}')
        
        ax1.plot(current_I_fact, 'k-', lw=2, label='Фактические данные')
        ax1.set_xlabel('День', fontsize=12)
        ax1.set_ylabel('Новые случаи', fontsize=12)
        ax1.set_title('Сравнение моделей для разных EPOP', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.4)

        best_trans = results[best_epop]['transmission']
        ax2.plot(best_trans, 'r-', lw=2, label=f'Модель (EPOP={best_epop:.0f})')
        ax2.plot(current_I_fact, 'b-', lw=2, label='Фактические данные')

        mae = results[best_epop]['MAE']
        rmse = results[best_epop]['RMSE']
        r2 = results[best_epop]['R2']
        ax2.text(0.02, 0.95, f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.4f}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_xlabel('День', fontsize=12)
        ax2.set_ylabel('Новые случаи', fontsize=12)
        ax2.set_title(f'Лучшая модель (EPOP={best_epop:.0f})', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.4)
        
        plt.suptitle(f'Подбор EPOP: лучшее значение = {best_epop:.0f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def find_optimal_epop_mu(force_recalculate=False):
    global current_best_R0_arr, current_I_fact, current_I_fact_abs, current_best_gamma, current_N, epop_optimized, epop_optimized_params, last_params, bayes_best_mu_global
    
    try:
                                  
        if current_best_R0_arr is None or current_I_fact is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала загрузите JSON и выполните симуляцию.")
            return None, None, None

        current_params = {
            'window': int(window_entry.get() or 5),
            'k': float(k_entry.get() or 0.055),
            'r0_max': float(r0_max_entry.get() or 20.0),
            'gamma': float(gamma_entry.get() or 0.074),
            'adapt_p': adapt_p_var.get() if adapt_p_var is not None else True
        }

        if epop_optimized is not None and bayes_best_mu_global is not None and not force_recalculate:
            if epop_optimized_params == current_params:
                print(f"✅ Используем оптимизированные параметры:")
                print(f"   EPOP = {epop_optimized:.0f}")
                print(f"   mu = {bayes_best_mu_global:.8f}")
                return epop_optimized, bayes_best_mu_global, None

        try:
            epop_min = float(epop_min_entry.get() or 10000)
            epop_max = float(epop_max_entry.get() or 100000)
            epop_step = float(epop_step_entry.get() or 5000)
        except:
            epop_min = 10000
            epop_max = 100000
            epop_step = 5000
        
        try:
            mu_min = float(mu_min_grid_entry.get() or 0.0)
            mu_max = float(mu_max_grid_entry.get() or 0.025)
            mu_step = float(mu_step_grid_entry.get() or 0.001)
        except:
            mu_min = 0.0
            mu_max = 0.025
            mu_step = 0.001

        epop_values = np.arange(epop_min, epop_max + epop_step/2, epop_step)
        mu_values = np.arange(mu_min, mu_max + mu_step/2, mu_step)
        
        print("\n" + "="*70)
        print("🔍 СОВМЕСТНЫЙ ПОДБОР EPOP И mu")
        print("="*70)
        print(f"   EPOP: [{epop_min:.0f} - {epop_max:.0f}], шаг: {epop_step:.0f}")
        print(f"   mu:    [{mu_min:.6f} - {mu_max:.6f}], шаг: {mu_step:.6f}")
        print(f"   Всего комбинаций: {len(epop_values) * len(mu_values)}")
        print("-"*70)
        
        best_epop = None
        best_mu = None
        best_error = np.inf
        results = {}

        gamma = current_best_gamma if current_best_gamma is not None else float(gamma_entry.get() or 0.074)
        window = int(window_entry.get() or 5)
        k_val = float(k_entry.get() or 0.055)
        r0_max_val = float(r0_max_entry.get() or 20.0)
        use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True

        I_fact_abs = current_I_fact
        
        total_combinations = len(epop_values) * len(mu_values)
        counter = 0
        
        for epop_candidate in epop_values:
            for mu_candidate in mu_values:
                counter += 1
                print(f"\n🔄 [{counter}/{total_combinations}] EPOP={epop_candidate:.0f}, mu={mu_candidate:.6f}")

                I_fact_norm = I_fact_abs / epop_candidate

                params_copy = last_params.copy()
                params_copy['N'] = epop_candidate
                params_copy['mu'] = mu_candidate

                S, I, transmission, R0, p, mu_arr = sir_euler_grid(
                    params_copy, 
                    I_fact_norm,
                    window=window,
                    p_min=0.0,
                    p_max=0.4,
                    p_step=0.001,
                    k_val=k_val,
                    r0_max=r0_max_val,
                    adapt_p=use_adapt_p,
                    mu_fixed=mu_candidate
                )

                trans_abs = transmission * epop_candidate

                mae = np.mean(np.abs(trans_abs - I_fact_abs))
                rmse = np.sqrt(np.mean((trans_abs - I_fact_abs) ** 2))
                ss_res = np.sum((I_fact_abs - trans_abs) ** 2)
                ss_tot = np.sum((I_fact_abs - np.mean(I_fact_abs)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-9)

                key = f"EPOP_{epop_candidate:.0f}_mu_{mu_candidate:.6f}"
                results[key] = {
                    'EPOP': epop_candidate,
                    'mu': mu_candidate,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'S': S,
                    'I': I,
                    'R0': R0,
                    'p': p,
                    'transmission': trans_abs
                }
                
                print(f"   MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
                
                if mae < best_error:
                    best_error = mae
                    best_epop = epop_candidate
                    best_mu = mu_candidate
                    print(f"   ✅ НОВЫЙ ЛУЧШИЙ! EPOP={epop_candidate:.0f}, mu={mu_candidate:.6f}, MAE={mae:.2f}")

        epop_optimized = best_epop
        bayes_best_mu_global = best_mu
        epop_optimized_params = current_params

        mu_entry.delete(0, tk.END)
        mu_entry.insert(0, f"{best_mu:.8f}")
        
        print("\n" + "="*70)
        print(f"✅ СОВМЕСТНЫЙ ПОДБОР ЗАВЕРШЕН!")
        print(f"   Лучший EPOP = {best_epop:.0f}")
        print(f"   Лучшее mu = {best_mu:.8f}")
        print(f"   Лучшая MAE = {best_error:.2f}")
        print(f"   R² = {results[f'EPOP_{best_epop:.0f}_mu_{best_mu:.6f}']['R2']:.4f}")
        print("="*70)

        text_area.insert(tk.END, f"\n✅ Оптимальные параметры:\n")
        text_area.insert(tk.END, f"   EPOP = {best_epop:.0f}\n")
        text_area.insert(tk.END, f"   mu = {best_mu:.8f}\n")
        text_area.insert(tk.END, f"   MAE = {best_error:.2f}\n")
        text_area.insert(tk.END, f"   R² = {results[f'EPOP_{best_epop:.0f}_mu_{best_mu:.6f}']['R2']:.4f}\n")
        text_area.see(tk.END)
        
        return best_epop, best_mu, results
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при подборе параметров: {str(e)}")
        return None, None, None

def plot_epop_mu_search_results(results, best_epop, best_mu):
    if results is None:
        return

    epop_values = sorted(set([r['EPOP'] for r in results.values()]))
    mu_values = sorted(set([r['mu'] for r in results.values()]))

    mae_matrix = np.zeros((len(mu_values), len(epop_values)))
    r2_matrix = np.zeros((len(mu_values), len(epop_values)))
    
    for i, mu in enumerate(mu_values):
        for j, epop in enumerate(epop_values):
            key = f"EPOP_{epop:.0f}_mu_{mu:.6f}"
            if key in results:
                mae_matrix[i, j] = results[key]['MAE']
                r2_matrix[i, j] = results[key]['R2']
            else:
                mae_matrix[i, j] = np.nan
                r2_matrix[i, j] = np.nan
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    im1 = ax1.imshow(mae_matrix, aspect='auto', cmap='viridis', 
                     origin='lower', interpolation='bilinear',
                     extent=[epop_values[0], epop_values[-1], mu_values[0], mu_values[-1]])
    ax1.scatter(best_epop, best_mu, color='red', s=100, marker='*', 
                label=f'Лучший: EPOP={best_epop:.0f}, mu={best_mu:.6f}')
    ax1.set_xlabel('EPOP', fontsize=12)
    ax1.set_ylabel('mu', fontsize=12)
    ax1.set_title('Тепловая карта MAE', fontsize=14, fontweight='bold')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='MAE')

    im2 = ax2.imshow(r2_matrix, aspect='auto', cmap='plasma', 
                     origin='lower', interpolation='bilinear',
                     extent=[epop_values[0], epop_values[-1], mu_values[0], mu_values[-1]])
    ax2.scatter(best_epop, best_mu, color='red', s=100, marker='*',
                label=f'Лучший: EPOP={best_epop:.0f}, mu={best_mu:.6f}')
    ax2.set_xlabel('EPOP', fontsize=12)
    ax2.set_ylabel('mu', fontsize=12)
    ax2.set_title('Тепловая карта R²', fontsize=14, fontweight='bold')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='R²')

    best_mu_data = [r for r in results.values() if abs(r['mu'] - best_mu) < 1e-9]
    best_mu_data.sort(key=lambda x: x['EPOP'])
    
    if best_mu_data:
        epop_best_mu = [d['EPOP'] for d in best_mu_data]
        mae_best_mu = [d['MAE'] for d in best_mu_data]
        r2_best_mu = [d['R2'] for d in best_mu_data]
        
        ax3.plot(epop_best_mu, mae_best_mu, 'b-o', lw=2, markersize=8, label=f'MAE (mu={best_mu:.6f})')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(epop_best_mu, r2_best_mu, 'r-s', lw=2, markersize=8, label=f'R² (mu={best_mu:.6f})')
        ax3.axvline(best_epop, color='green', linestyle='--', alpha=0.7,
                    label=f'Лучший EPOP={best_epop:.0f}')
        ax3.set_xlabel('EPOP', fontsize=12)
        ax3.set_ylabel('MAE', fontsize=12, color='blue')
        ax3_twin.set_ylabel('R²', fontsize=12, color='red')
        ax3.set_title(f'Зависимость метрик от EPOP при mu={best_mu:.6f}', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.4)

    best_key = f"EPOP_{best_epop:.0f}_mu_{best_mu:.6f}"
    if best_key in results:
        stats_text = f"""
    ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:
    
    EPOP = {best_epop:.0f}
    mu = {best_mu:.8f}
    
    КАЧЕСТВО:
    MAE = {results[best_key]['MAE']:.2f}
    RMSE = {results[best_key]['RMSE']:.2f}
    R² = {results[best_key]['R2']:.4f}
    
    ПРОВЕРЕНО:
    EPOP: {len(epop_values)} значений
    mu: {len(mu_values)} значений
    Всего: {len(results)} комбинаций
    """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                 fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Итоговая статистика', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Совместный подбор EPOP и mu\nЛучший EPOP={best_epop:.0f}, mu={best_mu:.8f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_4_graphs():
    global last_S, last_I, last_params, last_I_fact, current_json_path, current_best_R0_arr, current_best_p, current_transmission, current_I_fact, current_error_matrix, current_r0_grid, current_S
    
    try:
                                  
        if current_best_R0_arr is None or current_best_p is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала загрузите JSON и выполните симуляцию.")
            return

        days = len(current_best_R0_arr)
        t = np.arange(days)

        if current_best_gamma is not None:
            gamma = np.mean(current_best_gamma)
        else:
            try:
                gamma = float(gamma_entry.get() or 0.074)
            except:
                gamma = 0.074

        try:
            mu_val = float(mu_entry.get() or 0.001)
        except:
            mu_val = 0.001

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Анализ модели SIR (μ = {mu_val:.8f}, γ = {gamma:.6f})', 
                     fontsize=18, fontweight='bold')

        R_data = current_best_R0_arr
        R_mean = np.mean(R_data)
        R_std = np.std(R_data)
        R_max = np.max(R_data)
        R_min = np.min(R_data)
        R_last = R_data[-1]
        R_gt_1 = np.sum(R_data > 1.0)
        
        ax1.plot(t, R_data, 'b-', lw=2.5, label='R(t)')
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R = 1')
        ax1.axhline(R_mean, color='green', linestyle=':', alpha=0.7,
                   label=f'Среднее R = {R_mean:.4f}')
        ax1.fill_between(t, 0, R_data, alpha=0.1, color='blue')
        ax1.set_xlabel('День', fontsize=12)
        ax1.set_ylabel('R(t)', fontsize=12)
        ax1.set_title(f'1. Репродуктивное число R(t)\nСреднее = {R_mean:.4f}, Последнее = {R_last:.4f}', 
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.4)
        ax1.set_ylim([0, max(3, np.max(R_data) * 1.1)])

        stats_text_R = f'Max = {R_max:.4f}\nMin = {R_min:.4f}\nStd = {R_std:.4f}\nДней R>1 = {R_gt_1}'
        ax1.text(0.02, 0.95, stats_text_R, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        if current_error_matrix is None or current_r0_grid is None:
            try:
                window = int(window_entry.get())
                k_val = float(k_entry.get())
                r0_max_val = float(r0_max_entry.get())
            except:
                window = 5
                k_val = 0.055
                r0_max_val = 20.0

            if last_S is not None and last_I is not None and last_params is not None:
                error_matrix, r0_grid, _, _ = build_r0_day_matrix(
                    last_params, last_I_fact, last_S, last_I,
                    window=window, k_val=k_val, r0_max=r0_max_val
                )
                current_error_matrix = error_matrix
                current_r0_grid = r0_grid
            else:
                ax2.text(0.5, 0.5, 'Нет данных для тепловой матрицы\nЗапустите симуляцию сначала',
                        ha='center', va='center', fontsize=14, transform=ax2.transAxes)
                ax2.axis('off')
        else:
            error_matrix = current_error_matrix
            r0_grid = current_r0_grid

            min_val = np.min(error_matrix[error_matrix > 0]) if np.any(error_matrix > 0) else 1
            ratio = np.max(error_matrix) / (min_val + 1e-10)
            
            if ratio > 10000:
                data_plot = np.log10(np.where(error_matrix > 0, error_matrix, min_val / 100))
                cbar_label = 'log10(Ошибка)'
            else:
                data_plot = error_matrix
                cbar_label = 'Ошибка'
            
            im = ax2.imshow(data_plot, aspect='auto', cmap='viridis', 
                           origin='lower', interpolation='nearest')
            
            ax2.set_xlabel('День', fontsize=12)
            ax2.set_ylabel('R0', fontsize=12)
            ax2.set_title('2. Тепловая матрица ошибок R0×Дни', fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label(cbar_label, fontsize=10)

            if len(r0_grid) > 0:
                best_r0_idx = np.argmin(np.mean(error_matrix, axis=1))
                ax2.axhline(y=best_r0_idx, color='red', linestyle='--', alpha=0.5, 
                           label=f'Оптимальный R0 ≈ {r0_grid[best_r0_idx]:.2f}')
                ax2.legend(loc='upper right')

        p_data = current_best_p
        p_mean = np.mean(p_data)
        p_std = np.std(p_data)
        p_max = np.max(p_data)
        p_min = np.min(p_data)
        p_last = p_data[-1]
        p_active_mean = np.mean(p_data[p_data > 0.0001]) if np.any(p_data > 0.0001) else 0
        p_active_days = np.sum(p_data > 0.0001)
        p_integral = np.sum(p_data)
        
        p_threshold = 0.01
        p_masked = np.where(p_data >= p_threshold, p_data, 0)
        
        ax3.plot(t, p_data, 'g-', lw=2, label='p(t) (адаптивная)', alpha=0.5)
        ax3.plot(t, p_masked, 'g-', lw=2.5, label=f'p(t) (порог > {p_threshold})')
        ax3.axhline(0, color='red', linestyle='--', alpha=0.7, label='p = 0')
        ax3.axhline(p_mean, color='blue', linestyle=':', alpha=0.7,
                   label=f'Среднее p = {p_mean:.6f}')

        if p_active_days > 0:
            ax3.axhline(p_active_mean, color='orange', linestyle=':', alpha=0.7,
                       label=f'Среднее p (активные) = {p_active_mean:.6f}')

        use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True
        if use_adapt_p:
            ax3.axvline(x=430, color='purple', linestyle='--', alpha=0.7,
                       label='Начало адаптации p (день 430)')

        if np.any(p_data > 0.001):
            max_idx = np.argmax(p_data)
            ax3.plot(max_idx, p_data[max_idx], 'ro', markersize=8,
                    label=f'Max p = {p_data[max_idx]:.6f}')
        
        ax3.set_xlabel('День', fontsize=12)
        ax3.set_ylabel('p(t)', fontsize=12)

        title_p = f'3. Иммунная нагрузка p(t)\n'
        title_p += f'Среднее = {p_mean:.6f}, Последнее = {p_last:.6f}'
        if p_active_days > 0:
            title_p += f', Активных дней = {p_active_days}'
        ax3.set_title(title_p, fontsize=14, fontweight='bold')
        
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.4)

        p_max_display = max(p_data.max() * 1.1, 0.05)
        ax3.set_ylim([0, p_max_display])

        stats_text_p = f'Max = {p_max:.6f}\nMin = {p_min:.6f}\nStd = {p_std:.6f}\n'
        stats_text_p += f'Суммарная p = {p_integral:.6f}\n'
        stats_text_p += f'Дней p>0 = {p_active_days}'
        ax3.text(0.98, 0.95, stats_text_p, transform=ax3.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        if current_S is not None:
            S_data = current_S
        elif last_S is not None:
            S_data = last_S
        else:
            ax4.text(0.5, 0.5, 'Нет данных для S(t)\nЗапустите симуляцию сначала',
                    ha='center', va='center', fontsize=14, transform=ax4.transAxes)
            ax4.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            plt.show()
            return

        S_mean = np.mean(S_data)
        S_std = np.std(S_data)
        S_max = np.max(S_data)
        S_min = np.min(S_data)
        S_last = S_data[-1]
        S0 = S_data[0]
        S_min_day = np.argmin(S_data)
        S_decline = S0 - S_last
        
        ax4.plot(t, S_data, 'm-', lw=2.5, label='S(t)')
        ax4.axhline(S_mean, color='blue', linestyle=':', alpha=0.7,
                   label=f'Среднее S = {S_mean:.4f}')
        ax4.axhline(S0, color='green', linestyle='--', alpha=0.5,
                   label=f'S₀ = {S0:.4f}')
        ax4.axhline(S_min, color='red', linestyle='--', alpha=0.5,
                   label=f'S_min = {S_min:.4f} (день {S_min_day})')
        ax4.fill_between(t, 0, S_data, alpha=0.1, color='magenta')

        ax4.plot(S_min_day, S_min, 'ro', markersize=8,
                label=f'Минимум S = {S_min:.4f}')
        
        ax4.set_xlabel('День', fontsize=12)
        ax4.set_ylabel('S(t) (доля восприимчивых)', fontsize=12)
        ax4.set_title(f'4. Динамика восприимчивых S(t)\nS₀ = {S0:.4f}, S_мин = {S_min:.4f}, S_кон = {S_last:.4f}', 
                      fontsize=14, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.4)
        ax4.set_ylim([0, 1.05])

        stats_text_S = f'Max = {S_max:.4f}\nMin = {S_min:.4f}\nStd = {S_std:.4f}\n'
        stats_text_S += f'Падение S = {S_decline:.4f} ({S_decline/S0*100:.1f}%)\n'
        stats_text_S += f'День минимума = {S_min_day}'
        ax4.text(0.02, 0.05, stats_text_S, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при построении 4 графиков: {str(e)}")

def create_instruction_text():
    instruction = """
    📖 ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ
    ====================================
    
    1. Загрузить JSON-файл
    2. Ввести минимум, максимум населения и шаг (Поделить разницу между минимумом и максимумом на число шагов)
    3. Запустить симуляцию
    4. Подобрать эффективное население (EPOP)
    5. Запустить симуляцию еще раз
    6. Запустить тепловую диаграмму
    7. Нажать кнопку "Четыре графика"
    8. Получить таблицы R(t) и Y(t)
    9. Запустить программу регрессии 
    10. Загрузить Excel-файл и выполнить регрессию
    """
    return instruction

def find_optimal_epop(force_recalculate=False):
    global current_best_R0_arr, current_I_fact, current_I_fact_abs, current_best_gamma, current_N, epop_optimized, epop_optimized_params, last_params
    
    try:
                                  
        if current_best_R0_arr is None or current_I_fact is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала загрузите JSON и выполните симуляцию.")
            return None, None

        current_params = {
            'window': int(window_entry.get() or 5),
            'k': float(k_entry.get() or 0.055),
            'r0_max': float(r0_max_entry.get() or 20.0),
            'gamma': float(gamma_entry.get() or 0.074),
            'mu': float(mu_entry.get() or 0.001),
            'adapt_p': adapt_p_var.get() if adapt_p_var is not None else True
        }

        if epop_optimized is not None and not force_recalculate:
            if epop_optimized_params == current_params:
                print(f"✅ Используем оптимизированный EPOP = {epop_optimized:.0f}")
                return epop_optimized, None

        try:
            epop_min = float(epop_min_entry.get() or 10000)
            epop_max = float(epop_max_entry.get() or 100000)
            epop_step = float(epop_step_entry.get() or 5000)
        except:
            epop_min = 10000
            epop_max = 100000
            epop_step = 5000

        epop_values = np.arange(epop_min, epop_max + epop_step/2, epop_step)
        
        print("\n" + "="*70)
        print("🔍 ПОДБОР ОПТИМАЛЬНОГО EPOP")
        print("="*70)
        print(f"   Диапазон: [{epop_min:.0f} - {epop_max:.0f}], шаг: {epop_step:.0f}")
        print(f"   Количество значений: {len(epop_values)}")
        print(f"   mu фиксированное: {current_params['mu']:.8f} (из GUI)")
        print("-"*70)
        
        best_epop = None
        best_error = np.inf
        results = {}

        gamma = current_best_gamma if current_best_gamma is not None else float(gamma_entry.get() or 0.074)
        window = int(window_entry.get() or 5)
        k_val = float(k_entry.get() or 0.055)
        r0_max_val = float(r0_max_entry.get() or 20.0)
        mu_val = float(mu_entry.get() or 0.001)
        use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True

        I_fact_abs = current_I_fact
        
        for idx, epop_candidate in enumerate(epop_values):
            print(f"\n🔄 [{idx+1}/{len(epop_values)}] EPOP = {epop_candidate:.0f}...")

            I_fact_norm = I_fact_abs / epop_candidate

            params_copy = last_params.copy()
            params_copy['N'] = epop_candidate

            S, I, transmission, R0, p, mu_arr = sir_euler_grid(
                params_copy, 
                I_fact_norm,
                window=window,
                p_min=0.0,
                p_max=0.4,
                p_step=0.001,
                k_val=k_val,
                r0_max=r0_max_val,
                adapt_p=use_adapt_p,
                mu_fixed=mu_val                        
            )

            trans_abs = transmission * epop_candidate

            mae = np.mean(np.abs(trans_abs - I_fact_abs))
            rmse = np.sqrt(np.mean((trans_abs - I_fact_abs) ** 2))
            r2 = 1 - np.sum((I_fact_abs - trans_abs)**2) / (np.sum((I_fact_abs - np.mean(I_fact_abs))**2) + 1e-9)
            
            results[epop_candidate] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'S': S,
                'I': I,
                'R0': R0,
                'p': p,
                'transmission': trans_abs
            }
            
            print(f"   MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
            
            if mae < best_error:
                best_error = mae
                best_epop = epop_candidate
                print(f"   ✅ НОВЫЙ ЛУЧШИЙ! EPOP = {epop_candidate:.0f}, MAE = {mae:.2f}")

        epop_optimized = best_epop
        epop_optimized_params = current_params
        
        print("\n" + "="*70)
        print(f"✅ ПОДБОР EPOP ЗАВЕРШЕН!")
        print(f"   Лучший EPOP = {best_epop:.0f}")
        print(f"   Лучшая MAE = {best_error:.2f}")
        print(f"   R² = {results[best_epop]['R2']:.4f}")
        print(f"   mu = {mu_val:.8f} (фиксированное)")
        print("="*70)

        text_area.insert(tk.END, f"\n✅ Оптимальный EPOP: {best_epop:.0f}\n")
        text_area.insert(tk.END, f"   MAE = {best_error:.2f}\n")
        text_area.insert(tk.END, f"   R² = {results[best_epop]['R2']:.4f}\n")
        text_area.insert(tk.END, f"   mu = {mu_val:.8f} (фиксированное)\n")
        text_area.see(tk.END)
        
        return best_epop, results
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при подборе EPOP: {str(e)}")
        return None, None

def plot_epop_search_results(results, best_epop, mu_val):
    if results is None:
        return
    
    epop_values = sorted(results.keys())
    mae_values = [results[epop]['MAE'] for epop in epop_values]
    rmse_values = [results[epop]['RMSE'] for epop in epop_values]
    r2_values = [results[epop]['R2'] for epop in epop_values]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.plot(epop_values, mae_values, 'b-o', lw=2, markersize=8, label='MAE')
    ax1.axvline(best_epop, color='red', linestyle='--', alpha=0.7, 
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax1.axhline(min(mae_values), color='green', linestyle='--', alpha=0.7,
                label=f'Лучшая MAE = {min(mae_values):.2f}')
    ax1.set_xlabel('EPOP', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title(f'Зависимость MAE от EPOP (μ={mu_val:.6f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(epop_values, rmse_values, 'r-o', lw=2, markersize=8, label='RMSE')
    ax2.axvline(best_epop, color='red', linestyle='--', alpha=0.7,
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax2.set_xlabel('EPOP', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title(f'Зависимость RMSE от EPOP (μ={mu_val:.6f})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    ax3.plot(epop_values, r2_values, 'g-o', lw=2, markersize=8, label='R²')
    ax3.axvline(best_epop, color='red', linestyle='--', alpha=0.7,
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax3.axhline(max(r2_values), color='blue', linestyle='--', alpha=0.7,
                label=f'Лучший R² = {max(r2_values):.4f}')
    ax3.set_xlabel('EPOP', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title(f'Зависимость R² от EPOP (μ={mu_val:.6f})', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.4)

    best_idx = np.argmin(mae_values)
    stats_text = f"""
    ОПТИМАЛЬНЫЙ EPOP: {best_epop:.0f}
    
    КАЧЕСТВО:
    MAE = {min(mae_values):.2f}
    RMSE = {rmse_values[best_idx]:.2f}
    R² = {r2_values[best_idx]:.4f}
    
    ПАРАМЕТРЫ:
    mu = {mu_val:.8f} (фиксированное)
    
    ПРОВЕРЕНО:
    {len(epop_values)} значений EPOP
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Итоговая статистика', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Подбор оптимального EPOP\nЛучший EPOP = {best_epop:.0f}, mu = {mu_val:.8f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def find_and_plot_epop():
    best_epop, results = find_optimal_epop(True)
    if best_epop is not None and results is not None:
        mu_val = float(mu_entry.get() or 0.001)

        plot_epop_search_results(results, best_epop, mu_val)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        epop_values = sorted(results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(epop_values))))

        step = max(1, len(epop_values) // 5)
        for i, epop in enumerate(epop_values[::step][:5]):
            trans = results[epop]['transmission']
            ax1.plot(trans, color=colors[i], alpha=0.7, label=f'EPOP={epop:.0f}')
        
        ax1.plot(current_I_fact, 'k-', lw=2, label='Фактические данные')
        ax1.set_xlabel('День', fontsize=12)
        ax1.set_ylabel('Новые случаи', fontsize=12)
        ax1.set_title('Сравнение моделей для разных EPOP', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.4)

        best_trans = results[best_epop]['transmission']
        ax2.plot(best_trans, 'r-', lw=2, label=f'Модель (EPOP={best_epop:.0f})')
        ax2.plot(current_I_fact, 'b-', lw=2, label='Фактические данные')

        mae = results[best_epop]['MAE']
        rmse = results[best_epop]['RMSE']
        r2 = results[best_epop]['R2']
        ax2.text(0.02, 0.95, f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.4f}\nμ = {mu_val:.6f}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_xlabel('День', fontsize=12)
        ax2.set_ylabel('Новые случаи', fontsize=12)
        ax2.set_title(f'Лучшая модель (EPOP={best_epop:.0f})', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.4)
        
        plt.suptitle(f'Подбор EPOP: лучшее значение = {best_epop:.0f}, mu = {mu_val:.8f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        global epop_optimized
        epop_optimized = best_epop

        messagebox.showinfo("Успешно", 
            f"Найден оптимальный EPOP:\n"
            f"EPOP = {best_epop:.0f}\n"
            f"mu = {mu_val:.8f} (фиксированное)\n"
            f"MAE = {mae:.2f}\n"
            f"R² = {r2:.4f}\n\n"
            f"Нажмите 'Запустить симуляцию' для применения.")

def find_optimal_epop_bayesian(force_recalculate=False):
    global mcmc_samples_R0, mcmc_samples_p, last_I_fact, last_params, current_best_gamma, adapt_p_var, current_N, last_I_fact_abs, epop_bayesian_optimized, epop_bayesian_optimized_params
    
    try:
                                  
        if mcmc_samples_R0 is None:
            messagebox.showwarning("Предупреждение", "Сначала запустите MCMC.")
            return None, None
        
        if last_I_fact is None or last_params is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите JSON-файл.")
            return None, None

        current_params = {
            'window': int(window_entry.get() or 5),
            'k': float(k_entry.get() or 0.055),
            'r0_max': float(r0_max_entry.get() or 20.0),
            'gamma': float(gamma_entry.get() or 0.074),
            'mu': float(mu_entry.get() or 0.001),
            'adapt_p': adapt_p_var.get() if adapt_p_var is not None else True,
            'n_sim': 10
        }

        if epop_bayesian_optimized is not None and not force_recalculate:
            if epop_bayesian_optimized_params == current_params:
                print(f"✅ Используем оптимизированный EPOP (байес) = {epop_bayesian_optimized:.0f}")
                return epop_bayesian_optimized, None

        try:
            epop_min = float(epop_min_entry.get() or 10000)
            epop_max = float(epop_max_entry.get() or 100000)
            epop_step = float(epop_step_entry.get() or 5000)
        except:
            epop_min = 10000
            epop_max = 100000
            epop_step = 5000

        epop_values = np.arange(epop_min, epop_max + epop_step/2, epop_step)
        
        print("\n" + "="*70)
        print("🔍 ПОДБОР ОПТИМАЛЬНОГО EPOP ДЛЯ БАЙЕСОВСКОЙ МОДЕЛИ")
        print("="*70)
        print(f"   Диапазон: [{epop_min:.0f} - {epop_max:.0f}], шаг: {epop_step:.0f}")
        print(f"   Количество значений: {len(epop_values)}")
        print(f"   Симуляций на каждое EPOP: {current_params['n_sim']}")
        print(f"   mu фиксированное: {current_params['mu']:.8f} (из GUI)")
        print("-"*70)
        
        best_epop = None
        best_error = np.inf
        results = {}

        gamma = current_best_gamma if current_best_gamma is not None else float(gamma_entry.get() or 0.074)
        window = int(window_entry.get() or 5)
        k_val = float(k_entry.get() or 0.055)
        r0_max_val = float(r0_max_entry.get() or 20.0)
        mu_val = float(mu_entry.get() or 0.001)
        use_adapt_p = adapt_p_var.get() if adapt_p_var is not None else True
        n_sim = 10

        if last_I_fact_abs is not None:
            I_fact_abs = last_I_fact_abs
        else:
            I_fact_abs = last_I_fact * (last_params.get('N', 1) if last_params is not None else 1)
        
        for idx, epop_candidate in enumerate(epop_values):
            print(f"\n🔄 [{idx+1}/{len(epop_values)}] EPOP = {epop_candidate:.0f}...")

            I_fact_norm = I_fact_abs / epop_candidate

            params_copy = last_params.copy()
            params_copy['N'] = epop_candidate

            result = generate_bayesian_trajectories(
                mcmc_samples_R0, mcmc_samples_p, I_fact_norm, params_copy,
                n_sim=n_sim,
                window=window,
                k_val=k_val,
                r0_max=r0_max_val,
                return_p=True,
                return_I=True,
                add_noise=True,
                noise_level=0.05,
                adapt_p=use_adapt_p,
                mu_fixed=mu_val,
                verbose=False
            )

            all_I = result['I']
            all_R0 = result['R0']
            
            all_transmission = []
            for i in range(len(all_I)):
                I_traj = all_I[i]
                R0_traj = all_R0[i]
                I_traj = np.clip(I_traj, 0, 0.5)
                S_traj = np.maximum(1.0 - I_traj, 0)
                trans_traj = R0_traj * gamma * S_traj * I_traj
                trans_traj = np.maximum(trans_traj, 0)
                all_transmission.append(trans_traj)
            
            all_transmission = np.array(all_transmission)
            mean_transmission = np.mean(all_transmission, axis=0) * epop_candidate

            min_len = min(len(mean_transmission), len(I_fact_abs))
            mean_transmission = mean_transmission[:min_len]
            I_fact_abs_trimmed = I_fact_abs[:min_len]

            mae = np.mean(np.abs(mean_transmission - I_fact_abs_trimmed))
            rmse = np.sqrt(np.mean((mean_transmission - I_fact_abs_trimmed) ** 2))
            ss_res = np.sum((I_fact_abs_trimmed - mean_transmission) ** 2)
            ss_tot = np.sum((I_fact_abs_trimmed - np.mean(I_fact_abs_trimmed)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-9)

            results[epop_candidate] = {
                'result': result,
                'mean_transmission': mean_transmission,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'mean_R0': result['mean_R0'],
                'mean_p': result['mean_p'] if 'mean_p' in result else None
            }
            
            print(f"   MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
            
            if mae < best_error:
                best_error = mae
                best_epop = epop_candidate
                print(f"   ✅ НОВЫЙ ЛУЧШИЙ! EPOP = {epop_candidate:.0f}, MAE = {mae:.2f}")

        epop_bayesian_optimized = best_epop
        epop_bayesian_optimized_params = current_params
        
        print("\n" + "="*70)
        print(f"✅ ПОДБОР EPOP ДЛЯ БАЙЕСА ЗАВЕРШЕН!")
        print(f"   Лучший EPOP = {best_epop:.0f}")
        print(f"   Лучшая MAE = {best_error:.2f}")
        print(f"   R² = {results[best_epop]['R2']:.4f}")
        print(f"   mu = {mu_val:.8f} (фиксированное)")
        print("="*70)

        text_area.insert(tk.END, f"\n✅ Оптимальный EPOP для байеса: {best_epop:.0f}\n")
        text_area.insert(tk.END, f"   MAE = {best_error:.2f}\n")
        text_area.insert(tk.END, f"   R² = {results[best_epop]['R2']:.4f}\n")
        text_area.insert(tk.END, f"   mu = {mu_val:.8f} (фиксированное)\n")
        text_area.see(tk.END)

        global epop_optimized
        epop_optimized = best_epop
        
        return best_epop, results
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при подборе EPOP для байеса: {str(e)}")
        return None, None

def plot_epop_bayesian_search_results(results, best_epop, mu_val):
    if results is None:
        return
    
    epop_values = sorted(results.keys())
    mae_values = [results[epop]['MAE'] for epop in epop_values]
    rmse_values = [results[epop]['RMSE'] for epop in epop_values]
    r2_values = [results[epop]['R2'] for epop in epop_values]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.plot(epop_values, mae_values, 'b-o', lw=2, markersize=8, label='MAE')
    ax1.axvline(best_epop, color='red', linestyle='--', alpha=0.7, 
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax1.axhline(min(mae_values), color='green', linestyle='--', alpha=0.7,
                label=f'Лучшая MAE = {min(mae_values):.2f}')
    ax1.set_xlabel('EPOP', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title(f'Зависимость MAE от EPOP (байес, μ={mu_val:.6f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(epop_values, rmse_values, 'r-o', lw=2, markersize=8, label='RMSE')
    ax2.axvline(best_epop, color='red', linestyle='--', alpha=0.7,
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax2.set_xlabel('EPOP', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title(f'Зависимость RMSE от EPOP (байес, μ={mu_val:.6f})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    ax3.plot(epop_values, r2_values, 'g-o', lw=2, markersize=8, label='R²')
    ax3.axvline(best_epop, color='red', linestyle='--', alpha=0.7,
                label=f'Лучший EPOP = {best_epop:.0f}')
    ax3.axhline(max(r2_values), color='blue', linestyle='--', alpha=0.7,
                label=f'Лучший R² = {max(r2_values):.4f}')
    ax3.set_xlabel('EPOP', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title(f'Зависимость R² от EPOP (байес, μ={mu_val:.6f})', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.4)

    best_idx = np.argmin(mae_values)

    best_result = results[best_epop]
    if 'mean_R0' in best_result and best_result['mean_R0'] is not None:
        t = np.arange(len(best_result['mean_R0']))
        ax4.plot(t, best_result['mean_R0'], 'b-', lw=2, label=f'R(t) (EPOP={best_epop:.0f})')
        ax4.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R=1')
        ax4.set_xlabel('День', fontsize=12)
        ax4.set_ylabel('R(t)', fontsize=12)
        ax4.set_title('R(t) для лучшего EPOP', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.4)
    
    plt.suptitle(f'Подбор оптимального EPOP для байесовской модели\n'
                 f'Лучший EPOP = {best_epop:.0f}, mu = {mu_val:.8f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def find_and_plot_epop_mu():
    best_epop, best_mu, results = find_optimal_epop_mu(True)
    if best_epop is not None and best_mu is not None and results is not None:
                                          
        plot_epop_mu_search_results(results, best_epop, best_mu)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        best_key = f"EPOP_{best_epop:.0f}_mu_{best_mu:.6f}"
        best_trans = results[best_key]['transmission']

        sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])
        colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(sorted_results))))
        
        for i, (key, data) in enumerate(sorted_results[:5]):
            ax1.plot(data['transmission'], color=colors[i], alpha=0.7, 
                    label=f'EPOP={data["EPOP"]:.0f}, mu={data["mu"]:.6f}')
        
        ax1.plot(current_I_fact, 'k-', lw=2, label='Фактические данные')
        ax1.set_xlabel('День', fontsize=12)
        ax1.set_ylabel('Новые случаи', fontsize=12)
        ax1.set_title('Сравнение лучших моделей', fontsize=14)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.4)

        ax2.plot(best_trans, 'r-', lw=2, label=f'Модель (EPOP={best_epop:.0f}, mu={best_mu:.8f})')
        ax2.plot(current_I_fact, 'b-', lw=2, label='Фактические данные')

        mae = results[best_key]['MAE']
        rmse = results[best_key]['RMSE']
        r2 = results[best_key]['R2']
        ax2.text(0.02, 0.95, f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.4f}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_xlabel('День', fontsize=12)
        ax2.set_ylabel('Новые случаи', fontsize=12)
        ax2.set_title(f'Лучшая модель (EPOP={best_epop:.0f}, mu={best_mu:.8f})', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.4)
        
        plt.suptitle(f'Совместный подбор EPOP и mu\nЛучшие: EPOP={best_epop:.0f}, mu={best_mu:.8f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        global epop_optimized, bayes_best_mu_global
        epop_optimized = best_epop
        bayes_best_mu_global = best_mu

        mu_entry.delete(0, tk.END)
        mu_entry.insert(0, f"{best_mu:.8f}")

        messagebox.showinfo("Успешно", 
            f"Найдены оптимальные параметры:\n"
            f"EPOP = {best_epop:.0f}\n"
            f"mu = {best_mu:.8f}\n\n"
            f"Нажмите 'Запустить симуляцию' для применения.")

def grid_search_mu_grid(params, I_fact, I_fact_abs, window=5, p_min=0.0, p_max=0.4, p_step=0.001,
                        k_val=0.055, r0_max=10.0, adapt_p=True,
                        mu_min=0.0, mu_max=0.02, mu_step=0.003):
    days = len(I_fact)
    N = params.get('N', 1)
    
    mu_grid = np.arange(mu_min, mu_max + mu_step/2, mu_step)
    
    print(f"\n🔍 ГРУБЫЙ ПОДБОР ГЛОБАЛЬНОГО mu (СЕТОЧНАЯ МОДЕЛЬ)")
    print(f"   Диапазон: [{mu_min:.6f} - {mu_max:.6f}], шаг: {mu_step:.6f}")
    print(f"   Количество значений: {len(mu_grid)}")
    print("-" * 60)
    
    best_mu = None
    best_error = np.inf
    best_results = None
    all_results = {}
    
    for idx, mu_candidate in enumerate(mu_grid):
        print(f"\n🔄 mu = {mu_candidate:.6f} ({idx+1}/{len(mu_grid)})...")
        
        params_copy = params.copy()
        params_copy['mu'] = mu_candidate

        S, I, transmission, R0, p, mu_arr = sir_euler_grid(
            params_copy, I_fact,
            window=window,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            k_val=k_val,
            r0_max=r0_max,
            adapt_p=adapt_p,
            mu_fixed=mu_candidate,
        )
        
        trans_abs = transmission * N
        error_mae = np.mean(np.abs(trans_abs - I_fact_abs))
        error_rmse = np.sqrt(np.mean((trans_abs - I_fact_abs) ** 2))
        
        all_results[mu_candidate] = {
            'S': S,
            'I': I,
            'transmission': transmission,
            'R0': R0,
            'p': p,
            'mu': mu_arr,
            'trans_abs': trans_abs,
            'MAE': error_mae,
            'RMSE': error_rmse
        }
        
        r0_mean = np.mean(R0)
        p_mean = np.mean(p[p > 0.0001]) if np.any(p > 0.0001) else 0
        p_nonzero = np.sum(p > 0.0001)
        
        print(f"   MAE = {error_mae:.4f}, RMSE = {error_rmse:.4f}")
        print(f"   Среднее R0 = {r0_mean:.4f}, Среднее p = {p_mean:.6f}")
        
        if error_mae < best_error:
            best_error = error_mae
            best_mu = mu_candidate
            best_results = (S, I, transmission, R0, p, mu_arr)
            print(f"   ✅ НОВЫЙ ЛУЧШИЙ! mu = {mu_candidate:.6f}, MAE = {error_mae:.4f}")
    
    print("\n" + "=" * 60)
    print(f"✅ ПОДБОР ДЛЯ СЕТКИ ЗАВЕРШЕН!")
    print(f"   Лучшее mu = {best_mu:.8f}")
    print(f"   Лучшая MAE = {best_error:.4f}")
    print("=" * 60)
    
    return best_mu, best_results, all_results

def grid_search_mu_bayesian(mcmc_samples_R0, mcmc_samples_p, params, I_fact, I_fact_abs,
                            window=5, p_min=0.0, p_max=0.4, p_step=0.001,
                            k_val=0.055, r0_max=10.0, adapt_p=True,
                            mu_min=0.0, mu_max=0.02, mu_step=0.003,
                            n_sim=10):
    days = len(I_fact)
    N = params.get('N', 1)
    
    mu_grid = np.arange(mu_min, mu_max + mu_step/2, mu_step)
    
    print(f"\n🔍 ГРУБЫЙ ПОДБОР ГЛОБАЛЬНОГО mu (БАЙЕСОВСКАЯ МОДЕЛЬ)")
    print(f"   Диапазон: [{mu_min:.6f} - {mu_max:.6f}], шаг: {mu_step:.6f}")
    print(f"   Количество значений: {len(mu_grid)}")
    print(f"   Симуляций на каждое mu: {n_sim}")
    print("-" * 60)
    
    best_mu = None
    best_error = np.inf
    best_result = None
    all_results = {}
    
    for idx, mu_candidate in enumerate(mu_grid):
        print(f"\n🔄 mu = {mu_candidate:.6f} ({idx+1}/{len(mu_grid)})...")

        result = generate_bayesian_trajectories(
            mcmc_samples_R0, mcmc_samples_p, I_fact, params,
            n_sim=n_sim,
            window=window,
            k_val=k_val,
            r0_max=r0_max,
            add_noise=True,
            noise_level=0.05,
            return_p=True,
            return_I=True,
            adapt_p=adapt_p,
            mu_fixed=mu_candidate,
            verbose=False
        )

        gamma = params['gamma']
        all_I = result['I']
        all_R0 = result['R0']
        
        all_transmission = []
        for i in range(len(all_I)):
            I_traj = all_I[i]
            R0_traj = all_R0[i]
            I_traj = np.clip(I_traj, 0, 0.5)
            S_traj = np.maximum(1.0 - I_traj, 0)
            trans_traj = R0_traj * gamma * S_traj * I_traj
            trans_traj = np.maximum(trans_traj, 0)
            all_transmission.append(trans_traj)
        
        all_transmission = np.array(all_transmission)
        mean_transmission = np.mean(all_transmission, axis=0) * N

        error_mae = np.mean(np.abs(mean_transmission - I_fact_abs))
        error_rmse = np.sqrt(np.mean((mean_transmission - I_fact_abs) ** 2))

        all_results[mu_candidate] = {
            'result': result,
            'mean_transmission': mean_transmission,
            'MAE': error_mae,
            'RMSE': error_rmse
        }
        
        print(f"   MAE = {error_mae:.4f}, RMSE = {error_rmse:.4f}")
        
        if error_mae < best_error:
            best_error = error_mae
            best_mu = mu_candidate
            best_result = result
            print(f"   ✅ НОВЫЙ ЛУЧШИЙ! mu = {mu_candidate:.6f}, MAE = {error_mae:.4f}")
    
    print("\n" + "=" * 60)
    print(f"✅ ПОДБОР ДЛЯ БАЙЕСА ЗАВЕРШЕН!")
    print(f"   Лучшее mu = {best_mu:.8f}")
    print(f"   Лучшая MAE = {best_error:.4f}")
    print("=" * 60)
    
    return best_mu, best_result, all_results

def plot_mu_search_results(all_results, best_mu, title):
    mu_values = sorted(all_results.keys())
    mae_values = [all_results[mu]['MAE'] for mu in mu_values]
    rmse_values = [all_results[mu]['RMSE'] for mu in mu_values]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.plot(mu_values, mae_values, 'b-o', lw=2, markersize=8, label='MAE')
    ax1.axvline(best_mu, color='red', linestyle='--', alpha=0.7, 
                label=f'Лучшее mu = {best_mu:.6f}')
    ax1.axhline(min(mae_values), color='green', linestyle='--', alpha=0.7,
                label=f'Лучшая MAE = {min(mae_values):.4f}')
    ax1.set_xlabel('mu', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title(f'Зависимость MAE от mu ({title})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(mu_values, rmse_values, 'r-o', lw=2, markersize=8, label='RMSE')
    ax2.axvline(best_mu, color='red', linestyle='--', alpha=0.7,
                label=f'Лучшее mu = {best_mu:.6f}')
    ax2.set_xlabel('mu', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title(f'Зависимость RMSE от mu ({title})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    ax3.bar(range(len(mu_values)), mae_values, color='blue', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(mu_values)))
    ax3.set_xticklabels([f'{mu:.4f}' for mu in mu_values], rotation=45, ha='right')
    ax3.axhline(min(mae_values), color='red', linestyle='--', alpha=0.7,
                label=f'Минимум = {min(mae_values):.4f}')
    ax3.set_xlabel('mu', fontsize=12)
    ax3.set_ylabel('MAE', fontsize=12)
    ax3.set_title(f'Сравнение MAE для разных mu', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.4, axis='y')

    best_idx = np.argmin(mae_values)
    stats_text = f"""
    Лучшее mu = {best_mu:.8f}
    Лучшая MAE = {min(mae_values):.4f}
    Лучшая RMSE = {rmse_values[best_idx]:.4f}
    Проверено значений: {len(mu_values)}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=14, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Итоговая статистика', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Результаты грубого подбора mu ({title})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_mu_grid_search_comparison(grid_results, bayesian_results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    if grid_results:
        mu_values_grid = sorted(grid_results.keys())
        mae_values_grid = [grid_results[mu]['MAE'] for mu in mu_values_grid]
        
        ax1.plot(mu_values_grid, mae_values_grid, 'b-o', lw=2, markersize=8, label='Сетка')
        best_mu_grid = min(grid_results.items(), key=lambda x: x[1]['MAE'])[0]
        best_mae_grid = min(grid_results.items(), key=lambda x: x[1]['MAE'])[1]['MAE']
        ax1.axvline(best_mu_grid, color='blue', linestyle='--', alpha=0.7,
                    label=f'Лучшее mu (сетка) = {best_mu_grid:.6f}')
        ax1.axhline(best_mae_grid, color='blue', linestyle=':', alpha=0.7,
                    label=f'Лучшая MAE = {best_mae_grid:.4f}')
    ax1.set_xlabel('mu', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('Зависимость MAE от mu (СЕТКА)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    if bayesian_results:
        mu_values_bayes = sorted(bayesian_results.keys())
        mae_values_bayes = [bayesian_results[mu]['MAE'] for mu in mu_values_bayes]
        
        ax2.plot(mu_values_bayes, mae_values_bayes, 'r-o', lw=2, markersize=8, label='Байес')
        best_mu_bayes = min(bayesian_results.items(), key=lambda x: x[1]['MAE'])[0]
        best_mae_bayes = min(bayesian_results.items(), key=lambda x: x[1]['MAE'])[1]['MAE']
        ax2.axvline(best_mu_bayes, color='red', linestyle='--', alpha=0.7,
                    label=f'Лучшее mu (байес) = {best_mu_bayes:.6f}')
        ax2.axhline(best_mae_bayes, color='red', linestyle=':', alpha=0.7,
                    label=f'Лучшая MAE = {best_mae_bayes:.4f}')
    ax2.set_xlabel('mu', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Зависимость MAE от mu (БАЙЕС)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    if grid_results and bayesian_results:
                               
        best_grid = min(grid_results.items(), key=lambda x: x[1]['MAE'])
        best_bayes = min(bayesian_results.items(), key=lambda x: x[1]['MAE'])
        
        labels = ['Сетка', 'Байес']
        mus = [best_grid[0], best_bayes[0]]
        maes = [best_grid[1]['MAE'], best_bayes[1]['MAE']]
        
        bars = ax3.bar(labels, maes, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        for bar, mu, mae in zip(bars, mus, maes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'mu={mu:.6f}\nMAE={mae:.4f}', ha='center', va='bottom', fontsize=9)
        ax3.set_ylabel('MAE', fontsize=12)
        ax3.set_title('Сравнение лучших решений', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.4, axis='y')

    stats_text = ""
    if grid_results:
        best_grid = min(grid_results.items(), key=lambda x: x[1]['MAE'])
        stats_text += f"СЕТКА:\n"
        stats_text += f"  Лучшее mu = {best_grid[0]:.8f}\n"
        stats_text += f"  MAE = {best_grid[1]['MAE']:.4f}\n"
        stats_text += f"  RMSE = {best_grid[1]['RMSE']:.4f}\n\n"
    if bayesian_results:
        best_bayes = min(bayesian_results.items(), key=lambda x: x[1]['MAE'])
        stats_text += f"БАЙЕС:\n"
        stats_text += f"  Лучшее mu = {best_bayes[0]:.8f}\n"
        stats_text += f"  MAE = {best_bayes[1]['MAE']:.4f}\n"
        stats_text += f"  RMSE = {best_bayes[1]['RMSE']:.4f}\n"
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Итоговая статистика', fontsize=14, fontweight='bold')
    
    plt.suptitle('Сравнение грубого подбора mu для сетки и байеса', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_thermal_matrix():
    global last_S, last_I, last_params, last_I_fact, current_json_path, current_error_matrix, current_r0_grid
    
    try:
        print("\n" + "="*60)
        print("🔍 ДИАГНОСТИКА ПЕРЕД ПОСТРОЕНИЕМ ТЕПЛОВОЙ МАТРИЦЫ")
        print("="*60)
        print(f"   last_S:         {last_S is not None} (длина: {len(last_S) if last_S is not None else 0})")
        print(f"   last_I:         {last_I is not None} (длина: {len(last_I) if last_I is not None else 0})")
        print(f"   last_params:    {last_params is not None}")
        print(f"   last_I_fact:    {last_I_fact is not None} (длина: {len(last_I_fact) if last_I_fact is not None else 0})")
        print(f"   current_json_path: {current_json_path is not None}")
        print("="*60)
        
        if last_S is None or last_I is None:
            print("❌ Нет данных last_S или last_I")
            if current_json_path is None:
                messagebox.showwarning("Предупреждение", 
                    "Сначала загрузите JSON-файл (📂 Загрузить JSON), затем запустите симуляцию (▶ Запустить симуляцию).")
                return
            print("🔄 Автоматический запуск симуляции...")
            run_simulation(current_json_path)
            if last_S is None or last_I is None:
                messagebox.showerror("Ошибка", 
                    "Симуляция не создала данные. Проверьте JSON-файл и параметры модели.")
                return
        
        if len(last_S) == 0 or len(last_I) == 0:
            messagebox.showerror("Ошибка", "Данные симуляции пустые. Запустите симуляцию заново.")
            return
        
        if last_params is None:
            messagebox.showerror("Ошибка", "Нет параметров модели. Загрузите JSON-файл.")
            return
        
        if last_I_fact is None or len(last_I_fact) == 0:
            messagebox.showerror("Ошибка", "Нет фактических данных. Загрузите JSON-файл.")
            return
        
        print("✅ Все данные есть, строим тепловую матрицу...")
        
        try:
            window = int(window_entry.get())
        except:
            window = 5
        
        try:
            k_val = float(k_entry.get())
        except:
            k_val = 0.055
        
        try:
            r0_max_val = float(r0_max_entry.get())
        except:
            r0_max_val = 20.0
        
        print(f"\n🔄 Построение тепловой матрицы...")
        print(f"   Window: {window}, k: {k_val:.6f}, R0_max: {r0_max_val:.1f}")
        print(f"   Дней: {len(last_I_fact)}")
        
        error_matrix, r0_grid, best_r0_per_day, min_error_per_day = build_r0_day_matrix(
            last_params, last_I_fact, last_S, last_I, 
            window=window, k_val=k_val, r0_max=r0_max_val
        )
        
        current_error_matrix = error_matrix
        current_r0_grid = r0_grid
        
        days = len(last_I_fact)
        n_rows, n_cols = error_matrix.shape
        
        print(f"   ✅ Размер матрицы: {n_rows} строк × {n_cols} столбцов")
        print(f"   📊 Диапазон ошибок: {np.min(error_matrix):.6f} ... {np.max(error_matrix):.6f}")
        
        y_limit = 10.0
        
        r0_mask = (r0_grid >= 0) & (r0_grid <= y_limit)
        
        if not np.any(r0_mask):
            r0_mask = np.zeros(len(r0_grid), dtype=bool)
            r0_mask[:min(10, len(r0_grid))] = True
        
        error_matrix_plot = error_matrix[r0_mask, :]
        r0_grid_plot = r0_grid[r0_mask]
        
        n_rows_plot = error_matrix_plot.shape[0]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        min_val = np.min(error_matrix_plot[error_matrix_plot > 0]) if np.any(error_matrix_plot > 0) else 1
        ratio = np.max(error_matrix_plot) / (min_val + 1e-10)
        
        if ratio > 10000:
            print("   ⚠️ Используется логарифмическая шкала (разброс > 10000)")
            data_plot = np.log10(np.where(error_matrix_plot > 0, error_matrix_plot, min_val / 100))
            cbar_label = 'log10(Ошибка)'
        else:
            data_plot = error_matrix_plot
            cbar_label = 'Ошибка'
        
        im = ax.imshow(data_plot, aspect='auto', cmap='viridis', 
                       origin='lower', interpolation='nearest',
                       extent=[0, n_cols, r0_grid_plot[0], r0_grid_plot[-1]])
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
        
        ax.set_xlabel('День (t)', fontsize=12, fontweight='bold')
        ax.set_ylabel('R0', fontsize=12, fontweight='bold')
        
        ax.set_ylim([0, y_limit])
        
        step_x = max(1, n_cols // 20)
        x_ticks = np.arange(0, n_cols, step_x)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{d}' for d in x_ticks], rotation=45, ha='right', fontsize=8)
        
        y_ticks = np.linspace(0, y_limit, min(11, n_rows_plot + 1))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=8)
        
        ax.set_title(
            f'Тепловая матрица ошибок R0 × Дни (R0 от 0 до {y_limit:.0f})\n'
            f'Window = {window}, k = {k_val:.6f}, R0_max = {r0_max_val:.1f}',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        
        plt.tight_layout()
        
        output_file = f"thermal_matrix_window{window}_k{k_val:.6f}_ylim{y_limit:.0f}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Тепловая матрица сохранена: {output_file}")
        
        plt.show()
        
        print("\n✅ Тепловая матрица успешно построена!")
        print(f"   Ограничение по Y: 0 ... {y_limit:.0f}")
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Ошибка: {error_detail}")
        messagebox.showerror("Ошибка", f"Ошибка при построении тепловой матрицы: {str(e)}")

def select_file():
    global current_json_path, last_params, last_I_fact, last_I_fact_abs, current_I_fact_abs
    
    file_path = filedialog.askopenfilename(
        title="Выберите JSON-файл",
        filetypes=[("JSON Model", "*.jsonModel"), ("JSON files", "*.json"), ("All files", "*.*")]
    )
    if file_path:
        current_json_path = file_path
        result = load_data_from_json(file_path)
        if result[0] is not None:
            params, I_fact, I_fact_abs = result
            last_params = params
            last_I_fact = I_fact
            last_I_fact_abs = I_fact_abs
            
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, f"✅ Загружен: {os.path.basename(file_path)}\n")
            text_area.insert(tk.END, f"   Дней: {len(I_fact_abs)}\n")
            text_area.insert(tk.END, f"   N = {params.get('N', 0):.0f}\n")
            text_area.insert(tk.END, f"   gamma = {params['gamma']:.6f}\n")
            text_area.insert(tk.END, f"   R0 = {params['R0']:.4f}\n")
            text_area.insert(tk.END, "\n▶ Нажмите '▶ Запустить симуляцию'\n")

def bind_mousewheel(widget, scroll_direction=1):
    def on_mousewheel(event):
        if event.num == 4:
            delta = -1 * scroll_direction
        elif event.num == 5:
            delta = 1 * scroll_direction
        else:
            delta = -1 * (event.delta // 120) * scroll_direction
        try:
            widget.yview_scroll(int(delta), "units")
        except:
            pass
    
    if hasattr(widget, 'bind'):
        widget.bind("<MouseWheel>", on_mousewheel)
        widget.bind("<Button-4>", on_mousewheel)
        widget.bind("<Button-5>", on_mousewheel)
        widget.bind("<Shift-MouseWheel>", on_mousewheel)
        
        def bind_children(children):
            for child in children:
                try:
                    if hasattr(child, 'bind'):
                        child.bind("<MouseWheel>", on_mousewheel)
                        child.bind("<Button-4>", on_mousewheel)
                        child.bind("<Button-5>", on_mousewheel)
                    if hasattr(child, 'winfo_children'):
                        bind_children(child.winfo_children())
                except:
                    pass
        try:
            bind_children(widget.winfo_children())
        except:
            pass

def bind_mousewheel_to_window(window):
    def find_and_bind(widget):
        try:
            if isinstance(widget, (scrolledtext.ScrolledText, tk.Text, tk.Listbox, tk.Canvas)):
                bind_mousewheel(widget)
            if hasattr(widget, 'winfo_children'):
                for child in widget.winfo_children():
                    find_and_bind(child)
        except:
            pass
    find_and_bind(window)

def bind_mousewheel_to_all_windows():
    def on_new_window(event):
        try:
            if hasattr(event.widget, 'winfo_toplevel'):
                bind_mousewheel_to_window(event.widget.winfo_toplevel())
        except:
            pass
    bind_mousewheel_to_window(root)
    root.bind("<Map>", on_new_window)

def show_weights():
    try:
        window = int(window_entry.get())
        if window < 1:
            raise ValueError
    except:
        messagebox.showerror("Ошибка", "Window должен быть целым числом ≥ 1")
        return
    plot_weights(window)

current_best_R0_arr = None
current_best_p = None
current_best_mu = None
current_best_gamma = None
current_transmission = None
current_I_fact = None
current_I_fact_abs = None
current_json_path = None
current_N = None
current_S = None
current_I = None
current_params = None

current_best_R0_arr = None
current_best_p = None
current_best_gamma = None
current_I_fact = None
current_N = None

last_S = None
last_I = None
last_params = None
last_I_fact = None
last_I_fact_abs = None
last_window = 15
last_p_min = 0.0
last_p_max = 0.4
last_p_step = 0.001
last_best_R0 = None
last_best_gamma = None
last_k = 0.01
last_r0_max = 20.0
last_best_mu = None
epop_optimized = None
epop_optimized_params = {}
epop_min_entry = None
epop_max_entry = None
epop_step_entry = None

mcmc_samples_R0 = None
mcmc_samples_p = None
last_mcmc = None
noisy_curves_storage = None
current_error_matrix = None
current_r0_grid = None

mu_min_grid_entry = None
mu_max_grid_entry = None
mu_step_grid_entry = None
use_grid_search_var = None                 

saved_bayesian_results = None                        
saved_bayesian_params = {} 

bayes_best_mu_global = None
grid_best_mu_global = None

epop_bayesian_optimized = None
epop_bayesian_optimized_params = {}

import tkinter.ttk as ttk
from tkinter import font as tkfont

root = tk.Tk()
root.title("SIR-модель — Байесовский анализ")
root.geometry("1080x820")
root.configure(bg='#f8f9fa')

adapt_p_var = tk.BooleanVar(value=True)

style = ttk.Style()
style.theme_use('clam')

style.configure('TFrame', background='#f8f9fa')
style.configure('TLabelframe', background='#f8f9fa')
style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8)
style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), padding=8)

title_frame = tk.Frame(root, bg='#2c3e50')
title_frame.pack(fill=tk.X)

tk.Label(title_frame, text="SIR-модель", 
         font=("Segoe UI", 18, "bold"), fg="white", bg='#2c3e50').pack(pady=12)

tk.Label(title_frame, text="Адаптивная модель", 
         font=("Segoe UI", 11), fg="#bdc3c7", bg='#2c3e50').pack(pady=2)

param_frame = ttk.LabelFrame(root, text=" Параметры модели ", padding=15)
param_frame.pack(fill=tk.X, padx=20, pady=12)

param_grid = tk.Frame(param_frame)
param_grid.pack()

tk.Label(param_grid, text="Window:", font=("Segoe UI", 10)).grid(row=0, column=0, padx=8, pady=6, sticky='e')
window_entry = ttk.Entry(param_grid, width=8, font=("Consolas", 11))
window_entry.insert(0, "15")
window_entry.grid(row=0, column=1, padx=5)

tk.Label(param_grid, text="k:", font=("Segoe UI", 10)).grid(row=0, column=2, padx=8, pady=6, sticky='e')
k_entry = ttk.Entry(param_grid, width=10, font=("Consolas", 11))
k_entry.insert(0, "0.0055")
k_entry.grid(row=0, column=3, padx=5)

tk.Label(param_grid, text="R0_max:", font=("Segoe UI", 10)).grid(row=0, column=4, padx=8, pady=6, sticky='e')
r0_max_entry = ttk.Entry(param_grid, width=10, font=("Consolas", 11))
r0_max_entry.insert(0, "20.0")
r0_max_entry.grid(row=0, column=5, padx=5)

tk.Label(param_grid, text="gamma:", font=("Segoe UI", 10)).grid(row=0, column=6, padx=8, pady=6, sticky='e')
gamma_entry = ttk.Entry(param_grid, width=10, font=("Consolas", 11))
gamma_entry.insert(0, "0.074")
gamma_entry.grid(row=0, column=7, padx=5)

tk.Label(param_grid, text="mu (фиксированное):", font=("Segoe UI", 10)).grid(row=1, column=0, padx=8, pady=6, sticky='e')
mu_entry = ttk.Entry(param_grid, width=12, font=("Consolas", 11))
mu_entry.insert(0, "0.00013")
mu_entry.grid(row=1, column=1, padx=5)

tk.Label(param_grid, text="", font=("Segoe UI", 10)).grid(row=1, column=2, padx=8, pady=6)

tk.Label(param_grid, text="", font=("Segoe UI", 10)).grid(row=2, column=0, padx=8, pady=6)

adapt_p_check = tk.Checkbutton(
    param_grid, 
    text="Адаптировать p(t) (включается с дня 430)", 
    variable=adapt_p_var,
    font=("Segoe UI", 10)
)
adapt_p_check.grid(row=2, column=1, columnspan=4, padx=5, pady=6, sticky='w')

tk.Label(param_grid, text="EPOP_min:", font=("Segoe UI", 10)).grid(row=5, column=0, padx=8, pady=6, sticky='e')
epop_min_entry = ttk.Entry(param_grid, width=12, font=("Consolas", 11))
epop_min_entry.insert(0, "10000")
epop_min_entry.grid(row=5, column=1, padx=5)

tk.Label(param_grid, text="EPOP_max:", font=("Segoe UI", 10)).grid(row=5, column=2, padx=8, pady=6, sticky='e')
epop_max_entry = ttk.Entry(param_grid, width=12, font=("Consolas", 11))
epop_max_entry.insert(0, "100000")
epop_max_entry.grid(row=5, column=3, padx=5)

tk.Label(param_grid, text="EPOP_step:", font=("Segoe UI", 10)).grid(row=5, column=4, padx=8, pady=6, sticky='e')
epop_step_entry = ttk.Entry(param_grid, width=12, font=("Consolas", 11))
epop_step_entry.insert(0, "5000")
epop_step_entry.grid(row=5, column=5, padx=5)

btn_frame = tk.Frame(root, bg='#f8f9fa')
btn_frame.pack(fill=tk.X, padx=20, pady=15)

g1 = ttk.LabelFrame(btn_frame, text=" Основное управление ", padding=10)
g1.pack(side=tk.LEFT, fill=tk.Y, padx=8)

ttk.Button(g1, text="📂 Загрузить JSON", style='Accent.TButton', 
           command=select_file).pack(pady=4, fill=tk.X)

ttk.Button(g1, text="▶ Запустить симуляцию", 
           command=run_simulation_from_gui).pack(pady=4, fill=tk.X)

ttk.Button(g1, text="📊 Показать веса ошибки", 
           command=show_weights).pack(pady=4, fill=tk.X)

g2 = ttk.LabelFrame(btn_frame, text=" Сеточная модель (Grid) ", padding=10)
g2.pack(side=tk.LEFT, fill=tk.Y, padx=8)

g2_inner = tk.Frame(g2)
g2_inner.pack(fill=tk.X)

g2_col1 = tk.Frame(g2_inner)
g2_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

ttk.Button(g2_col1, text="📊 Показать R и p", 
           command=show_r_and_p).pack(pady=2, fill=tk.X)

ttk.Button(g2_col1, text="💾 Выгрузить S(t) и I(t)",
           command=export_S_I_from_grid).pack(pady=2, fill=tk.X)

ttk.Button(g2_col1, text="📈 λ раздельный (общая пораженность)", 
           command=analyze_lambda_grid_split_with_total).pack(pady=2, fill=tk.X)

ttk.Button(g2_col1, text="📊 Таблицы R(t) и Y(t)", 
           command=show_r_y_tables).pack(pady=2, fill=tk.X)

g2_col2 = tk.Frame(g2_inner)
g2_col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)

ttk.Button(g2_col2, text="🔥 Тепловая", 
           command=plot_thermal_matrix).pack(pady=2, fill=tk.X)

ttk.Button(g2_col1, text="🎯 Подобрать EPOP", 
           command=find_and_plot_epop).pack(pady=2, fill=tk.X)

ttk.Button(g2_col2, text="📊 4 графика", 
           command=plot_4_graphs).pack(pady=2, fill=tk.X)

text_main_frame = ttk.Frame(root)
text_main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

left_frame = ttk.LabelFrame(text_main_frame, text=" Результаты ", padding=10)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

text_area = scrolledtext.ScrolledText(
    left_frame, 
    font=("Consolas", 10), 
    height=16,
    relief=tk.SUNKEN,
    bd=1
)
text_area.pack(fill=tk.BOTH, expand=True)

right_frame = ttk.LabelFrame(text_main_frame, text=" 📖 Инструкция ", padding=10)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

instruction_text = scrolledtext.ScrolledText(
    right_frame, 
    font=("Segoe UI", 9), 
    height=16,
    relief=tk.SUNKEN,
    bd=1,
    wrap=tk.WORD  
)
instruction_text.pack(fill=tk.BOTH, expand=True)

instruction_text.insert(tk.END, create_instruction_text())
instruction_text.config(state=tk.DISABLED)                            

bind_mousewheel_to_all_windows()
root.mainloop()