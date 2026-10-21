# regression_gui_advanced.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import sys
import warnings
warnings.filterwarnings('ignore')


class RegressionAdvancedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Регрессионный анализ R = R₀·Y^p (нормировка Y)")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 700)

        self.path_R_file = None
        self.path_data_file = None

        self.current_file = None
        self.r_data = None
        self.r_data_norm = None      # нормированный R (R / mean(R)) — опционально
        self.y_data = None           # Y после выбора (real/EPOP, model/EPOP, I_main, I_abs)
        self.y_data_norm = None      # Y / mean(Y) — основная нормировка
        self.real_data = None
        self.model_data = None
        self.i_mean_data = None
        self.i_abs_data = None
        self.epop = 60000
        self.results = None
        self.gamma = 0.074
        self.df_full = None
        self.df_R = None
        self.df_data = None
        self.days = None
        self.y_type_var = tk.StringVar(value="real")  # real | model | i_mean | i_abs

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_container = ttk.Frame(main_frame, width=460)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_container.pack_propagate(False)

        self.left_canvas = tk.Canvas(left_container, highlightthickness=0, width=440)
        self.left_scrollbar = ttk.Scrollbar(left_container, orient="vertical",
                                            command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(self.left_canvas)
        self.left_canvas.create_window((0, 0), window=left_panel, anchor="nw")

        def _on_left_configure(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def _on_left_canvas_configure(event):
            self.left_canvas.itemconfig(
                self.left_canvas.find_all()[0],
                width=event.width
            )

        left_panel.bind("<Configure>", _on_left_configure)
        self.left_canvas.bind("<Configure>", _on_left_canvas_configure)

        def _on_mousewheel(event):
            delta = 0
            if event.num == 4:
                delta = -1
            elif event.num == 5:
                delta = 1
            elif event.delta:
                delta = -1 if event.delta > 0 else 1
            if delta:
                self.left_canvas.yview_scroll(delta, "units")

        def _bind_wheel(_):
            self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.left_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.left_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_):
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")

        left_container.bind("<Enter>", _bind_wheel)
        left_container.bind("<Leave>", _unbind_wheel)

        title = ttk.Label(left_panel, text="📊 R = R₀·(Y)^p",
                          font=('Arial', 14, 'bold'))
        title.pack(pady=(0, 10))

        files_frame = ttk.LabelFrame(left_panel, text="Загрузка данных (2 файла)", padding=10)
        files_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(files_frame, text="Файл 1: R(t) — R_mean:",
                  font=('Arial', 9, 'bold')).pack(anchor='w', pady=(4, 2))

        r_row = ttk.Frame(files_frame)
        r_row.pack(fill=tk.X)
        self.btn_load_R = ttk.Button(r_row, text="📁 Выбрать файл R",
                                     command=self.load_R_file)
        self.btn_load_R.pack(side=tk.LEFT)
        self.lbl_R_file = ttk.Label(r_row, text="(не выбран)",
                                    wraplength=280, foreground="gray")
        self.lbl_R_file.pack(side=tk.LEFT, padx=(8, 0))

        self.lbl_R_info = ttk.Label(files_frame, text="",
                                    wraplength=400, foreground="darkgreen")
        self.lbl_R_info.pack(anchor='w', pady=(2, 6))

        ttk.Label(files_frame, text="Файл 2: I_main / real / model:",
                  font=('Arial', 9, 'bold')).pack(anchor='w', pady=(4, 2))

        d_row = ttk.Frame(files_frame)
        d_row.pack(fill=tk.X)
        self.btn_load_data = ttk.Button(d_row, text="📁 Выбрать файл данных",
                                        command=self.load_data_file)
        self.btn_load_data.pack(side=tk.LEFT)
        self.lbl_data_file = ttk.Label(d_row, text="(не выбран)",
                                       wraplength=260, foreground="gray")
        self.lbl_data_file.pack(side=tk.LEFT, padx=(8, 0))

        self.lbl_data_info = ttk.Label(files_frame, text="",
                                       wraplength=400, foreground="darkgreen")
        self.lbl_data_info.pack(anchor='w', pady=(2, 6))

        ttk.Button(files_frame, text="🔗 Объединить и подготовить данные",
                   command=self.merge_files).pack(fill=tk.X, pady=(6, 2))

        self.merge_info = ttk.Label(files_frame, text="",
                                    wraplength=400, foreground="blue",
                                    justify="left")
        self.merge_info.pack(anchor='w', pady=(2, 0))

        params_frame = ttk.LabelFrame(left_panel, text="Параметры", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(params_frame, text="EPOP:").grid(row=0, column=0, sticky='w', pady=2)
        self.epop_entry = ttk.Entry(params_frame, width=15)
        self.epop_entry.insert(0, "60000")
        self.epop_entry.grid(row=0, column=1, sticky='w', pady=2, padx=(10, 0))

        ttk.Label(params_frame, text="γ:").grid(row=1, column=0, sticky='w', pady=2)
        self.gamma_entry = ttk.Entry(params_frame, width=15)
        self.gamma_entry.insert(0, "0.074")
        self.gamma_entry.grid(row=1, column=1, sticky='w', pady=2, padx=(10, 0))

        ttk.Button(params_frame, text="Применить EPOP",
                   command=self.apply_epop).grid(row=2, column=0, columnspan=2, pady=5)

        # === БЛОК: Нормировка Y ===
        norm_frame = ttk.LabelFrame(left_panel, text="Нормировка Y", padding=10)
        norm_frame.pack(fill=tk.X, pady=(0, 10))

        self.norm_y_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(norm_frame,
                        text="Нормировать Y: Y / mean(Y)",
                        variable=self.norm_y_var,
                        command=self.on_norm_y_change).pack(anchor='w', pady=2)

        self.norm_y_info_label = ttk.Label(norm_frame, text="",
                                           foreground="purple",
                                           wraplength=400,
                                           justify="left")
        self.norm_y_info_label.pack(anchor='w', pady=(2, 0))

        y_frame = ttk.LabelFrame(left_panel, text="Выбор Y для регрессии", padding=10)
        y_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(y_frame, text="Y = real / EPOP (новые случаи)",
                        variable=self.y_type_var, value="real",
                        command=self.on_y_type_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(y_frame, text="Y = model / EPOP",
                        variable=self.y_type_var, value="model",
                        command=self.on_y_type_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(y_frame, text="Y = I_main (доли инфицированных)",
                        variable=self.y_type_var, value="i_mean",
                        command=self.on_y_type_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(y_frame, text="Y = I_abs_main (чел. инфицированных)",
                        variable=self.y_type_var, value="i_abs",
                        command=self.on_y_type_change).pack(anchor='w', pady=2)

        self.y_info_label = ttk.Label(y_frame, text="", foreground="blue",
                                      wraplength=400)
        self.y_info_label.pack(anchor='w', pady=2)

        results_frame = ttk.LabelFrame(left_panel, text="Параметры регрессии", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(results_frame, text="R₀:").grid(row=0, column=0, sticky='w', pady=1)
        self.r0_label = ttk.Label(results_frame, text="-",
                                  font=('Arial', 10, 'bold'), foreground='blue')
        self.r0_label.grid(row=0, column=1, sticky='w', pady=1, padx=(10, 0))

        ttk.Label(results_frame, text="λ (p):").grid(row=1, column=0, sticky='w', pady=1)
        self.p_label = ttk.Label(results_frame, text="-",
                                 font=('Arial', 10, 'bold'), foreground='blue')
        self.p_label.grid(row=1, column=1, sticky='w', pady=1, padx=(10, 0))

        ttk.Label(results_frame, text="λ_up:").grid(row=2, column=0, sticky='w', pady=1)
        self.lambda_up_label = ttk.Label(results_frame, text="-",
                                         font=('Arial', 10, 'bold'), foreground='green')
        self.lambda_up_label.grid(row=2, column=1, sticky='w', pady=1, padx=(10, 0))

        ttk.Label(results_frame, text="λ_down:").grid(row=3, column=0, sticky='w', pady=1)
        self.lambda_down_label = ttk.Label(results_frame, text="-",
                                           font=('Arial', 10, 'bold'), foreground='red')
        self.lambda_down_label.grid(row=3, column=1, sticky='w', pady=1, padx=(10, 0))

        metrics_frame = ttk.LabelFrame(left_panel, text="Метрики качества", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        metrics = [
            ("R² (все):", "r2_label"),
            ("R² (up):", "r2_up_label"),
            ("R² (down):", "r2_down_label"),
            ("RMSE:", "rmse_label"),
            ("MAE:", "mae_label"),
            ("MAPE:", "mape_label"),
            ("Корреляция:", "corr_label")
        ]

        for i, (name, attr) in enumerate(metrics):
            ttk.Label(metrics_frame, text=name).grid(row=i, column=0, sticky='w', pady=1)
            label = ttk.Label(metrics_frame, text="-", font=('Arial', 9, 'bold'),
                              foreground='darkorange')
            label.grid(row=i, column=1, sticky='w', pady=1, padx=(10, 0))
            setattr(self, attr, label)

        self.run_btn = ttk.Button(left_panel, text="🚀 Выполнить регрессию",
                                  command=self.run_regression, state='disabled')
        self.run_btn.pack(fill=tk.X, pady=5)

        self.save_plot_btn = ttk.Button(left_panel, text="💾 Сохранить график",
                                        command=self.save_plot, state='disabled')
        self.save_plot_btn.pack(fill=tk.X, pady=5)

        self.export_btn = ttk.Button(left_panel, text="📊 Экспорт результатов",
                                     command=self.export_results, state='disabled')
        self.export_btn.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(left_panel, text="Готов к работе",
                                      relief=tk.SUNKEN, padding=5)
        self.status_label.pack(fill=tk.X, pady=(10, 10))

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(right_panel)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.clear_plot()

    def load_R_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл с R_mean (драйверы)",
            filetypes=[("Excel files", "*.xlsx *.xls"),
                       ("CSV files", "*.csv"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = self._read_table(path)
            self.df_R = df
            r_col = self._find_R_column(df)
            if r_col is None:
                messagebox.showerror("Ошибка", "Не найден столбец с R.")
                return
            day_col = self._find_day_column(df)
            self.lbl_R_file.config(
                text=f"{os.path.basename(path)} ({len(df)} стр.)",
                foreground="black"
            )
            rnum = pd.to_numeric(df[r_col], errors='coerce')
            self.lbl_R_info.config(
                text=f"R='{r_col}' | {rnum.min():.4f}…{rnum.max():.4f}"
                     + (f" | day='{day_col}'" if day_col else " | day: по индексу")
            )
            self.path_R_file = path
            self._invalidate_merge()
        except Exception as e:
            import traceback
            messagebox.showerror("Ошибка", f"{e}\n\n{traceback.format_exc()}")

    def load_data_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл с I_main / real / model",
            filetypes=[("Excel files", "*.xlsx *.xls"),
                       ("CSV files", "*.csv"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            df = self._read_table(path)
            self.df_data = df

            i_main_col = self._find_column(df, ['i_main', 'imain', 'i (main)', 'i main'])
            i_abs_main_col = self._find_column(df, ['i_abs_main', 'iabs_main', 'i_abs',
                                                    'i abs main', 'i_abs (main)'])
            real_col = self._find_column(df, ['real', 'факт', 'actual'])
            model_col = self._find_column(df, ['model', 'predicted'])
            day_col = self._find_day_column(df)

            found = []
            if i_main_col:      found.append(f"I_main='{i_main_col}'")
            if i_abs_main_col:  found.append(f"I_abs_main='{i_abs_main_col}'")
            if real_col:        found.append(f"real='{real_col}'")
            if model_col:       found.append(f"model='{model_col}'")
            if day_col:         found.append(f"day='{day_col}'")

            self.lbl_data_file.config(
                text=f"{os.path.basename(path)} ({len(df)} стр.)",
                foreground="black"
            )
            self.lbl_data_info.config(
                text="Найдено: " + ", ".join(found)
                if found else "⚠️ Ни один из ожидаемых столбцов не найден"
            )
            self.path_data_file = path
            self._invalidate_merge()
        except Exception as e:
            import traceback
            messagebox.showerror("Ошибка", f"{e}\n\n{traceback.format_exc()}")

    def _read_table(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        df = pd.read_csv(path, encoding='utf-8')
        if df.empty or len(df.columns) < 2:
            df = pd.read_csv(path, encoding='cp1251')
        if df.empty or len(df.columns) < 2:
            df = pd.read_csv(path, encoding='latin-1')
        return df

    def _find_R_column(self, df):
        for col in df.columns:
            c = str(col).lower().strip()
            if c in ('r_mean', 'r_actual', 'r', 'r(t)', 'rt'):
                return col
        for col in df.columns:
            c = str(col).lower().strip()
            if 'r_mean' in c or 'r_actual' in c or c == 'r':
                return col
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            return numeric_cols[1]
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        return None

    def _find_day_column(self, df):
        for col in df.columns:
            c = str(col).lower().strip()
            if c in ('day', 'день', 't', 'index', 'n'):
                return col
        return None

    def _find_column(self, df, names):
        for col in df.columns:
            c = str(col).lower().strip()
            if c in names:
                return col
        for col in df.columns:
            c = str(col).lower().strip()
            for n in names:
                if n in c:
                    return col
        return None

    def _invalidate_merge(self):
        self.r_data = None
        self.r_data_norm = None
        self.y_data = None
        self.y_data_norm = None
        self.real_data = None
        self.model_data = None
        self.i_mean_data = None
        self.i_abs_data = None
        self.results = None
        self.run_btn.config(state='disabled')
        self.save_plot_btn.config(state='disabled')
        self.export_btn.config(state='disabled')
        self.merge_info.config(text="")
        self.norm_y_info_label.config(text="")

    def merge_files(self):
        if self.df_R is None or self.df_data is None:
            messagebox.showwarning(
                "Нет данных",
                "Сначала загрузите оба файла:\n"
                "  1) R_mean\n"
                "  2) I_main / real"
            )
            return
        try:
            df_R = self.df_R.copy()
            df_D = self.df_data.copy()

            r_col = self._find_R_column(df_R)
            i_main_col = self._find_column(df_D, ['i_main', 'imain', 'i (main)', 'i main'])
            i_abs_main_col = self._find_column(df_D, ['i_abs_main', 'iabs_main', 'i_abs',
                                                      'i abs main', 'i_abs (main)'])
            real_col = self._find_column(df_D, ['real', 'факт', 'actual'])
            model_col = self._find_column(df_D, ['model', 'predicted'])
            epop_col = self._find_column(df_D, ['epop', 'epop_used'])

            day_col_R = self._find_day_column(df_R)
            day_col_D = self._find_day_column(df_D)

            n_R = len(df_R)
            n_D = len(df_D)
            n = min(n_R, n_D)
            if n == 0:
                messagebox.showerror("Ошибка", "Пустые данные.")
                return

            warn = ""
            if n_R != n_D:
                warn = f"⚠️ Длины разные: R={n_R}, data={n_D}; обрезаем до {n}\n"

            df_R = df_R.iloc[:n].reset_index(drop=True)
            df_D = df_D.iloc[:n].reset_index(drop=True)

            r_values = pd.to_numeric(df_R[r_col], errors='coerce').to_numpy(dtype=float)
            real_values = (pd.to_numeric(df_D[real_col], errors='coerce').to_numpy(dtype=float)
                           if real_col else np.zeros(n))
            model_values = (pd.to_numeric(df_D[model_col], errors='coerce').to_numpy(dtype=float)
                            if model_col else real_values.copy())
            i_main_values = (pd.to_numeric(df_D[i_main_col], errors='coerce').to_numpy(dtype=float)
                             if i_main_col else None)
            i_abs_main_values = (pd.to_numeric(df_D[i_abs_main_col], errors='coerce').to_numpy(dtype=float)
                                 if i_abs_main_col else None)

            if epop_col is not None:
                try:
                    self.epop = float(pd.to_numeric(df_D[epop_col],
                                                    errors='coerce').iloc[0])
                    self.epop_entry.delete(0, tk.END)
                    self.epop_entry.insert(0, f"{self.epop:.0f}")
                except Exception:
                    pass

            if day_col_R is not None:
                self.days = pd.to_numeric(df_R[day_col_R], errors='coerce').to_numpy(dtype=float)
            elif day_col_D is not None:
                self.days = pd.to_numeric(df_D[day_col_D], errors='coerce').to_numpy(dtype=float)
            else:
                self.days = np.arange(1, n + 1)

            mask = np.isfinite(r_values) & (r_values > 0)
            if real_col is not None:
                mask = mask & np.isfinite(real_values) & (real_values >= 0)
            if i_main_col is not None:
                mask = mask & np.isfinite(i_main_values)
            if i_abs_main_col is not None:
                mask = mask & np.isfinite(i_abs_main_values)

            self.r_data = r_values[mask]
            self.real_data = real_values[mask]
            self.model_data = model_values[mask]
            self.i_mean_data = i_main_values[mask] if i_main_values is not None else None
            self.i_abs_data = i_abs_main_values[mask] if i_abs_main_values is not None else None
            self.days = self.days[mask]

            if len(self.r_data) == 0:
                messagebox.showerror("Ошибка", "После фильтрации нет валидных точек.")
                return

            # Обновляем Y и его нормировку
            self.y_data = self.get_y_raw()
            self._update_y_norm()

            info = [warn + f"✅ Точек: {len(self.r_data)}",
                    f"R: {self.r_data.min():.4f}…{self.r_data.max():.4f}",
                    f"real: {self.real_data.min():.0f}…{self.real_data.max():.0f}"]
            if self.i_mean_data is not None:
                info.append(f"I_main: {self.i_mean_data.min():.6f}…{self.i_mean_data.max():.6f}")
            if self.i_abs_data is not None:
                info.append(f"I_abs_main: {self.i_abs_data.min():.2f}…{self.i_abs_data.max():.2f}")
            info.append(f"EPOP={self.epop:.0f}")
            self.merge_info.config(text="\n".join(info), foreground="blue")

            self.run_btn.config(state='normal')
            self.save_plot_btn.config(state='disabled')
            self.export_btn.config(state='disabled')
            self.results = None
            self.r0_label.config(text="-")
            self.p_label.config(text="-")
            self.lambda_up_label.config(text="-")
            self.lambda_down_label.config(text="-")
            for attr in ['r2_label', 'r2_up_label', 'r2_down_label',
                         'rmse_label', 'mae_label', 'mape_label', 'corr_label']:
                getattr(self, attr).config(text="-")

            self.y_info_label.config(text=f"Выбран: {self.get_y_label()}")
            self.plot_data()
            self.status_label.config(text=f"✅ Объединено: {len(self.r_data)} точек")

        except Exception as e:
            import traceback
            messagebox.showerror("Ошибка", f"{e}\n\n{traceback.format_exc()}")

    def _update_y_norm(self):
        """Обновляет Y (нормированный или сырой) в зависимости от чекбокса."""
        if self.real_data is None:
            self.y_data = None
            self.y_data_norm = None
            self.norm_y_info_label.config(text="")
            return

        y_raw = self.get_y_raw()
        if self.norm_y_var.get():
            mean_y = np.mean(y_raw)
            if np.isfinite(mean_y) and abs(mean_y) > 1e-12:
                self.y_data = y_raw / mean_y
                self.y_data_norm = self.y_data
                self.norm_y_info_label.config(
                    text=f"Y нормирован: mean(Y) = {mean_y:.6f}\n"
                         f"Y_norm: {self.y_data.min():.4f}…{self.y_data.max():.4f}"
                )
            else:
                self.y_data = y_raw.copy()
                self.y_data_norm = self.y_data
                self.norm_y_info_label.config(
                    text="⚠️ mean(Y) ≈ 0, нормировка не выполнена"
                )
        else:
            self.y_data = y_raw.copy()
            self.y_data_norm = self.y_data
            self.norm_y_info_label.config(
                text=f"Y без нормировки: {self.y_data.min():.6f}…{self.y_data.max():.6f}"
            )

    def get_y_raw(self):
        """Сырой Y (без нормировки) в зависимости от выбранного типа."""
        y = self.y_type_var.get()
        if y == "model" and self.model_data is not None:
            return self.model_data / self.epop
        if y == "i_mean" and self.i_mean_data is not None:
            return self.i_mean_data.copy()
        if y == "i_abs" and self.i_abs_data is not None:
            return self.i_abs_data.copy()
        return self.real_data / self.epop

    def get_y_data(self):
        """Актуальный Y (уже с учётом нормировки, если включена)."""
        return self.y_data

    def get_y_label(self):
        y = self.y_type_var.get()
        base = {
            "model":  "Y = model / EPOP",
            "i_mean": "Y = I_main (доли)",
            "i_abs":  "Y = I_abs_main (чел.)"
        }.get(y, "Y = real / EPOP")
        if self.norm_y_var.get():
            return base + " / mean(Y)"
        return base

    def on_norm_y_change(self):
        """Обработчик переключения чекбокса нормировки Y."""
        if self.real_data is None:
            return
        self._update_y_norm()
        self.y_info_label.config(text=f"Выбран: {self.get_y_label()}")
        self.plot_data()
        self.status_label.config(
            text="Нормировка Y: " + ("вкл" if self.norm_y_var.get() else "выкл")
        )
        # сбрасываем результаты
        self.results = None
        self.save_plot_btn.config(state='disabled')
        self.export_btn.config(state='disabled')

    def on_y_type_change(self):
        if self.real_data is not None:
            self._update_y_norm()
            self.plot_data()
            y_label = self.get_y_label()
            self.y_info_label.config(text=f"Выбран: {y_label}")
            self.status_label.config(text=f"Тип Y: {y_label}")
            # сбрасываем результаты
            self.results = None
            self.save_plot_btn.config(state='disabled')
            self.export_btn.config(state='disabled')

    def apply_epop(self):
        try:
            self.epop = float(self.epop_entry.get())
            self.status_label.config(text=f"EPOP = {self.epop:.0f}")
            if self.real_data is not None:
                self._update_y_norm()
                self.plot_data()
        except ValueError:
            messagebox.showerror("Ошибка", "Введите число для EPOP")

    def plot_data(self):
        if self.r_data is None or self.y_data is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.scatter(self.y_data, self.r_data, alpha=0.6, s=10, label='Данные')
        ax.set_xlabel(self.get_y_label())
        ax.set_ylabel('R(t)')
        ax.set_title(f'Исходные данные (EPOP={self.epop:.0f}, '
                     f'нормировка Y: {"вкл" if self.norm_y_var.get() else "выкл"})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5,
                'Загрузите 2 файла:\n1) R_mean\n2) I_main / real\nи нажмите «Объединить»',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='gray')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        self.canvas.draw()

    def run_regression(self):
        if self.r_data is None or self.real_data is None:
            return
        try:
            self.status_label.config(text="⏳ Регрессия...")
            self.root.update()
            try:
                self.gamma = float(self.gamma_entry.get())
            except Exception:
                self.gamma = 0.074

            # Обновляем Y с учётом нормировки
            self._update_y_norm()

            R_data = self.r_data.copy()
            Y_data = self.y_data.copy() if self.y_data is not None else self.get_y_raw()
            y_label = self.get_y_label()
            norm_note = "Y/mean(Y)" if self.norm_y_var.get() else "Y raw"

            results = self.analyze_lambda_split(
                R_data, Y_data, self.gamma,
                title=f"λ ({y_label})",
                show_plot=True, return_stats=True
            )
            self.results = results

            if results['all'] is not None:
                self.r0_label.config(text=f"{results['all']['R0_start']:.6f}")
                self.p_label.config(text=f"{results['all']['lambda']:.6f}")
                self.r2_label.config(text=f"{results['all']['r2']:.6f}")
                self.corr_label.config(text=f"{np.sqrt(results['all']['r2']):.6f}")
            if results['up'] is not None:
                self.lambda_up_label.config(text=f"{results['up']['lambda']:.6f}")
                self.r2_up_label.config(text=f"{results['up']['r2']:.6f}")
            if results['down'] is not None:
                self.lambda_down_label.config(text=f"{results['down']['lambda']:.6f}")
                self.r2_down_label.config(text=f"{results['down']['r2']:.6f}")

            if results['all'] is not None:
                r_pred = results['all']['R0_start'] * (Y_data ** results['all']['lambda'])
                rmse = np.sqrt(np.mean((R_data - r_pred) ** 2))
                mae = np.mean(np.abs(R_data - r_pred))
                m = np.abs(R_data) > 1e-12
                mape = (np.mean(np.abs((R_data[m] - r_pred[m]) / R_data[m])) * 100
                        if np.any(m) else np.nan)
                self.rmse_label.config(text=f"{rmse:.6f}")
                self.mae_label.config(text=f"{mae:.6f}")
                self.mape_label.config(text=f"{mape:.2f}%")

            self.save_plot_btn.config(state='normal')
            self.export_btn.config(state='normal')
            self.status_label.config(text=f"✅ Регрессия выполнена ({y_label}, {norm_note})")
        except Exception as e:
            import traceback
            messagebox.showerror("Ошибка", f"{e}\n\n{traceback.format_exc()}")
            self.status_label.config(text="❌ Ошибка")

    def analyze_lambda_split(self, R0_data, Y_data, gamma,
                             title="", show_plot=True, return_stats=False):
        """
        R0_data — R(t) (сырой).
        Y_data — уже подготовленный Y (нормированный или сырой — как передан).
        Никакой повторной нормировки здесь НЕ делается.
        """
        R0_data = np.asarray(R0_data, dtype=float)
        Y_data = np.asarray(Y_data, dtype=float)
        m = min(len(R0_data), len(Y_data))
        R0_data = R0_data[:m]
        Y_data = Y_data[:m]

        mask_up = np.zeros(m, dtype=bool)
        mask_down = np.zeros(m, dtype=bool)
        for t in range(1, m):
            if Y_data[t] > Y_data[t-1]:
                mask_up[t] = True
            elif Y_data[t] < Y_data[t-1]:
                mask_down[t] = True
        mask_up &= (Y_data > 1e-9)
        mask_down &= (Y_data > 1e-9)

        results = {}
        for key, msk in [('all', Y_data > 1e-9), ('up', mask_up), ('down', mask_down)]:
            Ys, R0s = Y_data[msk], R0_data[msk]
            if len(Ys) >= 3 and np.all(R0s > 0):
                slope, intercept, r_value, p_value, std_err = linregress(
                    np.log(Ys), np.log(R0s)
                )
                results[key] = {
                    'lambda': slope,
                    'R0_start': np.exp(intercept),
                    'r2': r_value ** 2,
                    'std_err': std_err,
                    'p_value': p_value,
                    'n_points': len(Ys),
                    'slope': slope,
                    'intercept': intercept
                }
            else:
                results[key] = None

        if show_plot:
            self.plot_regression_results(R0_data, Y_data,
                                          mask_up, mask_down, results, gamma)
        return results

    def plot_regression_results(self, R0_data, Y_data,
                                 mask_up, mask_down, results, gamma):
        self.fig.clear()
        y_label = self.get_y_label()
        t = np.arange(len(R0_data))
        Y_up = Y_data[mask_up]
        Y_down = Y_data[mask_down]
        R0_up = R0_data[mask_up]
        R0_down = R0_data[mask_down]

        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # ---- ax1: R(t) с трендами ----
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.plot(t, R0_data, 'k-', lw=1.5, alpha=0.5, label='R(t)')
        if len(Y_up):
            ax1.scatter(t[mask_up], R0_up, c='green', s=30, alpha=0.7, label='Y↑')
        if len(Y_down):
            ax1.scatter(t[mask_down], R0_down, c='red', s=30, alpha=0.7, label='Y↓')
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='R0=1')
        ax1.set_xlabel('День')
        ax1.set_ylabel('R(t)')
        ax1.set_title(f'R(t) с трендами ({y_label})', fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.4)

        # ---- ax2: log-log ----
        ax2 = self.fig.add_subplot(gs[0, 1])
        Y_all = Y_data[Y_data > 1e-9]
        R0_all = R0_data[Y_data > 1e-9]
        if len(Y_all) >= 3:
            ax2.scatter(np.log(Y_all), np.log(R0_all), c='blue', s=10, alpha=0.3, label='Все')
            if results['all'] is not None:
                s, b = results['all']['slope'], results['all']['intercept']
                ax2.plot(np.log(Y_all), b + s*np.log(Y_all), 'b--', lw=1.5,
                         label=f"λ={s:.4f}")
        if len(Y_up) >= 3 and results['up'] is not None:
            ax2.scatter(np.log(Y_up), np.log(R0_up), c='green', s=30, alpha=0.7, label='Y↑')
            s, b = results['up']['slope'], results['up']['intercept']
            ax2.plot(np.log(Y_up), b + s*np.log(Y_up), 'g-', lw=2, label=f"λ_up={s:.4f}")
        if len(Y_down) >= 3 and results['down'] is not None:
            ax2.scatter(np.log(Y_down), np.log(R0_down), c='red', s=30, alpha=0.7, label='Y↓')
            s, b = results['down']['slope'], results['down']['intercept']
            ax2.plot(np.log(Y_down), b + s*np.log(Y_down), 'r-', lw=2, label=f"λ_dn={s:.4f}")
        ax2.set_xlabel(f'log({y_label})')
        ax2.set_ylabel('log(R(t))')
        ax2.set_title('log-log', fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.4)

        # ---- ax3: λ bar ----
        ax3 = self.fig.add_subplot(gs[0, 2])
        labels, lambdas, r2s, colors = [], [], [], []
        for k, c in [('all', 'blue'), ('up', 'green'), ('down', 'red')]:
            if results[k] is not None:
                labels.append(k)
                lambdas.append(results[k]['lambda'])
                r2s.append(results[k]['r2'])
                colors.append(c)
        if labels:
            x = np.arange(len(labels))
            bars = ax3.bar(x, lambdas, color=colors, alpha=0.7, edgecolor='black')
            for bar, v, r2 in zip(bars, lambdas, r2s):
                ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f'{v:.4f}\nR²={r2:.3f}', ha='center', va='bottom', fontsize=8)
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels)
        ax3.axhline(0, color='k', alpha=0.3)
        ax3.set_ylabel('λ')
        ax3.set_title('λ по трендам', fontsize=11)
        ax3.grid(True, alpha=0.4, axis='y')

        # ---- ax4: Y(t) ----
        ax4 = self.fig.add_subplot(gs[1, 0])
        ax4.plot(t, Y_data, 'b-', lw=1.2, alpha=0.5, label=y_label)
        if len(Y_up):
            ax4.scatter(t[mask_up], Y_up, c='green', s=20, alpha=0.7)
        if len(Y_down):
            ax4.scatter(t[mask_down], Y_down, c='red', s=20, alpha=0.7)
        ax4.set_xlabel('День')
        ax4.set_ylabel(y_label)
        ax4.set_title('Y(t)', fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.4)

        # ---- ax5 (real) и ax6 (текстовая сводка) УДАЛЕНЫ ----

        # ---- Главный заголовок ----
        l_all = results['all']['lambda'] if results['all'] else float('nan')
        l_up = results['up']['lambda'] if results['up'] else float('nan')
        l_dn = results['down']['lambda'] if results['down'] else float('nan')

        norm_title = " | Y/mean(Y)" if self.norm_y_var.get() else " | Y raw"
        self.fig.suptitle(
            f'λ-анализ | {y_label} | γ={gamma:.4f} | '
            f'λ_all={l_all:.4f} | λ_up={l_up:.4f} | λ_down={l_dn:.4f} | '
            f'EPOP={self.epop:.0f}{norm_title}',
            fontsize=12, fontweight='bold'
        )
        self.fig.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        if self.results is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить график", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("All files", "*.*")]
        )
        if path:
            try:
                self.fig.savefig(path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("OK", f"Сохранено:\n{path}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def export_results(self):
        if self.results is None:
            return
        path = filedialog.asksaveasfilename(
            title="Экспорт результатов", defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            y_raw = self.get_y_raw()
            y_norm = self.y_data if self.y_data is not None else y_raw
            data = {
                'Y': y_norm,
                'Y_raw': y_raw,
                'Y_over_mean': y_raw / np.mean(y_raw) if np.mean(y_raw) != 0 else y_raw,
                'R_actual': self.r_data,
                'real': self.real_data,
            }
            if self.model_data is not None:
                data['model'] = self.model_data
            if self.i_mean_data is not None:
                data['I_main'] = self.i_mean_data
            if self.i_abs_data is not None:
                data['I_abs_main'] = self.i_abs_data

            df = pd.DataFrame(data)
            if path.endswith('.csv'):
                df.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine='openpyxl') as w:
                    df.to_excel(w, sheet_name='Данные', index=False)
                    params = []
                    for k in ['all', 'up', 'down']:
                        if self.results[k] is not None:
                            params.append({
                                'Тренд': k,
                                'λ': self.results[k]['lambda'],
                                'R0_start': self.results[k]['R0_start'],
                                'R²': self.results[k]['r2'],
                                'Std_err': self.results[k]['std_err'],
                                'P_value': self.results[k]['p_value'],
                                'N': self.results[k]['n_points'],
                            })
                    if params:
                        pd.DataFrame(params).to_excel(w, sheet_name='Параметры', index=False)
                    info = pd.DataFrame([{
                        'Нормировка Y': 'вкл' if self.norm_y_var.get() else 'выкл',
                        'mean(Y_raw)': float(np.mean(y_raw)),
                        'EPOP': self.epop,
                        'Y_type': self.y_type_var.get(),
                    }])
                    info.to_excel(w, sheet_name='Настройки', index=False)
            messagebox.showinfo("OK", f"Сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


def main():
    root = tk.Tk()
    app = RegressionAdvancedGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()