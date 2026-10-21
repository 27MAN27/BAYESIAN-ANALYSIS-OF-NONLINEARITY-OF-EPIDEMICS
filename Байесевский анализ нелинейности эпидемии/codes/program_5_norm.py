import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class ExcelPlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Просмотр Excel: R_mean + I_main + p_mean + abs cases")
        self.root.geometry("1280x800")

        # Основной файл
        self.df = None
        self.excel_path = None
        self.sheet_names = []
        self.xls = None

        # Файл сравнения (I_main)
        self.df_cmp = None
        self.cmp_path = None
        self.cmp_sheet_names = []
        self.cmp_xls = None

        # Третий файл (p_mean)
        self.df_p = None
        self.p_path = None
        self.p_sheet_names = []
        self.p_xls = None

        # === ЧЕТВЁРТЫЙ ФАЙЛ (абсолютные новые случаи, правая ось) ===
        self.df_abs = None
        self.abs_path = None
        self.abs_sheet_names = []
        self.abs_xls = None

        # Средние (для meta)
        self._last_main_mean = None
        self._last_cmp_mean = None
        self._last_p_mean = None
        self._last_abs_mean = None

        # Прозрачность линии p
        self.p_alpha_var = tk.DoubleVar(value=0.4)

        # Какие σ показывать
        self.show_1s_var = tk.BooleanVar(value=True)
        self.show_2s_var = tk.BooleanVar(value=True)
        self.show_3s_var = tk.BooleanVar(value=True)

        # Стиль CI: "fill" — заливка (как на картинке), "lines" — три линии
        self.ci_style_var = tk.StringVar(value="fill")

        # Ограничение левой оси Y (R/I/p)
        self.ylim_left_enabled_var = tk.BooleanVar(value=True)
        self.ylim_left_max_var = tk.StringVar(value="15")

        # Ограничение правой оси Y (abs cases)
        self.ylim_right_enabled_var = tk.BooleanVar(value=False)
        self.ylim_right_max_var = tk.StringVar(value="15")

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        left_container = ttk.Frame(self.root, padding=10)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        canvas = tk.Canvas(left_container, width=280, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical",
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        left.bind("<Configure>", _on_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --------- ОСНОВНОЙ ФАЙЛ ---------
        ttk.Label(left, text="Основной файл (R_mean)",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="Открыть Excel...",
                   command=self.open_file).pack(fill=tk.X, pady=3)

        ttk.Label(left, text="Лист:").pack(anchor=tk.W, pady=(5, 0))
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(left, textvariable=self.sheet_var,
                                        state="readonly", width=25)
        self.sheet_combo.pack(fill=tk.X)
        self.sheet_combo.bind("<<ComboboxSelected>>", lambda e: self.load_sheet())

        ttk.Label(left, text="Ось X:").pack(anchor=tk.W, pady=(8, 0))
        self.x_var = tk.StringVar(value="Day")
        self.x_combo = ttk.Combobox(left, textvariable=self.x_var,
                                    state="readonly", width=25)
        self.x_combo.pack(fill=tk.X)

        ttk.Label(left, text="Ось Y (значение):").pack(anchor=tk.W, pady=(8, 0))
        self.y_var = tk.StringVar(value="R_mean")
        self.y_combo = ttk.Combobox(left, textvariable=self.y_var,
                                    state="readonly", width=25)
        self.y_combo.pack(fill=tk.X)

        ttk.Label(left, text="Доверительный интервал (±σ):").pack(anchor=tk.W, pady=(8, 0))
        self.ci_var = tk.StringVar(value="3σ")
        ttk.Combobox(left, textvariable=self.ci_var, state="readonly", width=25,
                     values=["нет", "1σ", "2σ", "3σ"]).pack(fill=tk.X)

        self.norm_main_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Нормировать осн. (Y / Y_ср)",
                        variable=self.norm_main_var,
                        command=self.plot).pack(anchor=tk.W, pady=(6, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # --------- ФАЙЛ СРАВНЕНИЯ ---------
        ttk.Label(left, text="Файл сравнения (I_main)",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="Открыть Excel...",
                   command=self.open_cmp_file).pack(fill=tk.X, pady=3)

        ttk.Label(left, text="Лист:").pack(anchor=tk.W, pady=(5, 0))
        self.cmp_sheet_var = tk.StringVar()
        self.cmp_sheet_combo = ttk.Combobox(left, textvariable=self.cmp_sheet_var,
                                            state="readonly", width=25)
        self.cmp_sheet_combo.pack(fill=tk.X)
        self.cmp_sheet_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self.load_cmp_sheet())

        ttk.Label(left, text="Ось X (для сравнения):").pack(anchor=tk.W, pady=(8, 0))
        self.cmp_x_var = tk.StringVar(value="day")
        self.cmp_x_combo = ttk.Combobox(left, textvariable=self.cmp_x_var,
                                        state="readonly", width=25)
        self.cmp_x_combo.pack(fill=tk.X)

        ttk.Label(left, text="Ось Y (для сравнения):").pack(anchor=tk.W, pady=(8, 0))
        self.cmp_y_var = tk.StringVar(value="I_main")
        self.cmp_y_combo = ttk.Combobox(left, textvariable=self.cmp_y_var,
                                        state="readonly", width=25)
        self.cmp_y_combo.pack(fill=tk.X)

        self.norm_cmp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Нормировать сравн. (I / I_ср)",
                        variable=self.norm_cmp_var,
                        command=self.plot).pack(anchor=tk.W, pady=(6, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # --------- ТРЕТИЙ ФАЙЛ (p_mean) ---------
        ttk.Label(left, text="Файл p_mean",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="Открыть Excel...",
                   command=self.open_p_file).pack(fill=tk.X, pady=3)

        ttk.Label(left, text="Лист:").pack(anchor=tk.W, pady=(5, 0))
        self.p_sheet_var = tk.StringVar()
        self.p_sheet_combo = ttk.Combobox(left, textvariable=self.p_sheet_var,
                                          state="readonly", width=25)
        self.p_sheet_combo.pack(fill=tk.X)
        self.p_sheet_combo.bind("<<ComboboxSelected>>",
                                lambda e: self.load_p_sheet())

        ttk.Label(left, text="Ось X (для p):").pack(anchor=tk.W, pady=(8, 0))
        self.p_x_var = tk.StringVar(value="day")
        self.p_x_combo = ttk.Combobox(left, textvariable=self.p_x_var,
                                      state="readonly", width=25)
        self.p_x_combo.pack(fill=tk.X)

        ttk.Label(left, text="Ось Y (для p):").pack(anchor=tk.W, pady=(8, 0))
        self.p_y_var = tk.StringVar(value="p_mean")
        self.p_y_combo = ttk.Combobox(left, textvariable=self.p_y_var,
                                      state="readonly", width=25)
        self.p_y_combo.pack(fill=tk.X)

        self.norm_p_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Нормировать p (p / p_ср)",
                        variable=self.norm_p_var,
                        command=self.plot).pack(anchor=tk.W, pady=(6, 0))

        # ---- Доверительный интервал для p ----
        ttk.Label(left, text="Дов. интервал для p (±σ):").pack(anchor=tk.W, pady=(8, 0))
        self.p_ci_var = tk.StringVar(value="нет")
        self.p_ci_combo = ttk.Combobox(
            left, textvariable=self.p_ci_var, state="readonly", width=25,
            values=["нет", "1σ", "2σ", "3σ"]
        )
        self.p_ci_combo.pack(fill=tk.X)
        self.p_ci_combo.bind("<<ComboboxSelected>>", lambda e: self.plot())

        # ---- Прозрачность линии p ----
        ttk.Label(left, text="Прозрачность p:").pack(anchor=tk.W, pady=(8, 0))
        self.p_alpha_scale = ttk.Scale(
            left, from_=0.05, to=1.0,
            variable=self.p_alpha_var,
            orient=tk.HORIZONTAL,
            command=lambda _v: self.plot()
        )
        self.p_alpha_scale.pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # === ЧЕТВЁРТЫЙ ФАЙЛ: абсолютные новые случаи (правая ось) ===
        ttk.Label(left, text="Файл абс. новых случаев (правая ось)",
                  font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="Открыть Excel...",
                   command=self.open_abs_file).pack(fill=tk.X, pady=3)

        ttk.Label(left, text="Лист:").pack(anchor=tk.W, pady=(5, 0))
        self.abs_sheet_var = tk.StringVar()
        self.abs_sheet_combo = ttk.Combobox(left, textvariable=self.abs_sheet_var,
                                            state="readonly", width=25)
        self.abs_sheet_combo.pack(fill=tk.X)
        self.abs_sheet_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self.load_abs_sheet())

        ttk.Label(left, text="Ось X (для abs):").pack(anchor=tk.W, pady=(8, 0))
        self.abs_x_var = tk.StringVar(value="day")
        self.abs_x_combo = ttk.Combobox(left, textvariable=self.abs_x_var,
                                        state="readonly", width=25)
        self.abs_x_combo.pack(fill=tk.X)

        ttk.Label(left, text="Ось Y (для abs):").pack(anchor=tk.W, pady=(8, 0))
        self.abs_y_var = tk.StringVar(value="real")
        self.abs_y_combo = ttk.Combobox(left, textvariable=self.abs_y_var,
                                        state="readonly", width=25)
        self.abs_y_combo.pack(fill=tk.X)

        self.norm_abs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Нормировать abs (Y / Y_ср)",
                        variable=self.norm_abs_var,
                        command=self.plot).pack(anchor=tk.W, pady=(6, 0))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # --------- ОБЩИЕ НАСТРОЙКИ ---------
        ttk.Label(left, text="Стиль доверительного интервала:",
                  font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 3))
        ttk.Radiobutton(left, text="Заливка (fill_between)",
                        variable=self.ci_style_var, value="fill",
                        command=self.plot).pack(anchor=tk.W)
        ttk.Radiobutton(left, text="Линии (±lo, ±hi)",
                        variable=self.ci_style_var, value="lines",
                        command=self.plot).pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Показывать σ (для R и p):",
                  font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 3))
        ttk.Checkbutton(left, text="±1σ", variable=self.show_1s_var,
                        command=self.plot).pack(anchor=tk.W)
        ttk.Checkbutton(left, text="±2σ", variable=self.show_2s_var,
                        command=self.plot).pack(anchor=tk.W)
        ttk.Checkbutton(left, text="±3σ", variable=self.show_3s_var,
                        command=self.plot).pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # ---- Ограничение оси Y ----
        ttk.Label(left, text="Ограничение оси Y:",
                  font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 3))

        # левая ось
        left_ylim_row = ttk.Frame(left)
        left_ylim_row.pack(anchor=tk.W, fill=tk.X)
        ttk.Checkbutton(left_ylim_row, text="Левая ось (R/I/p) до:",
                        variable=self.ylim_left_enabled_var,
                        command=self.plot).pack(side=tk.LEFT)
        self.ylim_left_entry = ttk.Entry(left_ylim_row,
                                         textvariable=self.ylim_left_max_var,
                                         width=8)
        self.ylim_left_entry.pack(side=tk.LEFT, padx=(4, 0))
        self.ylim_left_entry.bind("<Return>", lambda e: self.plot())

        # правая ось
        right_ylim_row = ttk.Frame(left)
        right_ylim_row.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(right_ylim_row, text="Правая ось (abs) до:",
                        variable=self.ylim_right_enabled_var,
                        command=self.plot).pack(side=tk.LEFT)
        self.ylim_right_entry = ttk.Entry(right_ylim_row,
                                          textvariable=self.ylim_right_max_var,
                                          width=8)
        self.ylim_right_entry.pack(side=tk.LEFT, padx=(4, 0))
        self.ylim_right_entry.bind("<Return>", lambda e: self.plot())

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.log_y_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Логарифмическая шкала по Y (обе оси)",
                        variable=self.log_y_var, command=self.plot).pack(anchor=tk.W)

        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Сетка",
                        variable=self.grid_var, command=self.plot).pack(anchor=tk.W)

        self.marker_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Точки-маркеры",
                        variable=self.marker_var, command=self.plot).pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Построить", command=self.plot).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Сохранить график", command=self.save).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Экспорт в Excel", command=self.export_excel).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Выход", command=self.root.quit).pack(fill=tk.X, pady=3)

        self.info_label = ttk.Label(left, text="Файл не загружен",
                                    wraplength=230, font=("Arial", 9))
        self.info_label.pack(anchor=tk.W, pady=(15, 0))

        # Правая панель — график
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax2 = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

    # ---------------- Основной файл ----------------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите основной Excel-файл",
            filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.xls = pd.ExcelFile(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.excel_path = path
        self.sheet_names = self.xls.sheet_names
        self.sheet_combo["values"] = self.sheet_names
        if self.sheet_names:
            default = self.sheet_names[0]
            for s in self.sheet_names:
                tmp = self.xls.parse(s, nrows=0)
                if "R_mean" in tmp.columns:
                    default = s
                    break
            self.sheet_var.set(default)
            self.load_sheet()

        self._update_info()

    def load_sheet(self):
        if self.xls is None:
            return
        sheet = self.sheet_var.get()
        try:
            self.df = self.xls.parse(sheet)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать лист '{sheet}':\n{e}")
            return

        cols = list(self.df.columns)
        self.x_combo["values"] = cols
        self.y_combo["values"] = cols

        if "Day" in cols:
            self.x_var.set("Day")
        elif cols:
            self.x_var.set(cols[0])

        if "R_mean" in cols:
            self.y_var.set("R_mean")
        elif cols:
            self.y_var.set(cols[0])

        self._update_info()
        self.plot()

    # ---------------- Файл сравнения ----------------
    def open_cmp_file(self):
        path = filedialog.askopenfilename(
            title="Выберите Excel-файл для сравнения",
            filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.cmp_xls = pd.ExcelFile(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.cmp_path = path
        self.cmp_sheet_names = self.cmp_xls.sheet_names
        self.cmp_sheet_combo["values"] = self.cmp_sheet_names
        if self.cmp_sheet_names:
            default = self.cmp_sheet_names[0]
            for s in self.cmp_sheet_names:
                tmp = self.cmp_xls.parse(s, nrows=0)
                if "I_main" in tmp.columns:
                    default = s
                    break
            self.cmp_sheet_var.set(default)
            self.load_cmp_sheet()

        self._update_info()

    def load_cmp_sheet(self):
        if self.cmp_xls is None:
            return
        sheet = self.cmp_sheet_var.get()
        try:
            self.df_cmp = self.cmp_xls.parse(sheet)
        except Exception as e:
            messagebox.showerror("Ошибка",
                                 f"Не удалось прочитать лист '{sheet}':\n{e}")
            return

        cols = list(self.df_cmp.columns)
        self.cmp_x_combo["values"] = cols
        self.cmp_y_combo["values"] = cols

        for name in ("day", "Day", "Date"):
            if name in cols:
                self.cmp_x_var.set(name)
                break
        else:
            if cols:
                self.cmp_x_var.set(cols[0])

        for name in ("I_main", "I_abs_main", "R_mean"):
            if name in cols:
                self.cmp_y_var.set(name)
                break
        else:
            if cols:
                self.cmp_y_var.set(cols[0])

        self._update_info()
        self.plot()

    # ---------------- p_mean ----------------
    def open_p_file(self):
        path = filedialog.askopenfilename(
            title="Выберите Excel-файл с p_mean",
            filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.p_xls = pd.ExcelFile(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.p_path = path
        self.p_sheet_names = self.p_xls.sheet_names
        self.p_sheet_combo["values"] = self.p_sheet_names
        if self.p_sheet_names:
            default = self.p_sheet_names[0]
            for s in self.p_sheet_names:
                tmp = self.p_xls.parse(s, nrows=0)
                if "p_mean" in tmp.columns:
                    default = s
                    break
            self.p_sheet_var.set(default)
            self.load_p_sheet()

        self._update_info()

    def load_p_sheet(self):
        if self.p_xls is None:
            return
        sheet = self.p_sheet_var.get()
        try:
            self.df_p = self.p_xls.parse(sheet)
        except Exception as e:
            messagebox.showerror("Ошибка",
                                 f"Не удалось прочитать лист '{sheet}':\n{e}")
            return

        cols = list(self.df_p.columns)
        self.p_x_combo["values"] = cols
        self.p_y_combo["values"] = cols

        for name in ("day", "Day", "Date"):
            if name in cols:
                self.p_x_var.set(name)
                break
        else:
            if cols:
                self.p_x_var.set(cols[0])

        for name in ("p_mean", "p", "p_actual"):
            if name in cols:
                self.p_y_var.set(name)
                break
        else:
            if cols:
                self.p_y_var.set(cols[0])

        self._update_info()
        self.plot()

    # ---------------- Абсолютные новые случаи (правая ось) ----------------
    def open_abs_file(self):
        path = filedialog.askopenfilename(
            title="Выберите Excel-файл с абсолютными новыми случаями",
            filetypes=[("Excel files", "*.xlsx *.xls *.xlsm"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.abs_xls = pd.ExcelFile(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
            return

        self.abs_path = path
        self.abs_sheet_names = self.abs_xls.sheet_names
        self.abs_sheet_combo["values"] = self.abs_sheet_names
        if self.abs_sheet_names:
            default = self.abs_sheet_names[0]
            for s in self.abs_sheet_names:
                tmp = self.abs_xls.parse(s, nrows=0)
                if any(c in tmp.columns for c in
                       ("real", "Inew_abs_main", "Inew_abs", "I_abs_main")):
                    default = s
                    break
            self.abs_sheet_var.set(default)
            self.load_abs_sheet()

        self._update_info()

    def load_abs_sheet(self):
        if self.abs_xls is None:
            return
        sheet = self.abs_sheet_var.get()
        try:
            self.df_abs = self.abs_xls.parse(sheet)
        except Exception as e:
            messagebox.showerror("Ошибка",
                                 f"Не удалось прочитать лист '{sheet}':\n{e}")
            return

        cols = list(self.df_abs.columns)
        self.abs_x_combo["values"] = cols
        self.abs_y_combo["values"] = cols

        for name in ("day", "Day", "Date", "i", "index"):
            if name in cols:
                self.abs_x_var.set(name)
                break
        else:
            if cols:
                self.abs_x_var.set(cols[0])

        for name in ("real", "Inew_abs_main", "Inew_abs", "I_abs_main",
                     "cases", "new_cases"):
            if name in cols:
                self.abs_y_var.set(name)
                break
        else:
            if cols:
                self.abs_y_var.set(cols[0])

        self._update_info()
        self.plot()

    # ---------------- Info ----------------
    def _update_info(self):
        parts = []
        if self.excel_path:
            parts.append(f"Основной: {self.excel_path}")
            parts.append(f"  лист: {self.sheet_var.get()}, строк: "
                         f"{len(self.df) if self.df is not None else 0}")
        else:
            parts.append("Основной: не загружен")

        if self.cmp_path:
            parts.append(f"Сравнение: {self.cmp_path}")
            parts.append(f"  лист: {self.cmp_sheet_var.get()}, строк: "
                         f"{len(self.df_cmp) if self.df_cmp is not None else 0}")
        else:
            parts.append("Сравнение: не загружен")

        if self.p_path:
            parts.append(f"p_mean: {self.p_path}")
            parts.append(f"  лист: {self.p_sheet_var.get()}, строк: "
                         f"{len(self.df_p) if self.df_p is not None else 0}")
        else:
            parts.append("p_mean: не загружен")

        if self.abs_path:
            parts.append(f"abs cases: {self.abs_path}")
            parts.append(f"  лист: {self.abs_sheet_var.get()}, строк: "
                         f"{len(self.df_abs) if self.df_abs is not None else 0}")
        else:
            parts.append("abs cases: не загружен")

        self.info_label.config(text="\n".join(parts))

    # ---------------- Вспомогательное ----------------
    def _ci_columns_map(self):
        return [
            ("1σ", "R_lo1s", "R_hi1s"),
            ("2σ", "R_lo2s", "R_hi2s"),
            ("3σ", "R_lo3s", "R_hi3s"),
        ]

    def _ci_columns_map_p(self):
        return [
            ("1σ", "p_lo1s", "p_hi1s"),
            ("2σ", "p_lo2s", "p_hi2s"),
            ("3σ", "p_lo3s", "p_hi3s"),
        ]

    def _collect_series(self):
        result = {"x": None, "main": None, "cmp": None, "p": None, "abs": None}

        # ---- основной ----
        if self.df is not None:
            xcol = self.x_var.get()
            ycol = self.y_var.get()
            if xcol in self.df.columns and ycol in self.df.columns:
                df = self.df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce")
                df = df.dropna(subset=[xcol, ycol]).sort_values(by=xcol)
                x = df[xcol].to_numpy()
                y = df[ycol].to_numpy()

                ci_bands = []
                for level, lo_col, hi_col in self._ci_columns_map():
                    if (lo_col in self.df.columns
                            and hi_col in self.df.columns):
                        sub = self.df[[xcol, lo_col, hi_col]].apply(
                            pd.to_numeric, errors="coerce").dropna().sort_values(by=xcol)
                        merged = pd.merge(
                            pd.DataFrame({xcol: x}),
                            pd.DataFrame({xcol: sub[xcol].values,
                                          "lo": sub[lo_col].values,
                                          "hi": sub[hi_col].values}),
                            on=xcol, how="left"
                        )
                        ci_bands.append(
                            (level,
                             merged["lo"].to_numpy(),
                             merged["hi"].to_numpy())
                        )

                norm_main = self.norm_main_var.get()
                mean_main = None
                if norm_main:
                    finite_mask = np.isfinite(y)
                    if finite_mask.any():
                        mean_main = float(np.nanmean(y))
                        if mean_main != 0:
                            y = y / mean_main
                            ci_bands = [
                                (lvl, lo / mean_main, hi / mean_main)
                                for lvl, lo, hi in ci_bands
                            ]
                        else:
                            mean_main = None

                result["x"] = x
                result["main"] = {
                    "name": ycol + (" (норм.)" if norm_main and mean_main else ""),
                    "y": y,
                    "ci_bands": ci_bands,
                    "norm": norm_main and (mean_main is not None),
                    "mean": mean_main,
                }

        # ---- сравнение ----
        if self.df_cmp is not None:
            xcol_c = self.cmp_x_var.get()
            ycol_c = self.cmp_y_var.get()
            if xcol_c in self.df_cmp.columns and ycol_c in self.df_cmp.columns:
                df_c = self.df_cmp[[xcol_c, ycol_c]].apply(
                    pd.to_numeric, errors="coerce")
                df_c = df_c.dropna(subset=[xcol_c, ycol_c]).sort_values(by=xcol_c)
                x_c = df_c[xcol_c].to_numpy()
                y_c = df_c[ycol_c].to_numpy()

                norm_cmp = self.norm_cmp_var.get()
                mean_cmp = None
                if norm_cmp:
                    finite_mask = np.isfinite(y_c)
                    if finite_mask.any():
                        mean_cmp = float(np.nanmean(y_c))
                        if mean_cmp != 0:
                            y_c = y_c / mean_cmp
                        else:
                            mean_cmp = None

                result["cmp"] = {
                    "name": ycol_c + (" (норм.)" if norm_cmp and mean_cmp else ""),
                    "x": x_c,
                    "y": y_c,
                    "norm": norm_cmp and (mean_cmp is not None),
                    "mean": mean_cmp,
                }

        # ---- p_mean ----
        if self.df_p is not None:
            xcol_p = self.p_x_var.get()
            ycol_p = self.p_y_var.get()
            if xcol_p in self.df_p.columns and ycol_p in self.df_p.columns:
                df_p = self.df_p[[xcol_p, ycol_p]].apply(
                    pd.to_numeric, errors="coerce")
                df_p = df_p.dropna(subset=[xcol_p, ycol_p]).sort_values(by=xcol_p)
                x_p = df_p[xcol_p].to_numpy()
                y_p = df_p[ycol_p].to_numpy()

                ci_bands_p = []
                for level, lo_col, hi_col in self._ci_columns_map_p():
                    if (lo_col in self.df_p.columns
                            and hi_col in self.df_p.columns):
                        sub_p = self.df_p[[xcol_p, lo_col, hi_col]].apply(
                            pd.to_numeric, errors="coerce").dropna().sort_values(by=xcol_p)
                        merged_p = pd.merge(
                            pd.DataFrame({xcol_p: x_p}),
                            pd.DataFrame({xcol_p: sub_p[xcol_p].values,
                                          "lo": sub_p[lo_col].values,
                                          "hi": sub_p[hi_col].values}),
                            on=xcol_p, how="left"
                        )
                        ci_bands_p.append(
                            (level,
                             merged_p["lo"].to_numpy(),
                             merged_p["hi"].to_numpy())
                        )

                norm_p = self.norm_p_var.get()
                mean_p = None
                if norm_p:
                    finite_mask = np.isfinite(y_p)
                    if finite_mask.any():
                        mean_p = float(np.nanmean(y_p))
                        if mean_p != 0:
                            y_p = y_p / mean_p
                            ci_bands_p = [
                                (lvl, lo / mean_p, hi / mean_p)
                                for lvl, lo, hi in ci_bands_p
                            ]
                        else:
                            mean_p = None

                result["p"] = {
                    "name": ycol_p + (" (норм.)" if norm_p and mean_p else ""),
                    "x": x_p,
                    "y": y_p,
                    "ci_bands": ci_bands_p,
                    "norm": norm_p and (mean_p is not None),
                    "mean": mean_p,
                }

        # ---- abs cases ----
        if self.df_abs is not None:
            xcol_a = self.abs_x_var.get()
            ycol_a = self.abs_y_var.get()
            if xcol_a in self.df_abs.columns and ycol_a in self.df_abs.columns:
                df_a = self.df_abs[[xcol_a, ycol_a]].apply(
                    pd.to_numeric, errors="coerce")
                df_a = df_a.dropna(subset=[xcol_a, ycol_a]).sort_values(by=xcol_a)
                x_a = df_a[xcol_a].to_numpy()
                y_a = df_a[ycol_a].to_numpy()

                norm_abs = self.norm_abs_var.get()
                mean_abs = None
                if norm_abs:
                    finite_mask = np.isfinite(y_a)
                    if finite_mask.any():
                        mean_abs = float(np.nanmean(y_a))
                        if mean_abs != 0:
                            y_a = y_a / mean_abs
                        else:
                            mean_abs = None

                result["abs"] = {
                    "name": ycol_a + (" (норм.)" if norm_abs and mean_abs else ""),
                    "x": x_a,
                    "y": y_a,
                    "norm": norm_abs and (mean_abs is not None),
                    "mean": mean_abs,
                }

        if (result["main"] is None and result["cmp"] is None
                and result["p"] is None and result["abs"] is None):
            return None
        return result

    # ---------------- Plot ----------------
    def plot(self):
        data = self._collect_series()
        if data is None:
            return

        self._last_main_mean = data["main"]["mean"] if data["main"] else None
        self._last_cmp_mean = data["cmp"]["mean"] if data["cmp"] else None
        self._last_p_mean = data["p"]["mean"] if data["p"] else None
        self._last_abs_mean = data["abs"]["mean"] if data["abs"] else None

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax2 = None

        main = data["main"]
        cmp_ = data["cmp"]
        p_ = data["p"]
        abs_ = data["abs"]

        try:
            p_alpha = float(self.p_alpha_var.get())
        except Exception:
            p_alpha = 0.4
        p_alpha = max(0.0, min(1.0, p_alpha))

        show_1s = self.show_1s_var.get()
        show_2s = self.show_2s_var.get()
        show_3s = self.show_3s_var.get()

        def _level_enabled(level):
            if level == "1σ":
                return show_1s
            if level == "2σ":
                return show_2s
            if level == "3σ":
                return show_3s
            return False

        fill_alpha = {"1σ": 0.35, "2σ": 0.22, "3σ": 0.12}
        line_style = {
            "1σ": ("--", "-.", 0.85, 1.0),
            "2σ": ("--", "-.", 0.55, 0.9),
            "3σ": (":",  ":",  0.35, 0.8),
        }

        ci_style = self.ci_style_var.get()

        def _level_sort_key(level):
            return {"1σ": 0, "2σ": 1, "3σ": 2}.get(level, 99)

        # ---- ЛЕВАЯ ОСЬ: основной файл ----
        main_ci_labels = []
        if main is not None:
            x = data["x"]
            y = main["y"]

            bands = [b for b in main.get("ci_bands", [])
                     if _level_enabled(b[0])]

            if ci_style == "fill":
                ordered = sorted(bands, key=lambda b: _level_sort_key(b[0]),
                                 reverse=True)
                for level, lo_vals, hi_vals in ordered:
                    lbl = f"{main['name']} ± {level}"
                    self.ax.fill_between(x, lo_vals, hi_vals,
                                         color="#1f77b4",
                                         alpha=fill_alpha[level],
                                         label=lbl)
                    main_ci_labels.append(lbl)
            else:
                for level, lo_vals, hi_vals in bands:
                    ls_lo, ls_hi, a, lw = line_style[level]
                    lbl_lo = f"{main['name']}: −{level}"
                    lbl_hi = f"{main['name']}: +{level}"
                    self.ax.plot(x, lo_vals, color="#1f77b4",
                                 lw=lw, linestyle=ls_lo, alpha=a,
                                 label=lbl_lo)
                    self.ax.plot(x, hi_vals, color="#1f77b4",
                                 lw=lw, linestyle=ls_hi, alpha=a,
                                 label=lbl_hi)
                    main_ci_labels += [lbl_lo, lbl_hi]

            self.ax.plot(x, y, color="#1f77b4", lw=2.0,
                         label=main["name"])

        # ---- Сравнение (I_main) ----
        if cmp_ is not None:
            self.ax.plot(cmp_["x"], cmp_["y"], color="#d62728", lw=1.5,
                         label=cmp_["name"])

        # ---- p_mean ----
        p_label = None
        p_ci_labels = []

        if p_ is not None:
            p_label = p_["name"]
            bands_p = [b for b in p_.get("ci_bands", [])
                       if _level_enabled(b[0])]

            if ci_style == "fill":
                ordered_p = sorted(bands_p,
                                   key=lambda b: _level_sort_key(b[0]),
                                   reverse=True)
                for level, lo_vals, hi_vals in ordered_p:
                    lbl = f"{p_['name']} ± {level}"
                    self.ax.fill_between(p_["x"], lo_vals, hi_vals,
                                         color="#2ca02c",
                                         alpha=fill_alpha[level] * p_alpha,
                                         label=lbl)
                    p_ci_labels.append(lbl)
            else:
                for level, lo_vals, hi_vals in bands_p:
                    ls_lo, ls_hi, a, lw = line_style[level]
                    a_p = a * p_alpha
                    lbl_lo = f"{p_['name']}: −{level}"
                    lbl_hi = f"{p_['name']}: +{level}"
                    self.ax.plot(p_["x"], lo_vals, color="#2ca02c",
                                 lw=lw, linestyle=ls_lo, alpha=a_p,
                                 label=lbl_lo)
                    self.ax.plot(p_["x"], hi_vals, color="#2ca02c",
                                 lw=lw, linestyle=ls_hi, alpha=a_p,
                                 label=lbl_hi)
                    p_ci_labels += [lbl_lo, lbl_hi]

            self.ax.plot(p_["x"], p_["y"], color="#2ca02c", lw=1.6,
                         alpha=p_alpha,
                         label=p_label)

        # ---- ПРАВАЯ ОСЬ (abs cases) ----
        if abs_ is not None:
            self.ax2 = self.ax.twinx()
            self.ax2.plot(abs_["x"], abs_["y"], color="#ff7f0e", lw=1.6,
                          label=f"{abs_['name']} (правая ось)")
            self.ax2.set_ylabel(abs_["name"], color="#ff7f0e")
            self.ax2.tick_params(axis="y", labelcolor="#ff7f0e")
            if self.log_y_var.get():
                self.ax2.set_yscale("log")
            h1, l1 = self.ax.get_legend_handles_labels()
            h2, l2 = self.ax2.get_legend_handles_labels()
            leg = self.ax.legend(h1 + h2, l1 + l2, loc="best",
                                 fontsize=7, ncol=2)
        else:
            leg = self.ax.legend(loc="best", fontsize=7, ncol=2)

        if leg is not None:
            wanted = {p_label}
            wanted.update(p_ci_labels)
            wanted.discard(None)
            for handle, text in zip(leg.legend_handles, leg.get_texts()):
                if text.get_text() in wanted:
                    try:
                        handle.set_alpha(1.0)
                    except Exception:
                        pass

        # ---- оформление ----
        xlabel_parts = []
        if main is not None:
            xlabel_parts.append(self.x_var.get())
        if cmp_ is not None and self.cmp_x_var.get() not in xlabel_parts:
            xlabel_parts.append(self.cmp_x_var.get())
        if p_ is not None and self.p_x_var.get() not in xlabel_parts:
            xlabel_parts.append(self.p_x_var.get())
        if abs_ is not None and self.abs_x_var.get() not in xlabel_parts:
            xlabel_parts.append(self.abs_x_var.get())
        self.ax.set_xlabel(" / ".join(xlabel_parts) if xlabel_parts else "x")

        ylabel_parts = []
        if main is not None:
            ylabel_parts.append(main["name"])
        if cmp_ is not None:
            ylabel_parts.append(cmp_["name"])
        if p_ is not None:
            ylabel_parts.append(p_["name"])
        self.ax.set_ylabel(" / ".join(ylabel_parts) if ylabel_parts else "value")

        title_parts = []
        if main is not None:
            title_parts.append(f"{main['name']} ({self.sheet_var.get()})")
        if cmp_ is not None:
            title_parts.append(f"{cmp_['name']} ({self.cmp_sheet_var.get()})")
        if p_ is not None:
            title_parts.append(f"{p_['name']} ({self.p_sheet_var.get()})")
        if abs_ is not None:
            title_parts.append(f"{abs_['name']} ({self.abs_sheet_var.get()})")
        self.ax.set_title(" vs ".join(title_parts))

        if self.log_y_var.get():
            self.ax.set_yscale("log")
        if self.grid_var.get():
            self.ax.grid(True, which="both", alpha=0.3)

        # ---- ОГРАНИЧЕНИЕ ЛЕВОЙ ОСИ Y ----
        if self.ylim_left_enabled_var.get():
            try:
                ymax = float(self.ylim_left_max_var.get())
                self.ax.set_ylim(0, ymax)
            except Exception:
                pass

        # ---- ОГРАНИЧЕНИЕ ПРАВОЙ ОСИ Y ----
        if self.ax2 is not None and self.ylim_right_enabled_var.get():
            try:
                ymax = float(self.ylim_right_max_var.get())
                self.ax2.set_ylim(0, ymax)
            except Exception:
                pass

        self.fig.tight_layout()
        self.canvas.draw()
        self._update_info()

    # ---------------- Save graphic ----------------
    def save(self):
        if (self.df is None and self.df_cmp is None
                and self.df_p is None and self.df_abs is None):
            messagebox.showwarning("Внимание", "Сначала загрузите Excel-файл.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Готово", f"Сохранено: {path}")

    # ---------------- Export to Excel ----------------
    def export_excel(self):
        data = self._collect_series()
        if data is None:
            messagebox.showwarning("Внимание",
                                   "Нечего экспортировать: сначала постройте график.")
            return

        path = filedialog.asksaveasfilename(
            title="Куда сохранить Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile="exported_series.xlsx"
        )
        if not path:
            return

        try:
            main = data["main"]
            cmp_ = data["cmp"]
            p_ = data["p"]
            abs_ = data["abs"]

            frames = []
            x_names = {}

            if main is not None:
                x_main_name = self.x_var.get()
                df_main = pd.DataFrame({
                    x_main_name: data["x"],
                    main["name"]: main["y"],
                })
                for level, lo_vals, hi_vals in main.get("ci_bands", []):
                    df_main[f"{main['name']}_lo_{level}"] = lo_vals
                    df_main[f"{main['name']}_hi_{level}"] = hi_vals
                frames.append(df_main)
                x_names["main"] = x_main_name

            if cmp_ is not None:
                x_cmp_name = self.cmp_x_var.get()
                df_cmp = pd.DataFrame({
                    x_cmp_name: cmp_["x"],
                    cmp_["name"]: cmp_["y"],
                })
                frames.append(df_cmp)
                x_names["cmp"] = x_cmp_name

            if p_ is not None:
                x_p_name = self.p_x_var.get()
                df_p = pd.DataFrame({
                    x_p_name: p_["x"],
                    p_["name"]: p_["y"],
                })
                for level, lo_vals, hi_vals in p_.get("ci_bands", []):
                    df_p[f"{p_['name']}_lo_{level}"] = lo_vals
                    df_p[f"{p_['name']}_hi_{level}"] = hi_vals
                frames.append(df_p)
                x_names["p"] = x_p_name

            if abs_ is not None:
                x_abs_name = self.abs_x_var.get()
                df_abs = pd.DataFrame({
                    x_abs_name: abs_["x"],
                    abs_["name"]: abs_["y"],
                })
                frames.append(df_abs)
                x_names["abs"] = x_abs_name

            if len(frames) >= 2 and len(set(x_names.values())) == 1:
                df_out = frames[0]
                for f in frames[1:]:
                    df_out = pd.merge(df_out, f, on=list(x_names.values())[0], how="outer")
                df_out = df_out.sort_values(by=list(x_names.values())[0]).reset_index(drop=True)
            elif len(frames) == 1:
                df_out = frames[0].reset_index(drop=True)
            else:
                max_len = max(len(f) for f in frames)
                df_out = pd.DataFrame({"index": np.arange(max_len)})
                for f in frames:
                    df_out = pd.concat([df_out, f.reset_index(drop=True)], axis=1)

            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                df_out.to_excel(xw, sheet_name="data", index=False)

                meta = pd.DataFrame([{
                    "file_main": self.excel_path or "",
                    "sheet_main": self.sheet_var.get() if self.df is not None else "",
                    "x_main": self.x_var.get() if self.df is not None else "",
                    "y_main": self.y_var.get() if self.df is not None else "",
                    "norm_main": bool(self.norm_main_var.get()),
                    "mean_main": self._last_main_mean if self._last_main_mean is not None else "",
                    "ci_level": self.ci_var.get(),
                    "ci_style": self.ci_style_var.get(),
                    "show_1s": bool(self.show_1s_var.get()),
                    "show_2s": bool(self.show_2s_var.get()),
                    "show_3s": bool(self.show_3s_var.get()),
                    "ylim_left_enabled": bool(self.ylim_left_enabled_var.get()),
                    "ylim_left_max": self.ylim_left_max_var.get(),
                    "ylim_right_enabled": bool(self.ylim_right_enabled_var.get()),
                    "ylim_right_max": self.ylim_right_max_var.get(),
                    "file_cmp": self.cmp_path or "",
                    "sheet_cmp": self.cmp_sheet_var.get() if self.df_cmp is not None else "",
                    "x_cmp": self.cmp_x_var.get() if self.df_cmp is not None else "",
                    "y_cmp": self.cmp_y_var.get() if self.df_cmp is not None else "",
                    "norm_cmp": bool(self.norm_cmp_var.get()),
                    "mean_cmp": self._last_cmp_mean if self._last_cmp_mean is not None else "",
                    "file_p": self.p_path or "",
                    "sheet_p": self.p_sheet_var.get() if self.df_p is not None else "",
                    "x_p": self.p_x_var.get() if self.df_p is not None else "",
                    "y_p": self.p_y_var.get() if self.df_p is not None else "",
                    "norm_p": bool(self.norm_p_var.get()),
                    "mean_p": self._last_p_mean if self._last_p_mean is not None else "",
                    "alpha_p": float(self.p_alpha_var.get()),
                    "ci_level_p": self.p_ci_var.get(),
                    "file_abs": self.abs_path or "",
                    "sheet_abs": self.abs_sheet_var.get() if self.df_abs is not None else "",
                    "x_abs": self.abs_x_var.get() if self.df_abs is not None else "",
                    "y_abs": self.abs_y_var.get() if self.df_abs is not None else "",
                    "norm_abs": bool(self.norm_abs_var.get()),
                    "mean_abs": self._last_abs_mean if self._last_abs_mean is not None else "",
                }])
                meta.to_excel(xw, sheet_name="meta", index=False)

            messagebox.showinfo("Готово", f"Экспортировано: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ExcelPlotApp(root)
    root.mainloop()