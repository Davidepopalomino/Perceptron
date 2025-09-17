# app_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import numpy as np
import os, requests
from io import BytesIO
from perceptron import Perceptron
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PerceptronApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptrón Simple - Taller")
        self.geometry("1100x650")
        self.df = None
        self.perceptron = None
        self.errors = []
        self.training_thread = None
        self.train_running = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_df = None  # DataFrame con los datos de prueba
        self._create_widgets()

    def _create_widgets(self):
        # --- Panel izquierdo scrollable ---
        container = ttk.Frame(self)
        container.pack(side="left", fill="y")

        canvas = tk.Canvas(container, borderwidth=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.left_frame = ttk.Frame(canvas)

        self.left_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="y", expand=True)
        vscroll.pack(side="right", fill="y")

        # --- Botón para archivo local ---
        btn_load = ttk.Button(self.left_frame, text="Cargar dataset local", command=self.load_dataset)
        btn_load.pack(fill="x", pady=4)

        # --- Campo URL para nube ---
        ttk.Label(self.left_frame, text="URL Google Drive / nube:").pack(anchor="w", pady=(4,0))
        self.url_entry = ttk.Entry(self.left_frame)
        self.url_entry.pack(fill="x", pady=2)

        ttk.Label(self.left_frame, text="Formato archivo URL:").pack(anchor="w")
        self.format_combo = ttk.Combobox(self.left_frame, values=["csv", "excel", "json"], state="readonly")
        self.format_combo.pack(fill="x", pady=2)
        self.format_combo.set("csv")

        ttk.Button(self.left_frame, text="Cargar dataset desde URL", command=self.load_dataset_from_url).pack(fill="x", pady=2)

        self.lbl_file = ttk.Label(self.left_frame, text="Archivo: (ninguno)")
        self.lbl_file.pack(fill="x", pady=2)

        self.lbl_info = ttk.Label(self.left_frame, text="Entradas: -   Salidas: -   Patrones: -")
        self.lbl_info.pack(fill="x", pady=2)

        ttk.Label(self.left_frame, text="Columna objetivo (y):").pack(anchor="w", pady=(8,0))
        self.target_combo = ttk.Combobox(self.left_frame, state="readonly")
        self.target_combo.pack(fill="x", pady=2)

        # --- Información sobre división de datos (80/20 fijo) ---
        ttk.Label(self.left_frame, text="División de datos:").pack(anchor="w", pady=(8,0))
        ttk.Label(self.left_frame, text="80% para entrenamiento", foreground="blue").pack(anchor="w")
        ttk.Label(self.left_frame, text="20% para prueba/simulación", foreground="green").pack(anchor="w")

        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.left_frame, text="Normalizar entradas (min-max)", variable=self.normalize_var).pack(anchor="w", pady=4)

        ttk.Separator(self.left_frame).pack(fill="x", pady=6)
        ttk.Label(self.left_frame, text="Parámetros de entrenamiento:").pack(anchor="w")
        self.eta_var = tk.DoubleVar(value=0.1)
        ttk.Label(self.left_frame, text="Tasa de aprendizaje (η):").pack(anchor="w")
        ttk.Entry(self.left_frame, textvariable=self.eta_var).pack(fill="x", pady=2)

        self.maxiter_var = tk.IntVar(value=100)
        ttk.Label(self.left_frame, text="Máx. iteraciones:").pack(anchor="w")
        ttk.Entry(self.left_frame, textvariable=self.maxiter_var).pack(fill="x", pady=2)

        self.tol_var = tk.DoubleVar(value=0.001)
        ttk.Label(self.left_frame, text="Error máximo (ε) [MSE]:").pack(anchor="w")
        ttk.Entry(self.left_frame, textvariable=self.tol_var).pack(fill="x", pady=2)

        ttk.Separator(self.left_frame).pack(fill="x", pady=6)
        ttk.Label(self.left_frame, text="Inicialización pesos/bias:").pack(anchor="w")
        self.weights_entry = ttk.Entry(self.left_frame)
        self.weights_entry.pack(fill="x", pady=2)
        self.weights_entry.insert(0, "")
        ttk.Label(self.left_frame, text="Pesos (coma sep.) - dejar vacío = aleatorio").pack(anchor="w")

        self.bias_entry = ttk.Entry(self.left_frame)
        self.bias_entry.pack(fill="x", pady=2)
        self.bias_entry.insert(0, "")
        ttk.Label(self.left_frame, text="Bias / Umbral (dejar vacío = aleatorio)").pack(anchor="w")

        ttk.Label(self.left_frame, text="Algoritmo:").pack(anchor="w", pady=(8,0))
        self.alg_var = tk.StringVar(value="perceptron")
        ttk.Radiobutton(self.left_frame, text="Perceptron Rule (clásico)", variable=self.alg_var, value="perceptron").pack(anchor="w")
        ttk.Radiobutton(self.left_frame, text="Delta Rule (MSE)", variable=self.alg_var, value="delta").pack(anchor="w")

        self.btn_init = ttk.Button(self.left_frame, text="Inicializar perceptrón", command=self.initialize_perceptron)
        self.btn_init.pack(fill="x", pady=6)

        self.btn_train = ttk.Button(self.left_frame, text="Iniciar entrenamiento", command=self.start_training, state="disabled")
        self.btn_train.pack(fill="x", pady=4)

        self.btn_stop = ttk.Button(self.left_frame, text="Detener entrenamiento", command=self.stop_training, state="disabled")
        self.btn_stop.pack(fill="x", pady=2)

        ttk.Separator(self.left_frame).pack(fill="x", pady=6)
        ttk.Label(self.left_frame, text="Simulación / Prueba:").pack(anchor="w")
        
        # Combo para seleccionar filas de prueba (del 20% de datos de prueba)
        ttk.Label(self.left_frame, text="Seleccionar patrón de prueba:").pack(anchor="w", pady=(4,0))
        self.row_combo = ttk.Combobox(self.left_frame, state="readonly")
        self.row_combo.pack(fill="x", pady=2)
        ttk.Button(self.left_frame, text="Probar patrón seleccionado", command=self.test_selected_row).pack(fill="x", pady=2)

        ttk.Label(self.left_frame, text="Ingresar nuevo patrón (coma sep):").pack(anchor="w", pady=(6,0))
        self.newpattern_entry = ttk.Entry(self.left_frame)
        self.newpattern_entry.pack(fill="x", pady=2)
        ttk.Button(self.left_frame, text="Probar patrón nuevo", command=self.test_new_pattern).pack(fill="x", pady=2)

        # --- Panel central ---
        center = ttk.Frame(self, padding=8)
        center.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Error durante entrenamiento")
        self.ax.set_xlabel("Iteración")
        self.ax.set_ylabel("MSE")
        self.line, = self.ax.plot([], [])

        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Panel derecho ---
        right = ttk.Frame(self, padding=8, width=280)
        right.pack(side="left", fill="y")

        ttk.Label(right, text="Logs:").pack(anchor="w")

        log_frame = ttk.Frame(right)
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = tk.Text(log_frame, height=20, width=40, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")

        x_scroll = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        

        self.log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        # Expandir correctamente
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        ttk.Label(right, text="Evaluación:").pack(anchor="w", pady=(8,0))
        self.eval_text = tk.Text(right, height=6, width=40)
        self.eval_text.pack(fill="x")

    def log(self, msg):
        self.log_text.insert("end", str(msg) + "\n")
        self.log_text.see("end")

    def load_dataset(self):
        fname = filedialog.askopenfilename(title="Seleccionar dataset",
                                           filetypes=[("CSV","*.csv"),("Excel","*.xlsx;*.xls"),("JSON","*.json"),("All files","*.*")])
        if not fname:
            return
        self._load_dataframe(fname)

    def load_dataset_from_url(self):
        url = self.url_entry.get().strip()
        fmt = self.format_combo.get().strip().lower()
        if not url:
            messagebox.showwarning("Atención","Pegue primero una URL de Google Drive o nube")
            return
        try:
            if "drive.google.com" in url and "uc?export=download" not in url:
                import re
                m = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
                if m:
                    file_id = m.group(1)
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = requests.get(url)
            r.raise_for_status()
            content = BytesIO(r.content)
            if fmt == "csv":
                df = pd.read_csv(content)
            elif fmt == "json":
                df = pd.read_json(content)
            else:  # excel
                df = pd.read_excel(content, engine="openpyxl")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo desde la URL:\n{e}")
            return
        self._set_dataframe(df, f"URL: {url}")

    def _load_dataframe(self, fname):
        try:
            if fname.lower().endswith(".csv"):
                df = pd.read_csv(fname)
            elif fname.lower().endswith(".json"):
                df = pd.read_json(fname)
            else:
                df = pd.read_excel(fname)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")
            return
        self._set_dataframe(df, os.path.basename(fname))

    def _set_dataframe(self, df, name):
        self.df = df
        self.lbl_file.config(text=f"Archivo: {name}")
        cols = list(df.columns)
        self.target_combo['values'] = cols
        self.target_combo.set(cols[-1])
        n_inputs = len(cols) - 1
        n_outputs = 1
        n_patterns = len(df)
        self.lbl_info.config(text=f"Entradas: {n_inputs}   Salidas: {n_outputs}   Patrones: {n_patterns}")
        
        # Limpiar el combo de selección de patrones de prueba
        self.row_combo.set('')
        self.row_combo['values'] = []
        
        self.log(f"Dataset cargado: {name} ({len(df)} filas, {len(cols)} columnas)")
        self.btn_train.config(state="disabled")

    def initialize_perceptron(self):
        if self.df is None:
            messagebox.showwarning("Atención", "Carga primero un dataset")
            return
        
        # Dividir datos en entrenamiento (80%) y prueba (20%)
        train_percent = 0.8
        
        # Mezclar los datos antes de dividir
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calcular índices de división
        split_idx = int(len(df_shuffled) * train_percent)
        
        # Dividir en entrenamiento y prueba
        train_df = df_shuffled.iloc[:split_idx]
        test_df = df_shuffled.iloc[split_idx:]
        self.test_df = test_df  # Guardar para usar en pruebas
        
        target_col = self.target_combo.get()
        cols = [c for c in self.df.columns if c != target_col]
        
        # Preparar datos de entrenamiento
        X_train = train_df[cols].values
        y_train = train_df[target_col].values
        
        # Preparar datos de prueba
        X_test = test_df[cols].values
        y_test = test_df[target_col].values
        
        n_inputs = X_train.shape[1]
        self.perceptron = Perceptron(n_inputs)
        
        if self.normalize_var.get():
            self.perceptron.fit_normalizer(X_train)
            X_train_norm = self.perceptron.normalize(X_train)
            X_test_norm = self.perceptron.normalize(X_test)
        else:
            X_train_norm = X_train.astype(float)
            X_test_norm = X_test.astype(float)
            
        weights_text = self.weights_entry.get().strip()
        bias_text = self.bias_entry.get().strip()
        
        if weights_text:
            try:
                weights = [float(x.strip()) for x in weights_text.split(",")]
            except:
                messagebox.showerror("Error", "Pesos deben ser números separados por comas")
                return
            bias = None
            if bias_text:
                try:
                    bias = float(bias_text)
                except:
                    messagebox.showerror("Error", "Bias debe ser número")
                    return
            self.perceptron.initialize_weights(weights=weights, bias=bias, random_init=False)
        else:
            seed = None
            self.perceptron.initialize_weights(random_init=True, seed=seed)
            
        self.X_train = X_train_norm
        self.y_train = np.array(y_train, dtype=float)
        self.X_test = X_test_norm
        self.y_test = np.array(y_test, dtype=float)
        
        # Actualizar combo con índices de prueba
        test_indices = [str(i) for i in range(len(test_df))]
        self.row_combo['values'] = test_indices
        if test_indices:
            self.row_combo.set(test_indices[0])
        
        self.log(f"Perceptrón inicializado. Entradas={n_inputs}")
        self.log(f"Datos divididos: {len(train_df)} para entrenamiento (80%), {len(test_df)} para prueba (20%)")
        self.log(f"Pesos iniciales: {self.perceptron.weights}, bias={self.perceptron.bias:.4f}")
        
        self.errors = []
        self.ax.cla()
        self.ax.set_title("Error durante entrenamiento")
        self.ax.set_xlabel("Iteración")
        self.ax.set_ylabel("MSE")
        self.line, = self.ax.plot([], [])
        self.canvas.draw()
        self.btn_train.config(state="normal")

    def _training_callback(self, iteration, mse):
        self.errors.append(mse)

    def start_training(self):
        if self.perceptron is None or self.df is None:
            messagebox.showwarning("Atención", "Carga dataset e inicializa el perceptrón")
            return
        if self.train_running:
            return
        self.train_running = True
        self.btn_train.config(state="disabled")
        self.btn_stop.config(state="normal")
        eta = float(self.eta_var.get())
        max_iter = int(self.maxiter_var.get())
        tol = float(self.tol_var.get())
        alg = self.alg_var.get()
        X = self.X_train.copy()
        y = self.y_train.copy()
        
        def train_job():
            try:
                if alg == "perceptron":
                    self.perceptron.train_perceptron_rule(X, y, eta=eta, max_iter=max_iter, tol=tol, callback=self._training_callback)
                else:
                    self.perceptron.train_delta_rule(X, y, eta=eta, max_iter=max_iter, tol=tol, callback=self._training_callback)
            except Exception as e:
                self.log(f"Error en entrenamiento: {e}")
            finally:
                self.train_running = False
                self.after(0, lambda: self.btn_train.config(state="normal"))
                self.after(0, lambda: self.btn_stop.config(state="disabled"))
                
                # Evaluar en datos de entrenamiento y prueba
                train_evals = self.perceptron.evaluate(self.X_train, self.y_train)
                test_evals = self.perceptron.evaluate(self.X_test, self.y_test)
                
                self.after(0, lambda: self.show_evaluation(train_evals, test_evals))
                
        self.errors = []
        self.training_thread = threading.Thread(target=train_job, daemon=True)
        self.training_thread.start()
        self.after(200, self._update_plot)

    def _update_plot(self):
        if len(self.errors) > 0:
            xs = list(range(len(self.errors)))
            ys = self.errors
            self.ax.cla()
            self.ax.set_title("Error durante entrenamiento")
            self.ax.set_xlabel("Iteración")
            self.ax.set_ylabel("MSE")
            self.ax.plot(xs, ys, marker='o', linestyle='-')
            self.canvas.draw()
        if self.train_running:
            self.after(200, self._update_plot)

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Detener", "Detener detendrá después de la iteración actual (implementación básica).")
            self.train_running = False

    def show_evaluation(self, train_evals, test_evals):
        self.eval_text.delete("1.0", "end")
        self.eval_text.insert("end", "=== ENTRENAMIENTO (80%) ===\n")
        self.eval_text.insert("end", f"Accuracy: {train_evals['accuracy']*100:.2f}%\n")
        self.eval_text.insert("end", f"TP: {train_evals['tp']}\nTN: {train_evals['tn']}\nFP: {train_evals['fp']}\nFN: {train_evals['fn']}\n\n")
        
        self.eval_text.insert("end", "=== PRUEBA (20%) ===\n")
        self.eval_text.insert("end", f"Accuracy: {test_evals['accuracy']*100:.2f}%\n")
        self.eval_text.insert("end", f"TP: {test_evals['tp']}\nTN: {test_evals['tn']}\nFP: {test_evals['fp']}\nFN: {test_evals['fn']}\n")
        
        self.log("Entrenamiento finalizado.")
        self.log(f"Accuracy entrenamiento: {train_evals['accuracy']*100:.2f}%")
        self.log(f"Accuracy prueba: {test_evals['accuracy']*100:.2f}%")
        
        # Mostrar mensaje sobre calidad del aprendizaje
        if test_evals['accuracy'] >= 0.9:
            self.log("✓ La máquina ha aprendido bien (accuracy > 90%)")
        elif test_evals['accuracy'] >= 0.7:
            self.log("~ La máquina ha aprendido moderadamente bien (accuracy > 70%)")
        else:
            self.log("✗ La máquina necesita más entrenamiento (accuracy < 70%)")

    def test_selected_row(self):
        if self.perceptron is None or self.test_df is None:
            messagebox.showwarning("Atención", "Primero inicializa y entrena el perceptrón")
            return
            
        idx_text = self.row_combo.get()
        if idx_text == "":
            return
            
        idx = int(idx_text)
        target_col = self.target_combo.get()
        cols = [c for c in self.test_df.columns if c != target_col]
        
        # Obtener la fila del conjunto de prueba
        row = self.test_df.iloc[idx]
        x = row[cols].values.astype(float)
        
        # Normalizar y predecir
        x_in = self.perceptron.normalize(x)
        y_true = int(row[target_col])
        y_pred = int(self.perceptron.predict(x_in.reshape(1,-1))[0])
        
        # Mostrar resultado
        resultado = f"Patrón de prueba #{idx}\n\n"
        resultado += f"Entrada: {x.tolist()}\n"
        resultado += f"Valor esperado: {y_true}\n"
        resultado += f"Valor predicho: {y_pred}\n\n"
        
        if y_true == y_pred:
            resultado += "✓ PREDICCIÓN CORRECTA"
            messagebox.showinfo("Resultado de prueba", resultado)
        else:
            resultado += "✗ PREDICCIÓN INCORRECTA"
            messagebox.showerror("Resultado de prueba", resultado)
            
        self.log(f"Prueba patrón #{idx}: esperado={y_true}, predicho={y_pred}")

    def test_new_pattern(self):
        if self.perceptron is None:
            messagebox.showwarning("Atención", "Inicializa y entrena el perceptrón primero")
            return
            
        txt = self.newpattern_entry.get().strip()
        if not txt:
            return
            
        try:
            vals = [float(x.strip()) for x in txt.split(",")]
        except:
            messagebox.showerror("Error", "Patrón inválido (usar coma como separador)")
            return
            
        if len(vals) != self.perceptron.n_inputs:
            messagebox.showerror("Error", f"El patrón debe tener {self.perceptron.n_inputs} valores")
            return
            
        x = np.array(vals, dtype=float)
        x_in = self.perceptron.normalize(x)
        y_pred = int(self.perceptron.predict(x_in.reshape(1,-1))[0])
        
        messagebox.showinfo("Resultado", f"Patrón: {vals}\nPredicción: {y_pred}")
        self.log(f"Prueba patrón nuevo {vals} => pred={y_pred}")

if __name__ == "__main__":
    app = PerceptronApp()
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    app.mainloop()