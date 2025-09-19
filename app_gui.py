# app_gui_modern.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import numpy as np
import os, requests
import json
from io import BytesIO
from perceptron import Perceptron
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModernStyle:
    """Clase para manejar los estilos modernos de la aplicación"""
    
    # Colores del tema moderno
    PRIMARY = "#2E86AB"      # Azul principal
    SECONDARY = "#A23B72"    # Rosa/Morado
    SUCCESS = "#28A745"      # Verde éxito
    WARNING = "#FFC107"      # Amarillo advertencia
    DANGER = "#DC3545"       # Rojo peligro
    DARK = "#2C3E50"         # Gris oscuro
    LIGHT = "#ECF0F1"        # Gris claro
    WHITE = "#FFFFFF"
    
    # Colores de fondo
    BG_PRIMARY = "#F8F9FA"   # Fondo principal
    BG_SECONDARY = "#E9ECEF" # Fondo secundario
    BG_DARK = "#343A40"      # Fondo oscuro
    
    @staticmethod
    def configure_styles():
        """Configura los estilos modernos para ttk"""
        style = ttk.Style()
        
        # Configurar tema base
        style.theme_use('clam')
        
        # Estilo para botones principales
        style.configure('Modern.TButton',
                       background=ModernStyle.PRIMARY,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 8))
        
        style.map('Modern.TButton',
                 background=[('active', '#1e5f7a'),
                           ('pressed', '#1e5f7a')])
        
        # Estilo para botones de éxito
        style.configure('Success.TButton',
                       background=ModernStyle.SUCCESS,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 8))
        
        style.map('Success.TButton',
                 background=[('active', '#218838'),
                           ('pressed', '#218838')])
        
        # Estilo para botones de peligro
        style.configure('Danger.TButton',
                       background=ModernStyle.DANGER,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 8))
        
        style.map('Danger.TButton',
                 background=[('active', '#c82333'),
                           ('pressed', '#c82333')])
        
        # Estilo para marcos con bordes redondeados (simulado)
        style.configure('Card.TFrame',
                       background=ModernStyle.WHITE,
                       relief='flat',
                       borderwidth=1)
        
        # Estilo para etiquetas de título
        style.configure('Title.TLabel',
                       background=ModernStyle.WHITE,
                       foreground=ModernStyle.DARK,
                       font=('Segoe UI', 12, 'bold'))
        
        # Estilo para etiquetas de subtítulo
        style.configure('Subtitle.TLabel',
                       background=ModernStyle.WHITE,
                       foreground=ModernStyle.PRIMARY,
                       font=('Segoe UI', 10, 'bold'))
        
        # Estilo para etiquetas normales
        style.configure('Normal.TLabel',
                       background=ModernStyle.WHITE,
                       foreground=ModernStyle.DARK,
                       font=('Segoe UI', 9))
        
        # Estilo para campos de entrada
        style.configure('Modern.TEntry',
                       relief='flat',
                       borderwidth=1,
                       padding=(10, 5))
        
        # Estilo para combobox
        style.configure('Modern.TCombobox',
                       relief='flat',
                       borderwidth=1,
                       padding=(10, 5))

class PerceptronApp(tk.Tk):

    
    def __init__(self):
        super().__init__()
        self.title("🧠 Perceptrón Simple - Machine Learning Toolkit")
        self.geometry("1400x750")
        self.configure(bg=ModernStyle.BG_PRIMARY)
        
        # Configurar estilos modernos
        ModernStyle.configure_styles()
        
        # Variables de la aplicación
        self.df = None
        self.perceptron = None
        self.errors = []
        self.training_thread = None
        self.train_running = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_df = None
        
        self._create_widgets()

    def _create_widgets(self):
        """Crear la interfaz moderna"""
        
        # Header principal
        self._create_header()
        
        # Contenedor principal
        main_container = ttk.Frame(self, style='Card.TFrame')
        main_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Panel izquierdo (controles)
        self._create_left_panel(main_container)
        
        # Panel central (gráfico)
        self._create_center_panel(main_container)
        
        # Panel derecho (logs y evaluación)
        self._create_right_panel(main_container)

    def _create_header(self):
        """Crear header moderno"""
        header = tk.Frame(self, bg=ModernStyle.PRIMARY, height=70)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)
        
        # Título principal
        title_label = tk.Label(header, 
                              text="🧠 Perceptrón Simple",
                              font=("Segoe UI", 18, "bold"),
                              fg="white",
                              bg=ModernStyle.PRIMARY)
        title_label.pack(side="left", padx=20, pady=15)
        
        # Subtítulo
        subtitle_label = tk.Label(header,
                                
                                 font=("Segoe UI", 10),
                                 fg="#B8E0FF",
                                 bg=ModernStyle.PRIMARY)
        subtitle_label.pack(side="left", padx=(0, 20), pady=20)

    def _create_left_panel(self, parent):
        """Crear panel izquierdo con controles"""
        # Contenedor con scroll
        left_container = ttk.Frame(parent)
        left_container.pack(side="left", fill="y", padx=(0, 5))
        
        canvas = tk.Canvas(left_container, width=320, bg=ModernStyle.WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        
        self.left_frame = tk.Frame(canvas, bg=ModernStyle.WHITE)
        self.left_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- Sección: Carga de Datos ---
        self._create_data_section()
        
        # --- Sección: Configuración ---
        self._create_config_section()
        
        # --- Sección: Entrenamiento ---
        self._create_training_section()
        
        # --- Sección: Pruebas ---
        self._create_testing_section()
        
        # --- Sección: Guardar/Cargar Modelo ---
        self._create_model_io_section()

    def _create_section_header(self, title, icon=""):
        """Crear encabezado de sección estilizado"""
        section_frame = tk.Frame(self.left_frame, bg=ModernStyle.PRIMARY, height=40)
        section_frame.pack(fill="x", pady=(10, 0))
        section_frame.pack_propagate(False)
        
        label = tk.Label(section_frame,
                        text=f"{icon} {title}",
                        font=("Segoe UI", 11, "bold"),
                        fg="white",
                        bg=ModernStyle.PRIMARY)
        label.pack(side="left", padx=15, pady=10)
        
        return section_frame

    def _create_card_frame(self):
        """Crear marco tipo card"""
        card = tk.Frame(self.left_frame, bg=ModernStyle.WHITE, relief="solid", bd=1)
        card.pack(fill="x", padx=10, pady=(0, 10))
        return card

    def _create_data_section(self):
        """Sección de carga de datos"""
        self._create_section_header("Carga de Datos", "📁")
        card = self._create_card_frame()
        
        # Botón cargar archivo local
        btn_load = tk.Button(card,
                           text="📂 Cargar Dataset Local",
                           command=self.load_dataset,
                           bg=ModernStyle.PRIMARY,
                           fg="white",
                           font=("Segoe UI", 9, "bold"),
                           relief="flat",
                           cursor="hand2",
                           pady=8)
        btn_load.pack(fill="x", padx=15, pady=(15, 10))
        
        # Separador
        sep1 = tk.Frame(card, height=1, bg=ModernStyle.BG_SECONDARY)
        sep1.pack(fill="x", padx=15, pady=5)
        
        # URL section
        url_label = tk.Label(card, text="🌐 URL Google Drive / Nube:",
                           font=("Segoe UI", 9, "bold"),
                           bg=ModernStyle.WHITE,
                           fg=ModernStyle.DARK)
        url_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.url_entry = tk.Entry(card,
                                 font=("Segoe UI", 9),
                                 relief="solid",
                                 bd=1,
                                 highlightthickness=0)
        self.url_entry.pack(fill="x", padx=15, pady=5)
        
        # Formato
        format_frame = tk.Frame(card, bg=ModernStyle.WHITE)
        format_frame.pack(fill="x", padx=15, pady=5)
        
        tk.Label(format_frame, text="Formato:",
                font=("Segoe UI", 9),
                bg=ModernStyle.WHITE).pack(side="left")
        
        self.format_combo = ttk.Combobox(format_frame,
                                       values=["csv", "excel", "json"],
                                       state="readonly",
                                       width=10,
                                       style='Modern.TCombobox')
        self.format_combo.pack(side="right")
        self.format_combo.set("csv")
        
        # Botón cargar URL
        btn_url = tk.Button(card,
                          text="🔗 Cargar desde URL",
                          command=self.load_dataset_from_url,
                          bg=ModernStyle.SECONDARY,
                          fg="white",
                          font=("Segoe UI", 9, "bold"),
                          relief="flat",
                          cursor="hand2",
                          pady=8)
        btn_url.pack(fill="x", padx=15, pady=10)
        
        # Info del archivo
        self.lbl_file = tk.Label(card,
                               text="📄 Archivo: (ninguno)",
                               font=("Segoe UI", 9),
                               bg=ModernStyle.WHITE,
                               fg=ModernStyle.DARK,
                               wraplength=280)
        self.lbl_file.pack(anchor="w", padx=15, pady=5)
        
        self.lbl_info = tk.Label(card,
                               text="📊 Entradas: -   Salidas: -   Patrones: -",
                               font=("Segoe UI", 9),
                               bg=ModernStyle.WHITE,
                               fg=ModernStyle.PRIMARY,
                               wraplength=280)
        self.lbl_info.pack(anchor="w", padx=15, pady=(5, 15))

    def _create_config_section(self):
        """Sección de configuración"""
        self._create_section_header("Configuración", "⚙️")
        card = self._create_card_frame()
        
        # Columna objetivo
        tk.Label(card, text="🎯 Columna objetivo (y):",
                font=("Segoe UI", 9, "bold"),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=15, pady=(15, 5))
        
        self.target_combo = ttk.Combobox(card, state="readonly", style='Modern.TCombobox')
        self.target_combo.pack(fill="x", padx=15, pady=5)
        
        # Info división de datos
        info_frame = tk.Frame(card, bg="#E8F4FD", relief="solid", bd=1)
        info_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Label(info_frame, text="📊 División de Datos:",
                font=("Segoe UI", 9, "bold"),
                bg="#E8F4FD").pack(anchor="w", padx=10, pady=(8, 2))
        
        tk.Label(info_frame, text="• 80% para entrenamiento",
                font=("Segoe UI", 8),
                fg=ModernStyle.PRIMARY,
                bg="#E8F4FD").pack(anchor="w", padx=20)
        
        tk.Label(info_frame, text="• 20% para prueba/validación",
                font=("Segoe UI", 8),
                fg=ModernStyle.SUCCESS,
                bg="#E8F4FD").pack(anchor="w", padx=20, pady=(0, 8))
        
        # Normalización
        self.normalize_var = tk.BooleanVar(value=True)
        check_normalize = tk.Checkbutton(card,
                                       text="🔧 Normalizar entradas (min-max)",
                                       variable=self.normalize_var,
                                       font=("Segoe UI", 9),
                                       bg=ModernStyle.WHITE,
                                       fg=ModernStyle.DARK)
        check_normalize.pack(anchor="w", padx=15, pady=10)
        
        # Parámetros de entrenamiento
        params_frame = tk.LabelFrame(card,
                                   text=" 📐 Parámetros de Entrenamiento ",
                                   font=("Segoe UI", 9, "bold"),
                                   bg=ModernStyle.WHITE,
                                   fg=ModernStyle.DARK)
        params_frame.pack(fill="x", padx=15, pady=10)
        
        # Tasa de aprendizaje
        tk.Label(params_frame, text="Tasa de aprendizaje (η):",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.eta_var = tk.DoubleVar(value=0.1)
        eta_entry = tk.Entry(params_frame, textvariable=self.eta_var,
                           font=("Segoe UI", 9), relief="solid", bd=1)
        eta_entry.pack(fill="x", padx=10, pady=2)
        
        # Máximo de iteraciones
        tk.Label(params_frame, text="Máx. iteraciones:",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.maxiter_var = tk.IntVar(value=100)
        maxiter_entry = tk.Entry(params_frame, textvariable=self.maxiter_var,
                               font=("Segoe UI", 9), relief="solid", bd=1)
        maxiter_entry.pack(fill="x", padx=10, pady=2)
        
        # Error máximo
        tk.Label(params_frame, text="Error máximo (ε) [MSE]:",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.tol_var = tk.DoubleVar(value=0.001)
        tol_entry = tk.Entry(params_frame, textvariable=self.tol_var,
                           font=("Segoe UI", 9), relief="solid", bd=1)
        tol_entry.pack(fill="x", padx=10, pady=(2, 10))
        
        # Inicialización de pesos
        weights_frame = tk.LabelFrame(card,
                                    text=" ⚖️ Inicialización Pesos/Bias ",
                                    font=("Segoe UI", 9, "bold"),
                                    bg=ModernStyle.WHITE,
                                    fg=ModernStyle.DARK)
        weights_frame.pack(fill="x", padx=15, pady=10)
        
        tk.Label(weights_frame, text="Pesos (separados por comas):",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.weights_entry = tk.Entry(weights_frame,
                                    font=("Segoe UI", 9),
                                    relief="solid",
                                    bd=1)
        self.weights_entry.pack(fill="x", padx=10, pady=2)
        
        tk.Label(weights_frame, text="(vacío = aleatorio)",
                font=("Segoe UI", 7),
                fg="gray",
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10)
        
        tk.Label(weights_frame, text="Bias / Umbral:",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.bias_entry = tk.Entry(weights_frame,
                                 font=("Segoe UI", 9),
                                 relief="solid",
                                 bd=1)
        self.bias_entry.pack(fill="x", padx=10, pady=(2, 10))
        
        # Algoritmo
        alg_frame = tk.LabelFrame(card,
                                text=" 🤖 Algoritmo ",
                                font=("Segoe UI", 9, "bold"),
                                bg=ModernStyle.WHITE,
                                fg=ModernStyle.DARK)
        alg_frame.pack(fill="x", padx=15, pady=(10, 15))
        
        self.alg_var = tk.StringVar(value="perceptron")
        
        rb1 = tk.Radiobutton(alg_frame,
                           text="Perceptron Rule (clásico)",
                           variable=self.alg_var,
                           value="perceptron",
                           font=("Segoe UI", 8),
                           bg=ModernStyle.WHITE)
        rb1.pack(anchor="w", padx=10, pady=5)
        
        rb2 = tk.Radiobutton(alg_frame,
                           text="Delta Rule (MSE)",
                           variable=self.alg_var,
                           value="delta",
                           font=("Segoe UI", 8),
                           bg=ModernStyle.WHITE)
        rb2.pack(anchor="w", padx=10, pady=(0, 8))

    def _create_training_section(self):
        """Sección de entrenamiento"""
        self._create_section_header("Entrenamiento", "🎯")
        card = self._create_card_frame()
        
        # Botón inicializar
        self.btn_init = tk.Button(card,
                                text="🔄 Inicializar Perceptrón",
                                command=self.initialize_perceptron,
                                bg=ModernStyle.PRIMARY,
                                fg="white",
                                font=("Segoe UI", 10, "bold"),
                                relief="flat",
                                cursor="hand2",
                                pady=10)
        self.btn_init.pack(fill="x", padx=15, pady=15)
        
        # Botones de entrenamiento
        train_frame = tk.Frame(card, bg=ModernStyle.WHITE)
        train_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.btn_train = tk.Button(train_frame,
                                 text="▶️ Iniciar Entrenamiento",
                                 command=self.start_training,
                                 state="disabled",
                                 bg=ModernStyle.SUCCESS,
                                 fg="white",
                                 font=("Segoe UI", 9, "bold"),
                                 relief="flat",
                                 cursor="hand2",
                                 pady=8)
        self.btn_train.pack(fill="x", pady=5)
        
        self.btn_stop = tk.Button(train_frame,
                                text="⏹️ Detener Entrenamiento",
                                command=self.stop_training,
                                state="disabled",
                                bg=ModernStyle.DANGER,
                                fg="white",
                                font=("Segoe UI", 9, "bold"),
                                relief="flat",
                                cursor="hand2",
                                pady=8)
        self.btn_stop.pack(fill="x", pady=5)

    def _create_testing_section(self):
        """Sección de pruebas"""
        self._create_section_header("Simulación y Pruebas", "🧪")
        card = self._create_card_frame()
        
        # Prueba de patrones existentes
        test_frame = tk.LabelFrame(card,
                                 text=" 📊 Probar Patrones de Validación ",
                                 font=("Segoe UI", 9, "bold"),
                                 bg=ModernStyle.WHITE,
                                 fg=ModernStyle.DARK)
        test_frame.pack(fill="x", padx=15, pady=15)
        
        tk.Label(test_frame, text="Seleccionar patrón:",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.row_combo = ttk.Combobox(test_frame, state="readonly", style='Modern.TCombobox')
        self.row_combo.pack(fill="x", padx=10, pady=2)
        
        btn_test_row = tk.Button(test_frame,
                               text="🎯 Probar Patrón Seleccionado",
                               command=self.test_selected_row,
                               bg=ModernStyle.SECONDARY,
                               fg="white",
                               font=("Segoe UI", 8, "bold"),
                               relief="flat",
                               cursor="hand2",
                               pady=6)
        btn_test_row.pack(fill="x", padx=10, pady=(5, 10))
        
        # Prueba de patrones nuevos
        new_frame = tk.LabelFrame(card,
                                text=" ✨ Probar Patrón Personalizado ",
                                font=("Segoe UI", 9, "bold"),
                                bg=ModernStyle.WHITE,
                                fg=ModernStyle.DARK)
        new_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        tk.Label(new_frame, text="Nuevo patrón (separado por comas):",
                font=("Segoe UI", 8),
                bg=ModernStyle.WHITE).pack(anchor="w", padx=10, pady=(8, 2))
        
        self.newpattern_entry = tk.Entry(new_frame,
                                       font=("Segoe UI", 9),
                                       relief="solid",
                                       bd=1)
        self.newpattern_entry.pack(fill="x", padx=10, pady=2)
        
        btn_test_new = tk.Button(new_frame,
                               text="🚀 Probar Patrón Nuevo",
                               command=self.test_new_pattern,
                               bg=ModernStyle.WARNING,
                               fg="white",
                               font=("Segoe UI", 8, "bold"),
                               relief="flat",
                               cursor="hand2",
                               pady=6)
        btn_test_new.pack(fill="x", padx=10, pady=(5, 10))

    def _create_model_io_section(self):
        """Sección para guardar y cargar modelos"""
        self._create_section_header("Guardar/Cargar Modelo", "💾")
        card = self._create_card_frame()
        
        # Botón guardar modelo
        btn_save = tk.Button(card,
                           text="💾 Guardar Pesos y Umbral",
                           command=self.save_model,
                           bg=ModernStyle.PRIMARY,
                           fg="white",
                           font=("Segoe UI", 9, "bold"),
                           relief="flat",
                           cursor="hand2",
                           pady=8)
        btn_save.pack(fill="x", padx=15, pady=(15, 10))
        
        # Botón cargar modelo
        btn_load = tk.Button(card,
                           text="📂 Cargar Pesos y Umbral",
                           command=self.load_model,
                           bg=ModernStyle.SECONDARY,
                           fg="white",
                           font=("Segoe UI", 9, "bold"),
                           relief="flat",
                           cursor="hand2",
                           pady=8)
        btn_load.pack(fill="x", padx=15, pady=(0, 15))

    def _create_center_panel(self, parent):
        """Crear panel central con gráfico"""
        center_frame = tk.Frame(parent, bg=ModernStyle.WHITE, relief="solid", bd=1)
        center_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Header del gráfico
        graph_header = tk.Frame(center_frame, bg=ModernStyle.PRIMARY, height=50)
        graph_header.pack(fill="x")
        graph_header.pack_propagate(False)
        
        tk.Label(graph_header,
                text="📈 Evolución del Error Durante Entrenamiento",
                font=("Segoe UI", 12, "bold"),
                fg="white",
                bg=ModernStyle.PRIMARY).pack(side="left", padx=20, pady=15)
        
        # Gráfico
        graph_container = tk.Frame(center_frame, bg=ModernStyle.WHITE)
        graph_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.fig = Figure(figsize=(8, 6), facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Error MSE vs Iteraciones", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Iteración", fontsize=12)
        self.ax.set_ylabel("Error MSE", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#FAFAFA')
        
        self.line, = self.ax.plot([], [], color=ModernStyle.PRIMARY, linewidth=2, marker='o', markersize=4)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_right_panel(self, parent):
        """Crear panel derecho con logs y evaluación"""
        right_frame = tk.Frame(parent, bg=ModernStyle.WHITE, relief="solid", bd=1, width=450)
        right_frame.pack(side="left", fill="y", padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Header logs
        logs_header = tk.Frame(right_frame, bg=ModernStyle.DARK, height=50)
        logs_header.pack(fill="x")
        logs_header.pack_propagate(False)
        
        tk.Label(logs_header,
                text="📝 Registro de Actividad",
                font=("Segoe UI", 12, "bold"),
                fg="white",
                bg=ModernStyle.DARK).pack(side="left", padx=20, pady=15)
        
        # Área de logs con scrollbar horizontal y vertical
        log_container = tk.Frame(right_frame, bg=ModernStyle.WHITE)
        log_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.log_text = tk.Text(log_container,
                              height=15,
                              font=("Consolas", 9),
                              bg="#F8F9FA",
                              fg=ModernStyle.DARK,
                              relief="solid",
                              bd=1,
                              wrap="word")
        
        # Scrollbars para el área de logs
        log_v_scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        log_h_scrollbar = ttk.Scrollbar(log_container, orient="horizontal", command=self.log_text.xview)
        
        self.log_text.configure(yscrollcommand=log_v_scrollbar.set, xscrollcommand=log_h_scrollbar.set)
        
        # Grid layout para scrollbars
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_v_scrollbar.grid(row=0, column=1, sticky="ns")
        log_h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        log_container.grid_rowconfigure(0, weight=1)
        log_container.grid_columnconfigure(0, weight=1)
        
        # Área de evaluación
        eval_header = tk.Frame(right_frame, bg=ModernStyle.SUCCESS, height=40)
        eval_header.pack(fill="x")
        eval_header.pack_propagate(False)
        
        tk.Label(eval_header,
                text="📊 Evaluación del Modelo",
                font=("Segoe UI", 11, "bold"),
                fg="white",
                bg=ModernStyle.SUCCESS).pack(side="left", padx=20, pady=10)
        
        eval_container = tk.Frame(right_frame, bg=ModernStyle.WHITE)
        eval_container.pack(fill="x", padx=15, pady=15)
        
        self.eval_text = tk.Text(eval_container,
                               height=8,
                               font=("Consolas", 9),
                               bg="#E8F5E8",
                               fg=ModernStyle.DARK,
                               relief="solid",
                               bd=1)
        self.eval_text.pack(fill="x")

    def log(self, msg):
        """Añadir mensaje al log con timestamp"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}\n"
        self.log_text.insert("end", formatted_msg)
        self.log_text.see("end")

    def save_model(self):
        """Guardar pesos y umbral en un archivo JSON"""
        if self.perceptron is None:
            messagebox.showwarning("⚠️ Atención", "Primero inicializa o entrena el perceptrón")
            return
            
        fname = filedialog.asksaveasfilename(
            title="Guardar modelo",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not fname:
            return
            
        try:
            # Crear diccionario con los parámetros del modelo
            model_data = {
                "weights": self.perceptron.weights.tolist(),
                "bias": float(self.perceptron.bias),
                "n_inputs": self.perceptron.n_inputs,
                "normalization_params": {
                    "min": self.perceptron.norm_min.tolist() if hasattr(self.perceptron, 'norm_min') else [],
                    "max": self.perceptron.norm_max.tolist() if hasattr(self.perceptron, 'norm_max') else []
                },
                "algorithm": self.alg_var.get(),
                "normalize": self.normalize_var.get()
            }
            
            # Guardar en archivo JSON
            with open(fname, 'w') as f:
                json.dump(model_data, f, indent=4)
                
            self.log(f"💾 Modelo guardado: {os.path.basename(fname)}")
            messagebox.showinfo("✅ Éxito", f"Modelo guardado exitosamente en:\n{fname}")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"No se pudo guardar el modelo:\n{e}")
            self.log(f"❌ Error al guardar modelo: {e}")

    def load_model(self):
        """Cargar pesos y umbral desde un archivo JSON"""
        fname = filedialog.askopenfilename(
            title="Cargar modelo",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not fname:
            return
            
        try:
            # Leer archivo JSON
            with open(fname, 'r') as f:
                model_data = json.load(f)
                
            # Verificar que tenga los campos necesarios
            if "weights" not in model_data or "bias" not in model_data or "n_inputs" not in model_data:
                raise ValueError("El archivo no contiene un modelo válido")
                
            # Crear nuevo perceptrón
            n_inputs = model_data["n_inputs"]
            self.perceptron = Perceptron(n_inputs)
            
            # Establecer pesos y bias
            weights = np.array(model_data["weights"])
            bias = model_data["bias"]
            
            self.perceptron.initialize_weights(weights=weights, bias=bias, random_init=False)
            
            # Cargar parámetros de normalización si existen
            if "normalization_params" in model_data:
                norm_params = model_data["normalization_params"]
                if "min" in norm_params and "max" in norm_params and norm_params["min"] and norm_params["max"]:
                    self.perceptron.norm_min = np.array(norm_params["min"])
                    self.perceptron.norm_max = np.array(norm_params["max"])
            
            # Establecer algoritmo y normalización si están en el archivo
            if "algorithm" in model_data:
                self.alg_var.set(model_data["algorithm"])
                
            if "normalize" in model_data:
                self.normalize_var.set(model_data["normalize"])
            
            # Actualizar la interfaz
            self.weights_entry.delete(0, tk.END)
            self.weights_entry.insert(0, ", ".join(map(str, weights)))
            
            self.bias_entry.delete(0, tk.END)
            self.bias_entry.insert(0, str(bias))    
            
            self.log(f"📂 Modelo cargado: {os.path.basename(fname)}")
            self.log(f"⚖️ Pesos cargados: {weights}")
            self.log(f"🎯 Bias cargado: {bias}")
            
            # Habilitar botones de prueba
            if self.test_df is not None:
                test_indices = [str(i) for i in range(len(self.test_df))]
                self.row_combo['values'] = test_indices
                if test_indices:
                    self.row_combo.set(test_indices[0])
            
            messagebox.showinfo("✅ Éxito", f"Modelo cargado exitosamente desde:\n{fname}")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"No se pudo cargar el modelo:\n{e}")
            self.log(f"❌ Error al cargar modelo: {e}")

    # Resto de métodos (manteniendo la funcionalidad original)
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
            messagebox.showwarning("⚠️ Atención","Pegue primero una URL de Google Drive o nube")
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
            messagebox.showerror("❌ Error", f"No se pudo leer el archivo desde la URL:\n{e}")
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
            messagebox.showerror("❌ Error", f"No se pudo leer el archivo:\n{e}")
            return
        self._set_dataframe(df, os.path.basename(fname))

    def _set_dataframe(self, df, name):
        self.df = df
        self.lbl_file.config(text=f"📄 Archivo: {name}")
        cols = list(df.columns)
        self.target_combo['values'] = cols
        self.target_combo.set(cols[-1])
        n_inputs = len(cols) - 1
        n_outputs = 1
        n_patterns = len(df)
        self.lbl_info.config(text=f"📊 Entradas: {n_inputs}   Salidas: {n_outputs}   Patrones: {n_patterns}")
        
        # Limpiar el combo de selección de patrones de prueba
        self.row_combo.set('')
        self.row_combo['values'] = []
        
        self.log(f"✅ Dataset cargado: {name} ({len(df)} filas, {len(cols)} columnas)")
        self.btn_train.config(state="disabled")

    def initialize_perceptron(self):
        if self.df is None:
            messagebox.showwarning("⚠️ Atención", "Carga primero un dataset")
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
        n_outputs = 1  # Perceptrón simple tiene 1 salida
        
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
                if len(weights) != n_inputs:
                    messagebox.showerror("❌ Error", f"Debe ingresar {n_inputs} pesos (uno por cada entrada)")
                    return
            except:
                messagebox.showerror("❌ Error", "Pesos deben ser números separados por comas")
                return
            bias = None
            if bias_text:
                try:
                    bias = float(bias_text)
                except:
                    messagebox.showerror("❌ Error", "Bias debe ser número")
                    return
            self.perceptron.initialize_weights(weights=weights, bias=bias, random_init=False)
        else:
            # Inicialización automática: n_inputs pesos (uno por cada entrada)
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
        
        self.log(f"🤖 Perceptrón inicializado - Entradas={n_inputs}, Salidas={n_outputs}")
        self.log(f"📊 Datos divididos: {len(train_df)} entrenamiento, {len(test_df)} prueba")
        self.log(f"⚖️ Pesos iniciales: {self.perceptron.weights}")
        self.log(f"🎯 Bias inicial: {self.perceptron.bias:.4f}")
        
        self.errors = []
        self.ax.cla()
        self.ax.set_title("Error MSE vs Iteraciones", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Iteración", fontsize=12)
        self.ax.set_ylabel("Error MSE", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#FAFAFA')
        self.line, = self.ax.plot([], [], color=ModernStyle.PRIMARY, linewidth=2, marker='o', markersize=4)
        self.canvas.draw()
        self.btn_train.config(state="normal")

    def _training_callback(self, iteration, mse):
        self.errors.append(mse)

    def start_training(self):
        if self.perceptron is None or self.df is None:
            messagebox.showwarning("⚠️ Atención", "Carga dataset e inicializa el perceptrón")
            return
        if self.train_running:
            return
        
        self.train_running = True
        self.btn_train.config(state="disabled", bg="#6c757d")
        self.btn_stop.config(state="normal", bg=ModernStyle.DANGER)
        
        eta = float(self.eta_var.get())
        max_iter = int(self.maxiter_var.get())
        tol = float(self.tol_var.get())
        alg = self.alg_var.get()
        X = self.X_train.copy()
        y = self.y_train.copy()
        
        self.log(f"🚀 Iniciando entrenamiento - Algoritmo: {alg.upper()}")
        self.log(f"📈 Parámetros: η={eta}, max_iter={max_iter}, ε={tol}")
        
        def train_job():
            try:
                if alg == "perceptron":
                    self.perceptron.train_perceptron_rule(X, y, eta=eta, max_iter=max_iter, tol=tol, callback=self._training_callback)
                else:
                    self.perceptron.train_delta_rule(X, y, eta=eta, max_iter=max_iter, tol=tol, callback=self._training_callback)
            except Exception as e:
                self.log(f"❌ Error en entrenamiento: {e}")
            finally:
                self.train_running = False
                self.after(0, lambda: self.btn_train.config(state="normal", bg=ModernStyle.SUCCESS))
                self.after(0, lambda: self.btn_stop.config(state="disabled", bg="#6c757d"))
                
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
            self.ax.set_title("Error MSE vs Iteraciones", fontsize=14, fontweight='bold')
            self.ax.set_xlabel("Iteración", fontsize=12)
            self.ax.set_ylabel("Error MSE", fontsize=12)
            self.ax.grid(True, alpha=0.3)
            self.ax.set_facecolor('#FAFAFA')
            
            # Colores del gradiente según el error
            colors = ['#DC3545' if y > 0.5 else '#FFC107' if y > 0.1 else '#28A745' for y in ys]
            self.ax.plot(xs, ys, color=ModernStyle.PRIMARY, linewidth=2, marker='o', markersize=4)
            self.ax.scatter(xs, ys, c=colors, s=30, zorder=5)
            
            self.canvas.draw()
        if self.train_running:
            self.after(200, self._update_plot)

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("⏹️ Detener", "El entrenamiento se detendrá después de la iteración actual")
            self.train_running = False
            self.log("⏹️ Solicitud de detener entrenamiento recibida")

    def show_evaluation(self, train_evals, test_evals):
        self.eval_text.delete("1.0", "end")
        
        # Formatear evaluación con colores y emojis
        eval_content = "🎓 ENTRENAMIENTO (80%)\n"
        eval_content += "=" * 25 + "\n"
        eval_content += f"📊 Accuracy: {train_evals['accuracy']*100:.2f}%\n"
        eval_content += f"✅ TP (Verdaderos Positivos): {train_evals['tp']}\n"
        eval_content += f"✅ TN (Verdaderos Negativos): {train_evals['tn']}\n"
        eval_content += f"❌ FP (Falsos Positivos): {train_evals['fp']}\n"
        eval_content += f"❌ FN (Falsos Negativos): {train_evals['fn']}\n\n"
        
        eval_content += "🧪 VALIDACIÓN (20%)\n"
        eval_content += "=" * 25 + "\n"
        eval_content += f"📊 Accuracy: {test_evals['accuracy']*100:.2f}%\n"
        eval_content += f"✅ TP: {test_evals['tp']}\n"
        eval_content += f"✅ TN: {test_evals['tn']}\n"
        eval_content += f"❌ FP: {test_evals['fp']}\n"
        eval_content += f"❌ FN: {test_evals['fn']}\n"
        
        self.eval_text.insert("1.0", eval_content)
        
        self.log("🏁 Entrenamiento finalizado")
        self.log(f"📈 Accuracy entrenamiento: {train_evals['accuracy']*100:.2f}%")
        self.log(f"🎯 Accuracy validación: {test_evals['accuracy']*100:.2f}%")
        
        # Mostrar mensaje sobre calidad del aprendizaje con emojis
        if test_evals['accuracy'] >= 0.9:
            self.log("🎉 ¡Excelente! La IA ha aprendido muy bien (accuracy > 90%)")
        elif test_evals['accuracy'] >= 0.7:
            self.log("👍 Bien! La IA ha aprendido moderadamente (accuracy > 70%)")
        else:
            self.log("⚠️ La IA necesita más entrenamiento (accuracy < 70%)")

    def test_selected_row(self):
        if self.perceptron is None or self.test_df is None:
            messagebox.showwarning("⚠️ Atención", "Primero inicializa y entrena el perceptrón")
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
        
        # Mostrar resultado con diseño moderno
        resultado = f"🧪 PRUEBA DE PATRÓN #{idx}\n\n"
        resultado += f"📥 Entrada: {x.tolist()}\n"
        resultado += f"🎯 Esperado: {y_true}\n"
        resultado += f"🤖 Predicho: {y_pred}\n\n"
        
        if y_true == y_pred:
            resultado += "✅ PREDICCIÓN CORRECTA"
            messagebox.showinfo("🎉 Resultado Exitoso", resultado)
            self.log(f"✅ Prueba #{idx}: CORRECTA (esperado={y_true}, predicho={y_pred})")
        else:
            resultado += "❌ PREDICCIÓN INCORRECTA"
            messagebox.showerror("❌ Resultado Fallido", resultado)
            self.log(f"❌ Prueba #{idx}: INCORRECTA (esperado={y_true}, predicho={y_pred})")

    def test_new_pattern(self):
        if self.perceptron is None:
            messagebox.showwarning("⚠️ Atención", "Inicializa y entrena el perceptrón primero")
            return
            
        txt = self.newpattern_entry.get().strip()
        if not txt:
            return
            
        try:
            vals = [float(x.strip()) for x in txt.split(",")]
        except:
            messagebox.showerror("❌ Error", "Patrón inválido (usar coma como separador)")
            return
            
        if len(vals) != self.perceptron.n_inputs:
            messagebox.showerror("❌ Error", f"El patrón debe tener {self.perceptron.n_inputs} valores")
            return
            
        x = np.array(vals, dtype=float)
        x_in = self.perceptron.normalize(x)
        y_pred = int(self.perceptron.predict(x_in.reshape(1,-1))[0])
        
        resultado = f"🚀 PRUEBA DE PATRÓN PERSONALIZADO\n\n"
        resultado += f"📥 Patrón: {vals}\n"
        resultado += f"🤖 Predicción: {y_pred}\n\n"
        resultado += "✨ Análisis completado"
        
        messagebox.showinfo("🎯 Resultado de Predicción", resultado)
        self.log(f"🚀 Patrón personalizado {vals} => predicción={y_pred}")

if __name__ == "__main__":
    app = PerceptronApp()
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    app.mainloop()