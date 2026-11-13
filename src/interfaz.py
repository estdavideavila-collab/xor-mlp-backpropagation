"""Interfaz gráfica: configuración, entrenamiento animado, gráficas, validación de salidas y prueba de la red."""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from dataset import ENTRADAS, ORDENES, SALIDAS_DESEADAS, describir_orden
from entrenamiento import (CONVERGIO, LIMITE_EPOCAS, UMBRAL_CLASE, Configuracion, Entrenamiento,
                           backprop_disponible, mensaje_backprop_faltante, propagar)
from exportar import ARCHIVOS, exportar_todo
from graficas import dibujar_error, dibujar_frontera, dibujar_red

VELOCIDADES = ("1", "10", "25", "50", "100", "250", "1000")
VERDE = "#2e7d32"
ROJO = "#c62828"
GRIS = "#6b665c"


def carpeta_base():
    """Carpeta del .exe cuando está empaquetado; raíz del proyecto cuando se corre con Python."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def texto_pesos(pesos, titulo):
    W1, b1, W2, b2 = pesos.como_tupla()
    return "\n".join([
        titulo,
        "=" * len(titulo),
        "W1  pesos_oculta  (filas = entradas, columnas = neuronas ocultas)",
        f"{'':6}{'h1':>14}{'h2':>14}",
        f"{'x1':6}{W1[0, 0]:>14.6f}{W1[0, 1]:>14.6f}",
        f"{'x2':6}{W1[1, 0]:>14.6f}{W1[1, 1]:>14.6f}",
        "",
        "b1  sesgos_oculta  (sesgos de la capa oculta)",
        f"{'':6}{b1[0]:>14.6f}{b1[1]:>14.6f}",
        "",
        "W2  pesos_salida  (filas = neuronas ocultas, columna = salida)",
        f"{'':6}{'y':>14}",
        f"{'h1':6}{W2[0, 0]:>14.6f}",
        f"{'h2':6}{W2[1, 0]:>14.6f}",
        "",
        "b2  sesgo_salida  (sesgo de la neurona de salida)",
        f"{'':6}{b2[0]:>14.6f}",
        "",
    ])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptrón multicapa - XOR")
        self.geometry("1320x820")
        self.minsize(1120, 720)

        self.ent = None
        self.corriendo = False
        self._tarea = None

        defecto = Configuracion()
        self.var_tasa = tk.StringVar(value=f"{defecto.tasa_aprendizaje:g}")
        self.var_objetivo = tk.StringVar(value=f"{defecto.mse_objetivo:g}")
        self.var_maximas = tk.StringVar(value=str(defecto.epocas_maximas))
        self.var_semilla = tk.StringVar(value=str(defecto.semilla))
        self.var_orden = tk.StringVar(value=defecto.orden)
        self.var_velocidad = tk.StringVar(value="50")
        self.var_log = tk.BooleanVar(value=True)
        self.var_x1 = tk.IntVar(value=0)
        self.var_x2 = tk.IntVar(value=1)

        self.estado = {clave: tk.StringVar(value="-") for clave in
                       ("estado", "epoca", "iteraciones", "mse_inicial", "mse_actual")}
        self.desempeno = {clave: tk.StringVar(value="-") for clave in ("mse", "aciertos", "epocas")}
        self.var_orden_texto = tk.StringVar()
        self.var_prueba = tk.StringVar()

        self._construir()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self.inicializar()

    # ------------------------------------------------------------------ construcción de la ventana

    def _construir(self):
        estilo = ttk.Style(self)
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 16, "bold"))
        estilo.configure("Sub.TLabel", font=("Segoe UI", 9), foreground=GRIS)
        estilo.configure("Aviso.TLabel", font=("Segoe UI", 9, "bold"), foreground=ROJO)
        estilo.configure("Valor.TLabel", font=("Segoe UI", 10, "bold"))
        estilo.configure("Accion.TButton", font=("Segoe UI", 10, "bold"))
        estilo.configure("Resultado.TLabel", font=("Segoe UI", 20, "bold"))
        estilo.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        estilo.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        izquierda = ttk.Frame(self, padding=(16, 14, 10, 14))
        izquierda.pack(side="left", fill="y")
        derecha = ttk.Frame(self, padding=(4, 14, 14, 14))
        derecha.pack(side="left", fill="both", expand=True)

        ttk.Label(izquierda, text="Perceptrón multicapa", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(izquierda, text="Problema XOR · red 2-2-1 · backpropagation",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 12))
        if not backprop_disponible():
            ttk.Label(izquierda, style="Aviso.TLabel", wraplength=300, justify="left",
                      text="Falta backpropagation en src/backprop.py (parte de B). Se pueden ver los "
                           "pesos iniciales y probar el forward pass, pero no entrenar.").pack(
                anchor="w", pady=(0, 10))

        self._panel_configuracion(izquierda)
        self._panel_botones(izquierda)
        self._panel_estado(izquierda)
        self._panel_desempeno(izquierda)

        self.pestanas = ttk.Notebook(derecha)
        self.pestanas.pack(fill="both", expand=True)
        self._pestana_red()
        self._pestana_error()
        self._pestana_datos()
        self._pestana_prueba()
        self._pestana_pesos()

    def _panel_configuracion(self, padre):
        marco = ttk.LabelFrame(padre, text="Configuración", padding=10)
        marco.pack(fill="x")
        campos = [
            ("Tasa de aprendizaje (η)", self.var_tasa),
            ("MSE objetivo (parada)", self.var_objetivo),
            ("Épocas máximas", self.var_maximas),
            ("Semilla de los pesos", self.var_semilla),
        ]
        for fila, (texto, variable) in enumerate(campos):
            ttk.Label(marco, text=texto).grid(row=fila, column=0, sticky="w", pady=3)
            ttk.Entry(marco, textvariable=variable, width=14, justify="right").grid(
                row=fila, column=1, sticky="e", pady=3, padx=(10, 0))
        ttk.Label(marco, text="Orden de ingreso").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(marco, textvariable=self.var_orden, values=ORDENES, state="readonly", width=18).grid(
            row=4, column=1, sticky="e", pady=3, padx=(10, 0))
        ttk.Label(marco, text="Épocas por cuadro (velocidad)").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(marco, textvariable=self.var_velocidad, values=VELOCIDADES, state="readonly", width=8).grid(
            row=5, column=1, sticky="e", pady=3, padx=(10, 0))

    def _panel_botones(self, padre):
        marco = ttk.Frame(padre)
        marco.pack(fill="x", pady=12)
        marco.columnconfigure((0, 1), weight=1, uniform="b")
        ttk.Button(marco, text="Inicializar pesos", command=self.inicializar).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.btn_entrenar = ttk.Button(marco, text="▶  Entrenar", style="Accion.TButton",
                                       command=self.alternar_entrenamiento)
        self.btn_entrenar.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        ttk.Button(marco, text="Entrenar 1 época", command=self.una_epoca).grid(
            row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        ttk.Button(marco, text="Exportar resultados", command=self.exportar).grid(
            row=1, column=1, sticky="ew", padx=(4, 0), pady=2)

    def _filas_valores(self, marco, filas):
        etiquetas = {}
        for fila, (texto, variable) in enumerate(filas):
            ttk.Label(marco, text=texto).grid(row=fila, column=0, sticky="w", pady=2)
            etiqueta = ttk.Label(marco, textvariable=variable, style="Valor.TLabel")
            etiqueta.grid(row=fila, column=1, sticky="e", pady=2, padx=(12, 0))
            etiquetas[texto] = etiqueta
        marco.columnconfigure(1, weight=1)
        return etiquetas

    def _panel_estado(self, padre):
        marco = ttk.LabelFrame(padre, text="Estado del entrenamiento", padding=10)
        marco.pack(fill="x")
        etiquetas = self._filas_valores(marco, [
            ("Estado", self.estado["estado"]),
            ("Época", self.estado["epoca"]),
            ("Iteraciones", self.estado["iteraciones"]),
            ("MSE inicial", self.estado["mse_inicial"]),
            ("MSE actual", self.estado["mse_actual"]),
        ])
        self.lbl_estado = etiquetas["Estado"]

    def _panel_desempeno(self, padre):
        marco = ttk.LabelFrame(padre, text="Medidas de desempeño", padding=10)
        marco.pack(fill="x", pady=(12, 0))
        etiquetas = self._filas_valores(marco, [
            ("MSE ≤ objetivo", self.desempeno["mse"]),
            ("Patrones correctos", self.desempeno["aciertos"]),
            ("Épocas usadas", self.desempeno["epocas"]),
        ])
        self.lbl_cumple_mse = etiquetas["MSE ≤ objetivo"]
        self.lbl_aciertos = etiquetas["Patrones correctos"]
        ttk.Label(marco, text="Una salida ≥ 0.5 se toma como 1; menor, como 0.",
                  style="Sub.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _pestana_red(self):
        marco = ttk.Frame(self.pestanas, padding=8)
        self.pestanas.add(marco, text="  Red neuronal  ")
        ttk.Label(marco, text="Azul: peso positivo · Rojo: peso negativo · El grosor crece con la magnitud del peso",
                  style="Sub.TLabel").pack(side="bottom", anchor="w", pady=(4, 0))
        self.fig_red = Figure(figsize=(9, 7.4), dpi=100)
        self.ax_red_inicial, self.ax_red_actual = self.fig_red.subplots(2, 1)
        self.canvas_red = FigureCanvasTkAgg(self.fig_red, master=marco)
        self.canvas_red.get_tk_widget().pack(fill="both", expand=True)

    def _pestana_error(self):
        marco = ttk.Frame(self.pestanas, padding=8)
        self.pestanas.add(marco, text="  Error (MSE)  ")
        ttk.Checkbutton(marco, text="Escala logarítmica", variable=self.var_log,
                        command=self._dibujar_error).pack(anchor="w")
        self.fig_error = Figure(figsize=(11, 5), dpi=100)
        self.ax_error = self.fig_error.add_subplot(111)
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=marco)
        self.canvas_error.get_tk_widget().pack(fill="both", expand=True)

    def _pestana_datos(self):
        marco = ttk.Frame(self.pestanas, padding=14)
        self.pestanas.add(marco, text="  Dataset y salidas  ")
        ttk.Label(marco, text="Dataset: tabla de verdad del XOR (4 patrones)", style="Valor.TLabel").pack(anchor="w")
        ttk.Label(marco, textvariable=self.var_orden_texto, style="Sub.TLabel").pack(anchor="w", pady=(2, 10))
        columnas = ("x1", "x2", "deseada", "inicial", "actual", "clase", "correcta")
        titulos = ("x1", "x2", "Deseada", "Salida (pesos iniciales)", "Salida actual", "Clase (≥ 0.5)",
                   "¿Correcta?")
        self.tabla = ttk.Treeview(marco, columns=columnas, show="headings", height=4)
        for columna, titulo in zip(columnas, titulos):
            self.tabla.heading(columna, text=titulo)
            self.tabla.column(columna, anchor="center", width=190 if columna in ("inicial", "actual") else 95)
        self.tabla.tag_configure("bien", background="#e4f2e4")
        self.tabla.tag_configure("mal", background="#f9e3e1")
        self.tabla.pack(fill="x")
        ttk.Label(marco, style="Sub.TLabel", justify="left", text=(
            "Una iteración es un patrón presentado a la red (forward pass, backpropagation y actualización).\n"
            "Una época son los 4 patrones. Al final de cada época se mide el MSE = (1/4) · Σ (d − y)² con los "
            "pesos de ese momento.")).pack(anchor="w", pady=(10, 0))
        self.fig_frontera = Figure(figsize=(9, 4.2), dpi=100)
        self.ax_frontera_inicial, self.ax_frontera_actual = self.fig_frontera.subplots(1, 2)
        self.canvas_frontera = FigureCanvasTkAgg(self.fig_frontera, master=marco)
        self.canvas_frontera.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        ttk.Label(marco, style="Sub.TLabel", text=(
            "Zona azul: la red responde 1 · Zona roja: responde 0 · Línea negra: frontera y = 0.5. "
            "Una sola recta no puede separar los círculos de los cuadrados.")).pack(anchor="w")

    def _pestana_prueba(self):
        marco = ttk.Frame(self.pestanas, padding=18)
        self.pestanas.add(marco, text="  Probar la red  ")
        entradas = ttk.LabelFrame(marco, text="Entradas", padding=12)
        entradas.pack(side="left", fill="y", anchor="n")
        for fila, (texto, variable) in enumerate((("x1", self.var_x1), ("x2", self.var_x2))):
            ttk.Label(entradas, text=texto, style="Valor.TLabel").grid(row=fila, column=0, padx=(0, 10), pady=6)
            for valor in (0, 1):
                ttk.Radiobutton(entradas, text=str(valor), value=valor, variable=variable,
                                command=self.calcular_prueba).grid(row=fila, column=1 + valor, padx=6)
        ttk.Button(entradas, text="Calcular salida", style="Accion.TButton",
                   command=self.calcular_prueba).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(entradas, text="Usa los pesos actuales de la red.", style="Sub.TLabel").grid(
            row=3, column=0, columnspan=3, pady=(8, 0))

        resultado = ttk.LabelFrame(marco, text="Forward pass", padding=12)
        resultado.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.lbl_prueba = ttk.Label(resultado, style="Resultado.TLabel")
        self.lbl_prueba.pack(anchor="w")
        ttk.Label(resultado, textvariable=self.var_prueba, font=("Consolas", 11), justify="left").pack(
            anchor="w", pady=(10, 0))

    def _pestana_pesos(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="  Pesos  ")
        self.txt_pesos = tk.Text(marco, font=("Consolas", 11), relief="flat", padx=12, pady=10, wrap="none")
        self.txt_pesos.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ acciones

    def _leer_configuracion(self):
        try:
            tasa = float(self.var_tasa.get().replace(",", "."))
            objetivo = float(self.var_objetivo.get().replace(",", "."))
            maximas = int(self.var_maximas.get())
            semilla = int(self.var_semilla.get())
        except ValueError:
            raise ValueError("La tasa y el MSE objetivo deben ser números; las épocas máximas y la semilla, "
                             "números enteros.") from None
        if not 0 < tasa <= 10:
            raise ValueError("La tasa de aprendizaje debe ser mayor que 0 y como máximo 10 (por ejemplo 0.5).")
        if not 0 < objetivo < 1:
            raise ValueError("El MSE objetivo debe ser mayor que 0 y menor que 1 (por ejemplo 0.001).")
        if not 1 <= maximas <= 1_000_000:
            raise ValueError("Las épocas máximas deben estar entre 1 y 1 000 000.")
        if semilla < 0:
            raise ValueError("La semilla debe ser un entero mayor o igual a 0.")
        return Configuracion(tasa, objetivo, maximas, semilla, self.var_orden.get())

    def _configuracion_valida(self):
        try:
            return self._leer_configuracion()
        except ValueError as error:
            messagebox.showerror("Configuración no válida", str(error), parent=self)
            return None

    def inicializar(self):
        """Crea los pesos iniciales con la configuración escrita y deja la red lista para entrenar."""
        self._detener()
        config = self._configuracion_valida()
        if config is None:
            return
        self.ent = Entrenamiento(config)
        self.refrescar()

    def _preparar_entrenamiento(self):
        """Comprueba que la red esté lista y que la configuración escrita sea la que se está usando."""
        if not backprop_disponible():
            messagebox.showerror("Falta la retropropagación", mensaje_backprop_faltante(), parent=self)
            return False
        config = self._configuracion_valida()
        if config is None:
            return False
        if self.ent is None or self.ent.config != config:
            if self.ent is not None and self.ent.epocas > 0 and not messagebox.askyesno(
                    "Configuración cambiada",
                    "La configuración cambió desde que se inicializaron los pesos.\n"
                    "¿Reiniciar el entrenamiento con los nuevos valores?", parent=self):
                return False
            self.ent = Entrenamiento(config)
        if self.ent.terminado:
            messagebox.showinfo("Entrenamiento terminado",
                                "El entrenamiento ya terminó. Pulse «Inicializar pesos» para empezar de nuevo.",
                                parent=self)
            self.refrescar()
            return False
        return True

    def alternar_entrenamiento(self):
        if self.corriendo:
            self._detener()
            return
        if not self._preparar_entrenamiento():
            return
        self.corriendo = True
        self.btn_entrenar.configure(text="⏸  Pausar")
        self._ciclo()

    def _ciclo(self):
        self.ent.avanzar(int(self.var_velocidad.get()))
        self.refrescar()
        if self.ent.terminado:
            self._detener()
            self._avisar_fin()
        else:
            self._tarea = self.after(15, self._ciclo)

    def una_epoca(self):
        self._detener()
        if not self._preparar_entrenamiento():
            return
        self.ent.avanzar(1)
        self.refrescar()
        if self.ent.terminado:
            self._avisar_fin()

    def _detener(self):
        self.corriendo = False
        if self._tarea is not None:
            self.after_cancel(self._tarea)
            self._tarea = None
        if hasattr(self, "btn_entrenar"):
            self.btn_entrenar.configure(text="▶  Entrenar")

    def _avisar_fin(self):
        e = self.ent
        if e.estado == CONVERGIO:
            messagebox.showinfo("Entrenamiento terminado", (
                f"La red alcanzó el objetivo MSE ≤ {e.config.mse_objetivo:g}.\n\n"
                f"Épocas: {e.epocas}\nIteraciones: {e.iteraciones}\n"
                f"MSE final: {e.mse_actual:.8f}\nPatrones correctos: {e.aciertos()} de 4"), parent=self)
        else:
            messagebox.showwarning("Límite de épocas", (
                f"Se agotaron las {e.config.epocas_maximas} épocas sin llegar al objetivo.\n"
                f"MSE final: {e.mse_actual:.8f}\nPatrones correctos: {e.aciertos()} de 4\n\n"
                "La red quedó atrapada en un mínimo local. Pruebe otra semilla o tasa de aprendizaje."),
                parent=self)

    def exportar(self):
        if self.ent is None:
            return
        self._detener()
        if not self.ent.terminado and not messagebox.askyesno(
                "Entrenamiento sin terminar",
                "El entrenamiento no ha terminado. ¿Exportar el estado actual de todas formas?", parent=self):
            return
        sugerida = carpeta_base() / "resultados"
        carpeta = filedialog.askdirectory(parent=self, title="Carpeta donde guardar los resultados",
                                          initialdir=str(sugerida if sugerida.exists() else carpeta_base()))
        if not carpeta:
            return
        try:
            ruta = exportar_todo(self.ent, carpeta)
        except OSError as error:
            messagebox.showerror("No se pudo exportar", str(error), parent=self)
            return
        messagebox.showinfo("Resultados exportados", f"Se guardaron en:\n{ruta}\n\n" + "\n".join(ARCHIVOS),
                            parent=self)

    def calcular_prueba(self):
        if self.ent is None:
            return
        x = np.array([self.var_x1.get(), self.var_x2.get()], dtype=float)
        h, y = propagar(self.ent.pesos, x)
        salida = float(np.ravel(y)[0])
        clase = int(salida >= UMBRAL_CLASE)
        esperada = int(x[0]) ^ int(x[1])
        correcto = clase == esperada
        self.lbl_prueba.configure(text=f"y = {salida:.6f}  →  {clase}", foreground=VERDE if correcto else ROJO)
        self.var_prueba.set("\n".join([
            f"Entrada            x = ({int(x[0])}, {int(x[1])})",
            f"Capa oculta        h = sigmoide(x·W1 + b1) = [{h[0]: .6f}, {h[1]: .6f}]",
            f"Capa de salida     y = sigmoide(h·W2 + b2) = {salida: .6f}",
            "",
            f"Clase obtenida     {clase}   (y {'≥' if clase else '<'} 0.5)",
            f"XOR esperado       {esperada}",
            f"Resultado          {'CORRECTO' if correcto else 'INCORRECTO'}",
            "",
            f"Pesos de la época {self.ent.epocas}.",
        ]))

    # ------------------------------------------------------------------ refresco de la vista

    def refrescar(self):
        e = self.ent
        c = e.config
        self.estado["estado"].set(e.estado)
        self.estado["epoca"].set(f"{e.epocas}")
        self.estado["iteraciones"].set(f"{e.iteraciones}")
        self.estado["mse_inicial"].set(f"{e.mse_inicial:.8f}")
        self.estado["mse_actual"].set(f"{e.mse_actual:.8f}")
        color = VERDE if e.estado == CONVERGIO else ROJO if e.estado == LIMITE_EPOCAS else ""
        self.lbl_estado.configure(foreground=color)

        cumple = e.mse_actual <= c.mse_objetivo
        aciertos = e.aciertos()
        self.desempeno["mse"].set(f"{'Sí' if cumple else 'No'}  ({c.mse_objetivo:g})")
        self.desempeno["aciertos"].set(f"{aciertos} de 4  ({aciertos * 25}%)")
        self.desempeno["epocas"].set(f"{e.epocas} de {c.epocas_maximas}")
        self.lbl_cumple_mse.configure(foreground=VERDE if cumple else ROJO)
        self.lbl_aciertos.configure(foreground=VERDE if aciertos == 4 else ROJO)

        self.var_orden_texto.set(f"Orden de ingreso de los datos: {describir_orden(c.orden)}")
        self._llenar_tabla()
        self._dibujar_red()
        self._dibujar_frontera()
        self._dibujar_error()
        self._mostrar_pesos()
        self.calcular_prueba()

    def _llenar_tabla(self):
        self.tabla.delete(*self.tabla.get_children())
        iniciales = self.ent.salidas(self.ent.pesos_iniciales)
        actuales = self.ent.salidas()
        for i in range(len(ENTRADAS)):
            deseada = int(SALIDAS_DESEADAS[i, 0])
            clase = int(actuales[i] >= UMBRAL_CLASE)
            correcta = clase == deseada
            x1, x2 = ENTRADAS[i].astype(int)
            self.tabla.insert("", "end", tags=("bien" if correcta else "mal",), values=(
                x1, x2, deseada, f"{iniciales[i]:.6f}", f"{actuales[i]:.6f}", clase, "Sí" if correcta else "No"))

    def _dibujar_red(self):
        e = self.ent
        dibujar_red(self.ax_red_inicial, e.pesos_iniciales, "Pesos iniciales")
        titulo = f"Pesos finales (época {e.epocas})" if e.terminado else f"Pesos en la época {e.epocas}"
        dibujar_red(self.ax_red_actual, e.pesos, titulo)
        self.fig_red.tight_layout()
        self.canvas_red.draw_idle()

    def _dibujar_frontera(self):
        e = self.ent
        dibujar_frontera(self.ax_frontera_inicial, e.pesos_iniciales, "Con pesos iniciales")
        titulo = f"Con pesos finales (época {e.epocas})" if e.terminado else f"Época {e.epocas}"
        dibujar_frontera(self.ax_frontera_actual, e.pesos, titulo)
        self.fig_frontera.tight_layout()
        self.canvas_frontera.draw_idle()

    def _dibujar_error(self):
        if self.ent is None:
            return
        dibujar_error(self.ax_error, self.ent.historial_mse, self.ent.config.mse_objetivo, self.var_log.get())
        self.fig_error.tight_layout()
        self.canvas_error.draw_idle()

    def _mostrar_pesos(self):
        e = self.ent
        final = f"PESOS FINALES (época {e.epocas})" if e.terminado else f"PESOS ACTUALES (época {e.epocas})"
        self.txt_pesos.configure(state="normal")
        self.txt_pesos.delete("1.0", "end")
        self.txt_pesos.insert("1.0", texto_pesos(e.pesos_iniciales, "PESOS INICIALES") + "\n"
                              + texto_pesos(e.pesos, final))
        self.txt_pesos.configure(state="disabled")

    def _cerrar(self):
        self._detener()
        self.quit()
        self.destroy()
