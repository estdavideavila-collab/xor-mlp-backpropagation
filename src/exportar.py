"""Exporta los resultados de un entrenamiento: tablas en Markdown, CSV del error e imágenes."""

import csv
from datetime import datetime
from pathlib import Path

from matplotlib.figure import Figure

from dataset import ENTRADAS, SALIDAS_DESEADAS, describir_orden, nombre_patron
from entrenamiento import UMBRAL_CLASE
from graficas import dibujar_error, dibujar_frontera, dibujar_red

DEC_PESO = 6
DEC_MSE = 8

ARCHIVOS = (
    "resultados.md", "mse_por_epoca.csv", "grafica_error.png", "grafica_error_lineal.png",
    "red_pesos_iniciales.png", "red_pesos_finales.png", "frontera_decision.png",
)


def _f(valor, decimales=DEC_PESO):
    return f"{valor:.{decimales}f}"


def _tablas_pesos(pesos, decimales=DEC_PESO):
    W1, b1, W2, b2 = pesos.como_tupla()

    def f(valor):
        return _f(valor, decimales)

    return "\n".join([
        "**W1: pesos_oculta** (filas = entradas, columnas = neuronas ocultas)",
        "",
        "| | h1 | h2 |",
        "|---|---|---|",
        f"| x1 | {f(W1[0, 0])} | {f(W1[0, 1])} |",
        f"| x2 | {f(W1[1, 0])} | {f(W1[1, 1])} |",
        "",
        "**b1: sesgos_oculta** (sesgos de la capa oculta)",
        "",
        "| h1 | h2 |",
        "|---|---|",
        f"| {f(b1[0])} | {f(b1[1])} |",
        "",
        "**W2: pesos_salida** (filas = neuronas ocultas, columna = salida)",
        "",
        "| | y |",
        "|---|---|",
        f"| h1 | {f(W2[0, 0])} |",
        f"| h2 | {f(W2[1, 0])} |",
        "",
        "**b2: sesgo_salida** (sesgo de la neurona de salida)",
        "",
        "| y |",
        "|---|",
        f"| {f(b2[0])} |",
    ])


def _tabla_salidas(ent):
    iniciales = ent.salidas(ent.pesos_iniciales)
    finales = ent.salidas()
    filas = [
        "| Entrada (x1,x2) | Salida deseada | Salida con pesos iniciales | Salida final | Clase (umbral 0.5) | ¿Correcta? |",
        "|---|---|---|---|---|---|",
    ]
    for i in range(len(ENTRADAS)):
        deseada = int(SALIDAS_DESEADAS[i, 0])
        clase = int(finales[i] >= UMBRAL_CLASE)
        filas.append(
            f"| {nombre_patron(i)} | {deseada} | {_f(iniciales[i])} | {_f(finales[i])} | {clase} | "
            f"{'Sí' if clase == deseada else 'No'} |"
        )
    return "\n".join(filas)


def texto_resultados(ent):
    c = ent.config
    return f"""# Resultados del entrenamiento: XOR con perceptrón multicapa 2-2-1

Generado: {datetime.now():%Y-%m-%d %H:%M}

## Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | 2 entradas, 2 neuronas ocultas, 1 neurona de salida |
| Función de activación | Sigmoide en la capa oculta y en la salida |
| Pesos iniciales | Aleatorios uniformes en [-1, 1], redondeados a 2 decimales, semilla {c.semilla} |
| Tasa de aprendizaje (η) | {c.tasa_aprendizaje:g} |
| Orden de ingreso de los datos | {describir_orden(c.orden)} |
| Modo de actualización | En línea: pesos y sesgos se corrigen después de cada patrón |
| Criterio de parada | MSE ≤ {c.mse_objetivo:g} o {c.epocas_maximas} épocas, lo que ocurra primero |
| Estado al terminar | {ent.estado} |

## Estado inicial: pesos y sesgos

{_tablas_pesos(ent.pesos_iniciales, decimales=2)}

## Estado final: pesos y sesgos

{_tablas_pesos(ent.pesos)}

## Resumen del entrenamiento

| Dato | Valor |
|---|---|
| Tasa de aprendizaje | {c.tasa_aprendizaje:g} |
| Número de épocas | {ent.epocas} |
| Número total de iteraciones | {ent.iteraciones} (= {ent.epocas} épocas × 4 patrones) |
| MSE inicial | {_f(ent.mse_inicial, DEC_MSE)} |
| MSE final | {_f(ent.mse_actual, DEC_MSE)} |
| Patrones bien clasificados | {ent.aciertos()} de 4 |

Una iteración es la presentación de un patrón (forward pass, backpropagation y actualización de
pesos). Una época es la presentación de los 4 patrones. El MSE se mide al terminar cada época con
los pesos de ese momento: MSE = (1/4) · Σ (d − y)².

## Salidas de la red

{_tabla_salidas(ent)}

## Archivos

- `grafica_error.png`: disminución del MSE por época (escala logarítmica).
- `grafica_error_lineal.png`: la misma curva en escala lineal.
- `red_pesos_iniciales.png` y `red_pesos_finales.png`: diseño de la red con sus pesos.
- `frontera_decision.png`: lo que responde la red en todo el plano (x1, x2), antes y después de entrenar.
- `mse_por_epoca.csv`: valor del MSE en cada época (época 0 = antes de entrenar).
"""


def _guardar_figura(dibujar, ruta, tamano):
    fig = Figure(figsize=tamano, dpi=150)
    ax = fig.add_subplot(111)
    dibujar(ax)
    fig.tight_layout()
    fig.savefig(ruta, facecolor="white")


def exportar_todo(ent, carpeta):
    """Escribe todos los archivos de resultados en la carpeta y devuelve su ruta."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    (carpeta / "resultados.md").write_text(texto_resultados(ent), encoding="utf-8")

    with open(carpeta / "mse_por_epoca.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["epoca", "mse"])
        for epoca, mse in enumerate(ent.historial_mse):
            escritor.writerow([epoca, f"{mse:.10f}"])

    objetivo = ent.config.mse_objetivo
    _guardar_figura(lambda ax: dibujar_error(ax, ent.historial_mse, objetivo, escala_log=True),
                    carpeta / "grafica_error.png", (8, 4.5))
    _guardar_figura(lambda ax: dibujar_error(ax, ent.historial_mse, objetivo, escala_log=False),
                    carpeta / "grafica_error_lineal.png", (8, 4.5))
    _guardar_figura(lambda ax: dibujar_red(ax, ent.pesos_iniciales, "Red con pesos iniciales"),
                    carpeta / "red_pesos_iniciales.png", (7, 3.9))
    _guardar_figura(lambda ax: dibujar_red(ax, ent.pesos, f"Red con pesos finales (época {ent.epocas})"),
                    carpeta / "red_pesos_finales.png", (7, 3.9))

    fig = Figure(figsize=(9, 4.4), dpi=150)
    ax_inicial, ax_final = fig.subplots(1, 2)
    dibujar_frontera(ax_inicial, ent.pesos_iniciales, "Con pesos iniciales")
    dibujar_frontera(ax_final, ent.pesos, f"Con pesos finales (época {ent.epocas})")
    fig.suptitle("Regiones de decisión: azul = la red responde 1 · rojo = responde 0 · línea negra: y = 0.5",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(carpeta / "frontera_decision.png", facecolor="white")
    return carpeta
