"""Curva del MSE, diagrama de la red y regiones de decisión."""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from dataset import ENTRADAS, SALIDAS_DESEADAS
from entrenamiento import propagar

COLOR_POSITIVO = "#1f6fb4"
COLOR_NEGATIVO = "#c8423b"
COLOR_NEURONA = "#f4f1ea"
COLOR_BORDE = "#33302b"
COLOR_TEXTO = "#22201c"
COLOR_CURVA = "#1f6fb4"
COLOR_OBJETIVO = "#c8423b"

POSICIONES = {
    "x1": (0.0, 0.72), "x2": (0.0, 0.28),
    "h1": (1.0, 0.72), "h2": (1.0, 0.28),
    "y": (2.0, 0.50),
}
RADIO = 0.11


def _conexion(ax, origen, destino, peso, fraccion_etiqueta, desplazamiento):
    (x0, y0), (x1, y1) = POSICIONES[origen], POSICIONES[destino]
    color = COLOR_POSITIVO if peso >= 0 else COLOR_NEGATIVO
    grosor = 0.8 + min(abs(peso), 8.0) * 0.45
    ax.add_line(Line2D([x0, x1], [y0, y1], color=color, linewidth=grosor, alpha=0.85, zorder=1))
    tx = x0 + (x1 - x0) * fraccion_etiqueta
    ty = y0 + (y1 - y0) * fraccion_etiqueta + desplazamiento
    ax.text(tx, ty, f"{peso:.4f}", ha="center", va="center", fontsize=8.5, color=color, zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))


def _neurona(ax, nombre, sesgo=None, sesgo_arriba=False):
    x, y = POSICIONES[nombre]
    ax.add_patch(Circle((x, y), RADIO, facecolor=COLOR_NEURONA, edgecolor=COLOR_BORDE, linewidth=1.4, zorder=2))
    ax.text(x, y, nombre, ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_TEXTO, zorder=4)
    if sesgo is not None:
        color = COLOR_POSITIVO if sesgo >= 0 else COLOR_NEGATIVO
        if sesgo_arriba:
            ax.text(x, y + RADIO + 0.03, f"b = {sesgo:.4f}", ha="center", va="bottom", fontsize=8.5, color=color,
                    zorder=4)
        else:
            ax.text(x, y - RADIO - 0.03, f"b = {sesgo:.4f}", ha="center", va="top", fontsize=8.5, color=color,
                    zorder=4)


def dibujar_red(ax, pesos, titulo):
    """Diagrama 2-2-1 con pesos y sesgos."""
    ax.clear()
    W1, b1, W2, b2 = pesos.como_tupla()
    # Cruzadas: etiqueta cerca del origen para que no se encimen
    _conexion(ax, "x1", "h1", W1[0, 0], 0.5, 0.045)
    _conexion(ax, "x1", "h2", W1[0, 1], 0.24, 0.0)
    _conexion(ax, "x2", "h1", W1[1, 0], 0.24, 0.0)
    _conexion(ax, "x2", "h2", W1[1, 1], 0.5, -0.045)
    _conexion(ax, "h1", "y", W2[0, 0], 0.5, 0.045)
    _conexion(ax, "h2", "y", W2[1, 0], 0.5, -0.045)
    _neurona(ax, "x1")
    _neurona(ax, "x2")
    _neurona(ax, "h1", b1[0], sesgo_arriba=True)
    _neurona(ax, "h2", b1[1])
    _neurona(ax, "y", b2[0])
    for x, texto in ((0.0, "Entrada"), (1.0, "Capa oculta\n(sigmoide)"), (2.0, "Salida\n(sigmoide)")):
        ax.text(x, 1.0, texto, ha="center", va="bottom", fontsize=9, color="#6b665c")
    ax.set_title(titulo, fontsize=11, pad=24, color=COLOR_TEXTO)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.05, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def dibujar_error(ax, historial_mse, mse_objetivo, escala_log=True):
    """Curva del MSE por época con la línea del objetivo."""
    ax.clear()
    epocas = range(len(historial_mse))
    ax.plot(epocas, historial_mse, color=COLOR_CURVA, linewidth=1.8, label="MSE por época")
    ax.axhline(mse_objetivo, color=COLOR_OBJETIVO, linestyle="--", linewidth=1.2,
               label=f"Objetivo de parada (MSE ≤ {mse_objetivo:g})")
    ultima = len(historial_mse) - 1
    recuadro = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cfcabf", alpha=0.95)
    ax.plot([0], [historial_mse[0]], "o", color=COLOR_TEXTO, markersize=5)
    ax.annotate(f"MSE inicial = {historial_mse[0]:.6f}", (0, historial_mse[0]),
                textcoords="offset points", xytext=(10, -22), fontsize=8.5, bbox=recuadro)
    if ultima > 0:
        ax.plot([ultima], [historial_mse[-1]], "o", color=COLOR_OBJETIVO, markersize=5)
        ax.annotate(f"Época {ultima}\nMSE = {historial_mse[-1]:.8f}", (ultima, historial_mse[-1]),
                    textcoords="offset points", xytext=(-14, 34), ha="right", fontsize=8.5, bbox=recuadro,
                    arrowprops=dict(arrowstyle="-", color="#8a857b", linewidth=0.8))
    ax.set_yscale("log" if escala_log else "linear")
    ax.set_xlabel("Época")
    ax.set_ylabel("Error cuadrático medio (MSE)" + (" — escala logarítmica" if escala_log else ""))
    ax.set_title("Disminución del error durante el entrenamiento", fontsize=11)
    ax.set_xlim(0, max(ultima, 1))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)


def dibujar_frontera(ax, pesos, titulo):
    """Salida de la red en el plano (x1, x2) con los 4 patrones encima."""
    ax.clear()
    malla = np.linspace(-0.25, 1.25, 151)
    xx, yy = np.meshgrid(malla, malla)
    _, salida = propagar(pesos, np.column_stack([xx.ravel(), yy.ravel()]))
    zz = np.reshape(salida, xx.shape)
    ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.55, vmin=0, vmax=1)
    if zz.min() < 0.5 < zz.max():
        ax.contour(xx, yy, zz, levels=[0.5], colors=COLOR_TEXTO, linewidths=1.6)
    for (x1, x2), deseada in zip(ENTRADAS, SALIDAS_DESEADAS.ravel()):
        uno = deseada == 1
        ax.scatter(x1, x2, s=150, marker="o" if uno else "s", zorder=3, linewidths=1.5, edgecolors="white",
                   c=COLOR_POSITIVO if uno else COLOR_NEGATIVO)
        ax.annotate(f"({int(x1)},{int(x2)}) → {int(deseada)}", (x1, x2), textcoords="offset points",
                    xytext=(0, 13 if x2 == 1 else -19), ha="center", fontsize=9, fontweight="bold",
                    color=COLOR_TEXTO)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=10.5)
