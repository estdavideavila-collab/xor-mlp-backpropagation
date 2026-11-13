"""Punto de entrada: abre la ventana o, con --resultados [CARPETA], entrena y exporta."""

import argparse
import sys
from pathlib import Path

CARPETA_RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"


def generar_resultados(carpeta):
    from dataset import nombre_patron
    from entrenamiento import Configuracion, Entrenamiento, backprop_disponible, mensaje_backprop_faltante
    from exportar import exportar_todo

    if not backprop_disponible():
        print(mensaje_backprop_faltante(), file=sys.stderr)
        return 1

    ent = Entrenamiento(Configuracion()).entrenar_completo()
    carpeta = exportar_todo(ent, carpeta)

    print(f"Estado: {ent.estado}")
    print(f"Épocas: {ent.epocas}   Iteraciones: {ent.iteraciones}")
    print(f"MSE inicial: {ent.mse_inicial:.8f}   MSE final: {ent.mse_actual:.8f}")
    for i, y in enumerate(ent.salidas()):
        print(f"  {nombre_patron(i)} -> {y:.6f}")
    print(f"Resultados en: {carpeta}")
    return 0


def abrir_ventana():
    try:  # Texto nítido con escalado de Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    from interfaz import App
    App().mainloop()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Perceptrón multicapa 2-2-1 que aprende el XOR.")
    parser.add_argument("--resultados", nargs="?", const=CARPETA_RESULTADOS, metavar="CARPETA",
                        help="entrena sin abrir la ventana y exporta los resultados (por defecto en resultados/)")
    args = parser.parse_args()
    if args.resultados is not None:
        return generar_resultados(args.resultados)
    return abrir_ventana()


if __name__ == "__main__":
    sys.exit(main())
