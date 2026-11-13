"""Conjunto de datos del XOR y orden en que los patrones ingresan al entrenamiento."""

import numpy as np

# Tabla de verdad del XOR: la salida es 1 solo cuando las dos entradas son distintas.
ENTRADAS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

SALIDAS_DESEADAS = np.array([
    [0],
    [1],
    [1],
    [0],
], dtype=float)

ORDEN_SECUENCIAL = "Secuencial"
ORDEN_ALEATORIO = "Aleatorio por época"
ORDENES = (ORDEN_SECUENCIAL, ORDEN_ALEATORIO)


def nombre_patron(indice):
    """Texto del patrón, por ejemplo '(0,1)'."""
    x1, x2 = ENTRADAS[indice].astype(int)
    return f"({x1},{x2})"


def orden_de_ingreso(modo, generador):
    """Índices de los patrones en el orden en que entran a la red durante una época.

    Secuencial: siempre (0,0) -> (0,1) -> (1,0) -> (1,1), como en la tabla de verdad.
    Aleatorio por época: los cuatro patrones se barajan al iniciar cada época.
    """
    indices = np.arange(len(ENTRADAS))
    if modo == ORDEN_ALEATORIO:
        generador.shuffle(indices)
    return indices


def describir_orden(modo):
    """Frase corta del orden de ingreso para la interfaz y el informe."""
    if modo == ORDEN_ALEATORIO:
        return "Aleatorio: los 4 patrones se barajan al iniciar cada época"
    return "Secuencial: " + " -> ".join(nombre_patron(i) for i in range(len(ENTRADAS)))
