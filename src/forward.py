import numpy as np


def sigmoide(x):
    """
    Función de activación sigmoide.
    Convierte cualquier valor en un rango entre 0 y 1.
    """
    return 1 / (1 + np.exp(-x))


def forward_pass(entrada, pesos_oculta, sesgos_oculta, pesos_salida, sesgo_salida):
    """
    Realiza la propagación hacia adelante de una red neuronal
    con una capa oculta y una neurona de salida.
    """

    # Capa oculta
    z_oculta = np.dot(entrada, pesos_oculta) + sesgos_oculta
    salida_oculta = sigmoide(z_oculta)

    # Capa de salida
    z_salida = np.dot(salida_oculta, pesos_salida) + sesgo_salida
    salida_final = sigmoide(z_salida)

    return salida_oculta, salida_final


def calcular_mse(valor_real, valor_predicho):
    """
    Calcula el Error Cuadrático Medio.
    """
    return np.mean((valor_real - valor_predicho) ** 2)