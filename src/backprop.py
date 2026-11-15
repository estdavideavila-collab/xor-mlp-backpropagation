"""Retropropagación del error y actualización de pesos y sesgos de la red 2-2-1."""

import numpy as np


def derivada_sigmoide(salida):
    """Derivada de la sigmoide a partir de su salida: s · (1 - s)."""
    return salida * (1 - salida)


def delta_salida(salida_deseada, salida_final):
    """Delta de la neurona de salida: (d - y) · y · (1 - y)."""
    error = salida_deseada - salida_final
    return error * derivada_sigmoide(salida_final)


def delta_oculta(delta_de_salida, pesos_salida, salida_oculta):
    """Delta de cada neurona oculta: h · (1 - h) · Σ (w · δ salida)."""
    error_propagado = np.dot(pesos_salida, delta_de_salida)
    return error_propagado * derivada_sigmoide(salida_oculta)


def gradientes(entrada, salida_oculta, delta_de_salida, delta_de_oculta):
    """Gradientes de pesos y sesgos de cada capa."""
    gradiente_pesos_oculta = np.outer(entrada, delta_de_oculta)
    gradiente_sesgos_oculta = delta_de_oculta
    gradiente_pesos_salida = np.outer(salida_oculta, delta_de_salida)
    gradiente_sesgo_salida = delta_de_salida
    return (gradiente_pesos_oculta, gradiente_sesgos_oculta,
            gradiente_pesos_salida, gradiente_sesgo_salida)


def backpropagation(entrada, salida_deseada, salida_oculta, salida_final,
                    pesos_oculta, sesgos_oculta, pesos_salida, sesgo_salida,
                    tasa_aprendizaje):
    """Calcula los deltas de la salida y de la capa oculta y corrige pesos y sesgos."""
    entrada = np.asarray(entrada, dtype=float).ravel()
    salida_deseada = np.asarray(salida_deseada, dtype=float).ravel()
    salida_oculta = np.asarray(salida_oculta, dtype=float).ravel()
    salida_final = np.asarray(salida_final, dtype=float).ravel()

    delta_de_salida = delta_salida(salida_deseada, salida_final)
    delta_de_oculta = delta_oculta(delta_de_salida, pesos_salida, salida_oculta)

    (gradiente_pesos_oculta, gradiente_sesgos_oculta,
     gradiente_pesos_salida, gradiente_sesgo_salida) = gradientes(
        entrada, salida_oculta, delta_de_salida, delta_de_oculta)

    nuevos_pesos_oculta = pesos_oculta + tasa_aprendizaje * gradiente_pesos_oculta
    nuevos_sesgos_oculta = sesgos_oculta + tasa_aprendizaje * gradiente_sesgos_oculta
    nuevos_pesos_salida = pesos_salida + tasa_aprendizaje * gradiente_pesos_salida
    nuevo_sesgo_salida = sesgo_salida + tasa_aprendizaje * gradiente_sesgo_salida

    return nuevos_pesos_oculta, nuevos_sesgos_oculta, nuevos_pesos_salida, nuevo_sesgo_salida
