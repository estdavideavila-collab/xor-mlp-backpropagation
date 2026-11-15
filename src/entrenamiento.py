"""Ciclo de entrenamiento de la red 2-2-1: une forward (A) y backprop (B)."""

from dataclasses import dataclass

import numpy as np

from dataset import ENTRADAS, ORDEN_SECUENCIAL, SALIDAS_DESEADAS, orden_de_ingreso
from forward import calcular_mse, forward_pass

try:
    from backprop import backpropagation
    _ERROR_BACKPROP = None
except ImportError as error:  # B aún no sube backpropagation
    backpropagation = None
    _ERROR_BACKPROP = str(error)

N_ENTRADAS = 2
N_OCULTAS = 2
N_SALIDAS = 1

UMBRAL_CLASE = 0.5  # y >= 0.5 → clase 1

SIN_ENTRENAR = "Sin entrenar"
ENTRENANDO = "Entrenando"
CONVERGIO = "Convergió"
LIMITE_EPOCAS = "Detenido por límite de épocas"


class BackpropFaltante(RuntimeError):
    """Se intentó entrenar sin backpropagation."""


def backprop_disponible():
    return backpropagation is not None


def mensaje_backprop_faltante():
    return ("Falta la retropropagación: src/backprop.py debe definir la función backpropagation(...) "
            "(parte de B). Cuando esté en main, actualice esta rama con «git merge main».\n\n"
            f"Detalle: {_ERROR_BACKPROP}")


# --- Configuración y pesos ---

@dataclass(frozen=True)
class Configuracion:
    tasa_aprendizaje: float = 0.5
    mse_objetivo: float = 0.001
    epocas_maximas: int = 50_000
    semilla: int = 1
    orden: str = ORDEN_SECUENCIAL


@dataclass
class Pesos:
    """Pesos y sesgos con los nombres de forward_pass y backpropagation.

    pesos_oculta  W1 (2,2): fila = entrada, columna = neurona oculta
    sesgos_oculta b1 (2,)
    pesos_salida  W2 (2,1): fila = neurona oculta
    sesgo_salida  b2 (1,)
    """

    pesos_oculta: np.ndarray
    sesgos_oculta: np.ndarray
    pesos_salida: np.ndarray
    sesgo_salida: np.ndarray

    FORMAS = ((N_ENTRADAS, N_OCULTAS), (N_OCULTAS,), (N_OCULTAS, N_SALIDAS), (N_SALIDAS,))

    @classmethod
    def desde(cls, valores):
        """Arma Pesos desde 4 arreglos con las formas de FORMAS."""
        return cls(*(np.asarray(v, dtype=float).reshape(forma) for v, forma in zip(valores, cls.FORMAS)))

    def como_tupla(self):
        return self.pesos_oculta, self.sesgos_oculta, self.pesos_salida, self.sesgo_salida

    def copia(self):
        return Pesos(*(arreglo.copy() for arreglo in self.como_tupla()))


def inicializar_pesos(generador, decimales=2):
    """Pesos uniformes en [-1, 1], redondeados para verificarlos a mano."""
    return Pesos.desde([np.round(generador.uniform(-1.0, 1.0, forma), decimales) for forma in Pesos.FORMAS])


# --- Forward y MSE (A) ---

def propagar(pesos, entradas):
    """forward_pass de A; acepta un patrón (2,) o varios (n, 2)."""
    return forward_pass(entradas, *pesos.como_tupla())


def mse_del_dataset(pesos):
    """MSE sobre los 4 patrones: (1/4) · Σ (d - y)²."""
    _, salida_final = propagar(pesos, ENTRADAS)
    return float(calcular_mse(SALIDAS_DESEADAS, salida_final))


# --- Ciclo de entrenamiento ---

class Entrenamiento:
    """Pesos, contadores e historial del MSE de un entrenamiento."""

    def __init__(self, config):
        self.config = config
        # Un generador para pesos y barajado: misma semilla, mismo resultado
        self.generador = np.random.default_rng(config.semilla)
        self.pesos_iniciales = inicializar_pesos(self.generador)
        self.pesos = self.pesos_iniciales.copia()
        self.epocas = 0
        self.iteraciones = 0
        self.mse_inicial = mse_del_dataset(self.pesos)
        self.historial_mse = [self.mse_inicial]  # [0] = antes de entrenar
        self.estado = SIN_ENTRENAR

    @property
    def mse_actual(self):
        return self.historial_mse[-1]

    @property
    def terminado(self):
        return self.estado in (CONVERGIO, LIMITE_EPOCAS)

    def entrenar_patron(self, indice):
        """Una iteración: forward, backprop y actualización con un patrón."""
        if not backprop_disponible():
            raise BackpropFaltante(mensaje_backprop_faltante())
        entrada = ENTRADAS[indice]
        salida_deseada = SALIDAS_DESEADAS[indice]
        salida_oculta, salida_final = propagar(self.pesos, entrada)
        nuevos = backpropagation(entrada, salida_deseada, salida_oculta, salida_final,
                                 *self.pesos.como_tupla(), self.config.tasa_aprendizaje)
        self.pesos = Pesos.desde(nuevos)
        self.iteraciones += 1

    def entrenar_epoca(self):
        """Una época: los 4 patrones y luego el MSE de la red."""
        for indice in orden_de_ingreso(self.config.orden, self.generador):
            self.entrenar_patron(indice)
        self.epocas += 1
        self.historial_mse.append(mse_del_dataset(self.pesos))
        self._revisar_criterio_de_parada()

    def _revisar_criterio_de_parada(self):
        """Criterio de parada: MSE objetivo o límite de épocas."""
        if self.mse_actual <= self.config.mse_objetivo:
            self.estado = CONVERGIO
        elif self.epocas >= self.config.epocas_maximas:
            self.estado = LIMITE_EPOCAS
        else:
            self.estado = ENTRENANDO

    def avanzar(self, max_epocas):
        """Entrena hasta max_epocas más o hasta el criterio de parada."""
        for _ in range(max_epocas):
            if self.terminado:
                break
            self.entrenar_epoca()
        return self.terminado

    def entrenar_completo(self):
        while not self.terminado:
            self.avanzar(1000)
        return self

    def salidas(self, pesos=None):
        """Salida de la red para los 4 patrones, como vector."""
        _, salida_final = propagar(pesos if pesos is not None else self.pesos, ENTRADAS)
        return np.ravel(salida_final)

    def aciertos(self, pesos=None):
        """Patrones bien clasificados con el umbral 0.5."""
        clases = (self.salidas(pesos) >= UMBRAL_CLASE).astype(int)
        return int(np.sum(clases == SALIDAS_DESEADAS.ravel().astype(int)))
