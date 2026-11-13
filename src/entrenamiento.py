"""Ciclo de entrenamiento del perceptrón multicapa 2-2-1 y criterio de parada.

Integra las dos partes del algoritmo:
  - forward.py  (A): forward_pass y calcular_mse
  - backprop.py (B): backpropagation, que corrige pesos y sesgos

Una iteración = un patrón presentado a la red (forward pass + backpropagation + actualización).
Una época     = los 4 patrones del dataset presentados una vez (4 iteraciones).
"""

from dataclasses import dataclass

import numpy as np

from dataset import ENTRADAS, ORDEN_SECUENCIAL, SALIDAS_DESEADAS, orden_de_ingreso
from forward import calcular_mse, forward_pass

try:
    from backprop import backpropagation
    _ERROR_BACKPROP = None
except ImportError as error:  # backprop.py todavía no tiene la función de B
    backpropagation = None
    _ERROR_BACKPROP = str(error)

N_ENTRADAS = 2
N_OCULTAS = 2
N_SALIDAS = 1

UMBRAL_CLASE = 0.5  # una salida >= 0.5 se interpreta como 1; menor, como 0

SIN_ENTRENAR = "Sin entrenar"
ENTRENANDO = "Entrenando"
CONVERGIO = "Convergió"
LIMITE_EPOCAS = "Detenido por límite de épocas"


class BackpropFaltante(RuntimeError):
    """Se lanza al intentar entrenar sin la función backpropagation de backprop.py."""


def backprop_disponible():
    return backpropagation is not None


def mensaje_backprop_faltante():
    return ("Falta la retropropagación: src/backprop.py debe definir la función backpropagation(...) "
            "(parte de B). Cuando esté en main, actualice esta rama con «git merge main».\n\n"
            f"Detalle: {_ERROR_BACKPROP}")


@dataclass(frozen=True)
class Configuracion:
    tasa_aprendizaje: float = 0.5
    mse_objetivo: float = 0.001
    epocas_maximas: int = 50_000
    semilla: int = 1
    orden: str = ORDEN_SECUENCIAL


@dataclass
class Pesos:
    """Pesos y sesgos de la red, con los mismos nombres que usan forward_pass y backpropagation.

    pesos_oculta  (W1, 2x2): fila i = entrada x(i+1), columna j = neurona oculta h(j+1)
    sesgos_oculta (b1, 2):   sesgo de cada neurona oculta
    pesos_salida  (W2, 2x1): fila j = neurona oculta h(j+1) hacia la salida y
    sesgo_salida  (b2, 1):   sesgo de la neurona de salida
    """

    pesos_oculta: np.ndarray
    sesgos_oculta: np.ndarray
    pesos_salida: np.ndarray
    sesgo_salida: np.ndarray

    FORMAS = ((N_ENTRADAS, N_OCULTAS), (N_OCULTAS,), (N_OCULTAS, N_SALIDAS), (N_SALIDAS,))

    @classmethod
    def desde(cls, valores):
        """Arma los pesos a partir de 4 arreglos, llevándolos a la forma que espera la red."""
        return cls(*(np.asarray(v, dtype=float).reshape(forma) for v, forma in zip(valores, cls.FORMAS)))

    def como_tupla(self):
        return self.pesos_oculta, self.sesgos_oculta, self.pesos_salida, self.sesgo_salida

    def copia(self):
        return Pesos(*(arreglo.copy() for arreglo in self.como_tupla()))


def inicializar_pesos(generador, decimales=2):
    """Pesos y sesgos aleatorios uniformes en [-1, 1].

    Se redondean a pocos decimales para que el estado inicial se pueda escribir
    en el informe y verificar a mano el primer forward pass.
    """
    return Pesos.desde([np.round(generador.uniform(-1.0, 1.0, forma), decimales) for forma in Pesos.FORMAS])


def propagar(pesos, entradas):
    """forward_pass de A con estos pesos. Devuelve (salida_oculta, salida_final).

    Sirve para un patrón (forma (2,)) o para varios a la vez (forma (n, 2)).
    """
    return forward_pass(entradas, *pesos.como_tupla())


def mse_del_dataset(pesos):
    """Error cuadrático medio de la red sobre los 4 patrones: (1/4) · Σ (d - y)²."""
    _, salida_final = propagar(pesos, ENTRADAS)
    return float(calcular_mse(SALIDAS_DESEADAS, salida_final))


class Entrenamiento:
    """Estado completo de un entrenamiento: pesos, contadores e historial del error."""

    def __init__(self, config):
        self.config = config
        # Un solo generador con semilla fija: primero crea los pesos iniciales y luego,
        # si el orden es aleatorio, baraja los patrones de cada época. Misma semilla = mismo resultado.
        self.generador = np.random.default_rng(config.semilla)
        self.pesos_iniciales = inicializar_pesos(self.generador)
        self.pesos = self.pesos_iniciales.copia()
        self.epocas = 0
        self.iteraciones = 0
        self.mse_inicial = mse_del_dataset(self.pesos)
        self.historial_mse = [self.mse_inicial]  # posición 0 = antes de entrenar
        self.estado = SIN_ENTRENAR

    @property
    def mse_actual(self):
        return self.historial_mse[-1]

    @property
    def terminado(self):
        return self.estado in (CONVERGIO, LIMITE_EPOCAS)

    def entrenar_patron(self, indice):
        """Una iteración: el patrón entra, se calcula el error y se corrigen pesos y sesgos."""
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
        """Una época: los 4 patrones en el orden de ingreso; luego se mide el MSE de toda la red."""
        for indice in orden_de_ingreso(self.config.orden, self.generador):
            self.entrenar_patron(indice)
        self.epocas += 1
        self.historial_mse.append(mse_del_dataset(self.pesos))
        self._revisar_criterio_de_parada()

    def _revisar_criterio_de_parada(self):
        """Se detiene cuando el MSE llega al objetivo o cuando se agotan las épocas permitidas."""
        if self.mse_actual <= self.config.mse_objetivo:
            self.estado = CONVERGIO
        elif self.epocas >= self.config.epocas_maximas:
            self.estado = LIMITE_EPOCAS
        else:
            self.estado = ENTRENANDO

    def avanzar(self, max_epocas):
        """Entrena hasta max_epocas épocas más o hasta que se cumpla el criterio de parada."""
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
        """Salida de la red para los 4 patrones del dataset (vector de 4 valores)."""
        _, salida_final = propagar(pesos if pesos is not None else self.pesos, ENTRADAS)
        return np.ravel(salida_final)

    def aciertos(self, pesos=None):
        """Cuántos de los 4 patrones quedan bien clasificados con el umbral 0.5."""
        clases = (self.salidas(pesos) >= UMBRAL_CLASE).astype(int)
        return int(np.sum(clases == SALIDAS_DESEADAS.ravel().astype(int)))
