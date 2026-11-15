# Resultados del entrenamiento: XOR con perceptrón multicapa 2-2-1

Generado: 2026-11-15 17:12

## Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | 2 entradas, 2 neuronas ocultas, 1 neurona de salida |
| Función de activación | Sigmoide en la capa oculta y en la salida |
| Pesos iniciales | Aleatorios uniformes en [-1, 1], redondeados a 2 decimales, semilla 1 |
| Tasa de aprendizaje (η) | 0.5 |
| Orden de ingreso de los datos | Secuencial: (0,0) -> (0,1) -> (1,0) -> (1,1) |
| Modo de actualización | En línea: pesos y sesgos se corrigen después de cada patrón |
| Criterio de parada | MSE ≤ 0.001 o 50000 épocas, lo que ocurra primero |
| Estado al terminar | Convergió |

## Estado inicial: pesos y sesgos

**W1: pesos_oculta** (filas = entradas, columnas = neuronas ocultas)

| | h1 | h2 |
|---|---|---|
| x1 | 0.02 | 0.90 |
| x2 | -0.71 | 0.90 |

**b1: sesgos_oculta** (sesgos de la capa oculta)

| h1 | h2 |
|---|---|
| -0.38 | -0.15 |

**W2: pesos_salida** (filas = neuronas ocultas, columna = salida)

| | y |
|---|---|
| h1 | 0.66 |
| h2 | -0.18 |

**b2: sesgo_salida** (sesgo de la neurona de salida)

| y |
|---|
| 0.10 |

## Estado final: pesos y sesgos

**W1: pesos_oculta** (filas = entradas, columnas = neuronas ocultas)

| | h1 | h2 |
|---|---|---|
| x1 | 5.985607 | 5.474725 |
| x2 | -5.939778 | -5.195661 |

**b1: sesgos_oculta** (sesgos de la capa oculta)

| h1 | h2 |
|---|---|
| -3.332875 | 2.571202 |

**W2: pesos_salida** (filas = neuronas ocultas, columna = salida)

| | y |
|---|---|
| h1 | 8.330891 |
| h2 | -8.096275 |

**b2: sesgo_salida** (sesgo de la neurona de salida)

| y |
|---|
| 3.815574 |

## Resumen del entrenamiento

| Dato | Valor |
|---|---|
| Tasa de aprendizaje | 0.5 |
| Número de épocas | 3918 |
| Número total de iteraciones | 15672 (= 3918 épocas × 4 patrones) |
| MSE inicial | 0.25331092 |
| MSE final | 0.00099964 |
| Patrones bien clasificados | 4 de 4 |

Una iteración es la presentación de un patrón (forward pass, backpropagation y actualización de
pesos). Una época es la presentación de los 4 patrones. El MSE se mide al terminar cada época con
los pesos de ese momento: MSE = (1/4) · Σ (d − y)².

## Salidas de la red

| Entrada (x1,x2) | Salida deseada | Salida con pesos iniciales | Salida final | Clase (umbral 0.5) | ¿Correcta? |
|---|---|---|---|---|---|
| (0,0) | 0 | 0.570718 | 0.031717 | 0 | Sí |
| (0,1) | 1 | 0.535892 | 0.963357 | 1 | Sí |
| (1,0) | 1 | 0.561926 | 0.970814 | 1 | Sí |
| (1,1) | 0 | 0.529358 | 0.028249 | 0 | Sí |

## Archivos

- `grafica_error.png`: disminución del MSE por época (escala logarítmica).
- `grafica_error_lineal.png`: la misma curva en escala lineal.
- `red_pesos_iniciales.png` y `red_pesos_finales.png`: diseño de la red con sus pesos.
- `frontera_decision.png`: lo que responde la red en todo el plano (x1, x2), antes y después de entrenar.
- `mse_por_epoca.csv`: valor del MSE en cada época (época 0 = antes de entrenar).
