# Parte C: dataset, entrenamiento, interfaz y ejecutable

Rama: `integrante-c-app-entrenamiento`

## Autores

| Integrante | Nombre | Código |
|---|---|---|
| A | _(completar)_ | _(completar)_ |
| B | _(completar)_ | _(completar)_ |
| C | _(completar)_ | _(completar)_ |

Universidad Militar Nueva Granada · Ingeniería en Multimedia · Inteligencia Artificial · 2026-2

---

## Código de la parte C

| Archivo | Qué hace |
|---|---|
| `src/dataset.py` | Tabla de verdad del XOR y orden de ingreso de los patrones |
| `src/entrenamiento.py` | Pesos iniciales, ciclo de entrenamiento por épocas e iteraciones y criterio de parada. Une el `forward_pass` de A con el `backpropagation` de B |
| `src/graficas.py` | Gráfica del error, diagrama de la red con sus pesos y regiones de decisión |
| `src/exportar.py` | Escribe los resultados para el informe (tablas, CSV del error e imágenes) |
| `src/interfaz.py` | Ventana de la aplicación |
| `src/main.py` | Punto de entrada: abre la ventana o genera los resultados |
| `construir_exe.bat` | Construye el ejecutable de Windows |

Todo el aprendizaje está programado a mano. NumPy solo se usa para operar matrices y matplotlib para dibujar las gráficas; no se usa ninguna librería de redes neuronales.

### Ejecutar

```
pip install -r requirements.txt
python src/main.py
```

### Generar los resultados para A y B

```
python src/main.py --resultados
```

Entrena con la configuración por defecto y guarda en `resultados/`:

| Lo que se pasa a A y B | Archivo |
|---|---|
| Pesos y sesgos iniciales y finales | `resultados.md` (secciones «Estado inicial» y «Estado final») |
| Número de épocas e iteraciones | `resultados.md` (sección «Resumen del entrenamiento») |
| Error inicial y final (MSE) | `resultados.md` |
| Gráfica del error | `grafica_error.png` (escala logarítmica) y `grafica_error_lineal.png` |
| Diseño de la red con pesos iniciales y finales | `red_pesos_iniciales.png` y `red_pesos_finales.png` |
| Regiones de decisión antes y después de entrenar | `frontera_decision.png` |
| MSE de cada época | `mse_por_epoca.csv` |

El botón «Exportar resultados» de la app genera los mismos archivos.

### Construir el ejecutable

```
pip install -r requirements.txt pyinstaller
construir_exe.bat
```

Queda en `dist/XOR_PerceptronMulticapa.exe`: un solo archivo de unos 38 MB que incluye Python y las librerías, así que no hay que instalar nada para abrirlo. La carpeta `dist/` está en `.gitignore` para no subir el ejecutable al repositorio.

### Integración con A y B

`entrenamiento.py` usa las funciones de A tal como están en `src/forward.py`:

```python
salida_oculta, salida_final = forward_pass(entrada, pesos_oculta, sesgos_oculta, pesos_salida, sesgo_salida)
mse = calcular_mse(valor_real, valor_predicho)
```

Y espera que B defina en `src/backprop.py` esta función, que se llama una vez por cada patrón:

```python
def backpropagation(entrada, salida_deseada, salida_oculta, salida_final,
                    pesos_oculta, sesgos_oculta, pesos_salida, sesgo_salida,
                    tasa_aprendizaje):
    """Calcula los deltas de la salida y de la capa oculta y corrige pesos y sesgos."""
    ...
    return pesos_oculta, sesgos_oculta, pesos_salida, sesgo_salida   # ya actualizados
```

| Parámetro | Forma | Contenido |
|---|---|---|
| `entrada` | (2,) | x1, x2 del patrón |
| `salida_deseada` | (1,) | salida esperada del XOR para ese patrón |
| `salida_oculta` | (2,) | h1, h2 que devolvió `forward_pass` |
| `salida_final` | (1,) | y que devolvió `forward_pass` |
| `pesos_oculta` | (2, 2) | W1: fila = entrada, columna = neurona oculta |
| `sesgos_oculta` | (2,) | b1 |
| `pesos_salida` | (2, 1) | W2: fila = neurona oculta |
| `sesgo_salida` | (1,) | b2 |
| `tasa_aprendizaje` | número | η |

Mientras `backprop.py` no tenga esa función, la app abre, muestra los pesos iniciales y deja probar el forward pass, pero avisa que no puede entrenar. Cuando B la suba a `main`, se actualiza esta rama con `git merge main`.

---

## 4. Agente

### a. Data set: conjunto de datos a probar

El conjunto de datos es la tabla de verdad completa de la función XOR: cuatro patrones con dos entradas binarias y una salida deseada. La salida vale 1 solo cuando las dos entradas son distintas. Los mismos cuatro patrones se usan para entrenar y para comprobar la red, porque el XOR no tiene más casos posibles.

| Patrón | x1 | x2 | Salida deseada (d) |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 2 | 0 | 1 | 1 |
| 3 | 1 | 0 | 1 |
| 4 | 1 | 1 | 0 |

### b. Orden de ingreso de los datos al proceso de entrenamiento

Los patrones ingresan en orden secuencial, siguiendo la tabla de verdad: (0,0) → (0,1) → (1,0) → (1,1). El entrenamiento es en línea: después de cada patrón se hace el forward pass, se calcula el error, se retropropaga y se corrigen pesos y sesgos antes de presentar el siguiente patrón.

- **Iteración:** presentación de un patrón (forward pass + backpropagation + actualización).
- **Época:** presentación de los 4 patrones, es decir, 4 iteraciones.

Se eligió el orden secuencial porque, junto con la semilla fija de los pesos iniciales, hace que el entrenamiento sea reproducible: cada ejecución da los mismos pesos, épocas e iteraciones, y los valores del informe se pueden verificar. La app también permite el orden aleatorio por época, en el que los 4 patrones se barajan al iniciar cada época.

### c. Error estimado para establecer el momento de parada (medidas de desempeño)

Al terminar cada época se calcula el error cuadrático medio de la red sobre los 4 patrones:

MSE = (1/4) · Σ (d − y)²

El entrenamiento se detiene con el primero de estos dos criterios:

1. **MSE ≤ 0.001.** Con este valor, el error promedio por patrón es cercano a √0.001 ≈ 0.03, es decir, cada salida queda a unas tres centésimas de su valor deseado y lejos del umbral de decisión 0.5.
2. **50 000 épocas.** Es un límite de seguridad para el caso en que la red quede atrapada en un mínimo local y el MSE deje de bajar.

| Medida de desempeño | Qué indica | Valor buscado |
|---|---|---|
| MSE final | Qué tan cerca están las salidas de los valores deseados | ≤ 0.001 |
| Patrones correctos | Salidas que quedan del lado correcto del umbral (y ≥ 0.5 se toma como 1) | 4 de 4 (100 %) |
| Épocas | Cuántas veces se recorrió el dataset hasta parar | Las menos posibles |
| Iteraciones | Cuántas correcciones de pesos se hicieron (épocas × 4) | Las menos posibles |

### d. Tipo de agente

Durante el entrenamiento la red es un **agente que aprende** en un esquema de aprendizaje supervisado:

| Componente del agente que aprende | En la red XOR |
|---|---|
| Elemento de desempeño | La red 2-2-1 con sus pesos actuales, que produce la salida y con el forward pass |
| Crítico | La comparación de y con la salida deseada d, medida con el error (d − y) y el MSE |
| Elemento de aprendizaje | La retropropagación del error, que corrige pesos y sesgos con la regla delta generalizada |
| Generador de problemas | El dataset y el orden de ingreso, que deciden qué patrón se presenta en cada iteración |

Una vez entrenada, la red se comporta como un **agente reactivo simple**: percibe las dos entradas y responde con una función fija (los pesos finales), sin memoria de las entradas anteriores ni planificación.

---

## 5. Tipo de entorno

### a. Explicación

El entorno de la red es el problema XOR: los cuatro patrones de entrada y sus salidas deseadas. La red percibe las dos entradas completas de cada patrón, por lo que el entorno es totalmente observable. Es determinista porque la misma entrada siempre tiene la misma salida correcta, y es estático porque el dataset no cambia mientras la red entrena ni mientras responde. Es episódico: cada patrón se clasifica por separado y la respuesta a uno no depende de la respuesta al anterior. Es discreto, porque solo hay cuatro entradas posibles con valores 0 o 1 y dos clases de salida. Hay un solo agente y el entorno es conocido, ya que las salidas correctas se conocen de antemano y se usan para supervisar el aprendizaje.

### b. Tabla relacionada con la explicación

| Propiedad | Clasificación | Justificación en el XOR |
|---|---|---|
| Observabilidad | Totalmente observable | La red recibe x1 y x2 completos en cada patrón |
| Número de agentes | Agente único | Solo la red actúa sobre el problema |
| Determinismo | Determinista | Cada entrada tiene una única salida correcta, fijada por la tabla de verdad |
| Episodios | Episódico | Clasificar un patrón no depende de lo que se respondió en el patrón anterior |
| Dinámica | Estático | Los 4 patrones no cambian durante el entrenamiento ni durante la prueba |
| Estados y acciones | Discreto | 4 entradas posibles (valores 0 y 1) y 2 clases de salida |
| Conocimiento | Conocido | Las salidas deseadas se conocen y guían el aprendizaje supervisado |

---

## 8. Conclusiones

> Los valores de esta sección salen de la configuración por defecto (red 2-2-1, sigmoide, η = 0.5, semilla 1, orden secuencial, parada en MSE ≤ 0.001). Hay que confirmarlos con `python src/main.py --resultados` cuando el backpropagation de B esté integrado.

1. **El perceptrón multicapa resuelve el XOR, que no es linealmente separable.** Con una capa oculta de dos neuronas sigmoides, la red pasó de clasificar 2 de 4 patrones con los pesos iniciales (MSE = 0.2533) a clasificar los 4 correctamente, con un MSE final de 0.00099964 en 3918 épocas (15 672 iteraciones).

2. **La capa oculta transforma el espacio de entrada.** Con los pesos finales, h1 solo se activa con la entrada (1,0) y h2 solo se apaga con la entrada (0,1). Cada neurona oculta traza una recta en el plano (x1, x2), y la neurona de salida combina esas dos rectas. Así separa las dos clases, cosa que una sola recta no puede hacer, y la frontera de decisión deja de ser lineal (`frontera_decision.png`).

3. **El error no baja de forma uniforme.** Durante las primeras ~500 épocas el MSE se mantiene cerca de 0.25, el valor que se obtiene cuando la red responde alrededor de 0.5 en todos los patrones. Luego cae rápido: la red clasifica bien los 4 patrones desde la época 978, el MSE baja de 0.1 en la época 1003 y de 0.01 en la 1393. Las épocas restantes, hasta la 3918, sirven para llevar las salidas más cerca de 0 y 1 y alcanzar el objetivo de 0.001.

4. **El resultado depende de los pesos iniciales, de la tasa de aprendizaje y del orden de ingreso.** Al probar las semillas 1 a 20 con η = 0.5, 14 convergieron (entre 3480 y 6798 épocas) y 6 quedaron atrapadas en un mínimo local con MSE ≈ 0.125 y 3 de 4 patrones correctos. Con la semilla 1, η = 1.0 convergió en 1947 épocas y η = 2.0 en 1029, mientras que η = 0.1 y η = 0.25 se quedaron en un mínimo local. Con orden aleatorio y semilla 1 la red tampoco convergió. Por eso el límite de 50 000 épocas es necesario como segundo criterio de parada.

5. **La división del algoritmo en módulos permitió trabajar en paralelo.** El forward pass y el MSE (parte A), la retropropagación y la actualización de pesos (parte B), y el dataset, el ciclo de entrenamiento y la interfaz (parte C) se conectan por funciones con parámetros acordados. Todo el proceso inteligente está programado a mano con NumPy, sin librerías de redes neuronales.

---

## 9. Referencias

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.
- Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Haykin, S. (2009). *Neural networks and learning machines* (3rd ed.). Pearson.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- Minsky, M., & Papert, S. (1969). *Perceptrons: An introduction to computational geometry*. MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://doi.org/10.1038/323533a0
- Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.
- Universidad Militar Nueva Granada. (2026). *Aplicaciones redes neuronales 2026-2* [Guía de la asignatura Inteligencia Artificial].
