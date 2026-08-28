# Proyecto módulo 2: clasificación del conjunto Abalone

Este proyecto analiza el conjunto de datos **Abalone** y entrena una regresión
logística multiclase para predecir el sexo del abulón: `I` (infant), `M`
(male) o `F` (female). El modelo usa la estrategia *one-vs-all*: aprende un
clasificador binario por cada clase y elige la probabilidad más alta.

## Contenido

- `main.py`: análisis exploratorio, transformación, entrenamiento y gráficas.
- `abalone.data`: datos de entrada, sin encabezados.
- `requirements.txt`: dependencias de Python.

## Requisitos

- Python 3.10 o superior.
- Version sin framework: `numpy`, `pandas`, `matplotlib` y `seaborn`.

## Instalación en un entorno virtual

Desde la carpeta raíz del proyecto, crea el entorno:

```powershell
python -m venv .venv
```

Actívalo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias del archivo `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

Con el entorno virtual activado y estando en la raíz del proyecto:

```powershell
python main.py
```

El programa imprime resultados del análisis y del modelo, y abre gráficas. El
archivo `abalone.data` debe permanecer en la misma carpeta que `main.py`.

## Funcionamiento general

1. `main()` asigna nombres a las columnas y carga `abalone.data`.
2. Se ejecuta el análisis exploratorio (`EDA`) sobre una copia para no alterar
   los datos originales.
3. `data_transform()` limpia observaciones no válidas y convierte `Sex` a tres
   columnas binarias: `Sex_I`, `Sex_M` y `Sex_F`.
4. `correlation_analysis()` presenta un mapa de calor de correlaciones.
5. `logistic_regression_analysis()` divide los datos, los estandariza, entrena
   el modelo y evalúa las predicciones.
6. Se muestran la matriz de confusión y las curvas de Binary Cross Entropy
   (BCE) de entrenamiento, validación y prueba.

La función `main()` es el punto de entrada: encadena todos estos pasos cuando
se ejecuta `python main.py`.

## Datos usados por el modelo

La variable objetivo es `Sex`, convertida a formato *one-hot*:

```text
I -> [1, 0, 0]
M -> [0, 1, 0]
F -> [0, 0, 1]
```

Las características son:

```text
Length, Diameter, Height, Whole weight, Shucked weight,
Viscera weight, Shell weight, Rings
```

## Funciones principales

### Carga, exploración y transformación

| Función | Parámetros | Descripción |
| --- | --- | --- |
| `read_dataset(dataset, col_names)` | Ruta del archivo y lista de nombres de columna. | Lee el archivo con Pandas y retorna un `DataFrame`. |
| `scatter_plot(x, y, xlabel, ylabel, title)` | Dos secuencias de datos y textos para los ejes/título. | Dibuja una gráfica de dispersión. |
| `EDA(df)` | `DataFrame` original. | Muestra valores nulos, duplicados, revisiones de pesos y gráficas exploratorias. |
| `data_transform(df)` | `DataFrame` de Abalone. | Elimina registros inválidos o atípicos, muestra estadísticas y aplica *one-hot encoding* a `Sex`. |
| `correlation_analysis(df)` | `DataFrame` transformado. | Calcula correlaciones numéricas y las presenta en un mapa de calor. |

### Componentes matemáticos del modelo

| Función | Parámetros | Descripción |
| --- | --- | --- |
| `sigmoid(z)` | Número o arreglo NumPy `z`. | Aplica `1 / (1 + exp(-z))` y convierte cada valor lineal en una probabilidad entre 0 y 1. |
| `binary_cross_entropy(y_true, y_hat)` | Etiqueta real y probabilidad predicha, escalares o arreglos de la misma forma. | Calcula la pérdida BCE individual: `-(y log(y_hat) + (1-y) log(1-y_hat))`. |
| `distributed_split(x, y, train_size=0.70, val_size=0.15, seed=1)` | Matrices de características y etiquetas, proporciones y semilla. | Divide de forma estratificada en entrenamiento, validación y prueba; conserva aproximadamente la proporción de cada clase. |
| `train(x_train, y_train, datasets, epochs=3000, alpha=0.001)` | Datos de entrenamiento, diccionario de conjuntos, épocas y tasa de aprendizaje. | Entrena los tres modelos mediante gradiente descendente. Retorna `weights`, `biases` e `history`. |
| `predict(x, weights, biases)` | Características y parámetros ya entrenados. | Calcula las probabilidades `y_hat` y retorna tanto las probabilidades como el índice de la clase con mayor probabilidad. |
| `confusion_matrix(y_true, y_hat, class_names)` | Etiquetas de la variable y reales, índices predichos y nombres de clase. | Calcula la exactitud y muestra una matriz de confusión con Seaborn. |

### Flujo de regresión logística

`logistic_regression_analysis(df, epochs=5000, learning_rate=0.001)` coordina
el modelo completo:

1. Declara las columnas objetivo y características.
2. Convierte el `DataFrame` a arreglos NumPy de tipo `float`, necesarios para
   los cálculos matriciales del modelo.
3. Divide los datos en 70 % entrenamiento, 15 % validación y 15 % prueba.
4. Estandariza las características con la media y desviación estándar de
   entrenamiento. Así se evita que `Rings`, cuya escala es mayor, domine los
   ajustes de pesos. Validación y prueba usan esos mismos valores para evitar
   fuga de información.
5. Llama a `train()` y conserva los pesos, sesgos e historial BCE.
6. Predice sobre prueba, imprime tamaños y BCE final, y muestra la matriz de
   confusión.
7. Grafica la BCE por época. El eje vertical está limitado al intervalo de 0 a
   1 para facilitar la comparación visual.

## Métricas y salidas

- **BCE**: mide el error entre las probabilidades predichas y las etiquetas
  reales; valores más bajos son mejores.
- **Exactitud**: proporción de observaciones de prueba clasificadas
  correctamente.
- **Matriz de confusión**: filas = clase real, columnas = clase predicha. La
  diagonal contiene los aciertos.

## Ajustes frecuentes

Puedes modificar los argumentos de la llamada final en `main.py`:

```python
logistic_regression_analysis(
    dataframe,
    epochs=5000,
    learning_rate=0.001,
)
```

Un valor mayor de `epochs` implica más iteraciones; `learning_rate` controla
el tamaño de cada actualización de los parámetros.
