# Inicializar librerias necesarias.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
================================================================================
    Funcion: read_dataset.
    Sirve para recivir un archivo (generalmente .data o .csv, pero acepta otros 
    formatos), tambien recibe los nombres de columnas para los datasets que 
    los tienen en un archivo aparte.

    @param dataset -> file: archivo con el dataset a cargar
    @param col_names -> list<String>: lista con los nombres de las columnas
    @return df -> dataFrame: dataframe de pandas con la informacion del dataset
================================================================================
"""
def read_dataset(dataset, col_names):
    df = pd.read_csv(dataset, names=col_names)
    return df
"""
================================================================================
    Funcion: scatter_plot.
    Toma dos variables y crea una grafia de dispersion (scatter plot) que 
    imprime en pantalla con los nombres de los ejes y titulo del grafico.

    @param x -> list: lista con los valores del eje x
    @param y -> list: lista con los valores del eje y
    @param xlabel -> String: nombre del eje x
    @param ylabel -> String: nombre del eje y
    @param title -> String: titulo del grafico
================================================================================
"""
def scatter_plot(x, y, xlabel, ylabel, title):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
"""
================================================================================
    Funcion: data_transform.
    Sirve para encontrar resultados como datos nulos, duplicados y cuales pesos
    son iguales a la suma de los demas pesos. Imprime los resultados en consola.

    @param dataframe -> dataFrame: dataframe de pandas con la informacion.
================================================================================
"""
def data_transform(df):
    print("===================================================================")
    print("\nProceso de transformación de datos")
    print("\nEliminar registros con valores inválidos o atípicos:")
    ## Existen solo 0.04% de valores de 0 en la columna de Height, por lo que se
    ## puede eliminar esos registros.
    size_before_delete = len(df)
    print(f"\nTamaño antes de eliminar: {size_before_delete}")
    height_zero = df["Height"] == 0
    df = df[df["Height"] != 0]
    print(f"\nRegistros eliminados por altura de 0: {height_zero.sum()}")
    
    ## Eliminar observaciones cuya altura sea mayor a 0.5.
    height_outliers = df["Height"] >= 0.5
    df = df[df["Height"] <= 0.5]
    print(f"Registros eliminados por altura mayor a 0.5: {height_outliers.sum()}")
    
    ## El diámetro de un abulón no puede ser mayor que su longitud.
    diameter_outliers = df["Diameter"] > df["Length"]
    df = df[df["Diameter"] <= df["Length"]]
    print(
        "Registros eliminados por diámetro mayor a la longitud: "
        f"{diameter_outliers.sum()}"
    )
    ## Eliminar las diferencias porcentuales de peso negativas mayores al 10%
    weight_difference = (
        df["Whole weight"]
        - df["Shucked weight"]
        - df["Viscera weight"]
        - df["Shell weight"]
    )
    weight_difference_percentage = (
        weight_difference / df["Whole weight"] * 100
    )
    negative_weight_outliers = weight_difference_percentage < -10
    df = df[weight_difference_percentage > -10]

    print(
        "Registros eliminados por diferencia negativa de peso mayor a 10%: "
        f"{negative_weight_outliers.sum()}"
    )
    

    size_after_delete = len(df)
    print(f"Tamaño después de eliminar: {size_after_delete}")

    ## Mostrar las estadísticas descriptivas de los datos transformados.
    print("\nEstadísticas descriptivas de los datos transformados:")
    print(df.describe())

    ## Mostrar la distribución final de la variable a predecir.
    sex_distribution = pd.DataFrame({
        "Count": df["Sex"].value_counts(),
        "Percentage": df["Sex"].value_counts(normalize=True) * 100
    })
    sex_distribution.index.name = "Sex"
    print("\nDistribución variable Sex:")
    print(sex_distribution)

    ## Aplicar one hot encoding para la variable de Sex
    df = pd.get_dummies(df, columns=["Sex"], dtype=int)

    return df
"""
================================================================================
    Funcion: EDA (Exploratory Data Analysis).
================================================================================
"""
def EDA(df):
    print("===================================================================")
    print("\nAnalisis exploratorio de datos (EDA):")

    ### Datos nulos
    print("\nMostrar la cantidad de datos nulos:")
    print(df.isna().sum())

    ### Datos duplicados
    print("\nMostrar los elementos duplicados: ", df.duplicated().sum())

    ### Calcular la diferencia entre el peso total y los pesos restantes
    df["Weight difference"] = (
        df["Whole weight"]
        - df["Shucked weight"]
        - df["Viscera weight"]
        - df["Shell weight"]
    )

    ### Verificar si el peso total es igual a la suma de los demas pesos
    test_weight = pd.Series(
        np.isclose(df["Weight difference"], 0),
        index=df.index
    )
    weight_counts = test_weight.value_counts()
    weight_percentages = weight_counts / len(df) * 100
    
    weight_result = pd.DataFrame({
        "Count": weight_counts,
        "Percentage": weight_percentages
    })

    print("\nComparar si el peso total es igual a la suma de los demas pesos:")
    print(weight_result)

    ## Entender mejor las diferencias de pesos.
    df["Weight difference percentage"] = (df["Weight difference"] / 
                                          df["Whole weight"]) * 100
    print("\nMostrar la diferencia de pesos porcentuales:")
    print(df["Weight difference percentage"].describe())

    print("\nMostrar la informacion del dataset:")
    print(df.info())

    print("\nMostrar las estadisticas del dataset:")
    print(df.describe())

    ## Se encontró un valor de 0 para altura, verificar si hay más valores de 0
    ## en columnas para medidas, peso y anillos.
    print("\nVerificar valores menor o iguales a 0 en las columnas:")

    columns = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings", 
    ]

    zero_counts = (df[columns] <= 0).sum()
    zero_percentages = (df[columns] == 0).mean() * 100
    zeros_results = pd.DataFrame({
        "Zero count": zero_counts,
        "Zero percentage": zero_percentages
    })
    print(zeros_results)

    ## Verificar la distribución de la variable sexo.
    print("\nVerificar la distribución de la variable sexo:")
    sex_counts = df["Sex"].value_counts()
    sex_percentages = df["Sex"].value_counts(normalize=True) * 100
    sex_results = pd.DataFrame({
        "Count": sex_counts,
        "Percentage": sex_percentages
    })
    print(sex_results)

    ## Verificar las relaciones entre variables.
    ### Relación entre Sexo y Número de Anillos
    scatter_plot(
        df["Sex"], df["Rings"], 
        "Sexo", "Número de Anillos", 
        "Relación entre Sexo y Número de Anillos"
    )

    ### Relación entre Longitud y Altura
    scatter_plot(
        df["Length"], df["Height"], 
        "Longitud", "Altura", 
        "Relación entre Longitud y Altura"
    )

    ### Relación entre Altura y Peso Total
    scatter_plot(
        df["Height"], df["Whole weight"], 
        "Altura", "Peso Total", 
        "Relación entre Altura y Peso Total"
    )

    ### Relación entre Longitud y Peso Total
    scatter_plot(
        df["Length"], df["Whole weight"],
        "Longitud", "Peso Total",
        "Relacion entre Longitud y Peso Total"
    )

    ### Relación entre la suma de los pesos parciales y el peso total
    scatter_plot(
        df["Whole weight"] - df["Weight difference"], # Pesos restantes
        df["Whole weight"],
        "Suma de los pesos parciales", "Peso Total",
        "Peso total vs Suma de los pesos parciales"
    )

    ### Mostrar la distribucion de las diferencias de los pesos en abulones
    plt.hist(df["Weight difference"], bins=100)
    plt.xlabel("Diferencia de pesos")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de la diferencia entre peso total y suma de pesos")
    plt.show()

    ### Relacion entre diametro y altura
    scatter_plot(
        df["Diameter"], df["Height"],
        "Diametro", "Altura",
        "Relacion entre el diametro y la altura"
    )

    ### Relacion entre diametro y longitud
    scatter_plot(
        df["Length"], df["Diameter"],
        "Longitud", "Diametro",
        "Relación entre la longitud y el diametro"
    )

    ### Relacion entre diametro y peso total.
    scatter_plot(
        df["Diameter"], df["Whole weight"],
        "Diametro", "Peso Total",
        "Relación entre el diametro y el peso total"
    )

    ### Relacion entre diametro y sexo
    scatter_plot(
        df["Sex"], df["Diameter"],
        "Sexo", "Diametro",
        "Relación entre Sexo y Diametro"
    )

    ### Relacion entre diametro y numero de anillos
    scatter_plot(
        df["Diameter"], df["Rings"],
        "Diametro", "Anillos",
        "Relación entre diametro y número de anillos"
    )

    ## Encontrar valores atipicos
    ### Valores atipicos de la variable de Altura.
    height_counts = (df[["Height"]] > 0.5).sum()
    height_percentages = (df[["Height"]] > 0.5).mean() * 100
    height_outlier_results = pd.DataFrame({
        "Count": height_counts,
        "Percentage": height_percentages
    })

    print("\nValores atipicos cuya altura supera 0.5:")
    print(height_outlier_results)

    ### Valores atipicos del diametro y longitud
    diameter_outliers = df[["Diameter"]].gt(df["Length"], axis="index")
    diameter_counts = diameter_outliers.sum()
    diameter_percentages = diameter_outliers.mean() * 100
    diameter_outlier_results = pd.DataFrame({
        "Count": diameter_counts,
        "Percentage": diameter_percentages
    })
    print("\nValores atipicos con diametro mayor a longitud")
    print(diameter_outlier_results)

    ### Valores cuya suma de pesos es negativa (superan al peso total)
    negative_weight_counts = (df[["Weight difference"]] < 0).sum()
    negative_weight_percentages = (df[["Weight difference"]] < 0).mean() * 100
    negative_weight_outlier_results = pd.DataFrame({
        "Count": negative_weight_counts,
        "Percentage": negative_weight_percentages
    })

    print("\nValores atipicos cuya diferencia de pesos es negativa:")
    print(negative_weight_outlier_results)

    ## Calcular las estadisticas de las diferencias de pesos negativas
    negative_weight_difference = df.loc[
        df["Weight difference"] < 0,
        ["Weight difference", "Weight difference percentage"]
    ]
    print("\nEstadisticas descriptivas de las diferencias negativas:")
    print(negative_weight_difference.describe())

    ## Separar las diferencias negativas por su magnitud porcentual
    negative_percentage = negative_weight_difference[
        "Weight difference percentage"
    ]

    negative_below_5 = negative_weight_difference.loc[
        negative_percentage >= -5
    ]
    negative_between_5_and_10 = negative_weight_difference.loc[
        (negative_percentage < -5) & (negative_percentage >= -10)
    ]
    negative_between_10_and_25 = negative_weight_difference.loc[
        (negative_percentage < -10) & (negative_percentage >= -25)
    ]
    negative_above_25 = negative_weight_difference.loc[
        negative_percentage < -25
    ]

    print("\nDiferencias negativas de hasta 5%:")
    print(negative_below_5.head())

    print("\nDiferencias negativas mayores a 5% y de hasta 10%:")
    print(negative_between_5_and_10.head())

    print("\nDiferencias negativas mayores a 1% y de hasta 25%:")
    print(negative_between_10_and_25.head())

    print("\nDiferencias negativas mayores a 25%:")
    print(negative_above_25.head())

    ## Resumir las diferencias negativas por rangos porcentuales
    negative_percentage_magnitude = negative_percentage.abs()

    negative_0_to_10_count = negative_percentage_magnitude.le(10).sum()
    negative_above_10_count = (negative_percentage_magnitude > 10).sum()

    negative_ranges_summary = pd.DataFrame({
        "Range": ["0% - 10%", ">10%"],
        "Count": [
            negative_0_to_10_count,
            negative_above_10_count
        ],
        "Percentage of dataset": [
            negative_0_to_10_count / len(df) * 100,
            negative_above_10_count / len(df) * 100
        ]
    })

    print("\nResumen de diferencias negativas por rango:")
    print(negative_ranges_summary)

"""
================================================================================
"""
def correlation_analysis(df):
    print("===================================================================")
    print("\nAnálisis de correlación y análisis multivariable")

    ## Crear el heatmap con todas las variables numéricas, incluidas las
    ## columnas Sex_F, Sex_I y Sex_M generadas mediante one-hot encoding.
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(13, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    plt.title("Heatmap de correlación de las variables")
    plt.tight_layout()
    plt.show()
"""
================================================================================
    Funcion: sigmoid
    Función de activación sigmoide, para retornar valores entre 0 y 1.

    @param z: funcion lineal a convertir con la sigmoide.
    @return num: valor convertido en el rango de 0 a 1.
================================================================================
"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
"""
================================================================================
    Funcion: binary_cross_entropy
    Funcion para calcular el error de los valores predichos.

    @y_true -> num: valor real de y.
    @y_hat -> num: valor predicho con la sigmoide de y. 
    @return num: error o perdida por cada dato.
================================================================================
"""
def binary_cross_entropy(y_true, y_hat):
    return -(
        y_true * np.log(y_hat)
        + (1.0 - y_true) * np.log(1.0 - y_hat)
    )
"""
================================================================================
    Funcion: distributed_split
    Devuelve las proporciones distribuidas para los conjuntos de train, test y 
    val.

    @param x -> matrix: valores de las variables x
    @param y -> matrix: valores de la variable y
    @param train_size -> float: porcentaje de tamaño asignado al conjunto de test
    @param val_size -> float:  porcentaje de tamaño asignado al conjunto de val
    @param seed -> num: valor escogido para la aleatoriedad
    @return result -> list<tuple>: 
================================================================================
"""
def distributed_split(x, y, train_size=0.70, val_size=0.15, seed=1):
    
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    num_class = y.shape[1]

    for class_index in range(num_class):
        class_indices = [
            row_index
            ## Regresa los indices con su valor con ayuda de enumerate
            for row_index, value in enumerate(y[:, class_index])
            if value == 1
        ]
        
        rng.shuffle(class_indices)
        train_end = int(len(class_indices) * train_size)
        val_end = train_end + int(len(class_indices) * val_size)
        splits[0].extend(class_indices[:train_end])
        splits[1].extend(class_indices[train_end:val_end])
        splits[2].extend(class_indices[val_end:])

    result = []
    for indices in splits:
        rng.shuffle(indices)
        result.append((x[indices], y[indices]))
    return result
"""
================================================================================
    Funcion: train.
    Train entrena al modelo para identificar si es I, M, F almacena los pesos y 
    bias en una lista para evaluar la opción mas probable utilizando la tecnica
    de one vs all.

    @param x_train -> matriz: matriz con las variables independientes para train.
    @param y_ train -> matriz: matriz las respuestas para train.
    @param datasets -> list<list>: lista con los conjutos de train, test y val.
    @param epochs -> int: numero de veces que el modelo entrena.
    @param alpha -> float: taza de aprendizaje del modelo, ajusta el movimiento.
    @return weight, biases, history -> retorna los pesos, bias para el modelo
        así como los errores en la variable history para graficar el BCE
================================================================================
"""
def train(x_train, y_train, datasets, epochs=3000, alpha=0.001):
    n_samples, n_features = x_train.shape
    n_classes = y_train.shape[1]
    weights = np.zeros((n_features, n_classes))
    biases = np.zeros(n_classes)
    history = {name: [] for name in datasets}

    for _ in range(epochs):
        predictions = sigmoid(np.dot(x_train, weights) + biases)
        loss = predictions - y_train

        theta_weight = np.dot(x_train.T, loss) / n_samples
        theta_bias = np.mean(loss, axis=0)
        weights -= alpha * theta_weight
        biases -= alpha * theta_bias

        for name, (x_split, y_split) in datasets.items():
            y_hat = sigmoid(np.dot(x_split, weights) + biases)
            history[name].append(
                float(np.mean(binary_cross_entropy(y_split, y_hat)))
            )

    return weights, biases, history
"""
================================================================================
    Funcion: predict.
    A partir de un modelo entrenado obtiene la predicción mayor para cada clase
    objetivo del dataset.

    @param x -> matrix: matriz con los valores de las variables independientes
    @param weights -> matrix: pesos por variable aprendidos durante train.
    @param bias -> matrix: sesgo por variable aprendido durantre train
    @return num: predicción mayor según los datos ingresados.
================================================================================
"""
def predict(x, weights, biases):
    y_hat = sigmoid(np.dot(x, weights) + biases)
    return np.argmax(y_hat, axis=1), y_hat
"""
================================================================================
    Funcion: confusion_matrix.
    Muestra la matriz de confusion con el resultado de las predicciones contra 
    los valores reales.

    @param y_true -> matrix: matriz con los valores reales de Sex.
    @param y_hat -> np.array: arreglo con el indice de la clase predicha (0,1,2)
    @param class_names -> list<String>: Lista con los nombres de las clases
================================================================================
"""
def confusion_matrix(y_true, y_hat, class_names):
    true_classes = np.argmax(y_true, axis=1) # Indice de la clase verdadera
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(confusion, (true_classes, y_hat), 1)
    accuracy = np.mean(true_classes == y_hat)

    print(f"\nExactitud en test: {accuracy:.4f}")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,     # muestra los valores en la matriz de confusion
        fmt="d",        # muestra los valores como enteros
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.show()
"""
================================================================================
    Funcion: logistic_regression_analsis()
    Realiza una regresión logistica, primero identifica las variables 
    dependientes e independientes, las convierte a arreglos de numpy para 
    efectuar los calculos. Separa los datos en train, val y test. Se 
    estandarizan los valores por que Rings maneja valores muy grandes respecto a 
    las demás medidas. Entrena al modelo utilizando regresión logistica con 
    sigmoide, binary cross entropy, gradiente descendente y one vs all. Despues 
    realiza las predicciones correspondientes. Finalmente imprime los resultados
    en una matriz de confusion y grafica de los errores del modelo.

================================================================================
"""
def logistic_regression_analysis(df, epochs=10000, learning_rate=0.001):
    class_names = ["I", "M", "F"]
    target_columns = ["Sex_I", "Sex_M", "Sex_F"]
    feature_columns = [
        "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight", "Rings",
    ]

    x = df[feature_columns].to_numpy(dtype=float)
    y = df[target_columns].to_numpy(dtype=float)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = distributed_split(x, y, seed=1)

    # Estandarización calculada exclusivamente con train.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    datasets = {
        "Train": (x_train, y_train),
        "Validación": (x_val, y_val),
        "Test": (x_test, y_test),
    }
    weights, biases, history = train(
        x_train, y_train, datasets, epochs, learning_rate
    )

    predictions, y_hat = predict(
        x_test, weights, biases
    )
    print("\nResumen de regresión logística")
    print("\nTamaños de los conjuntos:")
    print(f"  Entrenamiento: {len(x_train)}")
    print(f"  Validación: {len(x_val)}")
    print(f"  Prueba: {len(x_test)}")
    print("\nBCE final:")
    for name, losses in history.items():
        print(f"  {name}: {losses[-1]:.6f}")
    confusion_matrix(y_test, predictions, class_names)

    plt.figure(figsize=(9, 5))
    for name, losses in history.items():
        plt.plot(losses, label=name)
    plt.title("Regresión logística One-vs-All")
    plt.xlabel("Épocas")
    plt.ylabel("Error BCE")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    ## Extraer los datos.
    column_names = [
        "Sex", "Length", "Diameter", 
        "Height", "Whole weight", "Shucked weight", 
        "Viscera weight", "Shell weight", "Rings"
    ]
    dataframe = read_dataset("abalone.data", column_names)
    print(dataframe.head())

    ## Manda una copia para no "manchar" el dataset original
    EDA(dataframe.copy())

    dataframe = data_transform(dataframe)

    correlation_analysis(dataframe)

    ## Entrenar tres modelos: I vs todos, M vs todos y F vs todos.
    logistic_regression_analysis(dataframe)

if __name__ == "__main__":
    main()
