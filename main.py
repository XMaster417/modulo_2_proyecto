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
    print("\nAnalisis exploratiorio de datos (EDA):")

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

if __name__ == "__main__":
    main()
