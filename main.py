# Inicializar librerias necesarias.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print("\nEliminar columnas con valores menores o iguales a 0:")
    ## Existen solo 0.04% de valores de 0 en la columna de Height, por lo que se
    ## puede eliminar esos registros.
    size_before_delete = len(df)
    df = df[df["Height"] != 0]
    size_after_delete = len(df)
    print(f"\nTamaño antes de eliminar: {size_before_delete}")
    print(f"Tamaño después de eliminar: {size_after_delete}")

    ## Convertir la columna de Rings a Age (sumar +1.5)
    print("\nConvertir la columna de Rings a Age (sumar +1.5):")
    df["Aproximated age"] = df["Rings"] + 1.5
    print(df.head())

    ## Separar las columnas de sexo en columnas binarias (one hot encoding)
    print("\nAplicar one hot encoding a la columna de Sex (M, F, I):")
    df = pd.get_dummies(df, columns=["Sex"], dtype=int)
    print(df.head())

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

    ### Calular si el peso total es la suma de las columnas restantes de peso
    sum_weight = (
        df["Shucked weight"] +
        df["Viscera weight"] + 
        df["Shell weight"]
    )

    test_weight = df["Whole weight"] == sum_weight
    weight_counts = test_weight.value_counts()
    weight_percentages = weight_counts / len(test_weight) * 100
    
    weight_result = pd.DataFrame({
        "Count": weight_counts,
        "Percentage": weight_percentages
    })

    print("\nComparar si el peso total es igual a la suma de los demas pesos:")
    print(weight_result)

    ## Entender mejor las diferencias de pesos.
    df["Weight difference"] = df["Whole weight"] - sum_weight
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
        sum_weight, df["Whole weight"],
        "Suma de los pesos parciales", "Peso Total",
        "Peso total vs Suma de los pesos parciales"
    )

    ### Mostrar la distribucion de las diferencias de los pesos en abulones
    plt.hist(df["Weight difference"], bins=100)
    plt.xlabel("Weight difference")
    plt.ylabel("Frequency")
    plt.title("Distribution of weight differences")
    plt.show()
"""
================================================================================
"""
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

if __name__ == "__main__":
    main()