# Inicializar librerias necesarias.
import numpy as np
import pandas as pd

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
    Funcion: data_transform.
    Sirve para encontrar resultados como datos nulos, duplicados y cuales pesos
    son iguales a la suma de los demas pesos. Imprime los resultados en consola.

    @param dataframe -> dataFrame: dataframe de pandas con la informacion.
================================================================================
"""
def data_transform(df):
    print("===================================================================")
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
    counts = test_weight.value_counts()
    percentages = counts / len(test_weight) * 100
    
    weight_result = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    print("\nComparar si el peso total es igual a la suma de los demas pesos:")
    print(weight_result)

    print("\nConvertir la columna de Rings a Age (sumar +1.5):")
    df["Aproximated age"] = df["Rings"] + 1.5
    print(df.head())

    print("\nAplicar one hot encoding a la columna de Sex (M, F, I):")
    df = pd.get_dummies(df, columns=["Sex"])
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
    print("\nMostrar la informacion del dataset:")
    print(df.info())
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
    
    dataframe = data_transform(dataframe)

    EDA(dataframe)


    
if __name__ == "__main__":
    main()