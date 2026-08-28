import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import scatter_plot


"""
================================================================================
    Funcion: EDA (Exploratory Data Analysis).
    
    Realiza un analisis exploratorio de los datos, verificando relaciones entre 
    variables, analizando posibles causas de explicacion y graficando relaciones 
    entre variables.

    @param df -> pd.dataFrame: dataframe con los datos crudos.
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
        np.isclose(df["Weight difference"], 0), index=df.index
    )
    weight_counts = test_weight.value_counts()
    weight_percentages = weight_counts / len(df) * 100
    weight_result = pd.DataFrame({
        "Count": weight_counts,
        "Percentage": weight_percentages,
    })
    print("\nComparar si el peso total es igual a la suma de los demas pesos:")
    print(weight_result)

    ## Entender mejor las diferencias de pesos.
    df["Weight difference percentage"] = (
        df["Weight difference"] / df["Whole weight"]
    ) * 100
    print("\nMostrar la diferencia de pesos porcentuales:")
    print(df["Weight difference percentage"].describe())

    print("\nMostrar la informacion del dataset:")
    print(df.info())
    print("\nMostrar las estadisticas del dataset:")
    print(df.describe())

    ## Se encontro un valor de 0 para altura, verificar si hay mas valores de 0
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
        "Zero percentage": zero_percentages,
    })
    print(zeros_results)

    ## Verificar la distribucion de la variable sexo.
    print("\nVerificar la distribucion de la variable sexo:")
    sex_counts = df["Sex"].value_counts()
    sex_percentages = df["Sex"].value_counts(normalize=True) * 100
    sex_results = pd.DataFrame({
        "Count": sex_counts,
        "Percentage": sex_percentages,
    })
    print(sex_results)

    ## Verificar las relaciones entre variables.
    ### Relacion entre Sexo y Numero de Anillos
    scatter_plot(
        df["Sex"], df["Rings"],
        "Sexo", "Numero de Anillos",
        "Relacion entre Sexo y Numero de Anillos",
    )

    ### Relacion entre Longitud y Altura
    scatter_plot(
        df["Length"], df["Height"],
        "Longitud", "Altura", "Relacion entre Longitud y Altura",
    )

    ### Relacion entre Altura y Peso Total
    scatter_plot(
        df["Height"], df["Whole weight"],
        "Altura", "Peso Total", "Relacion entre Altura y Peso Total",
    )

    ### Relacion entre Longitud y Peso Total
    scatter_plot(
        df["Length"], df["Whole weight"],
        "Longitud", "Peso Total", "Relacion entre Longitud y Peso Total",
    )

    ### Relacion entre la suma de los pesos parciales y el peso total
    scatter_plot(
        df["Whole weight"] - df["Weight difference"],
        df["Whole weight"],
        "Suma de los pesos parciales", "Peso Total",
        "Peso total vs Suma de los pesos parciales",
    )

    ### Mostrar la distribucion de las diferencias de los pesos en abulones
    plt.hist(df["Weight difference"], bins=100)
    plt.xlabel("Diferencia de pesos")
    plt.ylabel("Frecuencia")
    plt.title("Distribucion de la diferencia entre peso total y suma de pesos")
    plt.show()

    ### Relacion entre diametro y altura
    scatter_plot(
        df["Diameter"], df["Height"],
        "Diametro", "Altura", "Relacion entre el diametro y la altura",
    )

    ### Relacion entre diametro y longitud
    scatter_plot(
        df["Length"], df["Diameter"],
        "Longitud", "Diametro", "Relacion entre la longitud y el diametro",
    )

    ### Relacion entre diametro y peso total.
    scatter_plot(
        df["Diameter"], df["Whole weight"],
        "Diametro", "Peso Total", "Relacion entre el diametro y el peso total",
    )

    ### Relacion entre diametro y sexo
    scatter_plot(
        df["Sex"], df["Diameter"],
        "Sexo", "Diametro", "Relacion entre Sexo y Diametro",
    )

    ### Relacion entre diametro y numero de anillos
    scatter_plot(
        df["Diameter"], df["Rings"],
        "Diametro", "Anillos", "Relacion entre diametro y numero de anillos",
    )

    ## Encontrar valores atipicos
    ### Valores atipicos de la variable de Altura.
    height_counts = (df[["Height"]] > 0.5).sum()
    height_percentages = (df[["Height"]] > 0.5).mean() * 100
    height_outlier_results = pd.DataFrame({
        "Count": height_counts,
        "Percentage": height_percentages,
    })
    print("\nValores atipicos cuya altura supera 0.5:")
    print(height_outlier_results)

    ### Valores atipicos del diametro y longitud
    diameter_outliers = df[["Diameter"]].gt(df["Length"], axis="index")
    diameter_counts = diameter_outliers.sum()
    diameter_percentages = diameter_outliers.mean() * 100
    diameter_outlier_results = pd.DataFrame({
        "Count": diameter_counts,
        "Percentage": diameter_percentages,
    })
    print("\nValores atipicos con diametro mayor a longitud")
    print(diameter_outlier_results)

    ### Valores cuya suma de pesos es negativa (superan al peso total)
    negative_weight_counts = (df[["Weight difference"]] < 0).sum()
    negative_weight_percentages = (df[["Weight difference"]] < 0).mean() * 100
    negative_weight_outlier_results = pd.DataFrame({
        "Count": negative_weight_counts,
        "Percentage": negative_weight_percentages,
    })
    print("\nValores atipicos cuya diferencia de pesos es negativa:")
    print(negative_weight_outlier_results)

    ## Calcular las estadisticas de las diferencias de pesos negativas
    negative_weight_difference = df.loc[
        df["Weight difference"] < 0,
        ["Weight difference", "Weight difference percentage"],
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
    print("\nDiferencias negativas mayores a 10% y de hasta 25%:")
    print(negative_between_10_and_25.head())
    print("\nDiferencias negativas mayores a 25%:")
    print(negative_above_25.head())

    ## Resumir las diferencias negativas por rangos porcentuales
    negative_percentage_magnitude = negative_percentage.abs()
    negative_0_to_10_count = negative_percentage_magnitude.le(10).sum()
    negative_above_10_count = (negative_percentage_magnitude > 10).sum()
    negative_ranges_summary = pd.DataFrame({
        "Range": ["0% - 10%", ">10%"],
        "Count": [negative_0_to_10_count, negative_above_10_count],
        "Percentage of dataset": [
            negative_0_to_10_count / len(df) * 100,
            negative_above_10_count / len(df) * 100,
        ],
    })
    print("\nResumen de diferencias negativas por rango:")
    print(negative_ranges_summary)


"""
================================================================================
    Funcion: correlation_analysis.
    Realiza una matriz de correlacion entre todas las variables utilizando 
    seaborn con un mapa de calor.

    @param df -> pd.dataFrame: dataframe con los datos de las variables.
================================================================================
"""
def correlation_analysis(df):
    print("===================================================================")
    print("\nAnalisis de correlacion y analisis multivariable")

    ## Crear el heatmap con todas las variables numericas, incluidas las
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
        linewidths=0.5,
    )
    plt.title("Heatmap de correlacion de las variables")
    plt.tight_layout()
    plt.show()