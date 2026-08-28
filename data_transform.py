import pandas as pd

"""
================================================================================
    Funcion: data_transform.
    Elimina registros invalidos o atipicos y muestra las estadisticas de los
    datos transformados. Dependiendo del tipo de modelo, aplica one hot
    encoding a la variable Sex o conserva la columna categorica.

    @param df -> dataFrame: dataframe de pandas con la informacion.
    @param model_type -> int: 0 para el modelo manual con one hot encoding;
        1 para un modelo con framework sin one hot encoding.
    @return df -> dataFrame: dataframe limpio y preparado para el modelo.
================================================================================
"""
def data_transform(df, model_type):
    if model_type not in (0, 1):
        raise ValueError(
            "model_type debe ser 0 (modelo manual) o 1 (modelo con framework)"
        )

    print("===================================================================")
    print("\nProceso de transformacion de datos")
    print("\nEliminar registros con valores invalidos o atipicos:")

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
    print(
        "Registros eliminados por altura mayor a 0.5: "
        f"{height_outliers.sum()}"
    )

    ## El diametro de un abulon no puede ser mayor que su longitud.
    diameter_outliers = df["Diameter"] > df["Length"]
    df = df[df["Diameter"] <= df["Length"]]
    print(
        "Registros eliminados por diametro mayor a la longitud: "
        f"{diameter_outliers.sum()}"
    )

    ## Eliminar las diferencias porcentuales de peso negativas mayores al 10%.
    weight_difference = (
        df["Whole weight"]
        - df["Shucked weight"]
        - df["Viscera weight"]
        - df["Shell weight"]
    )
    weight_difference_percentage = weight_difference / df["Whole weight"] * 100
    negative_weight_outliers = weight_difference_percentage < -10
    df = df[weight_difference_percentage > -10]

    print(
        "Registros eliminados por diferencia negativa de peso mayor a 10%: "
        f"{negative_weight_outliers.sum()}"
    )

    size_after_delete = len(df)
    print(f"Tamaño despues de eliminar: {size_after_delete}")

    ## Mostrar las estadisticas descriptivas de los datos transformados.
    print("\nEstadisticas descriptivas de los datos transformados:")
    print(df.describe())

    ## Mostrar la distribucion final de la variable a predecir.
    sex_distribution = pd.DataFrame({
        "Count": df["Sex"].value_counts(),
        "Percentage": df["Sex"].value_counts(normalize=True) * 100,
    })
    sex_distribution.index.name = "Sex"
    print("\nDistribucion variable Sex:")
    print(sex_distribution)

    ## Aplicar one hot encoding para Sex unicamente en el modelo manual.
    if model_type == 0:
        df = pd.get_dummies(df, columns=["Sex"], dtype=int)

    return df