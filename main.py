from data_transform import data_transform
from eda import EDA, correlation_analysis
from models.manual_logistic_regression import logistic_regression_analysis
from models.random_forest import random_forest_analysis
from utils import read_dataset

def main():
    ## Extraer los datos.
    column_names = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ]
    dataframe = read_dataset("abalone.data", column_names)
    print(dataframe.head())

    ## Manda una copia para no "manchar" el dataset original.
    EDA(dataframe.copy())

    ## Transformar los datos para el modelo manual con one hot encoding.
    df_manual = data_transform(dataframe, model_type=0)

    correlation_analysis(df_manual)

    ## Entrenar tres modelos: I vs todos, M vs todos y F vs todos.
    logistic_regression_analysis(df_manual)

    # Proceso de entrenar modelo con framework
    ## No agregar one hot encoding para este dataframe del modelo.
    df_framework = data_transform(dataframe, model_type=1)
    random_forest_analysis(df_framework)

if __name__ == "__main__":
    main()
