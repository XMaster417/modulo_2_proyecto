import matplotlib.pyplot as plt
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
    return pd.read_csv(dataset, names=col_names)


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
