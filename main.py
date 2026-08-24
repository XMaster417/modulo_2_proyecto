# Inicializar librerias necesarias.
import numpy as np
import pandas as pd

def read_dataset(dataset, col_names):
    df = pd.read_csv(dataset, names=col_names)
    return df
    
def main():
    column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    df = read_dataset("abalone.data", column_names)
    print(df.head())
    
if __name__ == "__main__":
    main()