import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_excel_sheets(excel_file, imprimir):

    """
    Parámetros:
    excel_file (string): Ruta del archivo Excel donde cada hoja tiene 3 columnas: X, Y, Error, de izquierda a derehca
    imprimir (Bool): Si imprime, se guarda en la carpeta Lab_Intermedio
    """
    # Cargar todas las hojas
    sheets = pd.read_excel(excel_file, sheet_name=None)
    data_dict = {}

    for sheet_name, df in sheets.items():
        if df.shape[1] < 3:
            print(f"La hoja '{sheet_name}' tiene menos de 3 columnas, se omitirá.")
            continue

        # Seleccionar las primeras tres columnas
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        error = df.iloc[:, 2]

        # Guardar los datos en el diccionario
        data_dict[sheet_name] = {'x': x.tolist(), 'y': y.tolist(), 'error': error.tolist()}

        # Crear la gráfica scatter con barras de error
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=error, fmt='o', capsize=3, label=f'Hoja: {sheet_name}', color='black', ecolor='red', markersize=3)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title(f'{sheet_name}')
        plt.legend()
        plt.grid()
        if imprimir == True:
            plt.savefig(f'/content/{sheet_name}.pdf', format='pdf')
        plt.show()

    return data_dict