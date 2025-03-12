import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#Cosas a mejorar en el código
#1.Modificar el codigo para que pueda aceptar varias fitting functions
#2.Modificar el código para que pueda imprimir a libertad las unidades dada un titulo de la forma "x(<unidades>)"
def plot_excel_sheets(excel_file, imprimir):

    """
    Carga y grafica las hojas de un archivo Excel.

    Args:
        filepath (str): Ruta del archivo Excel a cargar.
        show (bool, opcional): Si True, muestra las gráficas. Por defecto es True.

    Returns:
        None
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
        data_dict[sheet_name] = {'x': np.array(x.tolist()), 'y': np.array(y.tolist()), 'error': np.array(error.tolist()), 'name':sheet_name}

        # Crear la gráfica scatter con barras de error
        plt.figure(figsize=(6, 4))
        plt.errorbar(x, y, yerr=error, fmt='o', capsize=3, label=f'Hoja: {sheet_name}', color='black', ecolor='red', markersize=3)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.title(f'{sheet_name}')
        plt.legend()
        plt.grid()
        if imprimir == True:
            plt.savefig(f'{sheet_name}.pdf', format='pdf')
        plt.show()

    return data_dict


def calcular_incertidumbre(cov_matrix, column_names):
    """
    Calcula la incertidumbre (raíz de la varianza) para cada columna a partir de la matriz de covarianza.

    Parámetros:
    - cov_matrix: ndarray (matriz de covarianza cuadrada de tamaño NxN)
    - column_names: lista de strings (nombre de cada variable)

    Retorna:
    - Diccionario con la incertidumbre de cada columna
    """
    if len(column_names) != cov_matrix.shape[0]:
        raise ValueError("El número de nombres de columnas no coincide con el tamaño de la matriz de covarianza")

    incertidumbres = {column_names[i]: np.sqrt(cov_matrix[i, i]) for i in range(len(column_names))}
    return incertidumbres


def plot_fitted_curve(fitting_function, imprimir, titulo, **datasets):
    """
    Plotea múltiples conjuntos de datos con sus respectivos ajustes y residuales.
    
    Parámetros:
    - fitting_function: función de ajuste.
    - imprimir (bool): Si True, guarda la gráfica en PDF.
    - titulo (str): Nombre de la gráfica.
    - **datasets: Diccionario con múltiples conjuntos de datos.
      Cada conjunto debe incluir: "x", "y", "error", "popt", "formula_text", "name".
    
    Ejemplo de uso:
    plot_fitted_curve(fitting_function, dataset1={...}, dataset2={...})
    """
    
    # Crear la figura con dos subgráficos
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 2], 'hspace':0})

    colors = ['y', 'b', 'g', 'm', 'c', "#800080"]  # Paleta de colores para múltiples conjuntos

    for i, (label, data) in enumerate(datasets.items()):
        x, y, error, popt, formula_text, name = data["x"], data["y"], data["error"], data["popt"], data["formula_text"], data["name"]
        color = colors[i % len(colors)]  # Selecciona un color cíclico

        # Calcular el ajuste
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = fitting_function(x_fit, *popt)

        # Calcular los residuales normalizados
        residuals = (y - fitting_function(x, *popt)) / error

        # Graficar los datos con barras de error
        ax[0].errorbar(x, y, yerr=error, fmt="o", color=color, label=f"Datos {name}", capsize=8, ecolor="r")
        ax[0].plot(x_fit, y_fit, linestyle="--", color=color, label=f"Ajuste {name}: {formula_text}")

        # Graficar los residuales
        ax[1].scatter(x, residuals, color=color, label=f"Residuales {name}", s=7)

    # Configuración de la gráfica superior
    ax[0].set_ylabel("Radio (cm)")
    ax[0].legend()
    ax[0].set_title(titulo)

    # Configuración de la gráfica de residuales
    ax[1].axhline(0, color='black', linestyle='dashed')
    ax[1].set_xlabel("Desplazamiento (μm)")
    ax[1].set_ylabel("Residuales")
    ax[1].legend()

    # Guardar en PDF si imprimir es True
    if imprimir:
        plt.savefig(f'{titulo}.pdf', format='pdf')

    plt.show()