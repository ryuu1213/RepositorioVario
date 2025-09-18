import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# ------------------------
# Validación de datos
# ------------------------
def validar_datos(*arrays):
    longitud = len(arrays[0])
    for arr in arrays:
        if len(arr) != longitud:
            raise ValueError("Todos los arrays deben tener la misma longitud.")

# ------------------------
# Funciones de modelo y análisis
# ------------------------
def linear_function(x, m, c):
    return m * x + c

def calculate_chi_squared(observed, expected, errors):
    return np.sum(((observed - expected) / errors)**2)

# ------------------------
# Datos del Ejercicio 6.5
# ------------------------
displacement_6_5 = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85])
phase_6_5 = np.array([0.00, 0.21, 0.44, 0.67, 0.88, 1.1, 1.3, 1.5, 2.0, 2.24])
error_6_5 = np.array([0.05, 0.05, 0.05, 0.05, 0.09, 0.1, 0.2, 0.5, 0.1, 0.07])

validar_datos(displacement_6_5, phase_6_5, error_6_5)

# Ajuste lineal
popt_6_5, pcov_6_5 = curve_fit(
    linear_function, displacement_6_5, phase_6_5,
    sigma=error_6_5, absolute_sigma=True
)
slope_6_5, intercept_6_5 = popt_6_5
slope_err_6_5, intercept_err_6_5 = np.sqrt(np.diag(pcov_6_5))

# Chi-cuadrado y p-valor
predicted_phase_6_5 = linear_function(displacement_6_5, slope_6_5, intercept_6_5)
chi_squared_6_5 = calculate_chi_squared(phase_6_5, predicted_phase_6_5, error_6_5)
dof_6_5 = len(phase_6_5) - 2
p_value_6_5 = 1 - chi2.cdf(chi_squared_6_5, dof_6_5)

# ------------------------
# Datos del Ejercicio 7.8
# ------------------------
concentration_7_8 = np.array([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175])
theta_7_8 = np.array([10.7, 21.6, 32.4, 43.1, 53.9, 64.9, 75.4])
error_theta_7_8 = np.array([0.1] * len(theta_7_8))

validar_datos(concentration_7_8, theta_7_8, error_theta_7_8)

# Ajuste con parámetros dados
slope_7_8 = 431.7
intercept_7_8 = -0.03
predicted_theta_7_8 = linear_function(concentration_7_8, slope_7_8, intercept_7_8)
chi_squared_7_8 = calculate_chi_squared(theta_7_8, predicted_theta_7_8, error_theta_7_8)
dof_7_8 = len(theta_7_8) - 2
p_value_7_8 = 1 - chi2.cdf(chi_squared_7_8, dof_7_8)

# ------------------------
# Resultados
# ------------------------
alpha = 0.05

print("--- Evaluación del ajuste lineal para el Ejercicio 6.5 ---")
print(f"Pendiente: {slope_6_5:.4f} ± {slope_err_6_5:.4f}")
print(f"Intercepto: {intercept_6_5:.4f} ± {intercept_err_6_5:.4f}")
print(f"Chi-cuadrado (χ²): {chi_squared_6_5:.4f}")
print(f"Grados de libertad: {dof_6_5}")
print(f"P-valor: {p_value_6_5:.4f}")
print("Conclusión:", "No se rechaza el modelo lineal." if p_value_6_5 > alpha else "Se rechaza el modelo lineal.")

print("\n--- Evaluación del ajuste lineal para el Ejercicio 7.8 (con valores dados) ---")
print(f"Pendiente (dada): {slope_7_8}")
print(f"Intercepto (dado): {intercept_7_8}")
print(f"Chi-cuadrado (χ²): {chi_squared_7_8:.4f}")
print(f"Grados de libertad: {dof_7_8}")
print(f"P-valor: {p_value_7_8:.4f}")
print("Conclusión:", "No se rechaza el modelo lineal." if p_value_7_8 > alpha else "Se rechaza el modelo lineal.")
