import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

##### PRIMER PUNTO

# Datos experimentales
m = np.array([0.05, 0.2, 0.3, 0.35])  # kg
theta = np.array([-0.2, -0.9, -1.08, -1.14])  # rad
r = 0.0215  # m
g = 9.81  # m/s²

# Cálculo del torque: τ = 2mgr
tau = 2 * m * g * r  # N·m

# Regresión lineal τ vs θ
slope, intercept, r_value, p_value, std_err = linregress(theta, tau)

# Valor teórico
kappa_teo = 0.058  # N·m/rad

# Gráfica
plt.figure(figsize=(8,5))
plt.plot(theta, tau, 'o', label="Datos experimentales")
plt.plot(theta, slope * theta + intercept, '-', label=f'Ajuste lineal: κ = {slope:.4f} N·m/rad')
plt.xlabel("Δθ (rad)")
plt.ylabel("Torque (N·m)")
plt.title("Torque vs Desplazamiento Angular")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Resultados
print("=== ACTIVIDAD 1: Torque mecánico y constante de torsión ===")
print(f"Constante de torsión experimental (κ): {slope:.4f} N·m/rad")
print(f"Error estándar de κ: {std_err:.4f} N·m/rad")
print(f"Valor teórico de κ: {kappa_teo:.4f} N·m/rad")
print(f"Diferencia porcentual: {abs((slope - kappa_teo) / kappa_teo) * 100:.2f}%")

##########Actividad 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === Parte 1: Datos de tiempos y análisis experimental ===

# Datos experimentales
N = np.array([0, 1, 2, 3, 4])  # número de cuadrantes de latón
T = np.array([1.16, 1.24, 1.28, 1.40, 1.44])  # segundos
T2 = T**2  # T^2 en s²

# Constante de torsión (de actividad 1)
kappa = 0.058  # N·m/rad

# Regresión lineal T² vs N
slope, intercept, r_value, p_value, std_err = linregress(N, T2)

# Calcular momento de inercia inicial I0 y delta_I por cuadrante
I0_exp = (intercept * kappa) / (4 * np.pi**2)
delta_I_exp = (slope * kappa) / (4 * np.pi**2)

# === Parte 2: Cálculo teórico de ΔI para un cuadrante de latón ===

# Datos geométricos y físicos
R1 = 0.0206  # m (radio interno)
R2 = 0.0463  # m (radio externo)
e = 0.00261  # m (espesor del latón)
rho = 8500   # kg/m³ (densidad del latón)

# Volumen de un cuadrante de anillo
V = 0.25 * np.pi * (R2**2 - R1**2) * e  # m³

# Masa del cuadrante
M = rho * V  # kg

# Momento de inercia teórico del cuadrante: ΔI = (1/8) M (R1² + R2²)
delta_I_teo = (1/8) * M * (R1**2 + R2**2)

# === Parte 3: Visualización y reporte ===

# Gráfica T² vs N
plt.figure(figsize=(8,5))
plt.plot(N, T2, 'o', label="Datos experimentales")
plt.plot(N, slope*N + intercept, '-', label=f'Ajuste: T² = {slope:.3f}N + {intercept:.3f}')
plt.xlabel("Número de cuadrantes de latón (N)")
plt.ylabel("T² (s²)")
plt.title("Actividad 2: T² vs número de cuadrantes de latón")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Resultados
print("=== ACTIVIDAD 2: Análisis completo ===")
print(f"[REGRESIÓN] Pendiente de T² vs N: {slope:.4f} s²/cuadrante")
print(f"[REGRESIÓN] Intercepto: {intercept:.4f} s²")
print(f"[EXPERIMENTAL] Momento de inercia inicial I₀: {I0_exp:.6e} kg·m²")
print(f"[EXPERIMENTAL] ΔI por cuadrante (exp): {delta_I_exp:.6e} kg·m²")
print()
print("Cálculos geométricos del cuadrante:")
print(f"[GEOMETRÍA] Volumen del cuadrante: {V:.6e} m³")
print(f"[GEOMETRÍA] Masa del cuadrante: {M:.6f} kg")
print(f"[TEÓRICO] ΔI por cuadrante (teo): {delta_I_teo:.6e} kg·m²")
print()
diff_percent = abs((delta_I_exp - delta_I_teo)/delta_I_teo)*100
print(f"Diferencia porcentual entre ΔI_exp y ΔI_teo: {diff_percent:.2f}%")

############Actividad 3:
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === Parte 1: Datos experimentales ===
corriente = np.array([
    -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
     0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1
])
theta = np.array([
    2.42, 2.46, 2.50, 2.56, 2.62, 2.66, 2.72, 2.80, 2.86, 2.94,
    3.00, 3.08, 3.14, 3.20, 3.28, 3.32, 3.38, 3.44, 3.48, 3.52, 3.58
])  # posiciones de equilibrio

# Convertimos a desplazamiento angular desde el equilibrio central (theta_0 = 3.00)
delta_theta = theta - 3.00  # radianes relativos

# === Parte 2: Ajuste lineal ===
slope, intercept, r_value, p_value, std_err = linregress(corriente, delta_theta)

# === Parte 3: Cálculo del momento magnético ===
k = 3.22  # m·T/A
kappa = 0.058  # N·m/rad

# m = mu * k / kappa -> mu = m * kappa / k
mu = slope * kappa / k  # A·m²

# === Parte 4: Gráfica ===
plt.figure(figsize=(8,5))
plt.plot(corriente, delta_theta, 'o', label="Datos experimentales")
plt.plot(corriente, slope * corriente + intercept, '-', label=f"Ajuste lineal: Δθ = {slope:.4f}·i + {intercept:.4f}")
plt.xlabel("Corriente (A)")
plt.ylabel("Δθ (rad)")
plt.title("Actividad 3: Δθ vs Corriente")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Resultados ===
print("=== ACTIVIDAD 3: Torque magnético y momento magnético ===")
print(f"Pendiente Δθ vs i: {slope:.6f} rad/A")
print(f"Intercepto: {intercept:.6f} rad")
print(f"Momento magnético experimental μ: {mu:.4f} A·m²")
print("Valor teórico esperado: μ_teo = 13.5 A·m²")
print(f"Diferencia porcentual: {abs((mu - 13.5)/13.5)*100:.2f}%")

######## Actividad 4:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar archivo
archivo = "C://Users//ekkol//OneDrive//Documentos//Actividad 4.xlsx"
df = pd.read_excel(archivo)

# Extraer columnas
tiempos = [df['Tiempo'].dropna(), df['Tiempo.1'].dropna(), df['Tiempo.2'].dropna()]
amplitudes = [df['Amplitud'].dropna(), df['Amplitud.1'].dropna(), df['Amplitud.2'].dropna()]
titulos = ["Toma 1", "Toma 2", "Toma 3"]

def calcular_Q(t, A, titulo):
    A = np.abs(A)
    t = np.array(t)
    A = np.array(A)

    # Encuentra máximos locales (picos)
    from scipy.signal import find_peaks
    idx_peaks, _ = find_peaks(A)
    t_peaks = t[idx_peaks]
    A_peaks = A[idx_peaks]

    # Usamos primer y último pico para estimar Q
    if len(A_peaks) < 2:
        return titulo, None, None, None

    A0 = A_peaks[0]
    An = A_peaks[-1]
    n = len(A_peaks) - 1
    Q = np.pi * n / np.log(A0 / An)

    # Gráfica
    plt.plot(t, A, label="Amplitud total")
    plt.plot(t_peaks, A_peaks, "ro", label="Picos")
    plt.title(f"{titulo} - Estimación de Q")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (V)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return titulo, Q, A0, An

# Calcular Q para cada toma
resultados = [calcular_Q(tiempos[i], amplitudes[i], titulos[i]) for i in range(3)]

# Imprimir resultados
print("=== ACTIVIDAD 4: Factores de Calidad Q ===")
for titulo, Q, A0, An in resultados:
    if Q is not None:
        print(f"{titulo}: Q ≈ {Q:.2f} (A0 = {A0:.2f}, An = {An:.2f})")
    else:
        print(f"{titulo}: No se encontraron suficientes picos para calcular Q.")


##Actividad 5

