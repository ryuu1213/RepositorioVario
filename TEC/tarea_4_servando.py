import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parámetros
I0 = 1
beta = 7
alpha = 2

# Definiciones de intensidad
def I1(z):
    return I0 / (1 + beta * I0 * z)

def I2(z):
    return I0 * np.exp(-alpha * z)

# (a) Graficar
z_vals = np.linspace(0, 2, 500)  # rango de z para graficar

# (b) Resolver punto de cruce I1(z) = I2(z)
f = lambda z: I1(z) - I2(z)
z_cruce = fsolve(f, 0.5)[0]  # punto inicial ~0.5
print(f"Distancia donde ambas intensidades son iguales: z = {z_cruce:.4f}")

# (c) Porcentaje respecto a I0
I_cruce = I1(z_cruce)
porcentaje = I_cruce / I0 * 100
print(f"Intensidad en ese punto: I = {I_cruce:.4f} ({porcentaje:.2f} % de I0)")
plt.plot(z_vals, I1(z_vals), label=r"$I_1(z)=\frac{I_0}{1+\beta I_0 z}$ (2 fotones)")
plt.plot(z_vals, I2(z_vals), label=r"$I_2(z)=I_0 e^{-\alpha z}$ (Beer)")
plt.scatter(z_cruce, I1(z_cruce), label= f"(z = {z_cruce:.4f}, I={I1(z_cruce):-4f})")
plt.ylabel("Intensidad normalizada")
plt.title("Decaimiento de la intensidad en ambos materiales")
plt.legend()
plt.grid(True)
plt.show()
