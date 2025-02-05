import numpy as np
import matplotlib.pyplot as plt

"""
    1. Para evitar hacer overflow en algunos casos, no se escogio una unidad por default pues cambiaron mucho
    entre grpaficas
    
    2. Seguir las siguientes indicaciones:
        Para T = 5800 K, usar x = np.linspace(10, 4000, 100000), e I está en el orden de 10^12
        Para T = 120000 K, usar x = np.linspace(10, 100, 100000), e I está en el orden de 10^5
        Para T = 100 K, usar x = np.linspace(10, 150000, 100000), e I está en el orden de 10^21
"""
Orden = 21
def funcion_planck( l, T):
    lt = l*T
    a = 23373.45
    b = 143.85*(10**5)
    I = (a/(l**5))*(1/(np.exp(b/lt)-1))*(10**Orden) #Se puede multiplicar por un buen Factor.
    return I

def wien(T):
    x_m = 143.85*(10**5)/(4.97*T)
    return x_m

#Modifica manualmente la temperatura y el espaciado
T = 100
x = np.linspace(10, 150000, 100000)
y = funcion_planck(x,  T)

x_m = wien(T)
y_m = funcion_planck(x_m, T)

fig, ax = plt.subplots()
ax.plot(x, y, label='Funcion de Planck')
ax.plot()

ax.plot(x_m, y_m, 'ro', label=f'Lambda máximo ({x_m}, {y_m:.2f})')

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel(f'Intensity (10^{Orden} eV*nm*s^-1)')
ax.set_title(f'Ley de Planck Para T={T}')
ax.legend()
plt.show()


