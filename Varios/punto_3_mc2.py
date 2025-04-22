import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros del sistema
N = 150  # Tamaño de la malla
J = 0.2  # Energía de interacción
beta = 10  # Inverso de temperatura
iterations_per_frame = 50  # Reducido temporalmente para evitar bloqueos
frames = 500  # Cantidad de frames
total_iterations = 200_000  # Total de iteraciones a completar
spins = np.random.choice([-1, 1], size=(N, N))  # Inicialización aleatoria de espines

def calcular_delta_E(spins, i, j):
    """Calcula la diferencia de energía al invertir un espín (i, j)."""
    N = spins.shape[0]
    s = spins[i, j]
    
    # Condiciones de frontera periódicas
    arriba = spins[(i - 1) % N, j]
    abajo = spins[(i + 1) % N, j]
    izquierda = spins[i, (j - 1) % N]
    derecha = spins[i, (j + 1) % N]
    
    # Energía antes y después de cambiar el espín
    suma_vecinos = arriba + abajo + izquierda + derecha
    dE = 2 * J * s * suma_vecinos  # Diferencia de energía según el hint

    return dE

def metropolis_step(spins, beta):
    """Realiza un solo paso del algoritmo de Metropolis-Hastings."""
    N = spins.shape[0]
    i, j = np.random.randint(0, N, size=2)  # Seleccionar un espín aleatorio
    dE = calcular_delta_E(spins, i, j)

    # Verificación de la transición
    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        spins[i, j] *= -1  # Cambiar el espín
        return True  # Cambio aceptado
    return False  # Cambio rechazado

def update(frame):
    """Función para actualizar la animación."""
    global spins
    aceptados = 0  # Contador de cambios aceptados
    for _ in range(iterations_per_frame):
        if metropolis_step(spins, beta):
            aceptados += 1
    print(f"Frame {frame}: Cambios aceptados = {aceptados}")  # Depuración
    im.set_array(spins)
    return [im]

# Configuración de la animación
fig, ax = plt.subplots()
im = ax.imshow(spins, cmap='gray', animated=True)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Guardar la animación en un video (verifica que ffmpeg esté instalado)
ani.save("3.mp4", writer="ffmpeg", fps=30, dpi=300)

plt.show()
