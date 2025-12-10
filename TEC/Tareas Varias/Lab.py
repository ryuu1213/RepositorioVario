import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Función para cargar imagen y convertir a escala de grises
def load_gray(path):
    img = mpimg.imread(path).astype(float)
    if img.ndim == 3:  # Si es RGB
        img = img[..., :3]  # tomar solo canales R,G,B (ignorar alpha si existe)
        img = 0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]
    return img

# 1. Cargar imágenes de intensidades
I_H   = load_gray("TEC\\datos_lab\\H1.JPG")
I_V   = load_gray("TEC\\datos_lab\\V1.JPG")
I_45  = load_gray("TEC\\datos_lab\\D1.JPG")
I_135 = load_gray("TEC\\datos_lab\\A1.JPG")
I_R   = load_gray("TEC\\datos_lab\\Der1.JPG")
I_L   = load_gray("TEC\\datos_lab\\Izq1.JPG")

# 2. Calcular parámetros de Stokes en cada píxel
S0 = I_H + I_V
S1 = I_H - I_V
S2 = I_45 - I_135
S3 = I_R - I_L

# 3. Calcular parámetros derivados

# Crear máscara para región de interés (haz)
radio = S0 > np.max(S0) * 0.15  # Umbral de intensidad para definir el haz

# Calcular theta, chi y DoP_local
theta = 0.5 * np.arctan2(S2, S1)                  # Orientación [rad]
chi = 0.5 * np.arctan2(S3, np.sqrt(S1**2 + S2**2))  # Elipticidad
DoP_local = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)  # Grado de polarización local

#Mascara aplicada
theta_masked = np.where(radio, theta, 0)
DoP_local_masked = np.where(radio, DoP_local, 0)

# Cálculo de promedios globales (solo dentro de la región de interés)
S0_mean = np.mean(S0[radio])
S1_mean = np.mean(S1[radio])
S2_mean = np.mean(S2[radio])
S3_mean = np.mean(S3[radio])

dop_global = np.sqrt(S1_mean**2 + S2_mean**2 + S3_mean**2) / (S0_mean + 1e-12)
print(f"El grado de polarización global del haz es: {dop_global:.3f}")

# Centro del haz
y_indices, x_indices = np.nonzero(radio)
cy, cx = np.mean(y_indices), np.mean(x_indices)

# Radio efectivo del haz
max_radius = np.max(np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2))
r_circ = 0.5 * max_radius  # círculo a la mitad del radio

# Ángulos de muestreo
angles = np.deg2rad(np.arange(0, 360, 22.5))

# --- Visualización básica ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
im = axs[0,0].imshow(S0, cmap="inferno"); axs[0,0].set_title("S0 (Intensidad total)"); plt.colorbar(im, ax=axs[0,0])
im = axs[0,1].imshow(S1, cmap="inferno"); axs[0,1].set_title("S1 (H - V)"); plt.colorbar(im, ax=axs[0,1])
im = axs[1,0].imshow(S2, cmap="inferno"); axs[1,0].set_title("S2 (45° - 135°)"); plt.colorbar(im, ax=axs[1,0])
im = axs[1,1].imshow(S3, cmap="inferno"); axs[1,1].set_title("S3 (R - L)"); plt.colorbar(im, ax=axs[1,1])

for ax in axs.flat:
    ax.axis("image")
plt.tight_layout()
plt.savefig("TEC\\Imagenes_LAB\\parametros.png", dpi=300)
plt.close()

# --- Elipses representativas con fórmulas exactas ---
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(DoP_local_masked, cmap="inferno", vmin=0, vmax=1)   # Fondo con DoP
ax.set_title("Elipses de polarización (Stokes)")
ax.axis("image")

# Agregar colorbar asociada al fondo
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Grado de polarización local (DoP)")

t = np.linspace(0, 2*np.pi, 200)

for ang in angles:
    # Posición en círculo
    x = int(cx + r_circ * np.cos(ang))
    y = int(cy + r_circ * np.sin(ang))
    
    # Parámetros de Stokes en el punto
    Ip = S0[y, x]
    Q  = S1[y, x]
    U  = S2[y, x]
    V  = S3[y, x]
    
    # Cálculo de L y magnitudes
    L = Q + 1j*U
    modL = np.abs(L)
    
    A = np.sqrt(0.5*(Ip + modL))
    B = np.sqrt(0.5*(Ip - modL))
    th = 0.5 * np.angle(L)
    h = np.sign(V) if V != 0 else 1
    
    # Parametrización de la elipse
    X = A*np.cos(t)*np.cos(th) - B*np.sin(t)*np.sin(th)
    Y = A*np.cos(t)*np.sin(th) + B*np.sin(t)*np.cos(th)
    
    # Escala para visualización
    scale = 5
    ax.plot(x + scale*X, y + h*scale*Y, color="blue", lw=1.2)

plt.savefig("TEC\\Imagenes_LAB\\elipses.png", dpi=300)
plt.close()

# --- Histograma del grado de polarización local ---
# Tomamos solo los píxeles dentro de la máscara
dop_values = DoP_local[radio]

# Definir los bins (0, 0.1, 0.2, ..., 1.0)
bins = np.arange(0, 1.1, 0.1)

fig, ax = plt.subplots(figsize=(7,5))
counts, _, _ = ax.hist(dop_values, bins=bins, color="blue", edgecolor="black", alpha=0.7)

ax.set_title("Histograma del grado de polarización local")
ax.set_xlabel("DoP local")
ax.set_ylabel("Número de píxeles")

# Mostrar valores sobre las barras
for i, c in enumerate(counts):
    ax.text(bins[i] + 0.05, c + 5, str(int(c)), ha="center", va="bottom", fontsize=8)

plt.savefig("TEC\\Imagenes_LAB\\histograma_dop.png", dpi=300)


