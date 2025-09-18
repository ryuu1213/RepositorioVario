import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv

#Valores Necesarios
N = 2**8
lambda0 = 600e-9     
L = 13e-4
w0 = 1e-4
k = (2*np.pi)/lambda0
zrayleigh = (k*w0**2)/2
zmax = 1.5*zrayleigh
foco = 5e-2

#Valores especiales para el punto 3
w0_p3 = [0.5e-4, 1.5e-4, 3e-4]
kt_p3 = 8665
lambda0_p3 = 632.8e-9



x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
z1 = np.linspace(0, zmax, 100)
X, Y = np.meshgrid(x, y)
idx_y0 = N//2

def propagation(U, xs, ys, zs, lambda_):

    U  = np.asarray(U)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)

    dx   = xs[1] - xs[0]
    Nz   = len(zs)
    dz   = zs[1] - zs[0]
    xmax = xs.max()

    # Spatial frequencies
    k    = 2 * np.pi / lambda_
    kmax = np.pi / dx
    grid = np.arange(-N // 2, N // 2)       # -N/2 : N/2-1 (N elements)
    kxs  = kmax * (2.0 / N) * grid
    kys  = kxs.copy()

    # Expanded grids
    Xs  = xs[:, None]        # (N,1)
    Ys  = ys[None, :]        # (1,N)
    KXs = kxs[:, None]       # (N,1)
    KYs = kys[None, :]       # (1,N)

    # Radii and transverse k
    Rs  = np.sqrt(Xs**2 + Ys**2)            # (N,N) via broadcasting
    KTs = np.sqrt(KXs**2 + KYs**2)          # (N,N)

    # Propagators
    PropParaxial = np.exp(-1j * 0.5 * dz * (KTs**2) / k)
    PropFull     = np.exp( 1j * np.sqrt((k**2 - KTs**2) + 0j) * dz)  # unused, kept for parity
    Prop = PropParaxial

    # Allocate and initialize
    Uz = np.zeros((N, N, Nz), dtype=np.complex128)
    Uz[:, :, 0] = U

    # Initial spectrum (centered)
    F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(Uz[:, :, 0])))

    # March over z
    for iz in range(1, Nz):

        # Propagate one dz in k-space
        F = F * Prop

        # Field at this z (spatial domain, centered)
        Uz[:, :, iz] = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(F)))

    return Uz

#2.a)
    #ANALÍTICA---------------------------------------------------------------------------------------------------
def gauss_analitico(X, Y, z_vals, w0, lam): #EVALUA LA EXPRESION EN LOS PUNTOS PEDIDOS
    k  = 2*np.pi/lam
    zR = np.pi*w0**2/lam
    
    U = np.zeros((len(z_vals), *X.shape), dtype=complex)
    
    r2 = X**2 + Y**2
    
    for i, z in enumerate(z_vals):
        w_z = w0 * np.sqrt(1 + (z/zR)**2)
        R_z = np.inf if z == 0 else z * (1 + (zR/z)**2)
        gouy = np.arctan(z/zR)

        amplitude = (w0 / w_z) * np.exp(-r2 / w_z**2)
        phase = np.exp(1j * (k*z - gouy)) * np.exp(1j * k * r2 / (2 * R_z))

        U[i] = amplitude * phase
    
    return U

Uan = gauss_analitico(X, Y, z1, w0, lambda0)
Ian = np.abs(Uan)**2
Ianxz = Ian[:, idx_y0, :]

    #PROPAGADA---------------------------------------------------------------------------------------------------
def gaussian_wave(x, y, w0): #HAZ INICIAL
    return np.exp(-(x**2 + y**2) / w0**2)

U0 = gaussian_wave(X, Y, w0)
Uz = propagation(U0, x, y, z1, lambda0)
Ixz = np.abs(Uz[:, idx_y0, :])**2


#2.b)
    #ANALÍTICA---------------------------------------------------------------------------------------------------
def gauss_lente_analitico(X, Y, w0, lam, f, z_vals):
    k  = 2*np.pi/lam
    r2 = X**2 + Y**2
    q0 = -1j*np.pi*w0**2/lam
    ql = 1.0 / (1.0/q0 - 1.0/f)

    U = np.zeros((len(z_vals), *X.shape), dtype=complex)

    for i, z in enumerate(z_vals):
        qz = ql + z
        G  = 1.0 / (1.0 + z/ql)
        U[i] = G * np.exp(1j * k * r2 / (2.0 * qz))

    return U

Uan2 = gauss_lente_analitico(X, Y, w0, lambda0, foco, z1)
Ian2 = np.abs(Uan2)**2
Ian2xz = Ian2[:, idx_y0, :]

    #NUMERICA--------------------------------------------------------------------------------------------------------
#Utilizamos el funcion del haz gaussiano ya creada en 2.a)

def lente(x, y, f, k):
    return np.exp(-1j * (k/(2*f)) * (x**2 + y**2))

U02 = gaussian_wave(X, Y, w0)*lente(X, Y, foco, k)
Uz2 = propagation(U02, x, y, z1, lambda0)
Ixz2 = np.abs(Uz2[:, idx_y0, :])**2


#3) 
#m=0, w0 = 0.5 mm
    #ANALÍTICA---------------------------------------------------------------------------------------------------
def bessel_gauss_analitico(X, Y, z_vals, w0, lam, kt, m=0):
    k  = 2*np.pi/lam
    zR = np.pi*w0**2/lam
    r  = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    U = np.zeros((len(z_vals), *X.shape), dtype=complex)

    for i, z in enumerate(z_vals):
        w_z = w0 * np.sqrt(1 + (z/zR)**2)
        R_z = np.inf if z == 0 else z * (1 + (zR/z)**2)
        gouy = np.arctan(z/zR)

        gauss = (w0 / w_z) * np.exp(-r**2 / w_z**2)
        phase = np.exp(1j * (k*z - gouy)) * np.exp(1j * k * r**2 / (2*R_z))

        bessel = jv(m, kt * r) * np.exp(1j * m * phi)

        U[i] = gauss * bessel * phase

    return U


#NUMERICA--------------------------------------------------------------------------------------------------------
def bessel_gauss_initial(X, Y, w0, kt, m=0):
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    return np.exp(-(r**2)/w0**2) * jv(m, kt*r) * np.exp(1j*m*phi)

#------------------------------------------------------------------
#Gráficas
#------------------------------------------------------------------
#2.a)------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))

#Gráfica del Analítico
im1 = axs[0].imshow(Ixz, cmap="copper",
              aspect='auto', origin='lower',
              extent=[z1[0]*1e3, z1[-1]*1e3, x[0]*1e3, x[-1]*1e3])
axs[0].set_xlabel("z [mm]")
axs[0].set_ylabel("x [mm]")
axs[0].set_title("Evolución del haz Gaussiano (propagador)")

fig.colorbar(im1, ax=axs[0], label="Intensidad (u.a.)")

#Gráfica del Propagador
im2 =   axs[1].imshow(Ianxz.T, cmap="copper",
              aspect='auto', origin='lower',
              extent=[z1[0]*1e3, z1[-1]*1e3, x[0]*1e3, x[-1]*1e3])
axs[1].set_xlabel("z [mm]")
axs[1].set_ylabel("x [mm]")
axs[1].set_title("Evolución del haz Gaussiano (analítico)")

fig.colorbar(im2, ax=axs[1], label="Intensidad (u.a.)")

plt.tight_layout()
plt.savefig('Tec\\2_a.png', dpi=300)
plt.close()

#2.b)------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))

#Gráfica del Analítico
im1 = axs[0].imshow(Ixz2, cmap="copper",
              aspect='auto', origin='lower',
              extent=[z1[0]*1e3, z1[-1]*1e3, x[0]*1e3, x[-1]*1e3])
axs[0].set_xlabel("z [mm]")
axs[0].set_ylabel("x [mm]")
axs[0].set_title("Evolución del haz Gaussiano con Lente (propagador)")

fig.colorbar(im1, ax=axs[0], label="Intensidad (u.a.)")

#Gráfica del Propagador
im2 =   axs[1].imshow(Ian2xz.T, cmap="copper",
              aspect='auto', origin='lower',
              extent=[z1[0]*1e3, z1[-1]*1e3, x[0]*1e3, x[-1]*1e3])
axs[1].set_xlabel("z [mm]")
axs[1].set_ylabel("x [mm]")
axs[1].set_title("Evolución del haz Gaussiano con Lente (analítico)")

fig.colorbar(im2, ax=axs[1], label="Intensidad (u.a.)")

plt.tight_layout()
plt.savefig('Tec\\2_b.png', dpi=300)
plt.close()
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
#3)
for m in range(0,2):
    for w_0 in w0_p3:
        #Datos------------------------------------
        #Analítica
        Uan3 = bessel_gauss_analitico(X, Y, z1, w_0, lambda0_p3, kt_p3, m)
        Ian3 = np.abs(Uan3)**2
        Ian3xz = Ian3[:, idx_y0, :]
        Ian3xy0 = Ian3[0,:,:]
        Ian3xyf = Ian3[-1,:,:]

        #Propagada
        U03 = bessel_gauss_initial(X, Y, w_0, kt_p3, m)
        Uz3 = propagation(U03, x, y, z1, lambda0_p3)
        I3 = np.abs(Uz3)**2
        Ixz3 = I3[:, idx_y0, :]
        Ixy30 = I3[:,:,0]
        Ixy3f = I3[:,:,-1]

        #Gráficas------------------------------------
        fig, axs = plt.subplots(3, 2, figsize=(10,8))

        #EJE DE PROPAGACION
        #Propagador
        im1 = axs[0,0].imshow(Ixz3, cmap="copper",
                              aspect='auto', origin='lower',
                              extent=[z1[0]*1e3, z1[-1]*1e3, x[0]*1e3, x[-1]*1e3])
        axs[0,0].set_xlabel("z [mm]")
        axs[0,0].set_ylabel("x [mm]")
        axs[0,0].set_title("Evolución (propagador)")
        fig.colorbar(im1, ax=axs[0,0], label="Intensidad (u.a.)")

        # Analítico
        im2 = axs[0,1].imshow(Ian3xz.T, cmap="copper",
                              aspect='auto', origin='lower',
                              extent=[z1[0]*1e3, z1[-1]*1e3, x[0]*1e3, x[-1]*1e3])
        axs[0,1].set_xlabel("z [mm]")
        axs[0,1].set_ylabel("x [mm]")
        axs[0,1].set_title("Evolución (analítico)")
        fig.colorbar(im2, ax=axs[0,1], label="Intensidad (u.a.)")

        #PERFIL INICIAL
        #Propagador
        im3 = axs[1,0].imshow(Ixy30, cmap="copper",
                              aspect='auto', origin='lower',
                              extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
        axs[1,0].set_xlabel("x [mm]")
        axs[1,0].set_ylabel("y [mm]")
        axs[1,0].set_title("Perfil inicial (propagador)")
        fig.colorbar(im3, ax=axs[1,0], label="Intensidad (u.a.)")

        #Analítico
        im4 = axs[1,1].imshow(Ian3xy0, cmap="copper",
                              aspect='auto', origin='lower',
                              extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
        axs[1,1].set_xlabel("x [mm]")
        axs[1,1].set_ylabel("y [mm]")
        axs[1,1].set_title("Perfil inicial (analítico)")
        fig.colorbar(im4, ax=axs[1,1], label="Intensidad (u.a.)")

        #PERFIL FINAL
        #Propagador
        im5 = axs[2,0].imshow(Ixy3f, cmap="copper",
                              aspect='auto', origin='lower',
                              extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
        axs[2,0].set_xlabel("x [mm]")
        axs[2,0].set_ylabel("y [mm]")
        axs[2,0].set_title("Perfil final (propagador)")
        fig.colorbar(im5, ax=axs[2,0], label="Intensidad (u.a.)")

        #Analítico
        im6 = axs[2,1].imshow(Ian3xyf, cmap="copper",
                              aspect='auto', origin='lower',
                              extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
        axs[2,1].set_xlabel("x [mm]")
        axs[2,1].set_ylabel("y [mm]")
        axs[2,1].set_title("Perfil final (analítico)")
        fig.colorbar(im6, ax=axs[2,1], label="Intensidad (u.a.)")

        plt.tight_layout()
        plt.savefig(f'Tec\\3_m_{m}__w0_{w_0}.png', dpi=300)
        plt.close()

