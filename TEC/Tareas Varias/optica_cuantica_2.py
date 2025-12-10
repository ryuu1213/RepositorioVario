import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from math import factorial
from matplotlib import animation

def psi_modsq_truncated(q, t, r, phi, w, mmax):
    q = np.asarray(q, dtype=float)
    pref = (np.pi ** (-0.25)) / np.sqrt(np.cosh(r)) * np.exp(-q**2 / 2.0)
    z = np.exp(1j * (phi - 2.0 * w * t)) * np.tanh(r) / 4.0
    series = np.zeros_like(q, dtype=np.complex128)
    for m in range(0, mmax + 1):
        coeffs = [0]*(2*m) + [1]           # H_{2m}(q)
        H_2m = hermval(q, coeffs)          # Hermite (físicos)
        series += ((-1)**m / factorial(m)) * (z**m) * H_2m
    psi = pref * series
    return np.abs(psi)**2

#Parámetros
r, phi, w, mmax = 0.9, 0.0, 1.0, 25
q = np.linspace(-4, 4, 1200)
T = 2*np.pi / w


probe = np.linspace(0, T, 80, endpoint=False)
ymax = max(psi_modsq_truncated(q, t, r=r, phi=phi, w=w, mmax=mmax).max() for t in probe) * 1.05

#GIF de keyframes
T_breath = np.pi / w  # periodo del breathing (2ω)
t_min       = (phi + 0.0*np.pi)  / (2*w)
t_quarter   = (phi + 0.5*np.pi)  / (2*w)
t_max       = (phi + 1.0*np.pi)  / (2*w)
t_3quarter  = (phi + 1.5*np.pi)  / (2*w)
t_next_min  = t_min + T_breath
key_times  = [t_min, t_quarter, t_max, t_3quarter, t_next_min]
key_labels = ["min width", "quarter", "max width", "three-quarter", "next min"]

fig, ax = plt.subplots(figsize=(7,4))
(line,) = ax.plot([], [], lw=2)
ax.set_xlim(q.min(), q.max()); ax.set_ylim(0, ymax)
ax.set_xlabel("q"); ax.set_ylabel(r"$|\psi(q,t)|^2$  (trunc., $m\leq 5$)")
title = ax.set_title("Squeezed breathing — keyframes")

def init_key():
    line.set_data([], []); return line,

def update_key(i):
    t = key_times[i % len(key_times)]
    lab = key_labels[i % len(key_labels)]
    y = psi_modsq_truncated(q, t, r=r, phi=phi, w=w, mmax=mmax)
    line.set_data(q, y)
    title.set_text(f"{lab} (t = {t:.3f})")
    return line, title

ani_key = animation.FuncAnimation(fig, update_key, frames=len(key_times),
                                  init_func=init_key, blit=True, interval=800)
ani_key.save("squeezed_breathing_keyframes_v2.gif",
             writer=animation.PillowWriter(fps=1.25))
plt.close(fig)

#GIF de animación completa
fig2, ax2 = plt.subplots(figsize=(7,4))
(line2,) = ax2.plot([], [], lw=2)
ax2.set_xlim(q.min(), q.max()); ax2.set_ylim(0, ymax)
ax2.set_xlabel("q"); ax2.set_ylabel(r"$|\psi(q,t)|^2$  (trunc., $m\leq 5$)")
title2 = ax2.set_title("Squeezed vacuum breathing: full cycle")

frames = 180  
def init_full():
    line2.set_data([], []); return line2,

def update_full(i):
    t = i * T / frames
    y = psi_modsq_truncated(q, t, r=r, phi=phi, w=w, mmax=mmax)
    line2.set_data(q, y)
    title2.set_text(f"Squeezed vacuum breathing: t = {t:.3f}")
    return line2, title2

ani_full = animation.FuncAnimation(fig2, update_full, frames=frames,
                                   init_func=init_full, blit=True, interval=40)
ani_full.save("squeezed_breathing_full.gif",
              writer=animation.PillowWriter(fps=24))
plt.close(fig2)

#-------------------------------------------ESTADO COHERENTE GENERAL---------


def hermite_phys(n, x):
    coeffs = [0]*n + [1]
    return hermval(x, coeffs)

def psi_modsq_squeezed_coherent_correct(q, t, alpha=0.8+0.3j, r=0.9, theta=0.3, w=1.0, N=20):
    """
    |ψ(q,t)|^2 para D(α)S(ξ)|0>, ξ=re^{iθ}.
    ψ ∝ (1/√cosh r) e^{-q^2/2}
         * exp[-|α|^2/2 - (α*^2 e^{iθ} tanh r)/2]
         * Σ_{n=0}^N [((1/2) e^{i(θ-2ωt)} tanh r)^n / n!] H_n(q) H_n(χ),
    con χ = γ / sqrt(e^{iθ} sinh(2r)),  γ = α cosh r + α* e^{iθ} sinh r.
    """
    q = np.asarray(q, dtype=float)
    theta_t = theta - 2.0*w*t
    pref = (np.pi**(-0.25))/np.sqrt(np.cosh(r)) * np.exp(-q**2/2.0)
    G = np.exp(-0.5*np.abs(alpha)**2 - 0.5*(np.conj(alpha)**2)*np.exp(1j*theta)*np.tanh(r))
    gamma = alpha*np.cosh(r) + np.conj(alpha)*np.exp(1j*theta)*np.sinh(r)
    chi = gamma/np.sqrt(np.exp(1j*theta)*np.sinh(2.0*r)+1e-16)
    K = 0.5*np.exp(1j*theta_t)*np.tanh(r)

    series = np.zeros_like(q, dtype=np.complex128)
    for n in range(0, N+1):
        series += (K**n/factorial(n)) * hermite_phys(n, q) * hermite_phys(n, chi)

    psi = pref * G * series
    return np.abs(psi)**2

# Parámetros
alpha = 0.8+0.3j; r=0.9; theta=0.3; w=1.0; N=20
q = np.linspace(-4.5, 4.5, 1400); T = 2*np.pi/w

# Tiempos clave (breathing a 2ω ⇒ periodo π/w)
T_breath = np.pi/w
t_min, t_quarter, t_max, t_3quarter = (theta+0*np.pi)/(2*w), (theta+0.5*np.pi)/(2*w), (theta+1*np.pi)/(2*w), (theta+1.5*np.pi)/(2*w)
t_next_min = t_min + T_breath
key_times = [t_min, t_quarter, t_max, t_3quarter, t_next_min]
key_labels = ["min width_COHERENT","quarter_COHERENT","max width_COHERENT","three-quarter_COHERENT","next min_COHERENT"]

# Escala vertical común
probe = np.linspace(0, T, 80, endpoint=False)
ymax = max(psi_modsq_squeezed_coherent_correct(q,t,alpha,r,theta,w,N).max() for t in probe)*1.05

# Frames clave
for t, lab in zip(key_times, key_labels):
    y = psi_modsq_squeezed_coherent_correct(q,t,alpha,r,theta,w,N)
    plt.figure(figsize=(7,4)); plt.plot(q,y,lw=2)
    plt.xlim(q.min(),q.max()); plt.ylim(0,ymax)
    plt.xlabel("q"); plt.ylabel(r"$|\psi(q,t)|^2$  (trunc., $N\leq 5$)")
    plt.title(f"Squeezed coherent (corrected) — {lab} (t = {t:.3f})")
    plt.tight_layout(); plt.savefig(f"corr_key_{lab.replace(' ','_')}.png", dpi=140); plt.close()

# GIF completo
fig, ax = plt.subplots(figsize=(7,4))
(line,) = ax.plot([], [], lw=2)
ax.set_xlim(q.min(), q.max()); ax.set_ylim(0, ymax)
ax.set_xlabel("q"); ax.set_ylabel(r"$|\psi(q,t)|^2$  (trunc., $N\leq 5$)")
title = ax.set_title("Corrected squeezed coherent: full cycle")
frames = 180
def init(): line.set_data([], []); return line,
def update(i):
    t = i*T/frames
    y = psi_modsq_squeezed_coherent_correct(q,t,alpha,r,theta,w,N)
    line.set_data(q, y); title.set_text(f"Corrected squeezed coherent: t = {t:.3f}")
    return line, title
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=40)
ani.save("squeezed_coherent_full_correct.gif", writer=animation.PillowWriter(fps=24))
plt.close(fig)
