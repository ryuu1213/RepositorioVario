import numpy as np
import matplotlib.pyplot as plt


N = 300  
x = np.linspace(0, 2*np.pi, N)
y = np.linspace(0, 2*np.pi, N)
X, Y = np.meshgrid(x, y)



C = 2 * np.sin(X + Y)**2 - (np.sin(2*X) * np.sin(2*Y))



A = (1 + (np.sin(2*X) * np.sin(2*Y)/2) - np.sin(X + Y)**2) 
A = np.clip(A, 0, None)
sqrtA = np.sqrt(A)

lambda1 = (1 + sqrtA)**2
lambda2 = (1 - sqrtA)**2

den = lambda1 + lambda2
p1 = lambda1 / den
p2 = lambda2 / den

eps = 1e-12
p1_safe = np.clip(p1, eps, 1)
p2_safe = np.clip(p2, eps, 1)

S = - p1_safe * np.log2(p1_safe) - p2_safe * np.log2(p2_safe)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#Concurrencia
cont1 = axes[0].contourf(X, Y, C, levels=50, cmap='viridis')
axes[0].set_title("Concurrencia C(x,y)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(cont1, ax=axes[0])

# Líneas verticales multiplos de pi
for k in range(3):
    axes[0].axvline(k*np.pi, color='white', linestyle='--', linewidth=1.2)

# Líneas horizontales  multiplos de pi
for k in range(3):
    axes[0].axhline(k*np.pi, color='white', linestyle='--', linewidth=1.2)


#Entropía 
cont2 = axes[1].contourf(X, Y, S, levels=50, cmap='inferno')
axes[1].set_title("Entropía de von Neumann S(x,y)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(cont2, ax=axes[1])

# Líneas verticales en x = 0, π, 2π
for k in range(3):
    axes[1].axvline(k*np.pi, color='white', linestyle='--', linewidth=1.2)

# Líneas horizontales en y = 0, π, 2π
for k in range(3):
    axes[1].axhline(k*np.pi, color='white', linestyle='--', linewidth=1.2)

plt.tight_layout()
plt.show()


