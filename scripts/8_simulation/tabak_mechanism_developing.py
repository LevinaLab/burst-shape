"""
Simulation like Tabak, 2010

https://journals.physiology.org/doi/full/10.1152/jn.00857.2009
"""

import numpy as np
import matplotlib.pyplot as plt

w = 0.8         # Connectivity
theta_0 = 0.17  # Input for half activation
k_a = 0.05      # Spread of a_inf
theta_s = 0.2   # Activity for half depression
k_s = 0.05      # Spread of s_inf
tau_s = 250     # unclear but should be slow
a_inf = lambda z: 1 / (1 + np.exp(-z / k_a))
s_inf = lambda a_: 1 / (1 + np.exp((a_ - theta_s) / k_s))
noise_amp = 0.01

a0 = .4
s0 = .4


dt = 1
t = np.arange(0,4000,dt)
a = np.zeros_like(t, dtype=float)
s = np.zeros_like(t, dtype=float)
a[0], s[0] = a0, s0
noise = np.random.normal(scale=noise_amp, size=t.shape)

a_diff = lambda a_old, s_old: -a_old + a_inf(w * s_old * a_old - theta_0)
s_diff = lambda a_old, s_old: (1/tau_s) * (-s_old + s_inf(a_old))

for i, _ in enumerate(t):
    if i == 0:
        continue
    a[i] = a[i-1] + dt * a_diff(a[i-1], s[i-1]) + noise[i]
    s[i] = s[i-1] + dt * s_diff(a[i-1], s[i-1])

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(t, a, label='Activity')
ax.plot(t, s, '--', label='Adaptation')
ax.grid(True)
ax.legend()
ax.grid(True)
ax.legend()
ax.set_ylabel('value')
ax.set_xlabel('time')
fig.show()

# %% plot flow field
# choose grid around the simulated trajectory with a margin
a_margin = 0.05
s_margin = 0.05
a_min, a_max = max(0, a.min() - a_margin), a.max() + a_margin
s_min, s_max = max(0, s.min() - s_margin), s.max() + s_margin

A_vals = np.linspace(a_min, a_max, 25)
S_vals = np.linspace(s_min, s_max, 25)
A, S = np.meshgrid(A_vals, S_vals)

# evaluate vector field on the grid
dA = a_diff(A, S)
dS = s_diff(A, S)

# normalize for plotting arrows (keep magnitude for coloring)
magnitude = np.sqrt(dA**2 + dS**2)
eps = 1e-8
U = dA / (magnitude + eps)
V = dS / (magnitude + eps)

fig2, ax2 = plt.subplots(figsize=(6,6))
q = ax2.quiver(A, S, U, V, magnitude, cmap='plasma', scale=30, pivot='mid', width=0.005)
cb = fig2.colorbar(q, ax=ax2)
cb.set_label('vector speed (unnormalized)')

# nullclines: contours where dA==0 and dS==0
ax2.contour(A, S, dA, levels=[0], colors='C0', linewidths=2, linestyles='--')
ax2.contour(A, S, dS, levels=[0], colors='C1', linewidths=2, linestyles='-.')

# overlay trajectory from the simulation
ax2.plot(a, s, color='k', lw=1, label='trajectory')
ax2.scatter(a[0], s[0], color='green', label='start')
ax2.scatter(a[-1], s[-1], color='red', label='end')

ax2.set_xlabel('a (activity)')
ax2.set_ylabel('s (adaptation)')
ax2.set_title('Phase plane and flow field')
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1.15))
ax2.grid(True)
plt.show()