"""
Simulation like Vinogradov, 2024

https://doi.org/10.1101/2024.08.21.608974
"""

import numpy as np
import matplotlib.pyplot as plt

tau = 1e3  # or should it be 1?
J = 1
phi = lambda z: 1 / (1 + np.exp(-z))
b = 7 # [0.05, 20]
theta = 4  # [-10, 15]
tau_w = 2e4 # [200, 20000]
sigma = 1.5 # [0.01, 2]
x0 = 0
w0 = 0

a = 5 # unclear
A = 9 # unclear

dt = 5e-2
t = np.arange(0,50000,dt) / 1000
x = np.zeros_like(t, dtype=float)
w = np.zeros_like(t, dtype=float)
x[0], w[0] = x0, w0
noise = np.random.normal(0,sigma,len(t))

for i, _ in enumerate(t):
    if i == 0:
        continue
    # equations in paper
    # τ ˙x(t) = −x(t) + Aϕ[−a(Jx(t) − w(t) + θ)] + σξ(t),
    # τw ˙w(t) = −w(t) + bx(t)

    x[i] = x[i-1] + (dt / tau) * (-x[i-1] + A * phi(-a * (J * x[i-1] - w[i-1] + theta)) + noise[i])
    w[i] = w[i-1] + (dt / tau_w) * (-w[i-1] + b * x[i-1])

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(t, x, label='Activity')
ax.plot(t, w, '--', label='Adaptation')
ax.grid(True)
ax.legend()
ax.grid(True)
ax.legend()
ax.set_ylabel('value')
ax.set_xlabel('time')
fig.show()