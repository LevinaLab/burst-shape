"""
Lotka-Volterra model (predator-prey model)

copied from https://en.wikipedia.org/wiki/File:Lotka-Volterra_model_(1.1,_0.4,_0.4,_0.1).svg
"""

import numpy as np
import matplotlib.pyplot as plt

alpha = 1.1   #

beta = 0.4    # prey death rate (0.4)
gamma = 0.4   # predator death rate (0.4)
delta = 0.1   # predator growth rate (0.1)
x0 = 10       # initial prey population (10)
y0 = 10       # initial predator population (10)

dt = 1e-3
t = np.arange(0,40,dt)
x = np.zeros_like(t, dtype=float)
y = np.zeros_like(t, dtype=float)
x[0], y[0] = x0, y0

for i, _ in enumerate(t):
    if i == 0:
        continue
    x[i] = x[i-1] + (alpha * x[i-1] - beta * x[i-1] * y[i-1]) * dt
    y[i] = y[i-1] + (delta * x[i-1] * y[i-1] - gamma * y[i-1]) * dt

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(t, x, label='Prey')
ax.plot(t, y, '--', label='Predator')
ax.grid(True)
ax.legend()
ax.set_ylabel('population')
ax.set_xlabel('time')
fig.show()