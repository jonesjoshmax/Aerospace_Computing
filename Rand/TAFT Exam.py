import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

c = 1.
m = 5. / 100.
p = .00000001
thickness = 0.

split = np.arccos(-1 * ((2 * p) - 1))

theta = sp.symbols('theta')
x = (c / 2) * (1 - sp.cos(theta))
a = sp.symbols('a')

dzdx_front = (2 * m / (p ** 2)) * (p - x)
dzdx_rear = (2 * m / ((1 - p) ** 2)) * (p - x)


def integ(sub):
    return sp.integrate(dzdx_front * sub, (theta, 0, split)) + sp.integrate(dzdx_rear * sub, (theta, split, np.pi))


al0 = -1 / np.pi * integ(sp.cos(theta) - 1)
a1 = 2 / np.pi * integ(sp.cos(theta))
a2 = 2 / np.pi * integ(sp.cos(2 * theta))

aoa = np.linspace(-10, 10, 21)
cl = np.zeros(aoa.size)
cm_le = np.zeros(aoa.size)

for i in range(len(aoa)):
    cl[i] = 2 * np.pi * (aoa[i] - al0)
    cm_le[i] = -1 * ((cl[i] / 4) + ((np.pi / 4) * (a1 - a2)))
cl = cl * np.pi / 180
cm_le = cm_le * np.pi / 180
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('5% Camber')
ax1.set_title('Lift Coefficient')
ax1.set_xlabel('AOA')
ax1.set_ylabel('cL')
ax1.plot(aoa, cl, color='red', label='cL')
ax1.legend()
ax1.grid()
ax2.set_title('Moment Coefficient')
ax2.set_xlabel('AOA')
ax2.set_ylabel('cM')
ax2.plot(aoa, cm_le, color='red', label='cM Leading Edge')
ax2.legend()
ax2.grid()
fig.tight_layout()
plt.show()
