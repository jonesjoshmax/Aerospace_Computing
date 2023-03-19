import sympy as sp
import numpy as np
naca = ''
while len(naca) != 4:
    naca = str(input('NACA Number: '))

c = 1
m = float(naca[0]) / 100
p = float(naca[1]) / 10
thickness = (10 * float(naca[2])) + float(naca[3])

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

while True:
    a = np.radians(float(input('AOA Degrees: ')))

    cl = 2 * np.pi * (a - al0)
    cm_le = -1 * ((cl / 4) + ((np.pi / 4) * (a1 - a2)))
    cm_c4 = (np.pi / 4) * (a2 - a1)
    x_cp = (c / 4) * (1 + ((np.pi / cl) * (a1 - a2)))

    print('CL, CM, CM4, X CP')
    print('%.5f' % cl)
    print('%.5f' % cm_le)
    print('%.5f' % cl)
    print('%.5f' % cm_le)
