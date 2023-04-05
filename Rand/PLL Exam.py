import numpy as np
import matplotlib.pyplot as plt

# Aspect Ratio Max
n = 10
# AOA of Air Foil
aoa = 5
# AOA where CL = 0
aoa_ZeroLift = 0
# CD Where CL = 0
cdo = 0


def gaussElimin(a, b):
    n = len(b)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                b[i] = b[i] - lam * b[k]
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b


def PLL(n):
    if n >= 4:
        x = np.zeros([n, n])
        y = np.ones(n)
        theta = np.linspace(0, np.pi, n)
    else:
        x = np.zeros([5, 5])
        y = np.ones(5)
        theta = np.linspace(0, np.pi, 5)
    for i in range(x.shape[1]):
        x[0, i] = (i + 1) ** 2
        x[x.shape[0] - 1, i] = np.power(-1, i + 2) * (i + 1) ** 2
    for i in range(1, x.shape[0] - 1):
        for j in range(x.shape[1]):
            x[i, j] = ((2 * n) / np.pi + (j + 1) / np.sin(theta[i])) * np.sin((j + 1) * theta[i])
    y = y * (aoa - aoa_ZeroLift) * np.pi / 180
    return gaussElimin(x, y)[0]


cL = np.zeros(n)
cD = np.zeros(cL.size)
ar = np.linspace(1, n, n)
for r in range(cL.size):
    cL[r] = PLL(r + 1) * np.pi * ar[r]
    cD[r] = cL[r] ** 2 / (np.pi * 0.95 * ar[r])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Josh Jones')
ax1.set_title('Lift Coefficient')
ax1.set_xlabel('Aspect Ratio')
ax1.set_ylabel('cL')
ax1.plot(ar, cL, 'o', color='red', label='cL')
ax1.legend()
ax1.grid()
ax2.set_title('Drag Coefficient')
ax2.set_xlabel('Aspect Ratio')
ax2.set_ylabel('cD')
ax2.plot(ar, cD, 'o', color='red', label='cD')
ax2.legend()
ax2.grid()
fig.tight_layout()
plt.show()
