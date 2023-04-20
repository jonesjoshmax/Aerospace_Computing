import numpy as np
import matplotlib.pyplot as plt

a = .05
t = .003
L = 1.2
E = 280 * 1000000000
To = 75
sig = 200 * 1000000
alpha = 12 * (10 ** -6)
A = (a * t) + (t * (a - t))
a1 = a * t
a2 = (a - t) * t
zbar = (a1 * (a / 2) + a2 * (a / 2)) / (a1 + a2)
ybar = (a1 * (t / 2) + a2 * (((a - t) / 2) + t)) / (a1 + a2)
I = ((a * (t ** 3)) / 12) + (a1 * ((ybar - (t / 2)) ** 2)) + \
    ((t * (a - t) ** 3) / 12) + (a2 * ((((a - t) / 2) + t - ybar) ** 2))
Tc = To - (To * (ybar / .05))
a_mat = np.array([[0, 0, 1, 0, 0, 0],
                  [0, L, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [L / (E * A), 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, (L ** 3) / (6 * E * I), (L ** 2) / (2 * E * I), 0, L, 0]])
b_mat = np.array([[0],
                  [0],
                  [0],
                  [-alpha * Tc * L],
                  [0],
                  [-alpha * (Tc / (2 * (a - ybar))) * (L ** 2)]])
c_mat = np.linalg.solve(a_mat, b_mat)

plt.title('Normal Force Vs Bar Length')
plt.axhline(y=c_mat[0], color='red', xmin=0, xmax=1.2)
plt.xlim(0, 1.2)
plt.xlabel('Distance (m)')
plt.ylabel('Normal Force (N)')
plt.grid()
plt.tight_layout()
plt.show()

x = np.linspace(0, 1.2, 1200)
y = alpha * (Tc / (2 * (a - ybar))) * (x ** 2) + c_mat[4] * x

plt.title('Vertical Displacement Vs Bar Length')
plt.plot(x, y, color='red')
plt.xlim(0, 1.2)
plt.xlabel('Distance (m)')
plt.ylabel('Vertical Displacement (m)')
plt.grid()
plt.tight_layout()
plt.show()

ToMax = (sig / (alpha * E)) / (1 - (ybar / a))
