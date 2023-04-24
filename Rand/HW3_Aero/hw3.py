import numpy as np
import matplotlib.pyplot as plt

resolution = 1000
sources = np.array([[-2, 1], [-1, 2], [0, 1], [1, -3], [2, -1]])
free = 10
for r in range(sources.shape[0]):
    sources[r, 1] = free * sources[r, 1]

x_size = 4
y_size = 2
x = np.linspace(-x_size, x_size, resolution)
y = np.linspace(-y_size, y_size, resolution)
i, j = np.meshgrid(x, y)
freeflow = free * j
b = 0


def source(pos1, pos2, m1, m2, strength):
    z = strength / (2 * np.pi) * np.arctan2((m2 - pos2), (m1 - pos1))
    return z


for r in range(sources.shape[0]):
    a = source(sources[r, 0], 0, i.copy(), j.copy(), sources[r, 1])
    b = a + b
b = b + freeflow


plt.contour(i, j, b, levels=30, cmap='cool')
plt.title('Stream Function')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.colorbar()
plt.show()
