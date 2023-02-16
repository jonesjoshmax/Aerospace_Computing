import numpy as np
import matplotlib.pyplot as plt

# setting number of points in mesh
n = 1000
# dimensions of the streamline plot
x0, x1 = -3, 3
y0, y1 = -3, 3
x = np.linspace(x0, x1, n)
y = np.linspace(y0, y1, n)
# creating mesh for X and Y directions
X, Y = np.meshgrid(x, y)

velo = 10

free_stream = velo * Y


def vortex_stream(mag, xs, ys, X, Y):
    u = + mag / (2 * np.pi) * (Y - ys) / ((X - xs) ** 2 + (Y - ys) ** 2)
    v = - mag / (2 * np.pi) * (X - xs) / ((X - xs) ** 2 + (Y - ys) ** 2)

    return u, v


def vortex_contour(mag, xs, ys, X, Y):
    p = mag / (4 * np.pi) * np.log((X - xs) ** 2 + (Y - ys) ** 2)

    return p


def doublet_stream(mag, xd, yd, X, Y):
    u = - mag / (2 * np.pi) * ((X - xd) ** 2 - (Y - yd) ** 2) / ((X - xd) ** 2 + (Y - yd) ** 2) ** 2
    v = - mag / (2 * np.pi) * 2 * (X - xd) * (Y - yd) / ((X - xd) ** 2 + (Y - yd) ** 2) ** 2

    return u, v


def doublet_contour(mag, xs, ys, X, Y):
    p = - mag / (2 * np.pi) * (Y - ys) / ((X - xs) ** 2 + (Y - ys) ** 2)
    return p


def sink_stream(mag, xs, ys, X, Y):
    u = mag / (2 * np.pi) * np.cos(np.arctan2((Y - ys), (X - xs)))
    v = mag / (2 * np.pi) * np.sin(np.arctan2((Y - ys), (X - xs)))
    return u, v


def sink_contour(mag, xs, ys, X, Y):
    p = mag / (2 * np.pi) * np.arctan2((Y - ys), (X - xs))
    return p


u_dou, v_dou = doublet_stream(velo, 0, 0, X.copy(), Y.copy())
u_vor, v_vor = vortex_stream(velo, 0, 0, X.copy(), Y.copy())
u_sink, v_sink = sink_stream(velo, 0, 0, X.copy(), Y.copy())

p_d = doublet_contour(velo, 0, 0, X, Y)
p_s = sink_contour(velo, 0, 0, X, Y)
p_v = vortex_contour(velo, 0, 0, X, Y)

psi = p_v + p_d + free_stream

u = u_dou + u_vor
v = v_dou + v_vor

fig, ax = plt.subplots()
sf = ax.contour(X, Y, psi, levels=50, cmap='cool')
plt.colorbar(sf, label='Velocity')
ax.set_title('Stream Function')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

size = 10
plt.figure(figsize=(size, (y1 - y0) / (x1 - x0) * size))
plt.grid(True)
plt.title('Stream Function')
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.streamplot(X, Y, u, v, linewidth=1, arrowsize=1, arrowstyle='->')
plt.show()
