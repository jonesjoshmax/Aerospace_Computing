import sympy as sp
import numpy as np

a = .05
b = .01
r = .015

x_bar = 0.0206391704
y_bar = 0.064097926
x_1 = abs(x_bar - (b / 2))
x_2 = abs(x_bar - (a / 2))
x_3 = abs(x_bar - (a / 2))
y_1 = abs(y_bar - (b / 2))
y_2 = abs(y_bar - (a / 2))
y_3 = abs(y_bar - (a / 2))


def i_rec(base, height):
    return (base * (height ** 3)) / 12


def i_circ(radius):
    return (np.pi * (radius ** 4)) / 4


i_x1 = i_rec(b, a) + (a * b * (x_1 ** 2))
i_x2 = i_rec(a, a) + (a * a * (x_2 ** 2))
i_x3 = i_circ(r) + ((np.pi * (r ** 2)) * (x_3 ** 2))
i_y1 = i_rec(a, b) + (a * b * (y_1 ** 2))
i_y2 = i_rec(a, a) + (a * a * (y_2 ** 2))
i_y3 = i_circ(r) + ((np.pi * (r ** 2)) * (y_3 ** 2))
i_xy1 = a * b * x_1 * y_1
i_xy2 = a * a * x_2 * y_2
i_xy3 = np.pi * (r ** 2) * x_3 * y_3


i_x = i_x1 + i_x2 - i_x3
i_y = i_y1 + i_y2 - i_y3
i_xy = i_xy1 + i_xy2 - i_xy3
i = np.array([[i_x, i_xy], [i_xy, i_y]])
print(i)
