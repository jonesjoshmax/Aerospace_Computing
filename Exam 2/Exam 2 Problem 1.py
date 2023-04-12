import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.var('x', real=True)

expr = (sp.exp(2 * x) - 1) * sp.sin(3 * sp.pi * x) * sp.cos(4 * sp.pi * x)

x_vals = np.linspace(0, 1, 100)
y_vals = np.zeros(x_vals.size)
for i in range(x_vals.size):
    y_vals[i] = expr.subs(x, x_vals[i])

plt.plot(x_vals, y_vals, color='red')
plt.title('f(x)')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

