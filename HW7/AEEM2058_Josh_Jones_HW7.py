import sympy as sp
import numpy as np


def prob2():
    x = sp.var('x')
    u = sp.var('u')
    u_vals = np.linspace(0, 1, int((1 - 0) / .05 + 1))
    grand = (x ** 4 * sp.exp(x)) / ((sp.exp(x) - 1) ** 2)
    data = np.zeros(np.size(u_vals))
    for i in range(np.size(u_vals)):
        expr = (u ** 3) * sp.integrate(grand, (x, 0, 1 / u))
        data[i] = expr.subs(x, u_vals[i])
    print(data)


def prob3():
    x = sp.var('x')
    expr = sp.log(x) / (x ** 2 - 2 * x + 2)
    print(expr)
    return sp.integrate(expr, (x, 0, np.pi))


print(prob3())
