import numpy as np
import compFunc as cf
import sympy as sp
from scipy.optimize import fsolve


# T B M Input
def tbm_input():
    # Will index x as [theta, beta, mach]
    x = np.zeros(3, float)
    x_i = input('Are you solving for Theta (T), Beta (B), or Mach (M): ').lower()
    y = input('Degrees (D) or Radians(R): ').lower()
    if x_i == 't':
        x[0] = None
        x[1] = eval(input('Beta Value: '))
        x[2] = eval(input('Mach Value: '))
    elif x_i == 'b':
        x[1] = None
        x[0] = eval(input('Theta Value: '))
        x[2] = eval(input('Mach Value: '))
    elif x_i == 'm':
        x[2] = None
        x[0] = eval(input('Theta Value: '))
        x[1] = eval(input('Beta Value: '))
    else:
        cf.err('Idiot')
    if y == 'd':
        x[0] = np.radians(x[0])
        x[1] = np.radians(x[1])
    return x


# T B M Solve
def tbm(x):
    t = sp.Symbol('t')
    b = sp.Symbol('b')
    m = sp.Symbol('m')
    expr = (2 * sp.atan(b) * ((sp.Pow(m, 2) * sp.Pow(sp.sin(b), 2) - 1) /
                              (sp.Pow(m, 2) * (1.4 + sp.cos(2 * b)) + 2))) - sp.tan(t)
    if np.isnan(x[0]):
        expr = expr.subs(b, x[1]).subs(m, x[2])
        return sp.solvers.solve(expr, t)
    elif np.isnan(x[1]):
        expr = expr.subs(t, x[0]).subs(m, x[2])
        expr_np = sp.lambdify(b, expr)
        return fsolve(expr_np, 0)
    elif np.isnan(x[2]):
        expr = expr.subs(t, x[0]).subs(b, x[1])
        return sp.solvers.solve(expr, m)
    else:
        cf.err('Error in TBM')


# Problem 3
def prob3():
    # Problem set 4.1 Problem 19

    # Given values used to solve equation
    v = 335
    u = 2510
    m0 = 2800000
    m_dot = 13300
    g = 9.81

    # Defining the function f(t)
    def f(t):
        return u * np.log(m0 / (m0 - (m_dot * t))) - (g * t) - v

    # A little accuracy loop I whipped up
    # Initial values
    a = 0
    b = 10 * 10 ** 3
    dx = 100
    while True:
        # Runs root search until tolerance met
        x1, x2 = cf.rootsearch(f, a, b, dx)
        if x2 - x1 <= 0.0001:
            return x1
        else:
            dx = dx / 10
            a = x1
            b = x2


def prob4():
    # Prob 4
    # Question 3 in set 4.2
    a = np.array([-7992, 6588, -2178, 361, -30, 1], float)
    r = 6

    pr = cf.polyRoots(a)
    pd = cf.deflPoly(a, r)
    return pd, pr
    # Question 11 in set 4.2


print('The time it takes to reach 335 m/s is %.3f seconds.' % prob3())
i, j = prob4()
