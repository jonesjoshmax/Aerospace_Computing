import numpy as np
import compFunc as cf
import sympy as sp
from scipy.optimize import fsolve


def prob2():

    # Taking inputs for relevant values
    print('Return to skip value (Solving for this value)')
    # Choice to choose working in degrees or radians
    choice = input('Degrees (D) or Radians(R)').lower()
    find = input('Are you solving for Theta (T), Beta(B), or Mach(M):').lower()
    # If statements to take variables in that are known and set unknown to None
    if find == 't':
        theta = None
        beta = float(input('Beta Value: '))
        mach = float(input('Mach Value: '))
    elif find == 'b':
        theta = float(input('Theta Value: '))
        beta = None
        mach = float(input('Mach Value: '))
    elif find == 'm':
        theta = float(input('Theta Value: '))
        beta = float(input('Beta Value: '))
        mach = None
    # Conversion from degrees to radians for solver
    if choice == 'd':
        if mach is None:
            theta = np.radians(theta)
            beta = np.radians(beta)
        elif theta is None:
            beta = np.radians(beta)
        elif beta is None:
            theta = np.radians(theta)
    gamma = float(input('Gamma Value: '))

    # Creating some variables for future sympy expression
    t = sp.symbols('t')
    b = sp.symbols('b')
    m = sp.symbols('m')

    # Expression written out
    expr = (2 * sp.cot(b)) * ((((m ** 2.0) * (sp.sin(b) ** 2.0)) - 1.0) /
                              ((m ** 2.0) * (gamma + sp.cos(2.0 * b)) + 2.0)) - sp.tan(t)

    # If chain to solve for x variable
    # Also subs known values into equation
    if theta is None:
        expr = expr.subs(b, beta).subs(m, mach)
        var = t
    elif beta is None:
        expr = expr.subs(t, theta).subs(m, mach)
        var = b
    elif mach is None:
        expr = expr.subs(t, theta).subs(b, beta)
        var = m
    else:
        return 'error'

    # Taking derivative of the function with substituted variable
    expr_d = sp.diff(expr, var)

    # Converting the expression to a function that can be sent through the root search and newton functions
    f = sp.lambdify(var, expr)
    df = sp.lambdify(var, expr_d)

    # Root search used to get bounds for beta first value so it can be solve twice
    a1, b1 = cf.rootsearch(f, 0.0, np.pi / 2, .01)

    # Solving beta twice as it is a paraboloa
    if beta is None:
        r1 = cf.newtonRaphson(f, df, a1, b1)
        r2 = cf.newtonRaphson(f, df, np.pi / 2, b1)
    elif theta is None:
        r1 = cf.newtonRaphson(f, df, a1, b1)
        r2 = None
    # Mach solve set from 0 to 10000
    elif mach is None:
        r1 = cf.newtonRaphson(f, df, 0, 10000)
        r2 = None

    # Converting the radians back to degrees if user selected degrees
    if choice == 'd' and find != 'm':
        r1 = np.degrees(r1)
        if r2 is not None:
            r2 = np.degrees(r2)

    # Returning values
    return r1, r2


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

    # Question 11 in set 4.2
    b = np.array([-624, 4, 780, -5, -156, 1])
    pr2 = cf.polyRoots(b)
    r2 = 6
    return pr, pd, pr2, r2


b1, b2 = prob2()
print('The corresponding Beta values are %.2f and %.2f' % (b1, b2))
t1, t2 = prob2()
print('The corresponding Theta value is %.2f' % t1)
m1, m2 = prob2()
print('The corresponding Mach value is %.2f' % m1)

print('The time it takes to reach 335 m/s is %.3f seconds.' % prob3())

x, y, z, r = prob4()
print('Solving set 4.2 question 3, our roots are :\n', x)
print('The deflated polynomial is:')
print('(i - {0}) ({1}x**4 + {2}x**3 + {3}x**2 + {4}x + {5})'.format(r, y[-1], y[-2], y[-3], y[-4], y[-5]))
print('Solving set 4.2 question 11, our roots are:\n', z)
