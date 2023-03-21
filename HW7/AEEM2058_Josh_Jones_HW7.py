import sympy as sp
import numpy as np
import compFunc as cf
import matplotlib.pyplot as plt


def prob1():
    # 1 / sqrt(sin(x)) subbing sin(x) for t^2
    # 1 / t
    # t^2 = sin(x)
    # 2t dt = cos(x) dx
    # cos(x) dx = sqrt(1 - sin(x)^2) dx
    # dx = 2t dt / sqrt(1 - sin(x)^2)
    # integral of  (1 / t) * (2t / sqrt(1 - sin(x)^2) dt
    # integral of (1 /t) * (2t / sqrt(1 - t^4)) dt
    # integral of 2 / sqrt(1 - t^4) dt
    # upper bound = sqrt(sin(x))
    # upper bound = sqrt(sin(pi/4))
    def f(t):
        return 2 / np.sqrt(1 - (t ** 4))
    # returning the integral value to be printed
    i, n = cf.romberg(f, 0, np.sqrt(np.sin(np.pi / 4)))
    print('The value of this integral is %.5f' % i)


def prob2():
    # Initializing array for x and u values
    u = np.linspace(0.05, 1, 20)
    i = np.zeros(np.shape(u))

    # function from book
    def f(x):
        return ((x ** 4) * np.exp(x)) / ((np.exp(x) - 1) ** 2)
    # solving function at the x values and storing
    for j in range(u.size):
        i[j] = (u[j] ** 3) * cf.gaussQuad(f, 0, 1 / u[j], 100)
    # a little bit of plotting action
    plt.plot(u, i)
    plt.grid()
    plt.title("Debeye's Formula for Heate Capacity")
    plt.xlabel('u')
    plt.ylabel('g')
    plt.show()


def prob3():
    # function from book
    def f(x):
        return np.log(x) / ((x ** 2) - (2 * x) + 2)
    # printing both answers at different node numbers
    print('The value of the 2 node integral is %.5f' % cf.gaussQuad(f, 1, np.pi, 2))
    print('The value of the 4 node integral is %.5f' % cf.gaussQuad(f, 1, np.pi, 4))


def prob4():
    # initializing x for sympy then function from book
    x = sp.var('x')
    expr = ((2 * x) + 1) / sp.sqrt(x * (1 - x))
    # printing the integrated function
    print('The value of this integral is %.5f' % sp.N(sp.integrate(expr, (x, 0, 1))))


def prob5():
    # initializing x and y for sympy then function for book
    x = sp.var('x')
    y = sp.var('y')
    expr = x * y * (2 - (x ** 2)) * (2 - (x * y))
    # printing the integrated function
    print('The value of this integral is %.5f' % sp.N(sp.integrate(sp.integrate(expr, (x, y / 2 - 2, y / 2 + 2)), (y, -2, 2))))
