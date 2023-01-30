import numpy as np
import compFunc as cf
import time


def p17matrix(n):
    a = np.zeros((n, n)).astype('float64')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i, j] = 4
            elif i + 1 == j or j + 1 == i:
                a[i, j] = -1
            elif (i == n - 1 and j == 0) or (i == 0 and j == n - 1):
                a[i, j] = 1
    b = np.zeros((n, 1)).astype('float64')
    for i in range(n):
        if i == n - 1:
            b[i, 0] = 100
    return a, b


def p19matrix(n):
    a = np.zeros((n, n)).astype('float64')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i, j] = -4
            elif i + 1 == j or j + 1 == i:
                a[i, j] = 1
            elif i + 3 == j or j + 3 == i:
                a[i, j] = 1
            elif (i == n - 1 and j == 0) or (i == 0 and j == n - 1):
                a[i, j] = 1


def part2():
    # Matrix Formation for problem
    n = 20
    a, b = p17matrix(n)

    # Cramer Calculation
    t0 = time.perf_counter()
    x = cf.cramer(a.copy(), b.copy())
    t = time.perf_counter() - t0
    v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))

    # Gauss Pivot Calculation
    t0 = time.perf_counter()
    x = cf.gaussPivot(a.copy(), b.copy())
    t = time.perf_counter() - t0
    v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))

    # LUPivot Calculation
    t0 = time.perf_counter()
    a0, seq = cf.LUpivot(a.copy())
    x = cf.LUsolve(a0.copy(), b.copy(), seq.copy())
    t = time.perf_counter() - t0
    v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))

    # Gauss Seidel Calculation
    t0 = time.perf_counter()
    x = np.zeros(n)
    t = time.perf_counter() - t0

    # Conjugate Gradient Calculation
    t0 = time.perf_counter()
    x = np.zeros(n)
    x, numIter = cf.conjGrad(cf.Ax, x, b.copy())
    t = time.perf_counter() - t0


def ax19(T):
    n = len(T)
    Ax = np.zeros(n)
    m = np.sqrt(n)
    Ax[0] = -4.0*T[0]+T[1]+T[m]
    for k in range(1:m-1):
        Ax[k] = T[k-1]-4.0*T[k]+T[k+1]+T[k+m]
    k = m-1
    Ax[k] = T[m-2]-4.0*T[m-1]+T[2*m-1]

    return Ax
