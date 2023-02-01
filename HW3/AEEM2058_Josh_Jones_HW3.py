import numpy as np
import compFunc as cf
import time
import matplotlib.pyplot as plt
import pandas as pd


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


def gs17(x, omega):
    n = len(x)
    x[0] = omega * (x[1] - x[n - 1]) / 4 + (1 - omega) * x[0]
    for i in range(1, n - 1):
        x[i] = omega * (x[i - 1] + x[i + 1]) / 4 + (1 - omega) * x[i]
    x[n - 1] = omega * (100 - x[0] + x[n - 2]) / 4 + (1 - omega) * x[n - 1]
    return x


def cg17(v):
    n = len(v)
    Ax = np.zeros(n)
    Ax[0] = 4 * v[0] - v[1] + v[n - 1]
    Ax[1:n - 1] = -v[0:n - 2] + 4 * v[1:n - 1] - v[2:n]
    Ax[n - 1] = v[0] - v[n - 2] + 4 * v[n - 1]
    return Ax


def p19mesh(t):
    n = len(t)
    Ax = np.zeros(n)
    m = int(np.sqrt(n))
    Ax[0] = -4.0 * t[0] + t[1] + t[m]
    for k in range(1, m - 1):
        Ax[k] = t[k - 1] - 4.0 * t[k] + t[k + 1] + t[k + m]
    k = m - 1
    Ax[k] = t[m - 2] - 4.0 * t[m - 1] + t[2 * m - 1]
    for i in range(1, m - 1):
        k = i * m
        Ax[k] = t[k - m] - 4.0 * t[k] + t[k + 1] + t[k + m]
        for j in range(1, m - 1):
            k = i * m + j
            Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k] + t[k + m] + t[k + 1]
    k = (i + 1) * m - 1
    Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k] + t[k + m]
    k = (m - 1) * m
    Ax[k] = t[k - m] - 4.0 * t[k] + t[k + 1]
    for j in range(1, m - 1):
        k = (m - 1) * m + j
        Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k] + t[k + 1]
    k = pow(m, 2) - 1
    Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k]
    return Ax


def p19b(n):
    # Where input is n of n x n size matrix
    a = np.zeros([n, n])
    b = np.zeros(pow(n, 2))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if j == n - 1 and i == n - 1:
                b[(3 * i) + j] = -300
            elif i == n - 1:
                b[(3 * i) + j] = -200
            elif j == n - 1:
                b[(3 * i) + j] = -100
    return b


def part2():
    # Matrix Formation for problem
    n = 20
    a, b = p17matrix(n)

    # Row 1 = Runtime Row 2 = Accuracy (using 2 base norm)
    data = np.empty([2, 5])

    # Cramer Calculation
    t0 = time.perf_counter()
    x = cf.cramer(a.copy(), b.copy())
    data[0, 0] = time.perf_counter() - t0
    data[1, 0] = np.linalg.norm(np.dot(a.copy(), x) - b.copy())

    # Gauss Pivot Calculation
    t0 = time.perf_counter()
    x = cf.gaussPivot(a.copy(), b.copy())
    data[0, 1] = time.perf_counter() - t0
    data[1, 1] = np.linalg.norm(np.dot(a.copy(), x) - b.copy())

    # LUPivot Calculation
    t0 = time.perf_counter()
    a0, seq = cf.LUpivot(a.copy())
    x = cf.LUsolve(a0.copy(), b.copy(), seq.copy())
    data[0, 2] = time.perf_counter() - t0
    data[1, 2] = np.linalg.norm(np.dot(a.copy(), x) - b.copy())

    # Gauss Seidel Calculation
    t0 = time.perf_counter()
    x = np.zeros(n)
    x, numIter, omega = cf.gaussSeidel(gs17, x)
    data[0, 3] = time.perf_counter() - t0
    x = x.reshape([n, 1])
    data[1, 3] = np.linalg.norm(np.dot(a.copy(), x) - b.copy())

    # Conjugate Gradient Calculation
    t0 = time.perf_counter()
    x = np.zeros(n)
    b1 = np.zeros(n)
    b1[n - 1] = 100
    x, numIter = cf.conjGrad(cg17, x, b1.copy())
    data[0, 4] = time.perf_counter() - t0
    x = x.reshape([n, 1])
    data[1, 4] = np.linalg.norm(np.dot(a.copy(), x) - b.copy())

    return data


def part2visualization(data):
    xpoints = np.arange(5)
    xticks = ['Cramer', 'GaussPivot', 'LUPivot', 'GaussSeidel', 'ConjGrad']
    data[0] = 100000 * data[0]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Runtime and Accuracy', weight='bold')
    axs[0].stem(xpoints, data[0], 'r')
    axs[0].set_xticks(xpoints, xticks, rotation=45)
    axs[0].set_title('Function Runtime')
    axs[0].set_ylabel('Runtime (microseconds)')
    axs[1].stem(xpoints, data[1], 'r')
    axs[1].set_xticks(xpoints, xticks, rotation=45)
    axs[1].set_title('Function Accuracy')
    axs[1].set_ylabel('2-Norm of Residual')
    axs[1].set_ylim([-0.001, 0.1])
    plt.tight_layout()
    plt.show()

    # Creating a dataFrame from pandas to create table
    data = {'Runtime (microseconds)': [data[0, 0], data[0, 1], data[0, 2], data[0, 3], data[0, 4]],
            'Accuracy': [data[1, 0], data[1, 1], data[1, 2], data[1, 3], data[1, 4]]
            }
    # Tabled values with relevant indexes for left hand side
    df = pd.DataFrame(data, index=['Cramer', 'GaussPivot', 'LUPivot', 'GaussSeidel', 'ConjGrad'])
    print(df)


#o = part2()
#part2visualization(o)
bm = p19b(3)
bm = bm.astype(float)
x1 = np.zeros(9, dtype=float)
x1, numIter = cf.conjGrad(p19mesh, x1, bm.copy())
print(x1)
print(x1.reshape([3,3]))
