import numpy as np
import compFunc as cf
import time
import matplotlib.pyplot as plt
import pandas as pd


# Function to create problem 17 array from book
def p17matrix(n):
    # Zeros for a array
    a = np.zeros((n, n)).astype('float64')
    # Double nested loop to give vals to x and y pos of array
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # Indexes and creates relevant points to the problem
            if i == j:
                a[i, j] = 4
            elif i + 1 == j or j + 1 == i:
                a[i, j] = -1
            elif (i == n - 1 and j == 0) or (i == 0 and j == n - 1):
                a[i, j] = 1
    # Empty b array to populate
    b = np.zeros((n, 1)).astype('float64')
    # Loop to satisfy given condition of array
    for i in range(n):
        if i == n - 1:
            b[i, 0] = 100
    # Returning a and b array for calculation
    return a, b


# gaussSeidel Ax solve
def gs17(x, omega):
    n = len(x)
    # Setting First value
    x[0] = omega * (x[1] - x[n - 1]) / 4 + (1 - omega) * x[0]
    # Setting other repeated values
    for i in range(1, n - 1):
        x[i] = omega * (x[i - 1] + x[i + 1]) / 4 + (1 - omega) * x[i]
    # Setting last values
    x[n - 1] = omega * (100 - x[0] + x[n - 2]) / 4 + (1 - omega) * x[n - 1]
    # Feeding back altered input
    return x


# Conjugate Gradient mesh
def cg17(v):
    n = len(v)
    # Empty storage for incoming values
    Ax = np.zeros(n)
    # Relevant values for x and y pos
    Ax[0] = 4 * v[0] - v[1] + v[n - 1]
    Ax[1:n - 1] = -v[0:n - 2] + 4 * v[1:n - 1] - v[2:n]
    Ax[n - 1] = v[0] - v[n - 2] + 4 * v[n - 1]
    # Returning relevant ax values
    return Ax


# Mesh generation for problem 19
def p19mesh(t):
    n = len(t)
    # Mesh initialization
    Ax = np.zeros(n)
    # m for indexing on m x m size
    m = int(np.sqrt(n))
    # Setting first values
    Ax[0] = -4.0 * t[0] + t[1] + t[m]
    # Repeated values on p1
    for k in range(1, m - 1):
        Ax[k] = t[k - 1] - 4.0 * t[k] + t[k + 1] + t[k + m]
    k = m - 1
    # Constant value
    Ax[k] = t[m - 2] - 4.0 * t[m - 1] + t[2 * m - 1]
    # Repeated values for surrounding points
    for i in range(1, m - 1):
        k = i * m
        # Setting p2
        Ax[k] = t[k - m] - 4.0 * t[k] + t[k + 1] + t[k + m]
        for j in range(1, m - 1):
            k = i * m + j
            # Setting p3
            Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k] + t[k + m] + t[k + 1]
        k = (i + 1) * m - 1
        Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k] + t[k + m]
        k = (m - 1) * m
        Ax[k] = t[k - m] - 4.0 * t[k] + t[k + 1]
        for j in range(1, m - 1):
            k = (m - 1) * m + j
            # Setting p4
            Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k] + t[k + 1]
        k = pow(m, 2) - 1
        # setting p5
        Ax[k] = t[k - m] + t[k - 1] - 4.0 * t[k]
    # Returning mesh
    return Ax


# b matrix for p19
def p19b(n):
    # Where input is n of n x n size matrix
    # I did not need to do a =; just helped me think about this spatially
    a = np.zeros([n, n])
    b = np.zeros(pow(n, 2), float)
    # nested loop for my sanity and visualizing the process
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # Setting vector values at relative wall. 300 for corner piece
            if j == n - 1 and i == n - 1:
                b[(n * i) + j] = -300
            elif i == n - 1:
                b[(n * i) + j] = -200
            elif j == n - 1:
                b[(n * i) + j] = -100
    # Returning given b matrix
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

    # Retuning relevant data for plotting / creating tabel
    return data


# Visualization part for previous code
def part2visualization(data):
    # Setting points to nicely label the graph
    xpoints = np.arange(5)
    xticks = ['Cramer', 'GaussPivot', 'LUPivot', 'GaussSeidel', 'ConjGrad']
    # Converting to microseconds to be read easier
    data[0] = 100000 * data[0]
    # Subplotting the data because im sick like that
    fig, axs = plt.subplots(1, 2)
    # Main Title
    fig.suptitle('Runtime and Accuracy', weight='bold')
    # Stemplot for the run time
    axs[0].stem(xpoints, data[0], 'r')
    # X labels
    axs[0].set_xticks(xpoints, xticks, rotation=45)
    axs[0].set_title('Function Runtime')
    axs[0].set_ylabel('Runtime (microseconds)')
    # Similar to previous section, just for accuracy
    axs[1].stem(xpoints, data[1], 'r')
    axs[1].set_xticks(xpoints, xticks, rotation=45)
    axs[1].set_title('Function Accuracy')
    axs[1].set_ylabel('2-Norm of Residual')
    # One point was e-10 not e-13 so blew graph out of proportion. Making it look nice and all 0
    axs[1].set_ylim([-0.001, 0.1])
    # Squeezing the plots together
    plt.tight_layout()
    plt.show()

    # Creating a dataFrame from pandas to create table
    data = {'Runtime (microseconds)': [data[0, 0], data[0, 1], data[0, 2], data[0, 3], data[0, 4]],
            'Accuracy': [data[1, 0], data[1, 1], data[1, 2], data[1, 3], data[1, 4]]
            }
    # Tabled values with relevant indexes for left hand side
    df = pd.DataFrame(data, index=['Cramer', 'GaussPivot', 'LUPivot', 'GaussSeidel', 'ConjGrad'])
    print(df)


def part3():
    # Some data intializaion for the following code
    b9 = p19b(3)
    b1600 = p19b(40)
    # x and b arrays for relevant calcs
    x9 = np.zeros(9)
    x1600 = np.zeros(1600)
    # data storage (only 2 points for time)
    data = np.zeros(2)

    # 9 x 9 Solver
    # Same as conj grad from earlier
    t0 = time.perf_counter()
    x9, void = cf.conjGrad(p19mesh, x9, b9.copy())
    data[0] = time.perf_counter() - t0
    x9 = x9.reshape([3, 3])
    x = np.arange(1, 4)
    y = x
    # Contour plot (looks like lava to signify the HEAT)
    plt.contourf(x, y, x9, cmap='hot')
    # Titles and labels to help your eyes
    plt.colorbar(label='Degrees')
    plt.title('Temperature Gradient of 9 x 9')
    plt.gca().invert_yaxis()
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()

    # 40 x 40 Solver
    # Literally same as last section but higher resolution
    t0 = time.perf_counter()
    x1600, void = cf.conjGrad(p19mesh, x1600, b1600.copy())
    data[1] = time.perf_counter() - t0
    x1600 = x1600.reshape([40, 40])
    x = np.arange(1, 41)
    y = x
    plt.contourf(x, y, x1600, cmap='hot')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Degrees')
    plt.title('Temperature Gradient of 40 x 40')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()

    # Runtime stemplots
    # Similar to ones described earlier in function
    plt.stem(np.arange(1, 3), data, 'r')
    plt.title('Runtime Performance')
    plt.xticks(np.arange(1, 3), ['9 x 9', '40 x 40'], rotation=45)
    plt.ylabel('Runtime (seconds)')
    # Text labels to show the actual value because they are so far apart
    plt.text(1, data[0] + .05, data[0])
    plt.text(2 - .35, data[1] - .05, data[1])
    plt.show()

    # Order of operations calculation
    order_op = np.log(data[1] / data[0]) / np.log(40 / 3)
    print('Order of Operations = {}'.format(np.round(order_op, 2)))


o = part2()
part2visualization(o)
part3()
