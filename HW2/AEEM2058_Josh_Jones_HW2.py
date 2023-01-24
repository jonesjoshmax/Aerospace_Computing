import numpy as np
import compFunc as cf
import time
import random as r
import matplotlib.pyplot as plt
import pandas as pd


# Code for part 3 of homework
def part3():
    # Empty array to store future timing and accuracy values
    rt_Val = np.empty((3, 2))
    # Given arrays from problem 11
    a = np.array([
        [10, -2, -1, 2, 3, 1, -4, 7],
        [5, 11, 3, 10, -3, 3, 3, -4],
        [7, 12, 1, 5, 3, -12, 2, 3],
        [8, 7, -2, 1, 3, 2, 2, 4],
        [2, -15, -1, 1, 4, -1, 8, 3],
        [4, 2, 9, 1, 12, -1, 4, 1],
        [-1, 4, -7, -1, 1, 1, -1, -3],
        [-1, 3, 4, 1, 3, -4, 7, 6]])
    b = np.array([[0], [12], [-5], [3], [-25], [-26], [9], [-7]])
    # Flipped signs for b vector
    b = b * -1
    # Define datatype because LU function wont work unless float
    a = a.astype('float64')
    b = b.astype('float64')

    # Cramer Calculation
    # t0 & t call time values
    # v calculates residual vector and sums values from within
    # t and v values are stored for table creation
    # same idea with Gauss and LU, just different solvers
    t0 = time.perf_counter()
    x = cf.cramer(a.copy(), b.copy())
    t = time.perf_counter() - t0
    v = np.dot(a.copy(), x) - b.copy()
    v0 = 0
    for i in range(np.size(v)):
        v0 = v0 + pow(v[i], 2)
    v0 = np.sqrt(v0)
    rt_Val[0, 0] = t * 1000000
    rt_Val[0, 1] = v0

    # Gauss Calculation
    t0 = time.perf_counter()
    x = cf.gaussPivot(a.copy(), b.copy())
    t = time.perf_counter() - t0
    v = np.dot(a.copy(), x) - b.copy()
    v0 = 0
    for i in range(np.size(v)):
        v0 = v0 + pow(v[i], 2)
    v0 = np.sqrt(v0)
    rt_Val[1, 0] = t * 1000000
    rt_Val[1, 1] = v0

    # LU Calculation
    t0 = time.perf_counter()
    a0, seq = cf.LUpivot(a.copy())
    x = cf.LUsolve(a0.copy(), b.copy(), seq.copy())
    t = time.perf_counter() - t0
    v = np.dot(a.copy(), x) - b.copy()
    v0 = 0
    for i in range(np.size(v)):
        v0 = v0 + pow(v[i], 2)
    v0 = np.sqrt(v0)
    rt_Val[2, 0] = t * 1000000
    rt_Val[2, 1] = v0

    # Creating a dataFrame from pandas to create table
    data = {'Runtime (microseconds)': [rt_Val[0, 0], rt_Val[1, 0], rt_Val[2, 0]],
            'Accuracy': [rt_Val[0, 1], rt_Val[1, 1], rt_Val[2, 1]]
            }
    # Tabled values with relevant indexes for left hand side
    df = pd.DataFrame(data, index=['Cramer', 'Gauss', 'LU'])
    print(df)


# Code for part 4
def part4():
    # Cycles stores N for N x N matrices
    cycles = [2, 4, 6, 8, 12]
    # Empty arrays size of cycles for data storage and then future plotting
    runtime = np.empty((3, len(cycles))).astype('float64')
    validation = np.copy(runtime)
    # Array creation / loop for testing
    a_m = 5
    for h in range(len(cycles)):
        while True:
            n = cycles[h]
            # Empty array size of current N value from cycle, and proper sized b vector
            a = np.empty([n, n])
            b = np.empty(n)
            # Nested for loop that puts a random int value at each location for new matrices
            for i in range(n):
                for j in range(n):
                    a[i, j] = r.randint(-a_m, a_m)
                b[i] = r.randint(-a_m, a_m)
            # Flips from row to column vector
            b = b.T
            if np.linalg.det(a) == 0:
                continue
            elif np.linalg.det(a) != 0:
                break

        # Cramer Calculation
        # Same as in part 3
        # Except now the stored values are going to dynamic locations instead of static
        t0 = time.perf_counter()
        x = cf.cramer(a.copy(), b.copy())
        t = time.perf_counter() - t0
        v = np.dot(a.copy(), x) - b.copy()
        v0 = 0
        for i in range(np.size(v)):
            v0 = v0 + pow(v[i], 2)
        v0 = np.sqrt(v0)
        runtime[0, h] = t
        validation[0, h] = v0

        # Gauss Calculation
        t0 = time.perf_counter()
        x = cf.gaussPivot(a.copy(), b.copy())
        t = time.perf_counter() - t0
        v = np.dot(a.copy(), x) - b.copy()
        v0 = 0
        for i in range(np.size(v)):
            v0 = v0 + pow(v[i], 2)
        v0 = np.sqrt(v0)
        runtime[1, h] = t
        validation[1, h] = v0

        # LU Calculation
        t0 = time.perf_counter()
        a0, seq = cf.LUpivot(a.copy())
        x = cf.LUsolve(a0.copy(), b.copy(), seq.copy())
        t = time.perf_counter() - t0
        v = np.dot(a.copy(), x) - b.copy()
        v0 = 0
        for i in range(np.size(v)):
            v0 = v0 + pow(v[i], 2)
        v0 = np.sqrt(v0)
        runtime[2, h] = t
        validation[2, h] = v0

    # Converting the run time values to microseconds
    runtime = runtime / 0.000001

    return runtime, validation, cycles


# Part 4 code (plotting functions)
def part4_visualization(runtime, validation, cycles):
    # Creates even sized spacing for bar plot
    x = np.arange(len(cycles))
    width = .25
    # Starting figure with plotting area ax
    fig, ax = plt.subplots()
    # Creating bars and designating relative location / color / label
    cr = ax.bar(x - width, runtime[0, :], width, label='Cramer', color='tomato')
    ga = ax.bar(x, runtime[1, :], width, label='Gauss', color='bisque')
    lu = ax.bar(x + width, runtime[2, :], width, label='LU', color='lightskyblue')
    # Create title, labels, legend, and bin ticks
    ax.set_xlabel('N x N Matrix')
    ax.set_ylabel('Runtime (microseconds)')
    ax.set_xticks(x, cycles)
    ax.legend()
    ax.set_title('Runtime of N x N Matrices', weight='bold')
    # Labels bars to show run time in microseconds
    ax.bar_label(cr, padding=2)
    ax.bar_label(ga, padding=2)
    ax.bar_label(lu, padding=2)
    fig.tight_layout()
    plt.show()
    # Creates another figure for accuracy with plotting area ax2
    fig2, ax2 = plt.subplots()
    # Plots lines for the cramer, gauss, and lu accuracy
    ax2.plot(cycles, validation[0, :], label='Cramer', color='tomato', linewidth=3)
    ax2.plot(cycles, validation[1, :], label='Gauss', color='bisque', linewidth=3)
    ax2.plot(cycles, validation[2, :], label='LU', color='lightskyblue', linewidth=3)
    # Creates title, labels, and legend
    ax2.set_xlabel('N x N Matrix')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy of N x N Matrix', weight='bold')
    ax2.legend()
    plt.show()


# Run function for part 5
def part5():
    # Given array in problem set
    prob8_array = np.array([
        [27.58, 7.004, -7.004, 0.0, 0.0],
        [7.004, 29.570, -5.253, 0.0, -24.32],
        [-7.004, -5.253, 29.57, 0.0, 0.0],
        [0.0, 0.0, 0.0, 27.580, -7.004],
        [0.0, -24.32, 0.0, -7.004, 29.57]
    ]).astype('float64')
    # runs matrixInverse function stored at top of this file
    prob8_inverse = cf.matInv(prob8_array.copy())
    # Prints original matrix and inverse matrix
    print('Original Matrix:\n{}\nInverted Matrix:\n{}'.format(prob8_array, prob8_inverse))
    # Prints identity matrix to show that it is the inverse
    print('Dot Product of Original & Inverted:\n{}'.format(np.round(np.dot(prob8_inverse, prob8_array))))


# Functions ran for submission
part3()
rt, valid, cyc = part4()
part4_visualization(rt, valid, cyc)
part5()
