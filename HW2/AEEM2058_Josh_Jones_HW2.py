import numpy as np
import compFunc as cf
import time
import random as r
import matplotlib.pyplot as plt
import pandas as pd


def ab_refresh(a, b):
    a_ref = np.copy(a)
    b_ref = np.copy(b)
    return a_ref, b_ref


def matrixInverse(a):
    ai = np.empty(np.shape(a))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a_ref = np.copy(a)
            a_ref = np.delete(a_ref, i, 0)
            a_ref = np.delete(a_ref, j, 1)
            ai[i, j] = pow(-1, (i + j)) * np.linalg.det(a_ref)
    ai = np.transpose(ai)
    ai = (1 / np.dot(ai, a)[0, 0]) * ai
    return ai


def part3():
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
    b = b * -1
    a = a.astype('float64')
    b = b.astype('float64')

    # Cramer Calculation
    t0 = time.perf_counter()
    x = cf.cramer(a.copy(), b.copy())
    t = time.perf_counter() - t0
    v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))
    rt_Val[0, 0] = t * 1000000
    rt_Val[0, 1] = v

    # Gauss Calculation
    t0 = time.perf_counter()
    x = cf.gaussPivot(a.copy(), b.copy())
    t = time.perf_counter() - t0
    v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))
    rt_Val[1, 0] = t * 1000000
    rt_Val[1, 1] = v

    # LU Calculation
    t0 = time.perf_counter()
    a0, seq = cf.LUpivot(a.copy())
    x = cf.LUsolve(a0.copy(), b.copy(), seq.copy())
    t = time.perf_counter() - t0
    v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))
    rt_Val[2, 0] = t * 1000000
    rt_Val[2, 1] = v

    data = {'Runtime (microseconds)': [rt_Val[0, 0], rt_Val[1, 0], rt_Val[2, 0]],
            'Accuracy': [rt_Val[0, 1], rt_Val[1, 1], rt_Val[2, 1]]
            }
    df = pd.DataFrame(data, index=['Cramer', 'Gauss', 'LU'])
    print(df)


def part4():
    cycles = [2, 4, 6, 8, 12]
    runtime = np.empty((3, len(cycles))).astype('float64')
    validation = np.copy(runtime)
    # Array creation / loop for testing
    for h in range(len(cycles)):
        n = cycles[h]
        a = np.empty([n, n])
        b = np.empty(n)
        for i in range(n):
            for j in range(n):
                a[i, j] = r.randint(-15, 15)
            b[i] = r.randint(-15, 15)
        b = b.reshape((n, 1))

        # Cramer Calculation
        t0 = time.perf_counter()
        x = cf.cramer(a.copy(), b.copy())
        t = time.perf_counter() - t0
        v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))
        runtime[0, h] = t
        validation[0, h] = v

        # Gauss Calculation
        t0 = time.perf_counter()
        x = cf.gaussPivot(a.copy(), b.copy())
        t = time.perf_counter() - t0
        v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))
        runtime[1, h] = t
        validation[1, h] = v

        # LU Calculation
        t0 = time.perf_counter()
        a0, seq = cf.LUpivot(a.copy())
        x = cf.LUsolve(a0.copy(), b.copy(), seq.copy())
        t = time.perf_counter() - t0
        v = np.sum(np.array(np.dot(a.copy(), x) - b.copy()))
        runtime[2, h] = t
        validation[2, h] = v
    runtime = runtime / 0.000001

    return runtime, validation, cycles


def part4_visualization(runtime, validation, cycles):
    x = np.arange(len(cycles))
    width = .25
    fig, ax = plt.subplots()
    cr = ax.bar(x - width, runtime[0, :], width, label='Cramer', color='tomato')
    ga = ax.bar(x, runtime[1, :], width, label='Gauss', color='bisque')
    lu = ax.bar(x + width, runtime[2, :], width, label='LU', color='lightskyblue')
    ax.set_xlabel('N x N Matrix')
    ax.set_ylabel('Runtime (microseconds)')
    ax.set_xticks(x, cycles)
    ax.legend()
    ax.set_title('Runtime of N x N Matrices', weight='bold')
    ax.bar_label(cr, padding=2)
    ax.bar_label(ga, padding=2)
    ax.bar_label(lu, padding=2)
    fig.tight_layout()
    plt.show()
    fig2, ax2 = plt.subplots()
    ax2.plot(cycles, validation[0, :], label='Cramer', color='tomato', linewidth=3)
    ax2.plot(cycles, validation[1, :], label='Gauss', color='bisque', linewidth=3)
    ax2.plot(cycles, validation[2, :], label='LU', color='lightskyblue', linewidth=3)
    ax2.set_xlabel('N x N Matrix')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy of N x N Matrix', weight='bold')
    ax2.legend()
    plt.show()


def part5():
    prob8_array = np.array([
        [0, 2, 5, -1],
        [2, 1, 3, 0],
        [-2, -1, 3, 1],
        [3, 3, -1, 2]
    ])
    prob8_inverse = matrixInverse(prob8_array)
    print('Original Matrix:\n{}\nInverted Matrix:\n{}'.format(prob8_array, prob8_inverse))
    print('Dot Product of Original & Inverted:\n{}'.format(np.round(np.dot(prob8_inverse, prob8_array))))


part3()
rt, valid, cyc = part4()
part4_visualization(rt, valid, cyc)
part5()
