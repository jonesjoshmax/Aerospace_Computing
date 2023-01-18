import numpy as np
import compFunc as cf
import time
import random as r


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
    runtime = np.empty((3, 1))
    validation = np.copy(runtime)
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

    # Cramer Calculation
    a_ref, b_ref = ab_refresh(a, b)
    t0 = time.perf_counter()
    x = cf.cramer(a_ref, b_ref)
    a_ref, b_ref = ab_refresh(a, b)
    v = np.sum(np.array(np.dot(a_ref, x) - b_ref))
    t = time.perf_counter() - t0
    runtime[0, 0] = t
    validation[0, 0] = v

    # Gauss Calculation
    a_ref, b_ref = ab_refresh(a, b)
    t0 = time.perf_counter()
    x = cf.gaussPivot(a_ref, b_ref)
    a_ref, b_ref = ab_refresh(a, b)
    v = np.sum(np.array(np.dot(a_ref, x) - b_ref))
    t = time.perf_counter() - t0
    runtime[1, 0] = t
    validation[1, 0] = v

    # LU Calculation

    return runtime, validation


def part4():
    cycles = [2, 4, 6, 8, 12]
    runtime = np.empty((3, len(cycles)))
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
        a_ref, b_ref = ab_refresh(a, b)
        t0 = time.perf_counter()
        x = cf.cramer(a_ref, b_ref)
        a_ref, b_ref = ab_refresh(a, b)
        v = np.sum(np.array(np.dot(a_ref, x) - b_ref))
        t = time.perf_counter() - t0
        runtime[0, h] = t
        validation[0, h] = v

        # Gauss Calculation
        a_ref, b_ref = ab_refresh(a, b)
        t0 = time.perf_counter()
        x = cf.gaussPivot(a_ref, b_ref)
        a_ref, b_ref = ab_refresh(a, b)
        v = np.sum(np.array(np.dot(a_ref, x) - b_ref))
        t = time.perf_counter() - t0
        runtime[1, h] = t
        validation[1, h] = v

        # LU Function

    return runtime, validation


def lu_testing():
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

    a_ref, b_ref = ab_refresh(a, b)
    a0, seq = cf.LUdecomp(a_ref)
    x = cf.LUsolve(a0, b_ref, seq)

    return x


def part5():
    prob8_array = np.array([
        [0, 2, 5, -1],
        [2, 1, 3, 0],
        [-2, -1, 3, 1],
        [3, 3, -1, 2]
    ])
    prob8_inverse = matrixInverse(prob8_array)
    print(np.round(np.dot(prob8_inverse, prob8_array)))


part5()
