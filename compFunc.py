import numpy as np
import sys

'''err(string).
    Prints ’string’ and terminates program.
    30 Introduction to Python
'''


def err(string):
    print(string)
    input('Press return to exit')
    sys.exit()


'''swapRows(v,i,j).
    Swaps rows i and j of a vector or matrix [v].
    swapCols(v,i,j).
    Swaps columns of matrix [v].'''


def swapRows(v, i, j):
    if len(v.shape) == 1:
        v[i], v[j] = v[j], v[i]
    else:
        v[[i, j], :] = v[[j, i], :]


def swapCols(v, i, j):
    v[:, [i, j]] = v[:, [j, i]]


'''x = gaussPivot(a,b,tol=1.0e-12).
    Solves [a]{x} = {b} by Gauss elimination with
    scaled row pivoting'''


def gaussPivot(a, b, tol=1.0e-12):
    n = len(b)

    # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i, :]))

    for k in range(0, n - 1):

        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n, k]) / s[k:n]) + k
        if abs(a[p, k]) < tol: err('Matrix is singular')
        if p != k:
            swapRows(b, k, p)
            swapRows(s, k, p)
            swapRows(a, k, p)

    # Elimination
    for i in range(k + 1, n):
        if a[i, k] != 0.0:
            lam = a[i, k] / a[k, k]
            a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
            b[i] = b[i] - lam * b[k]
    if abs(a[n - 1, n - 1]) < tol: err('Matrix is singular')

    # Back substitution
    b[n - 1] = b[n - 1] / a[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b


''' a,seq = LUdecomp(a,tol=1.0e-9).
    LU decomposition of matrix [a] using scaled row pivoting.
    The returned matrix [a] = contains [U] in the upper
    triangle and the nondiagonal terms of [L] in the lower triangle.
    Note that [L][U] is a row-wise permutation of the original [a];
    the permutations are recorded in the vector {seq}.
    x = LUsolve(a,b,seq).
    Solves [L][U]{x} = {b}, where the matrix [a] = and the
    permutation vector {seq} are returned from LUdecomp.
'''


def LUdecomp(a, tol=1.0e-9):
    n = len(a)
    seq = np.array(range(n))

    # Set up scale factors
    s = np.zeros((n))
    for i in range(n):
        s[i] = max(abs(a[i, :]))

    for k in range(0, n - 1):

        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n, k]) / s[k:n]) + k
        if abs(a[p, k]) < tol: err('Matrix is singular')
        if p != k:
            swapRows(s, k, p)
            swapRows(a, k, p)
            swapRows(seq, k, p)

        # Elimination
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                a[i, k] = lam
    return a, seq


def LUsolve(a, b, seq):
    n = len(a)

    # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

    # Solution
    for k in range(1, n):
        x[k] = x[k] - np.dot(a[k, 0:k], x[0:k])
    x[n - 1] = x[n - 1] / a[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (x[k] - np.dot(a[k, k + 1:n], x[k + 1:n])) / a[k, k]
    return x
