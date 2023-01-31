import numpy as np
import math
import sys

## module error
''' 
    err(string).
    Prints 'string' and terminates program.
'''
def err(string):
    print(string)
    input('Press return to exit')
    sys.exit()


## module swap
''' 
    swapRows(v,i,j).
    Swaps rows i and j of a vector or matrix [v].
    
    swapCols(v,i,j).
    Swaps columns of matrix [v].
    
    swapCramer(a, b, i).
    Swaps i-th column of matrix [a] with array [b].
'''
def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]
    return

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]
    return

def swapCramer(a, b, i):
    import numpy as np
    ai = a.copy()
    ai[:, i] = np.transpose(b)
    return ai


'''
    x = gaussPivot(a,b,tol=1.0e-12).
    Solves [a]{x} = {b} by Gauss elimination with
    scaled row pivoting
'''


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


''' 
    a,seq = LUpivot(a,tol=1.0e-9).
    LU decomposition of matrix [a] using scaled row pivoting.
    The returned matrix [a] = contains [U] in the upper
    triangle and the nondiagonal terms of [L] in the lower triangle.
    Note that [L][U] is a row-wise permutation of the original [a];
    the permutations are recorded in the vector {seq}.
    x = LUsolve(a,b,seq).
    Solves [L][U]{x} = {b}, where the matrix [a] = and the
    permutation vector {seq} are returned from LUdecomp.
'''


def LUpivot(a, tol=1.0e-9):
    n = len(a)
    seq = np.array(range(n))

    # Set up scale factors
    s = np.zeros(n)
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


'''
    Cramer Function returns 
    linalg solutions to each
    value of set of equations
    where m_ref is the reference
    matrix [A] and m_val is the
    [b] matrix in [A][x] = [b]
'''


def cramer(m_ref, m_val):
    # Constant [a] determinate value for [a']/[a]. If [a] = 0 then an error is returned
    a = np.linalg.det(m_ref)
    if a == 0:
        err('No Solution')
    # Array for storing x values for solution
    out = np.empty(m_val.shape)
    for r in range(m_val.size):
        # Copies reference matrix for each different execution of det
        m_clone = np.copy(m_ref)
        for i in range(m_ref.shape[0]):
            for j in range(m_ref.shape[0]):
                if j == r:
                    # Rewrites matrix in [a'] form
                    m_clone[i, j] = m_val[i]
        # Saves x value in storage
        out[r] = np.linalg.det(m_clone) / a
    # Returns output for values calculated
    return out


## Matrix Inverse
'''
    Returns matrix inverse using
    LUPivot rather than my own
'''
def matInv(a):
    n = len(a[0])
    aInv = np.identity(n)
    a,seq = LUpivot(a)
    for i in range(n):
        aInv[:,i] = LUsolve(a,aInv[:,i],seq)
    return aInv


## module gaussSeidel
'''
    x,numIter,omega = gaussSeidel(iterEqs,x,tol = 1.0e-9)
    Gauss-Seidel method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply the
    function iterEqs(x,omega) that returns the improved {x},
    given the current {x} (’omega’ is the relaxation factor).
'''
def gaussSeidel(iterEqs,x,tol = 1.0e-9):
    omega = 1.0
    k = 10
    p = 1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(np.dot(x-xOld,x-xOld))
        if dx < tol: return x,i,omega
        # Compute relaxation factor after k+p iterations
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - pow((dx2/dx1), (1.0/p))))
    print('Gauss-Seidel failed to converge')


## module conjGrad
'''
    x, numIter = conjGrad(Av, x, b, tol=1.0e-9)
    Conjugate gradient method for solving[A]{x} = {b}.
    The matrix[A] should be sparse. User must supply the 
    function Av(v)  that returns the vector[A] {v}.
'''
def conjGrad(Av, x, b, tol=1.0e-9):
    n = len(b)
    r = b - Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = np.dot(s, r) / np.dot(s, u)
        x = x + alpha * s
        r = b - Av(x)
        if (math.sqrt(np.dot(r, r))) < tol:
            break
        else:
            beta = -np.dot(r, u) / np.dot(s, u)
            s = r + beta * s
    return x, i

## module newtonPoly
'''
    p = evalPoly(a,xData,x).
    Evaluates Newton’s polynomial p at x. The coefficient
    vector {a} can be computed by the function ’coeffts’.
    a = coeffts(xData,yData).
    Computes the coefficients of Newton’s polynomial.
'''
def evalPoly(a,xData,x):
    n = len(xData) - 1 # Degree of polynomial
    p = a[n]
    for k in range(1,n+1):
        p = a[n-k] + (x -xData[n-k])*p
    return p
def coeffts(xData,yData):
    m = len(xData) # Number of data points
    a = yData.copy()
    for k in range(1,m):
        a[k:m] = (a[k:m] - a[k-1])/(xData[k:m] - xData[k-1])
    return a