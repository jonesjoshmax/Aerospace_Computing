import numpy as np


def eigVer(a, w, v):
    for k in range(w.size):
        print('k:', k + 1)
        print('A*D_k:', np.round(np.dot(a, v[:, k]), 5))
        print('V*D_k:', np.round(np.dot(w[k], v[:, k]), 5))
        print('Vx - Dx:', np.round(np.dot(a, v[:, k]) - np.dot(w[k], v [:, k]), 5))


# Matrix initialization for either problem
a = np.array([[5, 3, 0, 0, 0],
              [0, 5, 0, 0, 0],
              [0, 0, 5, 0, 0],
              [0, 0, 0, 5, 1],
              [0, 0, 0, 0, 5]], float)
b = np.array([[51, 2, 3, 5, 7, 11],
              [1, 52, 3, 5, 7, 11],
              [1, 2, 53, 5, 7, 11],
              [1, 2, 3, 55, 7, 11],
              [1, 2, 3, 5, 57, 11],
              [1, 2, 3, 5, 7, 61]], float)

# Matrix 1
# Eigenvalues and Eigenvectors
w, v = np.linalg.eig(a)
# Printing the matrix, values, and vectors
print('Matrix A: \n', np.round(a, 3))
print('\n')
print('Eigenvalues of Matrix A: \n', np.round(w, 3))
print('Eigenvectors of Matrix A: \n', np.round(v, 3))
print('\n')
eigVer(a, w, v)
print('\n\n\n')

# Matrix 2
# Eigenvalues and Eigenvectors
w, v = np.linalg.eig(b)
# Printing the matrix, values, and vectors
print('Matrix B: \n', np.round(b, 3))
print('\n')
print('Eigenvalues of Matrix A: \n', np.round(w, 3))
print('Eigenvectors of Matrix A: \n', np.round(v, 3))
print('\n')
eigVer(b, w, v)
print('\n\n\n')

# New eigenvector is created from a multiple of the eigenvectors
# So we can add the 2 similar ones to create an eigenvector that should be right
v_new = v[:, 0] + v[:, 2]
v_new = v_new.T
# Same structure as eigVer function, just for one set of values
print('Made Up Eigenvector')
print('B*D_1 = ', np.round(np.dot(b, v_new), 3))
print('Vb*D_1 = ', np.round(np.dot(w[0], v_new), 3))
print('Vbx - D1x =', np.round(np.dot(b, v_new) - np.dot(w[0], v_new), 3))