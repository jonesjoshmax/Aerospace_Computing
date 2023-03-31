import numpy as np

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

w, x = np.linalg.eig(a)
y, z = np.linalg.eig(b)
l = np.linalg.eigvals(b)
for i in range(np.size(w)):
    print(np.linalg.det(a - w[i] * x))

