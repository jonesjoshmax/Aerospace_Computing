import numpy as np
import random as rand

a = np.zeros([6,6])
a[0] = [8, 5, 0, 7, 7, 6]
test = 0
while test == 0:
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            if i > 0:
                a[i, j] = rand.randint(0, 10)
    test = np.linalg.det(a)
sol = np.ones([6, 1]) * 2
inv_a = np.linalg.inv(a)
b = np.matmul(inv_a, sol)
r = np.linalg.norm(np.dot(a, b) - sol)

print('The A matrix is: ')
print(a)
print('The Solution matrix is: ')
print(sol)
print('The resulting x matrix is: ')
print(b)
print('The 2 Norm Residual is: %.8f' % r)
