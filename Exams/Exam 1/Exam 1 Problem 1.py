import numpy as np
import random as rand

# Zeros for A matrix
a = np.zeros([6, 6])
# Setting first row to my m number
a[0] = [8, 5, 0, 7, 7, 6]
# Initializing determinate storage value
test = 0
# While loop to find matrix with det != 0
while test == 0:
    # Double for loop for indexing 2-d array
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            # Only rand int rows after the first
            if i > 0:
                a[i, j] = rand.randint(0, 10)
    # Calculate the determinate now that array is full
    test = np.linalg.det(a)
# Creating array of 2s for the solution side
sol = np.ones([6, 1]) * 2
# Inverting to multiply to find x matrix
inv_a = np.linalg.inv(a)
# Multiplying the matrices
b = np.matmul(inv_a, sol)
# Residual calculations
r = np.linalg.norm(np.dot(a, b) - sol)

print('The A matrix is: ')
print(a)
print('The Solution matrix is: ')
print(sol)
print('The resulting x matrix is: ')
print(b)
print('The 2 Norm Residual is: %.8f' % r)
