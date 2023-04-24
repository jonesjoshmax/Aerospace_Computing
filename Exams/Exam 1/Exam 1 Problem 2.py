import sympy as sp
import numpy as np
import pandas as pd
import compFunc as cf

# Initializing variables for equation and subbing gamma 1.4
m = sp.var('m')
y = 1.4
# Sympy expression for the equation
expr = (1 + (((y - 1) / 2) * (m ** 2))) ** (y / (y - 1))
# Mach values for calculations
x = np.linspace(0, 2, 11)
# Zeros for pressure ratios
p_r = np.zeros(np.size(x))
# Pressure ratio calcs
for i in range(np.size(x)):
    p_r[i] = expr.subs(m, x[i])

# Creating a dataFrame from pandas to create table
data = {'Mach Number': x,
        'Pressure Ratio': p_r}
# Tabled values with relevant labels
df1 = pd.DataFrame(data)
blankIndex = [''] * len(df1)
df1.index = blankIndex
# Printing data frame
print(df1)

# Getting coefficients for polynomial estimates
c_linear = cf.polyFit(x, p_r, 1)
c_quad = cf.polyFit(x, p_r, 2)
c_cubic = cf.polyFit(x, p_r, 3)


# Function that prints the coeffs in a pretty manner
def printCoeffs(c):
    out = ''
    # Loops that print the coeffs and values with if statements to make it easier to read
    for j in range(np.size(c)):
        if j == 0:
            out = out + str(c[j])
        else:
            out = out + str(np.round(c[j], 5)) + 'x^' + str(j)
        if j + 1 != np.size(c):
            out = out + ' + '
    return out


# Evaluates the values at a specific points from the coeffs estimate
def calcCoeffs(c, x):
    out = 0
    # Loops to calculate
    for j in range(np.size(c)):
        out = out + (c[j] * (x ** j))
    return out


# Print statements to show results to user
print('The linear model is: ')
print(printCoeffs(c_linear) + '\n')
print('The quadratic model is: ')
print(printCoeffs(c_quad) + '\n')
print('The cubic model is: ')
print(printCoeffs(c_cubic) + '\n')
print('The real value at M = 1.35 is %.3f' % expr.subs(m, 1.35))
print('The linear estimate at M = 1.35 is: %.3f' % calcCoeffs(c_linear, 1.35))
print('The quadratic estimate at M = 1.35 is: %.3f' % calcCoeffs(c_quad, 1.35))
print('The cubic estimate at M = 1.35 is: %.3f' % calcCoeffs(c_cubic, 1.35))
