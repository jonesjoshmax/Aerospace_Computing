import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Setting sympy variables, real so we get real roots
x = sp.var('x', real=True)

# creating Sympy expression
expr = (sp.exp(2 * x) - 1) * sp.sin(3 * sp.pi * x) * sp.cos(4 * sp.pi * x)

print('Part A:')
# X array of 100 values from 0 to 1
x_vals = np.linspace(0, 1, 100)
# Empty y array
y_vals = np.zeros(x_vals.size)
# Solver loop for the values
for i in range(x_vals.size):
    y_vals[i] = expr.subs(x, x_vals[i])

# Plotting. Title, labels, grid
plt.plot(x_vals, y_vals, color='red')
plt.title('f(x)')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

# Solving for roots. Function is mirrored across y axis so we can say
# the roots are the same but negative on other side
print('Part B:')
# Root solving
a = sp.solve(expr, x)
r1 = round(-a[1], 6)
r2 = round(-a[2], 6)
r3 = round(-a[3], 6)
# Printing 3 roots, discluding the one at zero because its borring
print('Root:', r1)
print('Root:', r2)
print('Root:', r3)

# Creating derivatinve
expr_der = sp.diff(expr)

print('Part C:')
# Similar solver and list for derivative values but on the directed values
x_list = np.arange(0, 2.5, .5)
for i in range(x_list.size):
    # printing values
    print('X:', x_list[i], 'Y:{:.6f}'.format(expr_der.subs(x, x_list[i])))

print('Part D:')
# solving derivative y values for plottting again in another loop
y_vals = np.zeros(x_vals.size)
for i in range(x_vals.size):
    y_vals[i] = expr_der.subs(x, x_vals[i])

# same plotting scheme as earlier for the new values
plt.plot(x_vals, y_vals, color='red')
plt.title("f'(x)")
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

print('Part E:')
# integrating across the range
integral = sp.integrate(expr, (x, 0, .125))
# printing the value we are returned
print('The value of the integral is:', round(integral, 6))
