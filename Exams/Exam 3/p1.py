import numpy as np
import sympy as sp
import scipy as sci
import matplotlib.pyplot as plt

print('\nPart 1\n')
# Zeros for A matrix
a = np.array([[31, 3, 5, 6, 9, 10],
             [1, 33, 5, 6, 9, 10],
             [1, 3, 35, 6, 9, 10],
             [1, 3, 5, 36, 9, 10],
             [1, 3, 5, 6, 39, 10],
             [1, 3, 5, 6, 9, 40]])
b = np.array([[5], [5], [5], [5], [5], [5]])
x = np.linalg.solve(a, b)
print(f'A Matrix:\n{a}')
print(f'x Matrix:\n{x}')
print(f'A multiplied to x:\n{np.matmul(a, x)}')
print(f'The residual is:\n{np.linalg.norm(np.dot(a, x) - b)}')

print('\nPart 2\n')


def eigVer(a, w, v):
    for k in range(w.size):
        print('k:', k + 1)
        print('A*D_k:', np.round(np.dot(a, v[:, k]), 5))
        print('V*D_k:', np.round(np.dot(w[k], v[:, k]), 5))
        print('Vx - Dx:', np.round(np.dot(a, v[:, k]) - np.dot(w[k], v[:, k]), 5))


w, v = np.linalg.eig(a)
# Printing the matrix, values, and vectors
print('Matrix A: \n', np.round(a, 3))
print('\n')
print('Eigenvalues of Matrix A: \n', np.round(w, 3))
print('Eigenvectors of Matrix A: \n', np.round(v, 3))
eigVer(a, w, v)

print('\nPart 3\n')


def f(x):
    return 6 * np.sin((2 * np.pi * x / 9) + (np.pi / 8)) - 1.5 * np.cos(4 * np.pi * x / 5)


x = sp.Symbol('x', real=True)
expr = 6 * sp.sin((2 * sp.pi * x / 9) + (sp.pi / 8)) - 1.5 * sp.cos(4 * sp.pi * x / 5)
x_vals = np.linspace(-10, 10, 100)
y_vals = f(x_vals)
for i in range(x_vals.size):
    y_vals[i] = expr.subs(x, x_vals[i])
plt.plot(x_vals, y_vals, color='red')
plt.title('f(x)')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

x0 = 4
r = np.zeros(4)
for i in range(r.size):
    r[i] = sci.optimize.fsolve(f, (i + 1) * x0)
    print(f'Root {i}: {r[i]}')
    print(f'Y Value: {round(f(r[i]), 6)}')
print(f'The integral value is: {sci.integrate.quad(f, 0, r[0])[0]}')


print('\nPart 4\n')


def my_fft(xStart, xEnd, y, barWidth=.2, N=500, xLim=50):
    T = 1 / N
    x = np.linspace(xStart, xEnd, N, endpoint=False)
    y_1 = y(x)
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Function')
    ax[0].plot(x, y_1, color='red')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].grid()
    amp = abs(np.fft.fft(y_1))[:N // 2]
    freq = np.fft.fftfreq(N, T)[:N // 2]
    ax[1].set_title('FFT')
    ax[1].bar(freq, 2 * amp / N, width=barWidth, color='red')
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Amplitude')
    ax[1].grid()
    ax[1].set_xlim(0, xLim)
    plt.tight_layout()
    plt.show()
    return


# Interval to run FFT is 2 x LCM of denominators of X
# Interval is 2 x 9 * 5 = 90
print('Bad FFT')
my_fft(0, 82, f)
print('Good FFT')
my_fft(0, 90, f)
print(f'The correct interval to run the FFT is the 2 times the LCM of the function: 2 x 9 x 5 = 90')
