import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import compFunc as cf

m = sp.var('m')
y = 1.4
a = sp.var('a')

v = sp.sqrt((y + 1) / (y - 1)) * sp.atan(sp.sqrt((y - 1) / (y + 1) * (m ** 2 - 1))) - sp.atan(sp.sqrt(m ** 2 - 1))

v_1 = sp.diff(v, m)
v_2 = sp.diff(v_1, m)


def prob2():
    ma = np.linspace(1, 5, 100)
    vals = np.zeros([6, ma.size], float)
    diff = 1
    for i in range(len(ma)):
        if type(cf.central_first(v, m, ma[i], diff)) != sp.core.add.Add:
            vals[0, i] = cf.central_first(v, m, ma[i], diff)
        if type(cf.forward_first(v, m, ma[i], diff)) != sp.core.add.Add:
            vals[1, i] = cf.forward_first(v, m, ma[i], diff)
        if type(cf.forward2_first(v, m, ma[i], diff)) != sp.core.add.Add:
            vals[2, i] = cf.forward2_first(v, m, ma[i], diff)
        if type(cf.central_second(v, m, ma[i], diff)) != sp.core.add.Add:
            vals[3, i] = cf.central_second(v, m, ma[i], diff)
        if type(cf.forward_second(v, m, ma[i], diff)) != sp.core.add.Add:
            vals[4, i] = cf.forward_second(v, m, ma[i], diff)
        if type(cf.forward2_second(v, m, ma[i], diff)) != sp.core.add.Add:
            vals[5, i] = cf.forward2_second(v, m, ma[i], diff)

    ref = np.zeros([2, np.size(ma)])
    for i in range(len(ma)):
        ref[0, i] = v_1.subs(m, ma[i])
        ref[1, i] = v_2.subs(m, ma[i])

    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(ma, vals[0, :])
    ax[0, 0].plot(ma, ref[0, :], '--')
    ax[0, 0].set_title("Central v'(m)")
    ax[0, 0].grid()

    ax[0, 1].plot(ma, vals[3, :])
    ax[0, 1].plot(ma, ref[1, :], '--')
    ax[0, 1].set_title("Central v''(m)")
    ax[0, 1].grid()

    ax[1, 0].plot(ma, vals[1, :])
    ax[1, 0].plot(ma, ref[0, :], '--')
    ax[1, 0].set_title("Forward 1st Order v'(m)")
    ax[1, 0].grid()

    ax[1, 1].plot(ma, vals[4, :])
    ax[1, 1].plot(ma, ref[1, :], '--')
    ax[1, 1].set_title("Forward 1st Order v''(m)")
    ax[1, 1].grid()

    ax[2, 0].plot(ma, vals[2, :])
    ax[2, 0].plot(ma, ref[0, :], '--')
    ax[2, 0].set_title("Forward 2nd Order v'(m)")
    ax[2, 0].grid()

    ax[2, 1].plot(ma, vals[5, :])
    ax[2, 1].plot(ma, ref[1, :], '--')
    ax[2, 1].set_title("Forward 2nd Order v''(m)")
    ax[2, 1].grid()

    plt.tight_layout()
    plt.show()


def prob3():
    n = 16
    res = 50
    x_diff = np.zeros(n)
    for i in range(n):
        x_diff[i] = 0.1 / 10 ** i
    mach_list = np.linspace(1, 5, res)
    label_list = []
    for j in range(n):
        label_list.append('1e-' + str(j + 1))

    def errorFunc(func, ref):
        error = np.zeros([n, res])
        for j in range(error.shape[0]):
            for k in range(error.shape[1]):
                error[j, k] = abs(func(v, m, mach_list[k], x_diff[j]) - ref.subs(m, mach_list[k]))
        return error

    error_fc1 = errorFunc(cf.central_first, v_1)
    error_1ff1 = errorFunc(cf.forward_first, v_1)
    error_2ff1 = errorFunc(cf.forward2_first, v_1)
    error_fc2 = errorFunc(cf.central_second, v_2)
    error_1ff2 = errorFunc(cf.forward_second, v_2)
    error_2ff2 = errorFunc(cf.forward2_second, v_2)

    fig, ax = plt.subplots(3, 2)

    for j in range(error_fc1.shape[0]):
        ax[0, 0].plot(mach_list, error_fc1[j])
    ax[0, 0].set_title("Central v'(m)")
    ax[0, 0].grid()

    for j in range(error_1ff1.shape[0]):
        ax[0, 1].plot(mach_list, error_1ff1[j])
    ax[0, 1].set_title("Central v''(m)")
    ax[0, 1].grid()

    for j in range(error_2ff1.shape[0]):
        ax[1, 0].plot(mach_list, error_2ff1[j])
    ax[1, 0].set_title("Forward 1st Order v'(m)")
    ax[1, 0].grid()

    for j in range(error_fc2.shape[0]):
        ax[1, 1].plot(mach_list, error_fc2[j])
    ax[1, 1].set_title("Forward 1st Order v''(m)")
    ax[1, 1].grid()

    for j in range(error_1ff2.shape[0]):
        ax[2, 0].plot(mach_list, error_1ff2[j])
    ax[2, 0].set_title("Forward 2nd Order v'(m)")
    ax[2, 0].grid()

    for j in range(error_2ff2.shape[0]):
        ax[2, 1].plot(mach_list, error_2ff2[j])
    ax[2, 1].set_title("Forward 2nd Order v''(m)")
    ax[2, 1].grid()

    fig.legend(label_list, loc=5)

    plt.tight_layout()
    plt.show()


