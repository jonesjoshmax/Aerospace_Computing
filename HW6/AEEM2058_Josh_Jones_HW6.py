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
    # ax[2, 1].set_ylim(-20, 1)
    ax[2, 1].grid()

    plt.tight_layout()
    plt.show()


def prob3():
    ma = np.linspace(1, 5, 50)
    diff = np.zeros(16)
    vals = np.zeros([6 * diff.size, ma.size], float)
    for i in range(diff.size):
        diff[i] = 1 / (10 ** i)
    bounds = 10
    for i in range(diff.size):
        for j in range(ma.size):
            if type(cf.central_first(v, m, ma[j], diff[i])) != sp.core.add.Add and \
                    0 <= cf.central_first(v, m, ma[j], diff[i]) <= 1:
                vals[0 + (i * 6), j] = cf.central_first(v, m, ma[j], diff[i])
            if type(cf.forward_first(v, m, ma[j], diff[i])) != sp.core.add.Add and \
                    0 <= cf.forward_first(v, m, ma[j], diff[i]) <= 1:
                vals[1 + (i * 6), j] = cf.forward_first(v, m, ma[j], diff[i])
            if type(cf.forward2_first(v, m, ma[j], diff[i])) != sp.core.add.Add and \
                    0 <= cf.forward2_first(v, m, ma[j], diff[i]) <= 1:
                vals[2 + (i * 6), j] = cf.forward2_first(v, m, ma[j], diff[i])
            if type(cf.central_second(v, m, ma[j], diff[i])) != sp.core.add.Add and \
                    -bounds <= cf.central_second(v, m, ma[j], diff[i]) <= bounds:
                vals[3 + (i * 6), j] = cf.central_second(v, m, ma[j], diff[i])
            if type(cf.forward_second(v, m, ma[j], diff[i])) != sp.core.add.Add and \
                    -bounds <= cf.forward_second(v, m, ma[j], diff[i]) <= bounds:
                vals[4 + (i * 6), j] = cf.forward_second(v, m, ma[j], diff[i])
            if type(cf.forward2_second(v, m, ma[j], diff[i])) != sp.core.add.Add and \
                    -bounds <= cf.forward2_second(v, m, ma[j], diff[i]) <= bounds:
                vals[5 + (i * 6), j] = cf.forward2_second(v, m, ma[j], diff[i])
    error = np.zeros(vals.shape)
    ticker = 1
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if 1 <= ticker <= 3:
                error[i, j] = v_1.subs(m, ma[j]) - vals[i, j]
            elif 4 <= ticker <= 6:
                error[i, j] = v_2.subs(m, ma[j]) - vals[i, j]
        ticker = ticker + 1
        if ticker == 7:
            ticker = 1
    return error


est = prob3()
