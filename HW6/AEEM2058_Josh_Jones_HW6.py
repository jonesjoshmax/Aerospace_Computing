import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

m = sp.var('m')
y = 1.4
a = sp.var('a')

v = sp.sqrt((y + 1) / (y - 1)) * sp.atan(sp.sqrt((y - 1) / (y + 1) * (m ** 2 - 1))) - sp.atan(sp.sqrt(m ** 2 - 1))

v_1 = sp.diff(v, m)
v_2 = sp.diff(v_1, m)


# va = 1 / a * sp.atan(a * sp.sqrt(m ** 2 - 1)) - sp.atan(sp.sqrt(m ** 2 - 1))


def central_first(f, mach, diff):
    return (f.subs(m, mach + diff) - f.subs(m, mach - diff)) / (2 * diff)


def central_second(f, mach, diff):
    return (f.subs(m, mach + diff) - 2 * f.subs(m, mach) + f.subs(m, mach - diff)) / (diff ** 2)


def forward_first(f, mach, diff):
    return (f.subs(m, mach + diff) - f.subs(m, mach)) / diff


def forward_second(f, mach, diff):
    return (f.subs(m, mach + (2 * diff)) - 2 * f.subs(m, mach + diff)
            + f.subs(m, mach)) / (diff ** 2)


def forward2_first(f, mach, diff):
    return (-f.subs(m, mach + (2 * diff)) + 4 * f.subs(m, mach + diff)
            - 3 * f.subs(m, mach)) / (2 * diff)


def forward2_second(f, mach, diff):
    return (2 * f.subs(m, mach) - 5 * f.subs(m, mach + diff)
            + 4 * f.subs(m, mach + (2 * diff)) - f.subs(m, mach + (3 * diff))) / (diff ** 3)


def prob2():
    ma = np.linspace(1, 5, 50, float)
    vals = np.zeros([6, ma.size], float)
    diff = .01
    for r in range(len(ma)):
        if type(central_first(v, ma[r], diff)) != sp.core.add.Add:
            vals[0, r] = central_first(v, ma[r], diff)
        if type(forward_first(v, ma[r], diff)) != sp.core.add.Add:
            vals[1, r] = forward_first(v, ma[r], diff)
        if type(forward2_first(v, ma[r], diff)) != sp.core.add.Add:
            vals[2, r] = forward2_first(v, ma[r], diff)
        if type(central_second(v, ma[r], diff)) != sp.core.add.Add:
            vals[3, r] = central_second(v, ma[r], diff)
        if type(forward_second(v, ma[r], diff)) != sp.core.add.Add:
            vals[4, r] = forward_second(v, ma[r], diff)
        if type(forward2_second(v, ma[r], diff)) != sp.core.add.Add:
            vals[5, r] = forward2_second(v, ma[r], diff)

    ref = np.zeros([2, np.size(ma)])
    for r in range(len(ma)):
        ref[0, r] = v_1.subs(m, ma[r])
        ref[1, r] = v_2.subs(m, ma[r])

    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(ma, vals[0, :])
    ax[0, 0].plot(ma, ref[0, :])
    ax[0, 0].set_title("Central v'(m)")
    ax[0, 0].grid()

    ax[0, 1].plot(ma, vals[3, :])
    ax[0, 1].plot(ma, ref[1, :])
    ax[0, 1].set_title("Central v''(m)")
    ax[0, 1].grid()

    ax[1, 0].plot(ma, vals[1, :])
    ax[1, 0].plot(ma, ref[0, :])
    ax[1, 0].set_title("Forward 1st Order v'(m)")
    ax[1, 0].grid()

    ax[1, 1].plot(ma, vals[4, :])
    ax[1, 1].plot(ma, ref[1, :])
    ax[1, 1].set_title("Forward 1st Order v''(m)")
    ax[1, 1].grid()

    ax[2, 0].plot(ma, vals[2, :])
    ax[2, 0].plot(ma, ref[0, :])
    ax[2, 0].set_title("Forward 2nd Order v'(m)")
    ax[2, 0].grid()

    ax[2, 1].plot(ma, vals[5, :])
    ax[2, 1].plot(ma, ref[1, :])
    ax[2, 1].set_title("Forward 2nd Order v''(m)")
    ax[2, 1].set_ylim(-10, .5)
    ax[2, 1].grid()

    plt.tight_layout()
    plt.show()


prob2()
