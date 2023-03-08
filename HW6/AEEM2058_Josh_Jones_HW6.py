import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import compFunc as cf

# Initializing variable for sympy
m = sp.var('m')
y = 1.4
# Creating equation for reference
v = sp.sqrt((y + 1) / (y - 1)) * sp.atan(sp.sqrt((y - 1) / (y + 1) * (m ** 2 - 1))) - sp.atan(sp.sqrt(m ** 2 - 1))
# First and second derivative
v_1 = sp.diff(v, m)
v_2 = sp.diff(v_1, m)


# Problem 2
def prob2():
    # Initialization of arrays for data
    ma = np.linspace(1, 5, 100)
    vals = np.zeros([6, ma.size], float)
    # Original h value
    diff = 1
    # For loop to create data
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

    # reference data vectors using actual equation
    ref = np.zeros([2, np.size(ma)])
    for i in range(len(ma)):
        ref[0, i] = v_1.subs(m, ma[i])
        ref[1, i] = v_2.subs(m, ma[i])

    # Subplotting the data
    fig, ax = plt.subplots(3, 2)

    # Same subplot section used for each data set
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

    fig.legend(['Estimate', 'Real'], loc=5)

    # Making it tight and displaying it
    plt.tight_layout()
    plt.show()


# Problem 3
def prob3():
    # Initializing data for initializing arrays
    n = 16
    res = 50
    x_diff = np.zeros(n)
    # Creating data array value
    for j in range(n):
        x_diff[j] = 0.1 / 10 ** j
    # list of machs
    mach_list = np.linspace(1, 5, res)
    label_list = []
    # Labels for legend
    for j in range(n):
        label_list.append('1e-' + str(j + 1))

    # Function to calculate error. Absolute values
    def errorFunc(func, ref):
        error = np.zeros([n, res])
        for j in range(error.shape[0]):
            for k in range(error.shape[1]):
                error[j, k] = abs(func(v, m, mach_list[k], x_diff[j]) - ref.subs(m, mach_list[k]))
        return error

    # Using the error function to plot data
    error_fc1 = errorFunc(cf.central_first, v_1)
    error_1ff1 = errorFunc(cf.forward_first, v_1)
    error_2ff1 = errorFunc(cf.forward2_first, v_1)
    error_fc2 = errorFunc(cf.central_second, v_2)
    error_1ff2 = errorFunc(cf.forward_second, v_2)
    error_2ff2 = errorFunc(cf.forward2_second, v_2)

    # Plotting again
    fig, ax = plt.subplots(3, 2)

    # Runs through each output of data array
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

    # Legend and tight layout
    plt.tight_layout()
    plt.show()


# Problem 4
def prob4():
    # Subbing in m values for error comparison
    v_1.subs(m, 1.8)
    v_2.subs(m, 1.8)
    n = 16
    label_list = []
    # Arrays for data calculation
    x_diff = np.zeros(n)
    for j in range(n):
        x_diff[j] = 0.1 / 10 ** j

    # New error function to caculate absolute difference
    def errorFunc(func, ref):
        mach = 1.8
        error = np.zeros(n)
        for j in range(n):
            error[j] = abs(func(v, m, mach, x_diff[j]) - ref.subs(m, mach))
        return error

    # Plotting similar to last two sections
    fig, ax = plt.subplots(1, 2)

    # Plotting all 3, setting table, scale, and names
    ax[0].plot(np.log(x_diff), np.log(errorFunc(cf.central_first, v_1)), label='Central Diff')
    ax[0].plot(np.log(x_diff), np.log(errorFunc(cf.forward_first, v_1)), label='First Order Diff')
    ax[0].plot(np.log(x_diff), np.log(errorFunc(cf.forward2_first, v_1)), label='Second Order Diff')
    ax[0].set_title("First Derivative")
    ax[0].legend()
    ax[0].set_xscale('symLog')
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(16))
    ax[0].set_xlabel('Difference')
    ax[0].set_ylabel('Error')
    ax[0].grid()

    # Same as previous section
    ax[1].plot(np.log(x_diff), np.log(errorFunc(cf.central_second, v_2)), label='Central Diff')
    ax[1].plot(np.log(x_diff), np.log(errorFunc(cf.forward_second, v_2)), label='First Order Diff')
    ax[1].plot(np.log(x_diff), np.log(errorFunc(cf.forward2_second, v_2)), label='Second Order Diff')
    ax[1].set_title("Second Derivative")
    ax[1].legend()
    ax[1].set_xscale('symLog')
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(16))
    ax[1].set_xlabel('Difference')
    ax[1].set_ylabel('Error')
    ax[1].grid()

    plt.tight_layout()
    plt.show()


prob2()
prob3()
prob4()
