import numpy as np
import matplotlib.pyplot as plt
import compFunc as cf
import pandas as pd


# Arrays for part 2
set31prob15 = np.array([[-250, 0.0163], [-200, 0.318], [-100, 0.699], [0, 0.870], [100, 0.941], [300, 1.04]], float)

set31prob17 = np.array([[0.2, 103], [2, 13.9], [20, 2.72], [200, 0.800], [2000, 0.401], [20000, 0.433]], float)

# Arrays for part 3
set32prob5 = np.array([[1994, 356.8], [1995, 358.2], [1996, 360.3], [1997, 361.8],
                       [1998, 364.0], [1999, 365.7], [2000, 366.7], [2001, 368.2],
                       [2002, 370.5], [2003, 372.2], [2004, 374.9], [2005, 376.7],
                       [2006, 378.7], [2007, 381.0], [2008, 382.9], [2009, 384.7]], float)

set32prob16 = np.array([[0.0, 1.000], [0.5, 0.994], [1.0, 0.990], [1.5, 0.985],
                        [2.0, 0.979], [2.5, 0.977], [3.0, 0.972], [3.5, 0.969],
                        [4.0, 0.967], [4.5, 0.960], [5.0, 0.956], [5.5, 0.952]], float)


def part2():

    # Problem 15

    # Rational solve
    x_r = np.arange(-250, 500, 1)
    x_np = x_r
    y_r = np.zeros(np.size(x_r))
    y_np = np.zeros(np.size(x_np))
    for i in range(len(x_r)):
        y_r[i] = cf.rational(set31prob15[:, 0].copy(), set31prob15[:, 1].copy(), x_r[i])

    # Newton Poly Solve
    a = cf.coeffts(set31prob15[:, 0].copy(), set31prob15[:, 1].copy())
    for i in range(np.size(x_np)):
        y_np[i] = cf.evalPoly(a, set31prob15[:, 0].copy(), x_np[i])

    # Data Visualization for p15
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Problem Set 3.1 Problem 17')
    ax1.set_title('Rational Solve')
    ax1.set_xlabel('Temperature (C)')
    ax1.set_ylabel('Specific Heat (kJ/kgK)')
    ax1.plot(x_r, y_r, color='orange')
    ax1.grid()
    ax2.set_title('Newton Polynomial Solve')
    ax2.set_xlabel('Temperature (C)')
    ax2.set_ylabel('Specific Heat (kJ/kgK)')
    ax2.plot(x_np, y_np, color='red')
    ax2.grid()
    fig.tight_layout()
    plt.show()

    # Problem 17

    x_r = np.array([5, 50, 500, 5000], float)
    x_np = np.array([5, 50, 500, 5000], float)
    x_cs = np.array([5, 50, 500, 5000], float)
    y_r = np.zeros(np.size(x_r), float)
    y_np = np.zeros(np.size(x_np), float)
    y_cs = np.zeros(np.size(x_cs), float)

    # Rational Solve
    for i in range(len(x_r)):
        y_r[i] = np.exp(cf.rational(np.log(set31prob17[:, 0].copy()), np.log(set31prob17[:, 1].copy()),
                                    np.log(x_r[i])))

    # Cubic Spline Solve
    k = cf.curvatures(np.log(set31prob17[:, 0].copy()), np.log(set31prob17[:, 1].copy()))
    for i in range(np.size(x_cs)):
        y_cs[i] = np.exp(cf.evalSpline(np.log(set31prob17[:, 0].copy()),
                                       np.log(set31prob17[:, 1].copy()), k, np.log(x_cs[i])))

    # Creating a dataFrame from pandas to create table
    data = {'Reynolds Number': x_r,
            'CD Cubic Spline': y_cs,
            'CD Rational': y_r}
    # Tabled values with relevant indexes for left hand side
    df = pd.DataFrame(data)
    blankIndex = [''] * len(df)
    df.index = blankIndex
    print(df)


def part3():

    # Function to test different M values for min stDev
    def fitTest(xData, yData, n):
        temp = np.zeros(n)
        # Runs loop to check different stDev Values
        for i in range(n):
            coeffs = cf.polyFit(xData, yData, i + 1)
            temp[i] = cf.stdDev(coeffs, xData, yData)
        # Indexes minimum value
        index = np.argmin(temp)
        coeffs = cf.polyFit(xData, yData, index + 1)
        deviation = cf.stdDev(coeffs, xData, yData)
        # Returns min stDev values and the m size polynomial
        return coeffs, deviation, index + 1

    # Problem set 3.2
    test_range = 9

    # Problem 5
    coeffs_5, deviation_5, fit_5 = fitTest(set32prob5[:, 0].copy(), set32prob5[:, 1].copy(), test_range)
    cf.plotPoly(set32prob5[:, 0].copy(), set32prob5[:, 1].copy(), coeffs_5)
    print(deviation_5, fit_5)

    # Problem 16
    coeffs_16, deviation_16, fit_16 = fitTest(set32prob16[:, 0].copy(), set32prob16[:, 1].copy(), test_range)
    cf.plotPoly(set32prob16[:, 0].copy(), set32prob16[:, 1].copy(), coeffs_16)
    print(deviation_16, fit_16)


part2()

