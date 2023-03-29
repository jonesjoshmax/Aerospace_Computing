import numpy as np
import compFunc as cf


def y(x):
    return 2 * np.sin(2 * np.pi * x / 7) - 4 * np.sin(3 * np.pi * x / 5)


cf.my_fft(0, 70, y)
cf.my_fft(0, 31, y)
