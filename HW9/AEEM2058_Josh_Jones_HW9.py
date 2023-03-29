import numpy as np
import compFunc as cf
import matplotlib.pyplot as plt


def prob1():
    def y(x):
        return 2 * np.sin(2 * np.pi * x / 7) - 4 * np.sin(3 * np.pi * x / 5)
    cf.my_fft(0, 70, y)
    cf.my_fft(0, 31, y)


def prob2a():
    data_1 = np.genfromtxt('CupData1.dat', delimiter=',')[:, 0]
    data_2 = np.genfromtxt('CupData2.dat', delimiter=',')[:, 0]
    data_3 = np.genfromtxt('CupData3.dat', delimiter=',')[:, 0]
    N = 1024000
    T = 5 / N
    pf_1 = 20 * np.log10(2 / N * (np.abs(np.fft.fft(data_1))[:N // 2] / (2 * (10 ** -5))))
    pf_2 = 20 * np.log10(2 / N * (np.abs(np.fft.fft(data_2))[:N // 2] / (2 * (10 ** -5))))
    pf_3 = 20 * np.log10(2 / N * (np.abs(np.fft.fft(data_3))[:N // 2] / (2 * (10 ** -5))))
    tf = np.fft.fftfreq(N, T)[:N // 2]
    lW = 1
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    plt.plot(tf, pf_1, label='Data 1', linewidth=lW)
    plt.plot(tf, pf_2, label='Data 2', linewidth=lW)
    plt.plot(tf, pf_3, label='Data 3', linewidth=lW)
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return


def prob2b():
    def avg_fft(file_name):
        M = 250
        N = 1024000 // M
        T = 1 / (50 * N)
        data = np.genfromtxt(file_name, delimiter=',')[:, 0].reshape(M, N)
        pf_storage = np.zeros(N)
        pf = 2 / N * np.abs(np.fft.fft(data))
        for i in range(M):
            for j in range(N):
                pf_storage[j] = pf[i, j] ** 2 + pf_storage[j]
        pf_storage = 20 * np.log10(np.sqrt(pf_storage / M)[:N // 2] / (2 * (10 ** -5)))
        tf = np.fft.fftfreq(N, T)[:N // 2]
        return tf, pf_storage
    plt.title('Averaged FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    a, b = avg_fft('CupData1.dat')
    plt.plot(a, b, label='Data 1')
    a, b = avg_fft('CupData2.dat')
    plt.plot(a, b, label='Data 2')
    a, b = avg_fft('CupData3.dat')
    plt.plot(a, b, label='Data 3')
    plt.grid()
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    return


prob1()
prob2a()
prob2b()
