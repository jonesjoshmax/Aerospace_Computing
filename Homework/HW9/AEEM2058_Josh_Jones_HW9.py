import numpy as np
import compFunc as cf
import matplotlib.pyplot as plt


# Problem 1
def prob1():
    # Defining function to feed my function in my compFunc file
    def y(x):
        return 2 * np.sin(2 * np.pi * x / 7) - 4 * np.sin(3 * np.pi * x / 5)
    # Running good FFT
    cf.my_fft(0, 70, y)
    # Running bad FFT
    cf.my_fft(0, 31, y)


# Problem 2 a
def prob2a():
    # Parsing the 3 data sets for the first columns in each set
    data_1 = np.genfromtxt('CupData1.dat', delimiter=',')[:, 0]
    data_2 = np.genfromtxt('CupData2.dat', delimiter=',')[:, 0]
    data_3 = np.genfromtxt('CupData3.dat', delimiter=',')[:, 0]
    # Setting N and T (data points and time difference)
    N = 1024000
    T = 5 / N
    # Running FFTs on all three and scaling to proper value for SPL
    pf_1 = 20 * np.log10(2 / N * (np.abs(np.fft.fft(data_1))[:N // 2] / (2 * (10 ** -5))))
    pf_2 = 20 * np.log10(2 / N * (np.abs(np.fft.fft(data_2))[:N // 2] / (2 * (10 ** -5))))
    pf_3 = 20 * np.log10(2 / N * (np.abs(np.fft.fft(data_3))[:N // 2] / (2 * (10 ** -5))))
    # Getting frequency domain
    tf = np.fft.fftfreq(N, T)[:N // 2]
    # Plotting things, Title, Line Width, Labels, Legend, Scale
    lW = 1
    plt.title('FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    # Plotting the three different FFTs
    plt.plot(tf, pf_1, label='Data 1', linewidth=lW)
    plt.plot(tf, pf_2, label='Data 2', linewidth=lW)
    plt.plot(tf, pf_3, label='Data 3', linewidth=lW)
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return


# Problem 2 b
def prob2b():
    # Function for averaging the FFT across data bins
    def avg_fft(file_name):
        # Setting Bins, Spacing, and Time Differences
        M = 250
        N = 1024000 // M
        T = 1 / (50 * N)
        # Parses data set and obtains the proper sized matrix to do operations
        data = np.genfromtxt(file_name, delimiter=',')[:, 0].reshape(M, N)
        # Initializing storage matrix
        pf_storage = np.zeros(N)
        # Running FFT on the data set
        pf = 2 / N * np.abs(np.fft.fft(data))
        # Double nested Loop
        for i in range(M):
            for j in range(N):
                # Adding different iterations to be RMS
                pf_storage[j] = pf[i, j] ** 2 + pf_storage[j]
        # Doing data operations and finishing RMS
        pf_storage = 20 * np.log10(np.sqrt(pf_storage / M)[:N // 2] / (2 * (10 ** -5)))
        # Frequency domain
        tf = np.fft.fftfreq(N, T)[:N // 2]
        return tf, pf_storage
    # Plotting the previously obtained data
    # Legend, Title, Labels, Grid...
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
