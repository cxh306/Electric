import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from vmdpy import VMD

from data_process import load_data

if __name__ == '__main__':
    print('dataset processing...')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    dataset = load_data('澳大利亚.xlsx',"Sheet2")
    # split
    dataset = dataset.iloc[:, 1:]

    train = dataset[:int(len(dataset) * 0.6)]
    validate = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

    alpha = 2566  # moderate bandwidth constraint
    tau = 1e-2  # noise-tolerance (no strict fidxelity enforcement)
    K = 4  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7
    load = train['电力负荷']
    data = load.values.reshape(-1, 1)
    plt.figure(dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title('训练集')
    plt.subplot(2, 1, 2)
    plt.plot(abs(fftshift(fft(data.T))).T)
    plt.title('训练集频谱')
    plt.tight_layout()
    plt.show()
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    plt.figure(dpi=300)
    for i in range(K):
        plt.subplot(int(np.ceil((K + 1) / 2)), 2, i + 1)
        plt.plot(u[i, :])
        plt.title('IMF-' + str(i + 1))
    plt.subplot(int(np.ceil((K + 1) / 2)), 2, K + 1)
    plt.plot((data.T - np.sum(u, axis=0)).T)
    plt.title('残差',)
    plt.tight_layout()
    plt.show()
    plt.figure(dpi=300)
    for i in range(K):
        plt.subplot(int(np.ceil((K + 1) / 2)), 2, i + 1)
        plt.plot(abs(fftshift(fft(u[i, :]))))
        plt.title('IMF-' + str(i + 1))
    plt.subplot(int(np.ceil((K + 1) / 2)), 2, K + 1)
    plt.title('残差')
    plt.plot((abs(fftshift(fft(data.T - np.sum(u, axis=0))))).T)
    plt.tight_layout()
    plt.show()
