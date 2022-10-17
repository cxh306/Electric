# SVR预测
# 也可用于时间序列分析（ARIMA也可用于时间序列分析）
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = '../../dataset/澳大利亚.xlsx'
    data = pd.read_excel(path, sheet_name='Sheet2')
    data.drop(axis=1, columns=['日期'], inplace=True)
    load = data['电力负荷']
    N = len(data)
    X = list(range(len(data)))
    seq_len = 1
    out_len = 1
    X_train = X[0:int(N * 0.6)]
    Y_train = data.loc[X_train].to_numpy()
    X_val = X[int(N * 0.6):int(N * 0.8)]
    Y_val = data.loc[X_val].to_numpy()
    X_test = X[int(N * 0.8): N]
    Y_test = data.loc[X_test].to_numpy()
    X = np.zeros((len(Y_train) - seq_len - out_len + 1, 7))
    Y = np.zeros((len(Y_train) - seq_len - out_len + 1))
    for k in range(len(Y_train) - seq_len - out_len + 1):
        X[k] = Y_train[k:k + seq_len]
        Y[k] = Y_train[k + seq_len:k + seq_len + out_len, 6]
    # 高斯核函数
    print('SVR - RBF')
    svr_rbf = svm.SVR(kernel='rbf', gamma=1e-8, C=100)
    svr_rbf.fit(X, Y)
    # 线性核函数
    print('SVR - Linear')
    svr_linear = svm.SVR(kernel='linear', C=100)
    svr_linear.fit(X, Y)
    # 多项式核函数
    print('SVR - Polynomial')
    svr_poly = svm.SVR(kernel='poly', degree=2, C=100)
    svr_poly.fit(X, Y)
    print('Fit OK.')
    X = np.zeros((len(Y_test) - seq_len - out_len + 1, 7))
    Y = np.zeros((len(Y_test) - seq_len - out_len + 1))
    for k in range(0, len(Y_test) - seq_len - out_len + 1, out_len):
        X[k] = Y_test[k:k + seq_len]
        Y[k] = Y_test[k + seq_len:k + seq_len + out_len, 6]
    y_predict_rbf = svr_rbf.predict(X)
    y_predict_linear = svr_linear.predict(X)
    y_predict_Polynomial = svr_poly.predict(X)
    error_rbf = Y_test[seq_len:, 6] - y_predict_rbf
    error_linear = Y_test[seq_len:, 6] - y_predict_linear
    error_Polynomial = Y_test[seq_len:, 6] - y_predict_Polynomial

    plt.figure(figsize=(9, 8), facecolor='w')
    plt.grid(True)
    plt.subplot(2, 1, 1)
    plt.plot(X_test[seq_len:], y_predict_rbf, 'r-', linewidth=2, label='RBF Kernel')
    plt.plot(X_test, Y_test[:, 6], 'b', markersize=6)
    plt.subplot(2, 1, 2)
    plt.plot(X_test[seq_len:], error_rbf, 'b', markersize=6, label='残差')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 8), facecolor='w')
    plt.grid(True)
    plt.subplot(2, 1, 1)
    plt.plot(X_test[seq_len:], y_predict_Polynomial, 'r-', linewidth=2, label='Polynomial Kernel')
    plt.plot(X_test, Y_test[:, 6], 'b', markersize=6)
    plt.subplot(2, 1, 2)
    plt.plot(X_test[seq_len:], error_Polynomial, 'b', markersize=6, label='残差')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 8), facecolor='w')
    plt.grid(True)
    plt.subplot(2, 1, 1)
    plt.plot(X_test[seq_len:], y_predict_linear, 'r-', linewidth=2, label='Linear Kernel')
    plt.plot(X_test, Y_test[:, 6], 'b', markersize=6)
    plt.subplot(2, 1, 2)
    plt.plot(X_test[seq_len:], error_linear, 'b', markersize=6, label='残差')
    plt.tight_layout()
    plt.show()

