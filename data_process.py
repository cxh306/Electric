# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 20:11
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import os
import random

from PyEMD import EMD
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.dual import fft
from numpy.fft import fftshift
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from vmdpy import VMD

device = torch.device("cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(file_name, sheet_name) -> DataFrame:
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/dataset/' + file_name
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.fillna(df.mean(), inplace=True)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_transformer(args):
    seq_len, B, num = args.seq_len, args.batch_size, args.output_size
    print('data processing...')
    dataset = load_data("澳大利亚.xlsx", "Sheet2")
    # split
    dataset = dataset['电力负荷']

    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = train.max(), train.min()

    def process(data, batch_size, step_size, shuffle):
        load = data
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(0, len(data) - seq_len - num, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                # for c in range(2, 8):
                #     x.append(data[j][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=num, shuffle=False)

    return Dtr, Val, Dte, m, n

def nn_seq_vmd_lstm(seq_len, B, pred_len):
    print('dataset processing...')
    dataset = load_data("澳大利亚.xlsx", "Sheet2")

    load = dataset['电力负荷']
    decompose = vmd(load.values.reshape(-1, 1), K=4, alpha=2566)
    # decompose = vmd(load.values.reshape(-1, 1))
    train = decompose[:, :int(len(load) * 0.6)]
    val = decompose[:, int(len(load) * 0.6):int(len(load) * 0.8)]
    test = decompose[:, int(len(load) * 0.8):len(load)]

    plt.figure()
    for i in range(5):
        plt.subplot(3, 2, i + 1)
        plt.plot(test[i, :])
    plt.show()
    m = []
    n = []
    for i in range(train.shape[0]):
        m.append(train[i, :].max())
        n.append(train[i, :].min())

    def process(ds, batch_size, step_size):
        seq = []
        for i in range(0, len(ds) - seq_len - pred_len, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [ds[j]]
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + pred_len):
                train_label.append(ds[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = []
    Val = []
    Dte = []
    for i in range(decompose.shape[0]):
        data = (train[i, :] - n[i]) / (m[i] - n[i])
        Dtr.append(process(data, B, step_size=1))

        data = (val[i, :] - n[i]) / (m[i] - n[i])
        Val.append(process(data, B, step_size=1))

        data = (test[i, :] - n[i]) / (m[i] - n[i])
        Dte.append(process(data, B, step_size=pred_len))

    return np.array(Dtr), np.array(Val), np.array(Dte), np.array(m), np.array(n)


def nn_seq_emd_lstm(seq_len, B, pred_len):
    print('dataset processing...')
    dataset = load_data("澳大利亚.xlsx", "Sheet2")

    load = dataset['电力负荷']
    emd = EMD()
    decompose = emd(load.values)
    train = decompose[:, :int(len(load) * 0.6)]
    val = decompose[:, int(len(load) * 0.6):int(len(load) * 0.8)]
    test = decompose[:, int(len(load) * 0.8):len(load)]

    imfNum = decompose.shape[0]
    plt.figure()
    for i in range(imfNum):
        plt.subplot(int(imfNum / 2), 2, i + 1)
        plt.plot(decompose[i, :])
    plt.show()
    m = []
    n = []
    for i in range(train.shape[0]):
        m.append(train[i, :].max())
        n.append(train[i, :].min())

    def process(ds, batch_size, step_size):
        seq = []
        for i in range(0, len(ds) - seq_len - pred_len, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [ds[j]]
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + pred_len):
                train_label.append(ds[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = []
    Val = []
    Dte = []
    for i in range(decompose.shape[0]):
        data = (train[i, :] - n[i]) / (m[i] - n[i])
        Dtr.append(process(data, B, step_size=1))

        data = (val[i, :] - n[i]) / (m[i] - n[i])
        Val.append(process(data, B, step_size=1))

        data = (test[i, :] - n[i]) / (m[i] - n[i])
        Dte.append(process(data, B, step_size=pred_len))

    return np.array(Dtr), np.array(Val), np.array(Dte), np.array(m), np.array(n)


def vmd(data, alpha=1e3, tau=1e-2, K=5, DC=0, init=1, tol=1e-7):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    residual = data.T - np.sum(u, axis=0)
    # 波形
    plt.figure(dpi=300)
    for i in range(K):
        plt.subplot(int(np.ceil((K + 1) / 2)), 2, i + 1)
        plt.plot(u[i, :])
        plt.title('IMF-' + str(i + 1))
    plt.subplot(int(np.ceil((K + 1) / 2)), 2, K + 1)
    plt.plot(residual.T)
    plt.title('残差', )
    plt.tight_layout()
    plt.show()
    # 频谱
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

    return np.row_stack((u, residual))


# LSTM dataset processing.
def nn_seq_lstm(seq_len, B, pred_len):
    print('dataset processing...')
    dataset = load_data("澳大利亚.xlsx", "Sheet2")
    # split
    dataset = dataset.iloc[:, 2:]

    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[-1]]), np.min(train[train.columns[-1]])

    def process(ds, batch_size, step_size):
        load = ds[ds.columns[-1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        ds = ds.values.tolist()
        seq = []
        for i in range(0, len(ds) - seq_len - pred_len, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                # for c in range(5):
                #     x.append(ds[j][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + pred_len):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B, step_size=pred_len)

    return Dtr, Val, Dte, m, n




def nn_seq_bpnn(seq_len, B, pred_len):
    print('dataset processing...')
    dataset = load_data("澳大利亚.xlsx", "Sheet2")
    # split
    dataset = dataset.iloc[:, 2:]

    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[-1]]), np.min(train[train.columns[-1]])

    def process(ds, batch_size, step_size):
        load = ds[ds.columns[-1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        ds = ds.values.tolist()
        seq = []
        for i in range(0, len(ds) - seq_len - pred_len, step_size):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                train_seq.append(load[j])
                # for c in range(5):
                #     train_seq.append(dataset[j][c])
            for j in range(i + seq_len, i + seq_len + pred_len):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1)
    Val = process(val, B, step_size=1)
    Dte = process(test, B, step_size=pred_len)
    return Dtr, Val, Dte, m, n


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
