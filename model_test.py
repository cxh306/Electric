import os
import sys

from scipy.interpolate import make_interp_spline

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from itertools import chain

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_process import device, get_mape, setup_seed
from models import BPNN, BiLSTM, LSTM, TransformerModel

setup_seed(20)


def transformer_test(args, Dte, path, m, n):
    print('loading model...')
    model = TransformerModel(args).to(args.device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    pred = []
    y = []
    for batch_idx, (seq, target) in enumerate(Dte, 0):
        seq = seq.to(args.device)
        target = target.to(args.device)
        with torch.no_grad():
            y_pred = model(seq)
            target = list(chain.from_iterable(target.tolist()))
            y.extend(target)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)

    y = (m - n) * y + n
    pred = (m - n) * pred + n
    plot(y,pred)
    print('mape:', get_mape(y, pred))
    # # plot
    # x = [i for i in range(1, 151)]
    # x_smooth = np.linspace(np.min(x), np.max(x), 900)
    # y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    #
    # y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    # plt.grid(axis='y')
    # plt.legend()
    # plt.show()


def vmd_lstm_test(args, Dte, path, m, n):
    pred = []
    y = []
    for i in range(Dte.shape[0]):
        y_ret, pred_ret = lstm_test(args, Dte[i], path + str(i) + '.pkl', m[i], n[i], i)
        y.append(y_ret)
        pred.append(pred_ret)

    pred = np.array(pred)
    y = np.array(y)
    pred = np.sum(pred, axis=0)
    y = np.sum(y, axis=0)
    plt.plot(y)
    plt.show()
    print('mape:', get_mape(y, pred))
    plot(y, pred)
    return y, pred


def emd_lstm_test(args, Dte, path, m, n):
    pred = []
    y = []
    for i in range(Dte.shape[0]):
        y_ret, pred_ret = lstm_test(args, Dte[i], path + str(i) + '.pkl', m[i], n[i],i)
        y.append(y_ret)
        pred.append(pred_ret)

    pred = np.array(pred)
    y = np.array(y)
    pred = np.sum(pred, axis=0)
    y = np.sum(y, axis=0)
    plt.plot(y)
    plt.show()
    print('mape:', get_mape(y, pred))
    plot(y, pred)
    return y, pred


def bpnn_test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, seq_len, hidden_sizes = args.input_size, args.seq_len, args.hidden_sizes
    output_size = args.output_size
    model = BPNN(input_size, seq_len, hidden_sizes, output_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)

    return y, pred


def lstm_test(args, Dte, path, m, n, mode):
    pred = []
    y = []
    print('loading models...')
    input_size, num_layers = args.input_size, args.num_layers
    output_size = args.output_size
    hidden_size = 64
    if mode == 0:
        hidden_size = args.hidden_size0
    if mode == 1:
        hidden_size = args.hidden_size1
    if mode == 2:
        hidden_size = args.hidden_size2
    if mode == 3:
        hidden_size = args.hidden_size3
    if mode == 4:
        hidden_size = args.hidden_size4
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)

    y = y * (m - n) + n
    pred = pred * (m - n) + n

    print('mape:', get_mape(y, pred))
    plot(y, pred)

    return y, pred


def plot(y, pred):
    # plot
    size = len(y)
    x = [i for i in range(0, size)]
    x_smooth = np.linspace(np.min(x), np.max(x), size * 5)
    y_smooth = make_interp_spline(x, y[0:size])(x_smooth)
    plt.figure(dpi=300)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    y_smooth = make_interp_spline(x, pred[0:size])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()
