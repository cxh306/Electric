import os
import sys

from args import emd_lstm_args_parser
from data_process import setup_seed
from model_test import lstm_test, vmd_lstm_test, emd_lstm_test
from model_train import lstm_train, load_data, vmd_lstm_train, emd_lstm_train

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

setup_seed(20)
path = os.path.abspath(os.path.dirname(os.getcwd()))
VMD_LSTM_PATH = '/Users/cxh/Desktop/research/Electric/models/emd-lstm/lstm-imf'

if __name__ == '__main__':
    args = emd_lstm_args_parser()
    flag = 'emd-lstm'
    Dtr, Val, Dte, m, n = load_data(args, flag)
    emd_lstm_train(args, Dtr, Val, VMD_LSTM_PATH)
    emd_lstm_test(args, Dte, VMD_LSTM_PATH, m, n)
