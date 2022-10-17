import os
import sys

from args import vmd_lstm_args_parser
from data_process import setup_seed
from model_test import vmd_lstm_test
from model_train import load_data, vmd_lstm_train

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

setup_seed(20)
path = os.path.abspath(os.path.dirname(os.getcwd()))
VMD_LSTM_PATH = '/Users/cxh/Desktop/research/Electric/models/vmd-lstm/lstm-imf'

if __name__ == '__main__':
    args = vmd_lstm_args_parser()
    flag = 'vmd-lstm'
    Dtr, Val, Dte, m, n = load_data(args, flag)
    vmd_lstm_train(args, Dtr, Val, VMD_LSTM_PATH)
    vmd_lstm_test(args, Dte, VMD_LSTM_PATH, m, n)
