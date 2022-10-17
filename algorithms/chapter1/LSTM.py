import os
import sys

from args import lstm_args_parser
from data_process import setup_seed
from model_test import lstm_test
from model_train import lstm_train, load_data

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

setup_seed(20)
path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = '/Users/cxh/Desktop/research/Electric/models/lstm.pkl'

if __name__ == '__main__':
    args = lstm_args_parser()
    flag = 'lstm'
    Dtr, Val, Dte, m, n = load_data(args, flag)

    # 没有进行分解，mode默认为-1
    lstm_train(args, Dtr, Val, LSTM_PATH, -1)
    lstm_test(args, Dte, LSTM_PATH, m, n, -1)
