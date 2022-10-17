import os

from args import bpnn_args_parser
from model_test import bpnn_test
from model_train import bpnn_train, load_data

path = os.path.abspath(os.path.dirname(os.getcwd()))
BPNN_PATH = '/Users/cxh/Desktop/research/Electric/models/bpnn.pkl'

if __name__ == '__main__':
    args = bpnn_args_parser()
    flag = 'bpnn'
    Dtr, Val, Dte, m, n = load_data(args, flag)
    bpnn_train(args, Dtr, Val, BPNN_PATH)
    bpnn_test(args, Dte, BPNN_PATH, m, n)
