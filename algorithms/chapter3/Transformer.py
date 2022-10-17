# -*- coding:utf-8 -*-
import optuna
import torch
from torch import nn

from args import transformer_args_parser
from data_process import nn_seq_transformer
from model_test import transformer_test
from model_train import transformer_train, get_best_parameters

PATH = '/Users/cxh/Desktop/research/Electric/models/transformer'

if __name__ == '__main__':
    args = transformer_args_parser()
    Dtr, Val, Dte, m, n = nn_seq_transformer(args)
    get_best_parameters(args, Dtr, Val)
    transformer_train(args, Dtr, Val, PATH)
    transformer_test(args, Dte, PATH, m, n)
