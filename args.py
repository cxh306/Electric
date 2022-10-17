# -*- coding:utf-8 -*-
"""
@Time：2022/04/15 15:30
@Author：KI
@File：args.py
@Motto：Hungry And Humble
"""
import argparse
import torch


def transformer_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--seq_len', type=int, default=48, help='seq len')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--d_model', type=int, default=32, help='input dimension')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma')

    args = parser.parse_args()

    return args


# Single step scroll
def bpnn_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=128, help='input dimension')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=48, help='seq len')
    parser.add_argument('--hidden_sizes', type=list, default=[128], help='hidden size')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args

def vmd_lstm_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=70, help='epohchs')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=48, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')

    # # vmd
    parser.add_argument('--hidden_size0', type=int, default=32, help='hidden size0')
    parser.add_argument('--hidden_size1', type=int, default=64, help='hidden size1')
    parser.add_argument('--hidden_size2', type=int, default=64, help='hidden size2')
    parser.add_argument('--hidden_size3', type=int, default=64, help='hidden size3')
    parser.add_argument('--hidden_size4', type=int, default=128, help='hidden size4')


    # normal
    # parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')

    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args


def emd_lstm_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='epohchs')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=48, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')

    # # emd
    parser.add_argument('--hidden_size0', type=int, default=32, help='hidden size0')
    parser.add_argument('--hidden_size1', type=int, default=64, help='hidden size1')
    parser.add_argument('--hidden_size2', type=int, default=64, help='hidden size2')
    parser.add_argument('--hidden_size3', type=int, default=64, help='hidden size3')
    parser.add_argument('--hidden_size4', type=int, default=128, help='hidden size4')
    parser.add_argument('--hidden_size5', type=int, default=64, help='hidden size5')
    parser.add_argument('--hidden_size6', type=int, default=64, help='hidden size6')
    parser.add_argument('--hidden_size7', type=int, default=64, help='hidden size7')


    # normal
    # parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')

    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args


def lstm_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='epohchs')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=48, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size')


    # normal
    # parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')

    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    args = parser.parse_args()

    return args