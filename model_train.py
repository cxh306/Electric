import copy

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn import svm
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_process import device, nn_seq_bpnn, nn_seq_lstm, nn_seq_vmd_lstm, nn_seq_emd_lstm, nn_seq_transformer
from models import BPNN, BiLSTM, LSTM, TransformerModel

# 根据flag运行对应的预处理方法
def load_data(args, flag):
    if flag == 'bpnn':
        Dtr, Val, Dte, m, n = nn_seq_bpnn(seq_len=args.seq_len, B=args.batch_size, pred_len=args.output_size)
    if flag == 'lstm':
        Dtr, Val, Dte, m, n = nn_seq_lstm(seq_len=args.seq_len, B=args.batch_size, pred_len=args.output_size)
    if flag == 'vmd-lstm':
        Dtr, Val, Dte, m, n = nn_seq_vmd_lstm(seq_len=args.seq_len, B=args.batch_size, pred_len=args.output_size)
    if flag == 'emd-lstm':
        Dtr, Val, Dte, m, n = nn_seq_emd_lstm(seq_len=args.seq_len, B=args.batch_size, pred_len=args.output_size)
    if flag == 'transformer':
        Dtr, Val, Dte, m, n = nn_seq_transformer(args)
    return Dtr, Val, Dte, m, n

# loss计算
def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def transformer_train(args, Dtr, Val, path):
    model = TransformerModel(args).to(args.device)
    loss_function = nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('training...')
    epochs = args.epochs
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    final_val_loss = []
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(epochs):
        train_loss = []
        for batch_idx, (seq, target) in enumerate(Dtr, 0):
            seq, target = seq.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = loss_function(y_pred, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        final_val_loss.append(val_loss)
        model.train()

    state = {'model': best_model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)

    return np.mean(final_val_loss)

def get_best_parameters(args, Dtr, Val):
    def objective(trial):
        model = TransformerModel(args).to(args.device)
        loss_function = nn.MSELoss().to(args.device)
        optimizer = trial.suggest_categorical('optimizer',
                                              [torch.optim.SGD,
                                               torch.optim.RMSprop,
                                               torch.optim.Adam])(
            model.parameters(), lr=trial.suggest_loguniform('lr', 5e-4, 1e-2))
        print('training...')
        epochs = 10
        val_loss = 0
        for epoch in range(epochs):
            train_loss = []
            for batch_idx, (seq, target) in enumerate(Dtr, 0):
                seq, target = seq.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                y_pred = model(seq)
                loss = loss_function(y_pred, target)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            # validation
            val_loss = get_val_loss(args, model, Val)

            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
            model.train()

        return val_loss

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(func=objective, n_trials=5)
    pruned_trials = study.get_trials(deepcopy=False,
                                     states=tuple([TrialState.PRUNED]))
    complete_trials = study.get_trials(deepcopy=False,
                                       states=tuple([TrialState.COMPLETE]))
    best_trial = study.best_trial
    print('val_loss = ', best_trial.value)
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

def vmd_lstm_train(args, Dtr, Val, path):
    for i in range(Dtr.shape[0]):
        lstm_train(args, Dtr[i], Val[i], path + str(i) + '.pkl',i)


def emd_lstm_train(args, Dtr, Val, path):
    for i in range(Dtr.shape[0]):
        lstm_train(args, Dtr[i], Val[i], path + str(i) + '.pkl',i)


def bpnn_train(args, Dtr, Val, path):
    seq_len, input_size, hidden_sizes = args.seq_len, args.input_size, args.hidden_sizes
    output_size = args.output_size
    model = BPNN(input_size, seq_len, hidden_sizes, output_size).to(device)
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)


def lstm_train(args, Dtr, Val, path, mode):
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
    if mode == 5:
        hidden_size = args.hidden_size5
    if mode == 6:
        hidden_size = args.hidden_size6
    if mode == 7:
        hidden_size = args.hidden_size7

    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    train_loss_epoch = []
    val_loss_epoch = []
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        train_loss_epoch.append(np.mean(train_loss))
        val_loss_epoch.append(val_loss)
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)

# def svd_train(Dtr, kernel):
#     model = None
#     seq = [x[0] for x in Dtr]
#     label = [x[1] for x in Dtr]
#     if kernel == 'rbf':
#         # 高斯核函数
#         print('SVR - RBF')
#         model = svm.SVR(kernel='rbf', gamma=1e-8, C=100)
#         model.fit(seq[0], label[0])
#     elif kernel == 'linear':
#         # 线性核函数
#         print('SVR - Linear')
#         model = svm.SVR(kernel='linear', C=100)
#         model.fit(seq, label)
#     else:
#         # 多项式核函数
#         print('SVR - Polynomial')
#         model = svm.SVR(kernel='poly', degree=2, C=100)
#         model.fit(seq, label)
#     return model
