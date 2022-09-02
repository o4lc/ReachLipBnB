#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:45:41 2020

@author: mahyarfazlyab
"""

# import sys
# sys.path.append("../Python/")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import scipy.io
import os


def export2matlab(file_name, net, A, B, save_model=False):
    '''
    Export pytorch fully connected network to matlab

    '''

    num_layers = int((len(net) - 1) / 2)
    dim_in = float(net[0].weight.shape[1])
    dim_out = float(net[-1].weight.shape[0])
    hidden_dims = [float(net[2 * i].weight.shape[0]) for i in range(0, num_layers)]

    # network dimensions
    dims = [dim_in] + hidden_dims + [dim_out]

    # get weights
    # weights = np.zeros((num_layers+1,))
    weights = [net[2 * i].weight.detach().numpy().astype(np.float64) for i in range(0, num_layers + 1)]

    # get biases
    # biases = np.zeros((num_layers+1,))
    biases = [net[2 * i].bias.detach().numpy().astype(np.float64).reshape(-1, 1) for i in range(0, num_layers + 1)]

    activation = str(net[1])[0:-2].lower()

    # export network data to matlab
    data = {}
    data['net'] = {'weights': weights, 'biases': biases, 'dims': dims, 'activation': activation, 'name': file_name}
    data['AMatrix'] = A
    data['BMatrix'] = B

    scipy.io.savemat(file_name + '.mat', data)


def main():

    data = loadmat("quadRotorTrainData.mat")

    Xtrain = data['Xtrain']
    Ytrain = data['Ytrain']
    A = data['A']
    B = data['B']

    trainset = torch.utils.data.TensorDataset(torch.Tensor(Xtrain.T), torch.Tensor(Ytrain.T))

    net = nn.Sequential(
        nn.Linear(6, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3))

    train_batch_size = 128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=2)
    epoch = 200
    net.train()

    criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for t in range(epoch):
        for i, (X, Y) in enumerate(trainloader):
            out = net(X)
            loss = criterion(out, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (np.mod(t, 10) == 0):
            print('epoch: ', t, 'MSE loss: ', loss.item())

    export2matlab('quadRotor' , net, A, B, save_model=True)



if __name__ == '__main__':
    main()
