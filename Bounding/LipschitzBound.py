from typing import List

import numpy as np
import torch
import torch.nn as nn


def upperBoundWithLipschitz(network, queryCoefficient, inputLowerBound, inputUpperBound, device):
    weights = extractWeightsFromNetwork(network)
    dMatrix = torch.diag(torch.tensor(2., device=device) / (inputUpperBound - inputLowerBound))
    weights[0] = weights[0] @ dMatrix
    weights[-1] = queryCoefficient @ weights[-1]
    lipschitzConstant = calculateLipschitzConstant(weights, device)[-1]
    upperBound = queryCoefficient @ network((inputUpperBound + inputLowerBound) / torch.tensor(2., device=device)) + lipschitzConstant
    return upperBound


def calculateLipschitzConstant(weights: List[np.ndarray],
                               device=torch.device('cuda', 0)):
    """
    :param weights: Weights of the neural network starting from the first layer to the last.
    :return:
    """
    halfTensor = torch.tensor(0.5, device=device)
    ms = torch.zeros(len(weights), dtype=torch.float).to(device)
    ms[0] = torch.linalg.norm(weights[0], float('inf'))
    for i in range(1, len(weights)):
        multiplier = torch.tensor(1., device=device)
        temp = torch.tensor([0.]).to(device)
        for j in range(i, -1, -1):
            productMatrix = weights[i]
            for k in range(i - 1, j - 1, -1):
                productMatrix = productMatrix @ weights[k]
            if j > 0:
                multiplier *= halfTensor
                temp += multiplier * torch.linalg.norm(productMatrix, float('inf')) * ms[j - 1]
            else:
                temp += multiplier * torch.linalg.norm(productMatrix, float('inf'))
        ms[i] = temp
    return ms


def extractWeightsFromNetwork(network: nn.Module):

    weights = []
    for name, param in network.named_parameters():
        if "weight" in name:
            weights.append(param.detach().clone())
    return weights
