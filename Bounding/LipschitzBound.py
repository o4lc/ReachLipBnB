from typing import List

import numpy as np
import torch
import torch.nn as nn


class LipschitzBounding:
    def __init__(self,
                 network: nn.Module,
                 device=torch.device("cuda", 0)):
        self.network = network
        self.device = device
        self.weights = self.extractWeightsFromNetwork(network)
        self.calculatedLipschitzConstants = []

    def lowerBound(self,
                   queryCoefficient: torch.Tensor,
                   inputLowerBound: torch.Tensor,
                   inputUpperBound: torch.Tensor):
        # this function is not optimal for cases in which an axis is cut into unequal segments

        # I added .float() at the end
        dilationVector = ((inputUpperBound - inputLowerBound) / torch.tensor(2., device=self.device)).float()

        batchSize = dilationVector.shape[0]
        pointsThatNeedLipschitzConstantCalculation = [i for i in range(batchSize)]
        lipschitzConstants = -torch.ones(batchSize)
        locationOfUnavailableConstants = {}
        for batchCounter in reversed(range(dilationVector.shape[0])):
            foundLipschitzConstant = False
            for i in range(len(self.calculatedLipschitzConstants)):
                existingDilationVector, lipschitzConstant = self.calculatedLipschitzConstants[i]
                if torch.norm(dilationVector[batchCounter, :] - existingDilationVector) < 1e-8:

                    pointsThatNeedLipschitzConstantCalculation.remove(batchCounter)
                    if lipschitzConstant == 1:
                        locationOfUnavailableConstants[batchCounter] = i
                    else:
                        lipschitzConstants[batchCounter] = lipschitzConstant
                        foundLipschitzConstant = True

                    break
            if not foundLipschitzConstant:
                locationOfUnavailableConstants[batchCounter] = len(self.calculatedLipschitzConstants)
                # suppose we divide equally along an axes. Then the lipschitz constant of the two subdomains are gonna
                # be the same. By adding the dilationVector of one of the sides, we are preventing the calculation of
                # the lipschitz constant for both sides when they are exactly the same.
                self.calculatedLipschitzConstants.append((dilationVector[batchCounter, :], -1))

        if len(pointsThatNeedLipschitzConstantCalculation) != 0:
            # dMatrix = torch.diag(dilationVector)
            # Incorporate the query coefficient and the dilation matrix into the weights so that the whole problem is a
            # neural network
            newWeights = [w.repeat(len(pointsThatNeedLipschitzConstantCalculation), 1, 1) for w in self.weights]
            # w @ D is equivalent to w * dilationVector
            # newWeights[0] = newWeights[0] @ dMatrix
            newWeights[0] = newWeights[0] * dilationVector.unsqueeze(1)
            queryCoefficientRepeated = queryCoefficient.repeat(len(pointsThatNeedLipschitzConstantCalculation), 1, 1)
            # newWeights[-1] = queryCoefficient @ newWeights[-1]
            newWeights[-1] = torch.bmm(queryCoefficientRepeated, newWeights[-1])
            newCalculatedLipschitzConstants = self.calculateLipschitzConstant(newWeights, self.device)[:, -1]
            for i in range(len(newCalculatedLipschitzConstants)):
                self.calculatedLipschitzConstants[locationOfUnavailableConstants[
                    pointsThatNeedLipschitzConstantCalculation[i]]] =\
                    (dilationVector[pointsThatNeedLipschitzConstantCalculation[i]], newCalculatedLipschitzConstants[i])
            for unavailableBatch in locationOfUnavailableConstants.keys():
                lipschitzConstants[unavailableBatch] =\
                    self.calculatedLipschitzConstants[locationOfUnavailableConstants[unavailableBatch]][1]
        if torch.any(lipschitzConstants < 0):
            print("error. lipschitz constant hasn't been calculated")
            raise
        centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)

        lowerBound = queryCoefficient @ self.network(centerPoint) - lipschitzConstants
        return lowerBound

    @staticmethod
    def calculateLipschitzConstant(weights: List[torch.Tensor], device=torch.device("cuda", 0)):
        """
        :param weights: Weights of the neural network starting from the first layer to the last.
        :return:
        """
        batchSize = weights[0].shape[0]
        numberOfWeights = len(weights)
        with torch.no_grad():  # is this needed?
            halfTensor = torch.tensor(0.5, device=device)
            ms = torch.zeros(batchSize, numberOfWeights, dtype=torch.float).to(device)
            ms[:, 0] = torch.linalg.norm(weights[0], float('inf'), dim=(1, 2))
            for i in range(1, numberOfWeights):
                multiplier = torch.tensor(1., device=device)
                temp = torch.zeros(batchSize).to(device)
                for j in range(i, -1, -1):
                    productMatrix = weights[i]
                    for k in range(i - 1, j - 1, -1):
                        # productMatrix = productMatrix @ weights[k]
                        productMatrix = torch.bmm(productMatrix, weights[k])
                    if j > 0:
                        multiplier *= halfTensor
                        temp += multiplier * torch.linalg.norm(productMatrix, float('inf'), dim=(1, 2)) * ms[:, j - 1]
                    else:
                        temp += multiplier * torch.linalg.norm(productMatrix, float('inf'), dim=(1, 2))
                ms[:, i] = temp
        return ms


    @staticmethod
    def extractWeightsFromNetwork(network: nn.Module):
        weights = []
        for name, param in network.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone())
        return weights
