from typing import List

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


"""
We can make this function faster for the purpose of reachability analysis. In such uses, we would neet to save 
not only the lipschitz constant, but the query coefficient used to calculate the terms and with these, 
we would only need to recalculate the final step m_l for the new query coefficient 
"""


class LipschitzBounding:
    def __init__(self,
                 network: nn.Module,
                 device=torch.device("cuda", 0),
                 virtualBranching=False):
        self.network = network
        self.device = device
        self.weights = self.extractWeightsFromNetwork(self.network)
        self.calculatedLipschitzConstants = []
        self.performVirtualBranching = virtualBranching

    def lowerBound(self,
                   queryCoefficient: torch.Tensor,
                   inputLowerBound: torch.Tensor,
                   inputUpperBound: torch.Tensor,
                   virtualBranch=True,
                   timer=None):
        # print(virtualBranch)
        batchSize = inputUpperBound.shape[0]
        difference = inputUpperBound - inputLowerBound
        # print(difference)
        if virtualBranch and self.performVirtualBranching:
            numberOfVirtualBranches = 4
            maxIndices = torch.argmax(difference, 1)
            newLowers = [inputLowerBound[i, :].clone() for i in range(batchSize)
                         for _ in range(numberOfVirtualBranches)]
            newUppers = [inputUpperBound[i, :].clone() for i in range(batchSize)
                         for _ in range(numberOfVirtualBranches)]
            for i in range(batchSize):
                for j in range(numberOfVirtualBranches):
                    newUppers[numberOfVirtualBranches * i + j][maxIndices[i]] = \
                        newLowers[numberOfVirtualBranches * i + j][maxIndices[i]] +\
                        (i + 1) * difference[i, maxIndices[i]] / numberOfVirtualBranches
                    newLowers[numberOfVirtualBranches * i + j][maxIndices[i]] +=\
                        i * difference[i, maxIndices[i]] / numberOfVirtualBranches

            newLowers = torch.vstack(newLowers)
            newUppers = torch.vstack(newUppers)
            virtualBranchLowerBoundsExtra = self.lowerBound(queryCoefficient, newLowers, newUppers, False, timer=timer)
            timer.start("virtualBranchMin")
            virtualBranchLowerBounds = torch.Tensor([torch.min(
                virtualBranchLowerBoundsExtra[i * numberOfVirtualBranches:(i + 1) * numberOfVirtualBranches])
                for i in range(0, batchSize)])
            timer.pause("virtualBranchMin")
            # print("virtual done")
            # print(virtualBranchLowerBounds)


        # print("---------"*15)
        # this function is not optimal for cases in which an axis is cut into unequal segments

        # I added .float() at the end
        dilationVector = difference / torch.tensor(2., device=self.device)
        # print(dilationVector)

        timer.start("lipschitzSearch")
        batchesThatNeedLipschitzConstantCalculation = [i for i in range(batchSize)]
        lipschitzConstants = -torch.ones(batchSize, device=self.device)
        locationOfUnavailableConstants = {}
        previousDilation = None
        for batchCounter in range(batchSize):  # making it reversed might just help a tiny amount.
            foundLipschitzConstant = False
            if previousDilation is not None:
                if torch.norm(dilationVector[batchCounter, :] - previousDilation) < 1e-8:
                    if previousLipschitzConstant == -1:
                        locationOfUnavailableConstants[batchCounter] = len(self.calculatedLipschitzConstants) - 1
                    else:
                        lipschitzConstants[batchCounter] = previousLipschitzConstant
                    batchesThatNeedLipschitzConstantCalculation.remove(batchCounter)
                    foundLipschitzConstant = True
            if not foundLipschitzConstant:
                for i in range(len(self.calculatedLipschitzConstants) - 1, -1, -1):
                    existingDilationVector, lipschitzConstant = self.calculatedLipschitzConstants[i]
                    if torch.norm(dilationVector[batchCounter, :] - existingDilationVector) < 1e-8:
                        previousLipschitzConstant = lipschitzConstant
                        if lipschitzConstant == -1:
                            locationOfUnavailableConstants[batchCounter] = i
                        else:
                            lipschitzConstants[batchCounter] = lipschitzConstant
                        batchesThatNeedLipschitzConstantCalculation.remove(batchCounter)
                        foundLipschitzConstant = True

                        break
            if not foundLipschitzConstant:
                locationOfUnavailableConstants[batchCounter] = len(self.calculatedLipschitzConstants)
                # suppose we divide equally along an axes. Then the lipschitz constant of the two subdomains are gonna
                # be the same. By adding the dilationVector of one of the sides, we are preventing the calculation of
                # the lipschitz constant for both sides when they are exactly the same.
                self.calculatedLipschitzConstants.append((dilationVector[batchCounter, :], -1))
        # print(self.calculatedLipschitzConstants)
        # print(batchesThatNeedLipschitzConstantCalculation)
        timer.pause("lipschitzSearch")
        timer.start("lipschitzCalc")
        if len(batchesThatNeedLipschitzConstantCalculation) != 0:
            # print("running lipschitz calculations")
            # dMatrix = torch.diag(dilationVector)
            # Incorporate the query coefficient and the dilation matrix into the weights so that the whole problem is a
            # neural network
            newWeights = [w.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1) for w in self.weights]
            # w @ D is equivalent to w * dilationVector
            # newWeights[0] = newWeights[0] @ dMatrix
            newWeights[0] = newWeights[0] * dilationVector[batchesThatNeedLipschitzConstantCalculation, :].unsqueeze(1)
            queryCoefficientRepeated = queryCoefficient.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1)
            # newWeights[-1] = queryCoefficient @ newWeights[-1]

            newWeights[-1] = torch.bmm(queryCoefficientRepeated, newWeights[-1])

            newCalculatedLipschitzConstants = self.calculateLipschitzConstant(newWeights, self.device)[:, -1]
            for i in range(len(newCalculatedLipschitzConstants)):
                self.calculatedLipschitzConstants[locationOfUnavailableConstants[
                    batchesThatNeedLipschitzConstantCalculation[i]]] =\
                    (dilationVector[batchesThatNeedLipschitzConstantCalculation[i]], newCalculatedLipschitzConstants[i])
            # print(self.calculatedLipschitzConstants)
            for unavailableBatch in locationOfUnavailableConstants.keys():
                lipschitzConstants[unavailableBatch] =\
                    self.calculatedLipschitzConstants[locationOfUnavailableConstants[unavailableBatch]][1]
            # print(lipschitzConstants)
            # print(len(self.calculatedLipschitzConstants))
        timer.pause("lipschitzCalc")
        if torch.any(lipschitzConstants < 0):
            print("error. lipschitz constant hasn't been calculated")
            raise

        centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)
        with torch.no_grad():
            timer.start("lipschitzForwardPass")
            lowerBound = self.network(centerPoint) @ queryCoefficient - lipschitzConstants
            timer.pause("lipschitzForwardPass")
            # print(self.network(centerPoint) @ queryCoefficient, lipschitzConstants)
            # if torch.any(lipschitzConstants != lipschitzConstants[0]):
            #     print(lipschitzConstants)
        if virtualBranch and self.performVirtualBranching:
            lowerBound = torch.maximum(lowerBound, virtualBranchLowerBounds)
        # print(lowerBound)
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
            # print(ms)
        return ms


    @staticmethod
    def extractWeightsFromNetwork(network: nn.Module):
        weights = []
        for name, param in network.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone())
        return weights
