from typing import List

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import cvxpy as cp


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
        self.maxSearchDepth = 10
        self.performVirtualBranching = virtualBranching
        self.extractWeightsForMilp()

    def lowerBound(self,
                   queryCoefficient: torch.Tensor,
                   inputLowerBound: torch.Tensor,
                   inputUpperBound: torch.Tensor,
                   virtualBranch=True,
                   timer=None):
        batchSize = inputUpperBound.shape[0]
        difference = inputUpperBound - inputLowerBound
        if virtualBranch and self.performVirtualBranching:
            timer.start("virtualBranchPreparation")
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
                        (j + 1) * difference[i, maxIndices[i]] / numberOfVirtualBranches
                    newLowers[numberOfVirtualBranches * i + j][maxIndices[i]] +=\
                        j * difference[i, maxIndices[i]] / numberOfVirtualBranches

            newLowers = torch.vstack(newLowers)
            newUppers = torch.vstack(newUppers)
            timer.pause("virtualBranchPreparation")
            virtualBranchLowerBoundsExtra = self.lowerBound(queryCoefficient, newLowers, newUppers, False, timer=timer)
            timer.start("virtualBranchMin")
            virtualBranchLowerBounds = torch.Tensor([torch.min(
                virtualBranchLowerBoundsExtra[i * numberOfVirtualBranches:(i + 1) * numberOfVirtualBranches])
                for i in range(0, batchSize)]).to(self.device)
            timer.pause("virtualBranchMin")

        # this function is not optimal for cases in which an axis is cut into unequal segments
        dilationVector = difference / torch.tensor(2., device=self.device)

        timer.start("lipschitzSearch")
        batchesThatNeedLipschitzConstantCalculation = [i for i in range(batchSize)]
        lipschitzConstants = -torch.ones(batchSize, device=self.device)
        locationOfUnavailableConstants = {}
        for batchCounter in range(batchSize):  # making it reversed might just help a tiny amount.
            foundLipschitzConstant = False
            if not foundLipschitzConstant:
                for i in range(len(self.calculatedLipschitzConstants) - 1,
                               max(len(self.calculatedLipschitzConstants) - self.maxSearchDepth, -1), -1):
                    existingDilationVector, lipschitzConstant = self.calculatedLipschitzConstants[i]
                    # if torch.norm(dilationVector[batchCounter, :] - existingDilationVector) < 1e-8:
                    if torch.allclose(dilationVector[batchCounter, :], existingDilationVector, rtol=1e-3, atol=1e-7):
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
                self.calculatedLipschitzConstants.append([dilationVector[batchCounter, :], -1])
        timer.pause("lipschitzSearch")
        timer.start("lipschitzCalc")
        if len(batchesThatNeedLipschitzConstantCalculation) != 0:
            # Incorporate the query coefficient and the dilation matrix into the weights so that the whole problem is a
            # neural network
            """"""
            # Torch batch implementation
            newWeights = [w.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1) for w in self.weights]
            # w @ D is equivalent to w * dilationVector
            # newWeights[0] = newWeights[0] @ dMatrix
            newWeights[0] = newWeights[0] * dilationVector[batchesThatNeedLipschitzConstantCalculation, :].unsqueeze(1)
            # print(newWeights[0])
            queryCoefficientRepeated = queryCoefficient.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1)
            # newWeights[-1] = queryCoefficient @ newWeights[-1]

            newWeights[-1] = torch.bmm(queryCoefficientRepeated, newWeights[-1])
            # print(newWeights)
            newCalculatedLipschitzConstants = self.calculateLipschitzConstant(newWeights, self.device)[:, -1]
            """"""
            # # Numpy single batch implementation
            # newCalculatedLipschitzConstants = []
            # for i in range(len(batchesThatNeedLipschitzConstantCalculation)):
            #     newWeights = [w.numpy() for w in self.weights]
            #     newWeights[0] = newWeights[0] * dilationVector[batchesThatNeedLipschitzConstantCalculation[i]:
            #                                                    batchesThatNeedLipschitzConstantCalculation[i] + 1, :].numpy()
            #     newWeights[-1] = queryCoefficient.unsqueeze(0).numpy() @ newWeights[-1]
            #     # print(newWeights)
            #     newCalculatedLipschitzConstants.append(torch.from_numpy(self.calculateLipschitzConstantSingleBatchNumpy(newWeights))[-1])

            """"""

            for i in range(len(newCalculatedLipschitzConstants)):
                self.calculatedLipschitzConstants[locationOfUnavailableConstants[
                    batchesThatNeedLipschitzConstantCalculation[i]]][1] = newCalculatedLipschitzConstants[i]
            for unavailableBatch in locationOfUnavailableConstants.keys():
                lipschitzConstants[unavailableBatch] =\
                    self.calculatedLipschitzConstants[locationOfUnavailableConstants[unavailableBatch]][1]

            if len(batchesThatNeedLipschitzConstantCalculation) != 1:
                print(dilationVector[batchesThatNeedLipschitzConstantCalculation, :])
        timer.pause("lipschitzCalc")
        if torch.any(lipschitzConstants < 0):
            print("error. lipschitz constant hasn't been calculated")
            raise

        centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)
        with torch.no_grad():
            timer.start("lipschitzForwardPass")

            lowerBound = self.network(centerPoint) @ queryCoefficient - lipschitzConstants
            timer.pause("lipschitzForwardPass")

            # for batchCounter in range(batchSize):
            #     import time
            #     startTime = time.time()
            #     actualLowerBound = self.milpSolver(inputLowerBound[batchCounter, :].numpy(),
            #                                        inputUpperBound[batchCounter, :].numpy())
            #     print("took to run MILP", time.time() - startTime)
            #     print("**", actualLowerBound, lowerBound[batchCounter], inputLowerBound[batchCounter, :], inputUpperBound[batchCounter, :])
            #     raise
            #     if actualLowerBound < lowerBound[batchCounter]:
            #         print(actualLowerBound, lowerBound[batchCounter])
            #         raise
        if virtualBranch and self.performVirtualBranching:
            lowerBound = torch.maximum(lowerBound, virtualBranchLowerBounds)
        return lowerBound

    @staticmethod
    def calculateLipschitzConstant(weights: List[torch.Tensor], device=torch.device("cuda", 0)):
        """
        :param weights: Weights of the neural network starting from the first layer to the last.
        :return:
        """
        batchSize = weights[0].shape[0]
        numberOfWeights = len(weights)

        halfTensor = torch.tensor(0.5, device=device)
        ms = torch.zeros(batchSize, numberOfWeights, dtype=torch.float).to(device)
        ms[:, 0] = torch.linalg.norm(weights[0], float('inf'), dim=(1, 2))
        for i in range(1, numberOfWeights):
            multiplier = torch.tensor(1., device=device)
            temp = torch.zeros(batchSize).to(device)
            for j in range(i, -1, -1):
                productMatrix = weights[i]
                for k in range(i - 1, j - 1, -1):
                    productMatrix = torch.bmm(productMatrix, weights[k])
                if j > 0:
                    multiplier *= halfTensor
                    temp += multiplier * torch.linalg.norm(productMatrix, float('inf'), dim=(1, 2)) * ms[:, j - 1]
                else:
                    temp += multiplier * torch.linalg.norm(productMatrix, float('inf'), dim=(1, 2))
            ms[:, i] = temp
        return ms

    @staticmethod
    def calculateLipschitzConstantSingleBatchNumpy(weights):
        numberOfWeights = len(weights)
        ms = np.zeros(numberOfWeights, dtype=np.float64)
        ms[0] = np.linalg.norm(weights[0], float('inf'))
        for i in range(1, numberOfWeights):
            multiplier = 1.
            temp = 0.
            for j in range(i, -1, -1):
                productMatrix = weights[i]
                for k in range(i - 1, j - 1, -1):
                    productMatrix = productMatrix @ weights[k]
                    # if j == 0:
                    #     print(weights[k])
                if j > 0:
                    multiplier *= 0.5
                    temp += multiplier * np.linalg.norm(productMatrix, float('inf')) * ms[j - 1]
                else:
                    temp += multiplier * np.linalg.norm(productMatrix, float('inf'))
            ms[i] = temp
        # print(ms)
        return ms


    @staticmethod
    def extractWeightsFromNetwork(network: nn.Module):
        weights = []
        for name, param in network.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone())
        return weights

    def extractWeightsForMilp(self):
        weights = []
        bs = []
        for name, param in self.network.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone().numpy())
            if "bias" in name:
                bs.append(param.detach().clone().unsqueeze(1).numpy())

        self.bs = bs
        self.Ws = weights
        self.dimensionList = [w.shape for w in weights]

    @staticmethod
    def mixedIntegerConstraints(inputVariable, outputVariable, integerVariable,
                                constraintSet, inputLowerBound, inputUpperBound):
        constraintSet.append(outputVariable >= 0)
        constraintSet.append(outputVariable >= inputVariable)
        constraintSet.append(
            outputVariable <= inputVariable - cp.multiply(inputLowerBound[np.newaxis].transpose(), 1 - integerVariable))
        constraintSet.append(outputVariable <= cp.multiply(inputUpperBound[np.newaxis].transpose(), integerVariable))
        constraintSet.append(integerVariable >= 0)
        constraintSet.append(integerVariable <= 1)

    def milpSolver(self, l, u):
        s, t = self.propagateBoundsInNetwork(l, u, self.Ws, self.bs)
        xs = [cp.Variable((self.dimensionList[0][1], 1))]

        for i in range(1, len(self.dimensionList)):
            xs.append(cp.Variable((self.dimensionList[i][1], 1)))
        violatingPlanes = [np.array([[1., 2.]])]
        violatingObjectiveValues = [0]

        # Define constraints
        constraints = [xs[0] >= s[0][:, np.newaxis], xs[0] <= t[0][:, np.newaxis]]
        ########################## place method-specific code here #############################
        WXPlusBs = [w @ x + b for w, x, b in zip(self.Ws, xs, self.bs)]
        integerVariables = [cp.Variable((self.dimensionList[i][0], 1), integer=True) for i in range(len(self.dimensionList) - 1)]

        for i in range(len(self.dimensionList) - 1):
            self.mixedIntegerConstraints(WXPlusBs[i], xs[i + 1], integerVariables[i],
                                         constraints, s[i + 1], t[i + 1])

        newViolatingPlanes = []
        newViolatingPlanesCorrect = []
        for i, (violatingPlane, objectiveValue) in enumerate(zip(violatingPlanes, violatingObjectiveValues)):
            outputVariableMatrix = self.Ws[-1] @ xs[-1] + self.bs[-1]
            c = violatingPlane[0]
            objective = cp.Minimize(c @ outputVariableMatrix)
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False,
                           mosek_params={
                               "MSK_DPAR_MIO_REL_GAP_CONST": 1e-15,
                               "MSK_DPAR_MIO_TOL_ABS_RELAX_INT": 1e-9,
                               "MSK_DPAR_MIO_TOL_FEAS": 1e-9,
                               "MSK_DPAR_MIO_TOL_REL_GAP": 0,
                           }
                           )
            except:
                print(prob.status, end='\n\n')
                continue
            difference = abs(objectiveValue - objective.value)
            relative = abs(difference / objective.value)
            if relative < 1e-3:
                continue
            # print(objective.value)

        return objective.value

    @staticmethod
    def calculateBoundsAfterLinearTransformation(weight, bias, lowerBound, upperBound):
        """
        :param weight:
        :param bias: A (n * 1) matrix
        :param lowerBound: A vector and not an (n * 1) matrix
        :param upperBound: A vector and not an (n * 1) matrix
        :return:
        """

        outputLowerBound = (np.maximum(weight, 0) @ (lowerBound[np.newaxis].transpose())
                            + np.minimum(weight, 0) @ (upperBound[np.newaxis].transpose()) + bias).squeeze()
        outputUpperBound = (np.maximum(weight, 0) @ (upperBound[np.newaxis].transpose())
                            + np.minimum(weight, 0) @ (lowerBound[np.newaxis].transpose()) + bias).squeeze()

        return outputLowerBound, outputUpperBound


    def propagateBoundsInNetwork(self, l, u, weights, biases):
        relu = lambda x: np.maximum(x, 0)

        s, t = [l], [u]
        for i, (W, b) in enumerate(zip(weights, biases)):
            val1, val2 = s[-1], t[-1]
            if 0 < i:
                val1, val2 = relu(val1), relu(val2)
            if val1.shape == ():
                val1 = np.array([val1])
                val2 = np.array([val2])
            sTemp, tTemp = self.calculateBoundsAfterLinearTransformation(W, b, val1, val2)
            if sTemp.shape == ():
                sTemp = np.array([sTemp])
                tTemp = np.array([tTemp])
            s.append(sTemp)
            t.append(tTemp)
        return s, t

