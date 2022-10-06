from typing import List

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import cvxpy as cp
from scipy.linalg import block_diag


"""
We can make this function faster for the purpose of reachability analysis. In such uses, we would neet to save 
not only the lipschitz constant, but the query coefficient used to calculate the terms and with these, 
we would only need to recalculate the final step m_l for the new query coefficient 
"""


class LipschitzBounding:
    def __init__(self,
                 network: nn.Module,
                 device=torch.device("cuda", 0),
                 virtualBranching=False,
                 maxSearchDepth=10,
                 normToUse=2,
                 useTwoNormDilation=False,
                 useSdpForLipschitzCalculation=False,
                 numberOfVirtualBranches=4,
                 sdpSolverVerbose=False,
                 calculatedLipschitzConstants=[],
                 originalNetwork=None,
                 horizon=1):
        self.network = network
        self.device = device
        if originalNetwork:
            self.weights = self.extractWeightsFromNetwork(originalNetwork)
        else:
            self.weights = self.extractWeightsFromNetwork(self.network)
        self.calculatedLipschitzConstants = calculatedLipschitzConstants
        self.maxSearchDepth = maxSearchDepth
        self.performVirtualBranching = virtualBranching
        self.extractWeightsForMilp()
        self.normToUse = normToUse
        self.useTwoNormDilation = useTwoNormDilation
        self.useSdpForLipschitzCalculation = useSdpForLipschitzCalculation
        self.numberOfVirtualBranches = numberOfVirtualBranches
        if normToUse == 2:
            assert (not(self.useSdpForLipschitzCalculation and self.useTwoNormDilation))
        self.sdpSolverVerbose = sdpSolverVerbose
        self.horizon = horizon

    def lowerBound(self,
                   queryCoefficient: torch.Tensor,
                   inputLowerBound: torch.Tensor,
                   inputUpperBound: torch.Tensor,
                   virtualBranch=True,
                   timer=None):
        batchSize = inputUpperBound.shape[0]
        difference = inputUpperBound - inputLowerBound
        if virtualBranch and self.performVirtualBranching:
            self.startTime(timer, "lowerBound:virtualBranchPreparation")
            maxIndices = torch.argmax(difference, 1)
            newLowers = [inputLowerBound[i, :].clone() for i in range(batchSize)
                         for _ in range(self.numberOfVirtualBranches)]
            newUppers = [inputUpperBound[i, :].clone() for i in range(batchSize)
                         for _ in range(self.numberOfVirtualBranches)]
            for i in range(batchSize):
                for j in range(self.numberOfVirtualBranches):
                    newUppers[self.numberOfVirtualBranches * i + j][maxIndices[i]] = \
                        newLowers[self.numberOfVirtualBranches * i + j][maxIndices[i]] +\
                        (j + 1) * difference[i, maxIndices[i]] / self.numberOfVirtualBranches
                    newLowers[self.numberOfVirtualBranches * i + j][maxIndices[i]] +=\
                        j * difference[i, maxIndices[i]] / self.numberOfVirtualBranches

            newLowers = torch.vstack(newLowers)
            newUppers = torch.vstack(newUppers)
            self.pauseTime(timer, "lowerBound:virtualBranchPreparation")

            virtualBranchLowerBoundsExtra = self.lowerBound(queryCoefficient, newLowers, newUppers, False, timer=timer)
            self.startTime(timer, "lowerBound:virtualBranchMin")

            virtualBranchLowerBounds = torch.Tensor([torch.min(
                virtualBranchLowerBoundsExtra[i * self.numberOfVirtualBranches:(i + 1) * self.numberOfVirtualBranches])
                for i in range(0, batchSize)]).to(self.device)
            self.pauseTime(timer, "lowerBound:virtualBranchMin")

        # this function is not optimal for cases in which an axis is cut into unequal segments
        dilationVector = difference / torch.tensor(2., device=self.device)
        if (self.normToUse == 2 and not self.useTwoNormDilation) or self.normToUse == 1:
            if len(self.calculatedLipschitzConstants) == 0:
                self.startTime(timer, "lowerBound:lipschitzCalc")

                newWeights = [w.cpu().numpy() for w in self.weights]
                # print("==" ,queryCoefficient.unsqueeze(0).cpu().numpy().shape, newWeights[-1].shape)
                # newWeights[-1] = queryCoefficient.unsqueeze(0).cpu().numpy() @ newWeights[-1]
                # lipschitzConstant = torch.from_numpy(
                #     self.calculateLipschitzConstantSingleBatchNumpy(newWeights, normToUse=self.normToUse))[-1].to(
                #     self.device)
                # normalizer = (lipschitzConstant / 1) ** (1 / len(newWeights))
                normalizer = 1
                newWeights = [w / normalizer for w in newWeights]
                if self.useSdpForLipschitzCalculation and self.normToUse == 2:
                    num_neurons = sum([newWeights[i].shape[0] for i in range(len(newWeights) - 1)])
                    alpha = np.zeros((num_neurons, 1))
                    beta = np.ones((num_neurons, 1))
                    if self.horizon == 1:
                        # lipschitzConstant = torch.Tensor([lipSDP(newWeights, alpha, beta, verbose=self.sdpSolverVerbose)]).to(self.device)
                        # queryCoefficient = np.eye(6)
                        lipschitzConstant = torch.Tensor([lipSDP2(newWeights, alpha, beta,
                                                                  queryCoefficient.unsqueeze(0).cpu().numpy(),
                                                                # queryCoefficient,
                                                                  self.network.A.cpu().numpy(),
                                                                  self.network.B.cpu().numpy(),
                                                                  verbose=self.sdpSolverVerbose)]).to(self.device)
                        print(lipschitzConstant)
                    else:
                        l1 = torch.Tensor([lipSDP2(newWeights, alpha, beta,
                                                   queryCoefficient.unsqueeze(0).cpu().numpy(),
                                                   self.network.A.cpu().numpy(),
                                                   self.network.B.cpu().numpy(),
                                                   verbose=self.sdpSolverVerbose)]).to(self.device)
                        l2 = torch.Tensor([lipSDP2(newWeights, alpha, beta,
                                                   np.eye(self.network.A.shape[0]),
                                                   self.network.A.cpu().numpy(),
                                                   self.network.B.cpu().numpy(),
                                                   verbose=self.sdpSolverVerbose)]).to(self.device)
                        lipschitzConstant = l1 * l2 ** (self.horizon - 1)
                else:
                    lipschitzConstant = torch.from_numpy(
                        self.calculateLipschitzConstantSingleBatchNumpy(newWeights, normToUse=self.normToUse))[-1].to(
                        self.device)
                lipschitzConstant *= normalizer ** len(newWeights)
                if False:
                    print(lipschitzConstant)
                self.calculatedLipschitzConstants.append(lipschitzConstant)
                self.pauseTime(timer, "lowerBound:lipschitzCalc")
            else:
                lipschitzConstant = self.calculatedLipschitzConstants[0]
            multipliers = torch.linalg.norm(dilationVector, ord=self.normToUse, dim=1)
            additiveTerm = lipschitzConstant * multipliers
        elif self.normToUse == float("inf") or (self.normToUse == 2 and self.useTwoNormDilation):
            self.startTime(timer, "lowerBound:lipschitzSearch")
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
            self.pauseTime(timer, "lowerBound:lipschitzSearch")
            self.startTime(timer, "lowerBound:lipschitzCalc")
            if len(batchesThatNeedLipschitzConstantCalculation) != 0:
                if self.normToUse == 2:
                    normalizerDilationVector = torch.sqrt(difference.shape[1]) * dilationVector
                else:
                    normalizerDilationVector = dilationVector
                # Incorporate the query coefficient and the dilation matrix into the weights so that the whole problem is a
                # neural network
                """"""
                # Torch batch implementation
                newWeights = [w.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1) for w in self.weights]
                # w @ D is equivalent to w * dilationVector
                # newWeights[0] = newWeights[0] @ dMatrix

                newWeights[0] = newWeights[0] * normalizerDilationVector[batchesThatNeedLipschitzConstantCalculation, :].unsqueeze(1)
                # print(newWeights[0])
                queryCoefficientRepeated = queryCoefficient.repeat(len(batchesThatNeedLipschitzConstantCalculation), 1, 1)
                # newWeights[-1] = queryCoefficient @ newWeights[-1]

                newWeights[-1] = torch.bmm(queryCoefficientRepeated, newWeights[-1])
                # print(newWeights)
                newCalculatedLipschitzConstants = self.calculateLipschitzConstant(newWeights, self.device, self.normToUse)[:, -1]
                """"""
                # # Numpy single batch implementation
                # newCalculatedLipschitzConstants = []
                # for i in range(len(batchesThatNeedLipschitzConstantCalculation)):
                #     newWeights = [w.cpu().numpy() for w in self.weights]
                #     newWeights[0] = newWeights[0] * normalizerDilationVector[batchesThatNeedLipschitzConstantCalculation[i]:
                #                                                    batchesThatNeedLipschitzConstantCalculation[i] + 1, :].cpu().numpy()
                #     newWeights[-1] = queryCoefficient.unsqueeze(0).cpu().numpy() @ newWeights[-1]
                #     # print(newWeights)
                #     newCalculatedLipschitzConstants.append(torch.from_numpy(self.calculateLipschitzConstantSingleBatchNumpy(newWeights, normToUse=self.normToUse))[-1].to(self.device))

                """"""

                for i in range(len(newCalculatedLipschitzConstants)):
                    self.calculatedLipschitzConstants[locationOfUnavailableConstants[
                        batchesThatNeedLipschitzConstantCalculation[i]]][1] = newCalculatedLipschitzConstants[i]
                for unavailableBatch in locationOfUnavailableConstants.keys():
                    lipschitzConstants[unavailableBatch] =\
                        self.calculatedLipschitzConstants[locationOfUnavailableConstants[unavailableBatch]][1]

                # if len(batchesThatNeedLipschitzConstantCalculation) != 1:
                #     print(dilationVector[batchesThatNeedLipschitzConstantCalculation, :])
            self.pauseTime(timer, "lowerBound:lipschitzCalc")
            if torch.any(lipschitzConstants < 0):
                print("error. lipschitz constant hasn't been calculated")
                raise
            additiveTerm = lipschitzConstants

        centerPoint = (inputUpperBound + inputLowerBound) / torch.tensor(2., device=self.device)
        with torch.no_grad():
            self.startTime(timer, "lowerBound:lipschitzForwardPass")

            lowerBound = self.network(centerPoint) @ queryCoefficient - additiveTerm
            self.pauseTime(timer, "lowerBound:lipschitzForwardPass")

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
    def calculateLipschitzConstant(weights: List[torch.Tensor], device=torch.device("cuda", 0),
                                   normToUse=float("inf")):
        """
        :param weights: Weights of the neural network starting from the first layer to the last.
        :return:
        """
        batchSize = weights[0].shape[0]
        numberOfWeights = len(weights)

        halfTensor = torch.tensor(0.5, device=device)
        ms = torch.zeros(batchSize, numberOfWeights, dtype=torch.float).to(device)
        ms[:, 0] = torch.linalg.norm(weights[0], normToUse, dim=(1, 2))
        for i in range(1, numberOfWeights):
            multiplier = torch.tensor(1., device=device)
            temp = torch.zeros(batchSize).to(device)
            for j in range(i, -1, -1):
                productMatrix = weights[i]
                for k in range(i - 1, j - 1, -1):
                    productMatrix = torch.bmm(productMatrix, weights[k])
                if j > 0:
                    multiplier *= halfTensor
                    temp += multiplier * torch.linalg.norm(productMatrix, normToUse, dim=(1, 2)) * ms[:, j - 1]
                else:
                    temp += multiplier * torch.linalg.norm(productMatrix, normToUse, dim=(1, 2))
            ms[:, i] = temp
        return ms

    @staticmethod
    def startTime(timer, timerName):
        try:
            timer.start(timerName)
        except:
            pass

    @staticmethod
    def pauseTime(timer, timerName):
        try:
            timer.pause(timerName)
        except:
            pass

    @staticmethod
    def calculateLipschitzConstantSingleBatchNumpy(weights, normToUse=float("inf")):
        numberOfWeights = len(weights)
        ms = np.zeros(numberOfWeights, dtype=np.float64)
        ms[0] = np.linalg.norm(weights[0], normToUse)
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
                    temp += multiplier * np.linalg.norm(productMatrix, normToUse) * ms[j - 1]
                else:
                    temp += multiplier * np.linalg.norm(productMatrix, normToUse)
            ms[i] = temp
        # print(ms)
        return ms


    @staticmethod
    def extractWeightsFromNetwork(network: nn.Module):
        weights = []
        for name, param in network.Linear.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone())
        return weights

    def extractWeightsForMilp(self):
        weights = []
        bs = []
        for name, param in self.network.named_parameters():
            if "weight" in name:
                weights.append(param.detach().clone().cpu().numpy())
            if "bias" in name:
                bs.append(param.detach().clone().unsqueeze(1).cpu().numpy())

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
                prob.solve(solver=cp.MOSEK, warm_start=True, verbose=self.sdpSolverVerbose,
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


def lipSDP(weights, alpha, beta, verbose=False):
    num_layers = len(weights) - 1
    dim_in = weights[0].shape[1]
    dim_out = weights[-1].shape[0]
    dim_last_hidden = weights[-1].shape[1]
    hidden_dims = [weights[i].shape[0] for i in range(0, num_layers)]
    dims = [dim_in] + hidden_dims + [dim_out]
    num_neurons = sum(hidden_dims)

    # decision vars
    Lambda = cp.Variable((num_neurons, 1), nonneg=True)
    T = cp.diag(Lambda)
    rho = cp.Variable((1, 1), nonneg=True)

    A = weights[0]
    C = np.bmat([np.zeros((weights[-1].shape[0], dim_in + num_neurons - dim_last_hidden)), weights[-1]])
    D = np.bmat([np.eye(dim_in), np.zeros((dim_in, num_neurons))])

    for i in range(1, num_layers):
        A = block_diag(A, weights[i])

    A = np.bmat([A, np.zeros((A.shape[0], weights[num_layers].shape[1]))])
    B = np.eye(num_neurons)
    B = np.bmat([np.zeros((num_neurons, weights[0].shape[1])), B])
    A_on_B = np.bmat([[A], [B]])

    cons = [A_on_B.T @ cp.bmat(
        [[-2 * np.diag(alpha[:, 0]) @ np.diag(beta[:, 0]) @ T, np.diag(alpha[:, 0] + beta[:, 0]) @ T],
         [np.diag(alpha[:, 0] + beta[:, 0]) @ T, -2 * T]]) @ A_on_B + C.T @ C - rho * D.T @ D << 0]

    prob = cp.Problem(cp.Minimize(rho), cons)

    prob.solve(solver=cp.MOSEK, verbose=verbose)

    return np.sqrt(rho.value)[0][0]


def lipSDP2(weights, alpha, beta, coef, Asys, Bsys, verbose=False):
    # @TODO: Possible bug in weights input
    num_layers = len(weights) - 1
    dim_in = weights[0].shape[1]
    dim_out = weights[-1].shape[0]
    dim_last_hidden = weights[-1].shape[1]
    hidden_dims = [weights[i].shape[0] for i in range(0, num_layers)]
    dims = [dim_in] + hidden_dims + [dim_out]
    num_neurons = sum(hidden_dims)
    
    # decision vars
    Lambda = cp.Variable((num_neurons, 1), nonneg=True)
    T = cp.diag(Lambda)
    rho = cp.Variable((1, 1), nonneg=True)
    A = weights[0]
    # C = np.bmat([np.zeros((weights[-1].shape[0], dim_in + num_neurons - dim_last_hidden)), weights[-1]])
    E0 = np.bmat([np.eye(weights[0].shape[1]), np.zeros((weights[0].shape[1], dim_in + num_neurons - dim_in))])
    El = np.bmat([np.zeros((weights[-1].shape[1], dim_in + num_neurons - dim_last_hidden)), np.eye(weights[-1].shape[1])])
    # print(Asys.shape, E0.shape)
    # print(Bsys.shape, weights[-1].shape, El.shape)
    Asys = coef @ Asys
    Bsys = coef @ Bsys
    Cnew = Asys @ E0 + Bsys @ weights[-1] @ El

    C = Cnew

    D = np.bmat([np.eye(dim_in), np.zeros((dim_in, num_neurons))])

    for i in range(1, num_layers):
        A = block_diag(A, weights[i])

    A = np.bmat([A, np.zeros((A.shape[0], weights[num_layers].shape[1]))])
    B = np.eye(num_neurons)
    B = np.bmat([np.zeros((num_neurons, weights[0].shape[1])), B])
    A_on_B = np.bmat([[A], [B]])

    cons = [A_on_B.T @ cp.bmat(
        [[-2 * np.diag(alpha[:, 0]) @ np.diag(beta[:, 0]) @ T, np.diag(alpha[:, 0] + beta[:, 0]) @ T],
         [np.diag(alpha[:, 0] + beta[:, 0]) @ T, -2 * T]]) @ A_on_B + C.T @ C - rho * D.T @ D << 0]

    prob = cp.Problem(cp.Minimize(rho), cons)

    prob.solve(solver=cp.MOSEK, verbose=verbose)

    return np.sqrt(rho.value)[0][0]
