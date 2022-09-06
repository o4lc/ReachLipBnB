# from tabnanny import verbose

import torch

from packages import *
from utilities import Plotter
from BranchAndBoundNode import BB_node
from Bounding.LipschitzBound import LipschitzBounding
from Bounding.PgdUpperBound import PgdUpperBound
from Utilities.Timer import Timers


class BranchAndBound:
    def __init__(self, coordUp=None, coordLow=None, verbose=False, pgdStepSize=1e-3,
                 inputDimension=2, eps=0.1, network=None, queryCoefficient=None,
                 pgdIterNum=5, pgdNumberOfInitializations=2, device=torch.device("cuda", 0),
                 maximumBatchSize=256,  nodeBranchingFactor=2, branchNodeNum = 1,
                 scoreFunction='length',
                 virtualBranching=False, numberOfVirtualBranches=4,
                 maxSearchDepthLipschitzBound=10,
                 normToUseLipschitz=2, useTwoNormDilation=False, useSdpForLipschitzCalculation=False,
                 lipschitzSdpSolverVerbose=False, initialGD=False, previousLipschitzCalculations=[],
                 originalNetwork=None,
                 horizonForLipschitz=1
                 ):

        self.spaceNodes = [BB_node(np.infty, -np.infty, coordUp, coordLow, scoreFunction=scoreFunction)]
        self.bestUpperBound = None
        self.bestLowerBound = None
        self.initCoordUp = coordUp
        self.initCoordLow = coordLow
        self.verbose = verbose
        self.pgdIterNum = pgdIterNum
        self.pgdNumberOfInitializations = pgdNumberOfInitializations
        self.inputDimension = inputDimension
        self.eps = eps
        self.network = network
        self.queryCoefficient = queryCoefficient
        self.lowerBoundClass = LipschitzBounding(network, device, virtualBranching, maxSearchDepthLipschitzBound,
                                                 normToUseLipschitz, useTwoNormDilation, useSdpForLipschitzCalculation,
                                                 numberOfVirtualBranches, lipschitzSdpSolverVerbose,
                                                 previousLipschitzCalculations,
                                                 originalNetwork=originalNetwork,
                                                 horizon=horizonForLipschitz
                                                 )
        self.upperBoundClass = PgdUpperBound(network, pgdNumberOfInitializations, pgdIterNum, pgdStepSize,
                                             inputDimension, device, maximumBatchSize)
        self.nodeBranchingFactor = nodeBranchingFactor
        self.scoreFunction = scoreFunction
        self.branchNodeNum = branchNodeNum
        self.device = device
        self.maximumBatchSize = maximumBatchSize
        self.initialGD = initialGD
        self.timers = Timers(["lowerBound",
                              "lowerBound:lipschitzForwardPass", "lowerBound:lipschitzCalc",
                              "lowerBound:lipschitzSearch",
                              "lowerBound:virtualBranchPreparation", "lowerBound:virtualBranchMin",
                              "upperBound",
                              "bestBound",
                              "branch", "branch:prune", "branch:maxFind", "branch:nodeCreation",
                              ])

    def prune(self):
        # slightly faster since this starts deleting from the end of the list.
        for i in range(len(self.spaceNodes) - 1, -1, -1):
            if self.spaceNodes[i].lower > self.bestUpperBound:
                self.spaceNodes.pop(i)
                if self.verbose:
                    print('deleted')
        # for node in self.spaceNodes:
        #     if node.lower >= self.bestUpperBound:
        #         self.spaceNodes.remove(node)
        #         if self.verbose:
        #             print('deleted')

    def lowerBound(self, indices):
        lowerBounds = torch.vstack([self.spaceNodes[index].coordLower for index in indices])
        upperBounds = torch.vstack([self.spaceNodes[index].coordUpper for index in indices])
        return self.lowerBoundClass.lowerBound(self.queryCoefficient, lowerBounds, upperBounds, timer=self.timers)

    def upperBound(self, indices):
        return self.upperBoundClass.upperBound(indices, self.spaceNodes, self.queryCoefficient)

    def branch(self):
        # Prunning Function
        self.timers.start("branch:prune")
        self.prune()
        self.timers.pause("branch:prune")
        numNodesAfterPrune = len(self.spaceNodes)

        self.timers.start("branch:maxFind")
        #@TODO Choosing the node to branch -> this parts should be swaped with the sort idea
        scoreArray = torch.Tensor([self.spaceNodes[i].score for i in range(len(self.spaceNodes))])
        scoreArraySorted = torch.argsort(scoreArray)
        if len(self.spaceNodes) > self.branchNodeNum:
            maxIndices = scoreArraySorted[len(scoreArraySorted) - self.branchNodeNum : len(scoreArraySorted)]
        else:
            maxIndices = scoreArraySorted[:]


        deletedUpperBounds = []
        deletedLowerBounds = []
        nodes = []
        maxIndices, __ = torch.sort(maxIndices, descending=True)
        # print('\n\n\n', maxIndices, scoreArray, scoreArraySorted)
        for maxIndex in maxIndices:
            node = self.spaceNodes.pop(maxIndex)
            nodes.append(node)
            # print("!!!", maxIndex, node.upper, node.lower)
            for i in range(self.nodeBranchingFactor):
                deletedUpperBounds.append(node.upper)
                deletedLowerBounds.append(node.lower)
        deletedLowerBounds = torch.Tensor(deletedLowerBounds).to(self.device)
        deletedUpperBounds = torch.Tensor(deletedUpperBounds).to(self.device)
        self.timers.pause("branch:maxFind")
        for j in range(len(nodes) - 1, -1, -1):
            self.timers.start("branch:nodeCreation")
            coordToSplitSorted = torch.argsort(nodes[j].coordUpper - nodes[j].coordLower)
            coordToSplit = coordToSplitSorted[len(coordToSplitSorted) - 1]
            # print(coordToSplit)
            #@TODO This can be optimized by keeping the best previous 'x's in that space
            node = nodes[j]


            # '''
            # @TODO
            # Python Float Calculation Problem
            # Need to round up ?
            # '''
            parentNodeUpperBound = node.coordUpper
            parentNodeLowerBound = node.coordLower


            # Numpy can do this more efficently!
            newIntervals = torch.linspace(parentNodeLowerBound[coordToSplit],
                                                    parentNodeUpperBound[coordToSplit],
                                                    self.nodeBranchingFactor + 1)
            for i in range(self.nodeBranchingFactor):
                tempLow = parentNodeLowerBound.clone()
                tempHigh = parentNodeUpperBound.clone()

                tempLow[coordToSplit] = newIntervals[i]
                tempHigh[coordToSplit] = newIntervals[i+1]
                self.spaceNodes.append(BB_node(np.infty, -np.infty, tempHigh, tempLow, scoreFunction=self.scoreFunction))

                if torch.any(tempHigh - tempLow < 1e-8):
                    self.spaceNodes[-1].score = -1
            self.timers.pause("branch:nodeCreation")
        
        numNodesAfterBranch = len(self.spaceNodes)
        numNodesAdded = numNodesAfterBranch - numNodesAfterPrune + len(maxIndices)

        return [len(self.spaceNodes) - j for j in range(1, numNodesAdded + 1)], deletedUpperBounds, deletedLowerBounds

    def bound(self, indices, parent_ub, parent_lb):

        self.timers.start("lowerBound")
        lowerBounds = torch.maximum(self.lowerBound(indices), parent_lb)
        # print("Start bounds" + "--------" * 15)
        # print("## LOWER BOUNDS ", lowerBounds)
        self.timers.pause("lowerBound")
        self.timers.start("upperBound")
        upperBounds = self.upperBound(indices)
        # print("## UPPER BOUNDS ", upperBounds)
        # print("End bounds" + "--------" * 15)
        # print(upperBounds)
        self.timers.pause("upperBound")
        for i, index in enumerate(indices):
            self.spaceNodes[index].upper = upperBounds[i]
            self.spaceNodes[index].lower = lowerBounds[i]

    def run(self):
        if self.initialGD:
            initUpperBoundClass = PgdUpperBound(self.network, 10, 1000, 0.001,
                                             self.inputDimension, self.device, self.maximumBatchSize)


            self.bestUpperBound = torch.Tensor(initUpperBoundClass.upperBound([0], self.spaceNodes, self.queryCoefficient))
            print(self.bestUpperBound)
        else:
            self.bestUpperBound = torch.Tensor([torch.inf]).to(self.device)
        self.bestLowerBound = torch.Tensor([-torch.inf]).to(self.device)

        if self.verbose:
            plotter = Plotter()

        self.bound([0], self.bestUpperBound, self.bestLowerBound)
        if self.scoreFunction == "worstLowerBound" or self.scoreFunction == "bestLowerBound":
            self.spaceNodes[0].calc_score()
        while self.bestUpperBound - self.bestLowerBound >= self.eps:
            print(len(self.spaceNodes))
            # for i in range(len(self.spaceNodes)):
            #     if self.spaceNodes[i].lower > self.spaceNodes[i].upper:
            #         print("@@: ", i, self.spaceNodes[i].lower, self.spaceNodes[i].upper)
            #         raise
            self.timers.start("branch")
            indices, deletedUb, deletedLb = self.branch()
            # print(indices, end=" ")
            # print("------", end=" ")
            # print(deletedUb, end=" ** ** ")
            # print(deletedLb)
            # print([self.spaceNodes[i].coordLower for i in indices])
            # print([self.spaceNodes[i].coordUpper for i in indices])
            self.timers.pause("branch")

            self.bound(indices, deletedUb, deletedLb)

            if self.scoreFunction == "worstLowerBound" or self.scoreFunction == "bestLowerBound":
                minimumIndex = len(self.spaceNodes) - self.branchNodeNum * self.nodeBranchingFactor
                if minimumIndex < 0:
                    minimumIndex = 0
                # minimumIndex = 0
                maximumIndex = len(self.spaceNodes)
                for i in range(minimumIndex, maximumIndex):
                    self.spaceNodes[i].score = self.spaceNodes[i].calc_score()

                    # print(self.spaceNodes[i].lower)


            self.timers.start("bestBound")

            # TODO: make this better by keeping the previous one.
            self.bestUpperBound =\
                torch.minimum(self.bestUpperBound,
                              torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))])))
            self.bestLowerBound = torch.min(
                torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            self.timers.pause("bestBound")
            #1.08618
            print('Best LB', self.bestLowerBound, 'Best UB', self.bestUpperBound, "diff", self.bestUpperBound - self.bestLowerBound)
            if self.verbose:
                # print('Best LB', self.bestLowerBound, 'Best UB', self.bestUpperBound)
                plotter.plotSpace(self.spaceNodes, self.initCoordLow, self.initCoordUp)
                # print('--------------------')

        if self.verbose:
            plotter.showAnimation(self.spaceNodes)
        self.timers.pauseAll()
        self.timers.print()
        print(self.lowerBoundClass.calculatedLipschitzConstants)
        print("number of calculated lipschitz constants ", len(self.lowerBoundClass.calculatedLipschitzConstants))

        return self.bestLowerBound, self.bestUpperBound, self.spaceNodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.spaceNodes)):
            string += self.spaceNodes[i].__repr__() 
            string += "\n"

        return string


        