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
                 virtualBranching=False):
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
        self.lowerBoundClass = LipschitzBounding(network, device, virtualBranching)
        self.upperBoundClass = PgdUpperBound(network, pgdNumberOfInitializations, pgdIterNum, pgdStepSize,
                                             inputDimension, device, maximumBatchSize)
        self.nodeBranchingFactor = nodeBranchingFactor
        self.scoreFunction = scoreFunction
        self.branchNodeNum = branchNodeNum
        self.device = device
        self.maximumBatchSize = maximumBatchSize
        self.timers = Timers(["lowerBound", "upperBound", "branch", "prune", "maxFind", "nodeCreation", "bestBound"])

    def prune(self):
        # slightly faster since this starts deleting from the end of the list.
        for i in range(len(self.spaceNodes) - 1, -1, -1):
            if self.spaceNodes[i].lower >= self.bestUpperBound:
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
        return self.lowerBoundClass.lowerBound(self.queryCoefficient, lowerBounds, upperBounds)

    def upperBound(self, indices):
        return self.upperBoundClass.upperBound(indices, self.spaceNodes, self.queryCoefficient)

    def branch(self):
        # Prunning Function
        self.timers.start("prune")
        self.prune()
        self.timers.pause("prune")
        numNodesAfterPrune = len(self.spaceNodes)

        self.timers.start("maxFind")
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
            for i in range(self.nodeBranchingFactor):
                deletedUpperBounds.append(node.upper)
                deletedLowerBounds.append(node.lower)
        deletedLowerBounds = torch.Tensor(deletedLowerBounds)
        deletedUpperBounds = torch.Tensor(deletedUpperBounds)
        self.timers.pause("maxFind")
        for j in range(len(nodes)):
            self.timers.start("nodeCreation")
            coordToSplitSorted = torch.argsort(nodes[j].coordUpper - nodes[j].coordLower)
            coordToSplit = coordToSplitSorted[len(coordToSplitSorted) - 1]

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

            self.timers.pause("nodeCreation")
        
        numNodesAfterBranch = len(self.spaceNodes)
        numNodesAdded = numNodesAfterBranch - numNodesAfterPrune + len(maxIndices)

        return [len(self.spaceNodes) - j for j in range(1, numNodesAdded + 1)], deletedUpperBounds, deletedLowerBounds

    def bound(self, indices, parent_ub, parent_lb):
        self.timers.start("lowerBound")
        lowerBounds = torch.maximum(self.lowerBound(indices), parent_lb)
        self.timers.pause("lowerBound")
        self.timers.start("upperBound")
        upperBounds = self.upperBound(indices)
        self.timers.pause("upperBound")
        for i, index in enumerate(indices):
            self.spaceNodes[index].upper = upperBounds[i]
            self.spaceNodes[index].lower = lowerBounds[i]

    def run(self):
        self.bestUpperBound = torch.Tensor([torch.inf]).to(self.device)
        self.bestLowerBound = torch.Tensor([-torch.inf]).to(self.device)

        if self.verbose:
            plotter = Plotter()

        self.bound([0], self.bestUpperBound, self.bestLowerBound)
        while self.bestUpperBound - self.bestLowerBound >= self.eps:
            self.timers.start("branch")
            indices, deletedUb, deletedLb = self.branch()
            self.timers.pause("branch")

            self.bound(indices, deletedUb, deletedLb)
            self.timers.start("bestBound")
            self.bestUpperBound = torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))]))
            self.bestLowerBound = torch.min(torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            self.timers.pause("bestBound")
            if self.verbose:
                print('Best UB', self.bestLowerBound, 'Best LB', self.bestUpperBound)
                plotter.plotSpace(self.spaceNodes, self.initCoordLow, self.initCoordUp)
                print('--------------------')

        if self.verbose:
            plotter.showAnimation()
        self.timers.pauseAll()
        self.timers.print()
        return self.bestLowerBound, self.bestUpperBound, self.spaceNodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.spaceNodes)):
            string += self.spaceNodes[i].__repr__() 
            string += "\n"

        return string
        