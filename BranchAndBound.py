from tabnanny import verbose

import torch

from packages import *
from utilities import Plotter
from BranchAndBoundNode import BB_node
from Bounding.LipschitzBound import LipschitzBounding
from Bounding.PgdUpperBound import PgdUpperBound


class BranchAndBound:
    def __init__(self, coordUp=None, coordLow=None, verbose=False, pgdStepSize=1e-3,
                 inputDimension=2, eps=0.1, network=None, queryCoefficient=None,
                 pgdIterNum=5, pgdNumberOfInitializations=2, device=torch.device("cuda", 0),
                 branchingMethod='SimpleBranch', nodeBranchingFactor=2, branchNodeNum = 1,
                 scoreFunction='length'):
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
        self.lowerBoundClass = LipschitzBounding(network, device)
        self.upperBoundClass = PgdUpperBound(network, pgdNumberOfInitializations, pgdIterNum, pgdStepSize,
                                             inputDimension, device)
        self.branchingMethod = branchingMethod
        self.nodeBranchingFactor = nodeBranchingFactor
        self.scoreFunction = scoreFunction
        self.branchNodeNum = branchNodeNum
        self.device = device

    def prune(self):
        for node in self.spaceNodes:
            if node.lower >= self.bestUpperBound:
                self.spaceNodes.remove(node)
                if self.verbose:
                    print('deleted')

    def lowerBound(self, indices):
        lowerBounds = torch.vstack([self.spaceNodes[index].coordLower for index in indices])
        upperBounds = torch.vstack([self.spaceNodes[index].coordUpper for index in indices])
        return self.lowerBoundClass.lowerBound(self.queryCoefficient, lowerBounds, upperBounds)

    def upperBound(self, indices):
        return self.upperBoundClass.upperBound(indices, self.spaceNodes, self.queryCoefficient)

    def branch(self):
        # Prunning Function
        self.prune()

        # if self.branchMethod == 'SimpleBranch':
        #@TODO Choosing the node to branch -> this parts should be swaped with the sort idea
        scoreArray = torch.Tensor([self.spaceNodes[i].score for i in range(len(self.spaceNodes))])
        scoreArraySorted = torch.argsort(scoreArray)
        maxIndeces = scoreArraySorted[len(scoreArraySorted) - 1: len(scoreArraySorted) - self.branchNodeNum + 1]
        
        deletedUpperBounds = []
        deletedLowerBounds = []
        for maxIndex in maxIndeces:
            coordToSplitSorted = torch.argsort(self.spaceNodes[maxIndex].coordUpper - self.spaceNodes[maxIndex].coordLower)
            coordToSplit = coordToSplitSorted[len(coordToSplitSorted) - 1]
            
            #@TODO This can be optimized by keeping the best previous 'x's in that space
            node = self.spaceNodes.pop(maxIndex)
            deletedUpperBounds.append(node.upper)
            deletedLowerBounds.append(node.lower)

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
            return [len(self.spaceNodes) - j for j in range(1, self.nodeBranchingFactor + 1)], node.upper, node.lower
        
        else:
            print("Not Implemented Yet!")
            return None

    def bound(self, indices, parent_ub, parent_lb):
        lowerBounds = torch.maximum(self.lowerBound(indices), parent_lb)
        upperBounds = self.upperBound(indices)
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
            indices, deletedUb, deletedLb = self.branch()
            self.bound(indices, deletedUb, deletedLb)

            self.bestUpperBound = torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))]))
            self.bestLowerBound = torch.min(torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            # print(self.bestUpperBound, self.bestLowerBound)
            if self.verbose:
                print('Best UB', self.bestLowerBound, 'Best LB', self.bestUpperBound)
                plotter.plotSpace(self.spaceNodes, self.initCoordLow, self.initCoordUp)
                print('--------------------')

        if self.verbose:
            plotter.showAnimation()
        return self.bestLowerBound, self.bestUpperBound, self.spaceNodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.spaceNodes)):
            string += self.spaceNodes[i].__repr__() 
            string += "\n"

        return string
        