import numpy as np
import torch


class BB_node:
    def __init__(self, up=np.infty, low=-np.infty, coordUp: torch.Tensor=None, coordLow: torch.Tensor=None,
                        scoreFunction='length'):
        self.upper = up
        self.lower = low
        self.coordUpper = coordUp
        self.coordLower = coordLow
        self.scoreFunction = scoreFunction
        self.score = None
        self.score = self.calc_score()
    
    def calc_score(self, scoreFunction=None):
        #@TODO
        # Temp scoring function
        if scoreFunction is None:
            scoreFunction = self.scoreFunction
        dilation = self.coordUpper - self.coordLower
        if scoreFunction == 'length':
            return torch.max(dilation)
        
        elif scoreFunction == 'volume':
            return torch.prod(dilation)

        elif scoreFunction == 'condNum':
            return torch.max(dilation) / torch.min(dilation)
        elif scoreFunction == "worstLowerBound":
            return -self.lower
        elif scoreFunction == "bestLowerBound":
            return self.lower
        elif scoreFunction == "bestUpperBound":
            return -self.upper
        elif scoreFunction == "worstUpperBound":
            return self.upper
        elif scoreFunction == "averageBounds":
            return (self.upper - self.lower) / 2
        elif scoreFunction.find("*") >= 0:
            index = scoreFunction.find("*")
            return self.calc_score(scoreFunction[:index]) * self.calc_score(scoreFunction[index + 1:])
        elif scoreFunction == "weightedGap":
            w = .9
            return -self.lower * w + self.upper * (1 - w)

    def __repr__(self):
        return str(self.coordLower) + ' ≤ ' + str(self.coordUpper)