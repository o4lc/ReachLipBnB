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

        self.score = self.calc_score()
    
    def calc_score(self):
        #@TODO
        # Temp scoring function
        dilation = self.coordUpper - self.coordLower
        if self.scoreFunction == 'length':
            return torch.max(dilation)
        
        elif self.scoreFunction == 'volume':
            return torch.prod(dilation)

        elif self.scoreFunction == 'condNum':
            return torch.max(dilation) / torch.min(dilation)

    def __repr__(self):
        return str(self.coordLower) + ' ≤ ' + str(self.coordUpper)