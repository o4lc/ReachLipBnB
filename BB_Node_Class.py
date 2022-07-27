import numpy as np

class BB_node:
    def __init__(self, up=np.infty, low=-np.infty, coordUp=None, coordLow=None,
                        scoreFunction='length'):
        self.upper = up
        self.lower = low
        self.coordUpper = np.array(coordUp, dtype=float)
        self.coordLower = np.array(coordLow, dtype=float)
        self.scoreFunction = scoreFunction

        
        self.score = self.calc_score()
    
    def calc_score(self):
        #@TODO
        # Temp scoring function
        if self.scoreFunction == 'length':
            return np.max(self.coordUpper - self.coordLower)
        
        elif self.scoreFunction == 'volume':
            return np.prod(self.coordUpper - self.coordLower)

        elif self.scoreFunction == 'condNum':
            return np.max(self.coordUpper - self.coordLower) / np.min(self.coordUpper - self.coordLower)

    def __repr__(self):
        return str(self.coordLower) + ' ≤ ' + str(self.coordUpper)