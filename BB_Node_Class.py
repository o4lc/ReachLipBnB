import numpy as np

class BB_node:
    def __init__(self, up=None, low=None, coordUp=None, coordLow=None):
        self.upper = up
        self.lower = low
        self.coordUpper = np.array(coordUp, dtype=float)
        self.coordLower = np.array(coordLow, dtype=float)
        self.score = self.calc_score()
    
    def calc_score(self):
        #@TODO
        # Temp scoring function
        if self.upper == None:
            return None
        
        return np.max(self.coordUpper - self.coordLower)

    def __repr__(self):
        return str(self.coordLower) + ' ≤ ' + str(self.coordUpper)