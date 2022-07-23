import numpy as np

class BB_node:
    def __init__(self, up=None, low=None, coord_up=None, coord_low=None):
        self.upper = up
        self.lower = low
        self.coord_upper = np.array(coord_up, dtype=float)
        self.coord_lower = np.array(coord_low, dtype=float)
        self.score = self.calc_score()
    
    def calc_score(self):
        # Temp scoring function
        if self.upper == None:
            return None
        
        return np.max(self.coord_upper - self.coord_lower)

    def __repr__(self):
        return str(self.coord_lower) + ' ≤ ' + str(self.coord_upper)