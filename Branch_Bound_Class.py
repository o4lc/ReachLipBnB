from tabnanny import verbose
from packages import *
from utilities import Plotter
from BB_Node_Class import BB_node
from Bounding.LipschitzBound import LipschitzBounding

class Branch_Bound:
    def __init__(self, coordUp=None, coordLow=None, verbose=False, eta=1e-3, 
                        dim=2, eps=0.1, network=None, queryCoefficient=None,
                        pgdIterNum=5, batchNumber=2, device=torch.device("cuda", 0),
                        branch_method='SimpleBranch', branch_constant=2,
                        scoreFunction='length'):
        self.spaceNodes = [BB_node(np.infty, -np.infty, coordUp, coordLow, scoreFunction=scoreFunction)]
        self.BUB = None
        self.BLB = None
        self.initCoordUp = coordUp
        self.initCoordLow = coordLow
        self.verbose = verbose
        self.eta = eta
        self.pgdIterNum = pgdIterNum
        self.batchNumber = batchNumber
        self.dim = dim
        self.eps = eps
        self.network = network
        self.queryCoefficient = queryCoefficient
        self.lowerBoundClass = LipschitzBounding(network, device)
        self.branch_method = branch_method
        self.branch_constant = branch_constant
        self.scoreFunction = scoreFunction

    def prune(self):
        for node in self.spaceNodes:
            if node.lower >= self.BUB:
                self.spaceNodes.remove(node)
                if self.verbose:
                    print('deleted')

    def lowerBound(self, indices):
        lowerBounds = torch.from_numpy(np.array([self.spaceNodes[index].coordLower for index in indices]))
        upperBounds = torch.from_numpy(np.array([self.spaceNodes[index].coordUpper for index in indices]))
        return self.lowerBoundClass.lowerBound(self.queryCoefficient, lowerBounds, upperBounds)

    def upperBound(self, index):
        x0 = np.random.uniform(low = self.spaceNodes[index].coordLower, 
                                          high = self.spaceNodes[index].coordUpper,
                                            size=(self.batchNumber, self.dim))

        x = Variable(torch.from_numpy(x0.astype('float')).float(), requires_grad=True)
        
        # Gradient Descent
        for i in range(self.pgdIterNum):
            x.requires_grad = True
            for j in range(self.batchNumber):
                with torch.autograd.profiler.profile() as prof:
                    ll = self.queryCoefficient @ self.network.forward(x[j])
                    ll.backward()
                    # l.append(ll.data)

            with no_grad():
                gradient = x.grad.data
                x = x - self.eta * gradient

        # Projection
        x = torch.max(torch.min(x, torch.from_numpy(self.spaceNodes[index].coordUpper).float()),
                        torch.from_numpy(self.spaceNodes[index].coordLower).float())

        ub = torch.min(torch.Tensor([self.queryCoefficient @ self.network.forward(xx) for xx in x]))
        return ub

    def branch(self):
        # Prunning Function
        self.prune()

        if self.branch_method == 'SimpleBranch':
            #@TODO Choosing the node to branch -> this parts should be swaped with the sort idea
            maxScore, maxIndex = -1, -1
            for i in range(len(self.spaceNodes)):
                if self.spaceNodes[i].score > maxScore:
                    maxIndex = i
                    maxScore = self.spaceNodes[i].score

            coordToSplit = np.argmax(self.spaceNodes[maxIndex].coordUpper 
                                    - self.spaceNodes[maxIndex].coordLower)
        
            
            #@TODO This can be optimized by keeping the best previous 'x's in that space
            node = self.spaceNodes.pop(maxIndex)
            nodeLow = np.array(node.coordLower, dtype=float)
            nodeUp = np.array(node.coordUpper, dtype=float)

            '''
            @TODO
            Python Float Calculation Problem
            Need to round up ? 
            '''
            parentNodeUpperBound = np.array(nodeUp, dtype=float)
            parentNodeLowerBound = np.array(nodeLow, dtype=float)
            newIntervals = [x for x in np.linspace(parentNodeLowerBound[coordToSplit],
                                                    parentNodeUpperBound[coordToSplit],
                                                    self.branch_constant + 1)]
            for i in range(self.branch_constant):
                tempLow = parentNodeLowerBound
                tempLow[coordToSplit] = newIntervals[i]

                tempHigh = parentNodeUpperBound
                tempHigh[coordToSplit] = newIntervals[i+1]

                self.spaceNodes.append(BB_node(np.infty, -np.infty, tempHigh, tempLow, scoreFunction=self.scoreFunction))

            return [len(self.spaceNodes) - j for j in range(1, self.branch_constant + 1)], node.upper, node.lower
        
        else:
            print("Not Implemented Yet!")
            return None

    def bound(self, indices, parent_ub, parent_lb):
        lowerBounds = torch.maximum(self.lowerBound(indices), parent_lb)
        for i, index in enumerate(indices):
            self.spaceNodes[index].upper = self.upperBound(index)
            self.spaceNodes[index].lower = lowerBounds[i]
        # self.spaceNodes[index].upper = min(cost_upper, parent_ub)


    def run(self):
        self.BUB = torch.Tensor([torch.inf])
        self.BLB = torch.Tensor([-torch.inf])

        if self.verbose:
            plotter = Plotter()

        self.bound([0], self.BUB, self.BLB)
        while self.BUB - self.BLB >= self.eps:
            indeces, deletedUb, deletedLb = self.branch()
            self.bound(indeces, deletedUb, deletedLb)

            self.BUB = torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))]))
            self.BLB = torch.min(torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            
            if self.verbose:
                print('Best UB', self.BLB , 'Best LB' , self.BUB)
                plotter.plotSpace(self.spaceNodes, self.initCoordLow, self.initCoordUp)
                print('--------------------')

        if self.verbose:
            plotter.showAnimation()
        return self.BLB, self.BUB, self.spaceNodes

    def __repr__(self):
        string = 'These are the remaining nodes: \n'
        for i in range(len(self.spaceNodes)):
            string += self.spaceNodes[i].__repr__() 
            string += "\n"

        return string
        