from tabnanny import verbose
from packages import *
from utilities import Plotter
from BB_Node_Class import BB_node
from Bounding.LipschitzBound import LipschitzBounding

class Branch_Bound:
    def __init__(self, coordUp=None, coordLow=None, verbose=False, eta=1e-3, 
                        dim=2, eps=0.1, network=None, queryCoefficient=None,
                        pgdIterNum=5, batchNumber=2, device=torch.device("cuda", 0)):
        self.spaceNodes = [BB_node(np.infty, -np.infty, coordUp, coordLow)]
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

        # Choosing the node to branch
        maxScore, maxIndex = -1, -1
        for i in range(len(self.spaceNodes)):
            if self.spaceNodes[i].score > maxScore:
                maxIndex = i
                maxScore = self.spaceNodes[i].score

        coordToSplit = np.argmax(self.spaceNodes[maxIndex].coordUpper 
                                   - self.spaceNodes[maxIndex].coordLower)
        
        # This can be optimized by keeping the best previous 'x's in that space
        node = self.spaceNodes.pop(maxIndex)
        nodeLow = np.array(node.coordLower, dtype=float)
        nodeUp = np.array(node.coordUpper, dtype=float)


        newAxis = (node.coordUpper[coordToSplit] + 
                           node.coordLower[coordToSplit])/2

        nodeSplitU1 = np.array(nodeUp, dtype=float)
        nodeSplitU1[coordToSplit] = newAxis

        nodeSplitL2 = np.array(nodeLow, dtype=float)
        nodeSplitL2[coordToSplit] = newAxis

        self.spaceNodes.append(BB_node(np.infty, -np.infty, nodeSplitU1, nodeLow))
        self.spaceNodes.append(BB_node(np.infty, -np.infty, nodeUp, nodeSplitL2))
                
        return [len(self.spaceNodes) - 2, len(self.spaceNodes) - 1], node.upper, node.lower

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
        count = 0
        while self.BUB - self.BLB >= self.eps:
            # count += 1
            # if count == 3:
            #     break
            indeces, deletedUb, deletedLb = self.branch()
            print(indeces)
            self.bound(indeces, deletedUb, deletedLb)
            # for ind in indeces:
            #     # todo: add twice
            #     self.bound(ind, deletedUb, deletedLb)

            self.BUB = torch.min(torch.Tensor([self.spaceNodes[i].upper for i in range(len(self.spaceNodes))]))
            self.BLB = torch.min(torch.Tensor([self.spaceNodes[i].lower for i in range(len(self.spaceNodes))]))
            
            if self.verbose:
                print(self.BLB , self.BUB)
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
        