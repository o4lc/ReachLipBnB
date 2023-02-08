import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import no_grad
from torch.autograd.functional import jacobian


class PgdUpperBound:
    def __init__(self, network, numberOfInitializationPoints, numberOfPgdSteps, pgdStepSize,
                 inputSpaceDimension, device, maximumBatchSize):
        self.network = network
        self.pgdNumberOfInitializations = numberOfInitializationPoints
        self.pgdIterNum = numberOfPgdSteps
        self.inputDimension = inputSpaceDimension
        self.device = device
        self.pgdStepSize = pgdStepSize
        self.maximumBatchSize = maximumBatchSize

    def upperBound(self, indices, nodes, queryCoefficient):
        if self.pgdIterNum == 0:
            upperBounds = self.upperBoundViaRandomPoints(indices, nodes, queryCoefficient)
        else:
            upperBounds = []
            for index in indices:
                upperBounds.append(self.upperBoundPerIndexWithPgd(index, nodes, queryCoefficient))
        return upperBounds

    def upperBoundViaRandomPoints(self, indices, nodes, queryCoefficient):
        currentBatchSize = len(indices) * self.pgdNumberOfInitializations
        multiplier = torch.zeros(currentBatchSize, self.inputDimension, device=self.device)
        bias = torch.zeros(currentBatchSize, self.inputDimension, device=self.device)

        for i, index in enumerate(indices):
            offset = self.pgdNumberOfInitializations * i
            multiplier[offset: offset + self.pgdNumberOfInitializations, :] =\
                nodes[index].coordUpper - nodes[index].coordLower
            bias[offset: offset + self.pgdNumberOfInitializations, :] = nodes[index].coordLower
        x = multiplier \
            * torch.rand(currentBatchSize, self.inputDimension, device=self.device) \
            + bias

        # the center point implementation would only work if pgdNumberOfInitializations=1 (Duh)
        # ub = torch.vstack([nodes[i].coordUpper for i in indices])
        # lb = torch.vstack([nodes[i].coordLower for i in indices])
        # x = (ub + lb) / 2

        if currentBatchSize > self.maximumBatchSize:
            y = torch.zeros(currentBatchSize)
            i = 0
            while i < currentBatchSize:
                y[i:i + self.maximumBatchSize] = self.network(x[i:i + self.maximumBatchSize, :]) @ queryCoefficient
                i += self.maximumBatchSize
        else:
            with torch.no_grad():
                y = self.network(x) @ queryCoefficient
        upperBounds = []
        for i in range(len(indices)):
            offset = self.pgdNumberOfInitializations * i
            upperBounds.append(torch.min(y[offset: offset + self.pgdNumberOfInitializations]))
        return upperBounds

    def upperBoundPerIndexWithPgd(self, index, nodes, queryCoefficient, x0=None):

        if x0 is None:
            x0 = (nodes[index].coordUpper - nodes[index].coordLower) \
                 * torch.rand(self.pgdNumberOfInitializations, self.inputDimension, device=self.device) \
                 + nodes[index].coordLower
        # print(x0.shape)
        # x0 = (nodes[index].coordUpper + nodes[index].coordLower).unsqueeze(0) / 2
        # print(x0.shape)
        x = Variable(x0, requires_grad=True)

        # Gradient Descent
        for i in range(self.pgdIterNum):
            x.requires_grad = True
            for j in range(self.pgdNumberOfInitializations):
                with torch.autograd.profiler.profile() as prof:
                    ll = queryCoefficient @ self.network.forward(x[j])
                    ll.backward()
                    # l.append(ll.data)
        
            with no_grad():
                gradient = x.grad.data
                x = x - self.pgdStepSize * gradient

        # Projection
        x = torch.clamp(x, nodes[index].coordLower, nodes[index].coordUpper)

        upperBound = torch.min(self.network(x) @ queryCoefficient)
        return upperBound