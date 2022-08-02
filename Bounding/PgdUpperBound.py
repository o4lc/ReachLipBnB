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
        multiplier = torch.zeros(currentBatchSize, self.inputDimension)
        bias = torch.zeros(currentBatchSize, self.inputDimension)

        for i, index in enumerate(indices):
            offset = self.pgdNumberOfInitializations * i
            multiplier[offset: offset + self.pgdNumberOfInitializations, :] =\
                nodes[index].coordUpper - nodes[index].coordLower
            bias[offset: offset + self.pgdNumberOfInitializations, :] = nodes[index].coordLower
        x = multiplier \
            * torch.rand(currentBatchSize, self.inputDimension, device=self.device) \
            + bias
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

    def upperBoundPerIndexWithPgd(self, index, nodes, queryCoefficient):
        x0 = (nodes[index].coordUpper - nodes[index].coordLower) \
             * torch.rand(self.pgdNumberOfInitializations, self.inputDimension, device=self.device) \
             + nodes[index].coordLower

        x = Variable(x0, requires_grad=True)

        # Batch Gradient Descent
        for i in range(self.pgdIterNum):
            x.requires_grad = True
            with torch.autograd.profiler.profile() as prof:
                def loss_reducer(x):
                    return self.network.forward(x) @ queryCoefficient

                gradient = jacobian(loss_reducer, x)

            with no_grad():
                x = x - self.pgdStepSize * gradient.sum(-self.pgdNumberOfInitializations)


        # Per Sample Gradient Descent
        # for i in range(self.pgdIterNum):
        #     x.requires_grad = True
        #     with torch.autograd.profiler.profile() as prof:
        #         fmodel, params, buffers = make_functional_with_buffers(self.network)
        #         print(fmodel)
        #         return None


        #     with no_grad():
        #         x = x - self.eta * gradient.sum(-self.pgdNumberOfInitializations)

        # # Gradient Descent
        # for i in range(self.pgdIterNum):
        #     x.requires_grad = True
        #     for j in range(self.pgdNumberOfInitializations):
        #         with torch.autograd.profiler.profile() as prof:
        #             ll = queryCoefficient @ self.network.forward(x[j])
        #             ll.backward()
        #             # l.append(ll.data)
        #
        #     with no_grad():
        #         gradient = x.grad.data
        #         x = x - self.pgdStepSize * gradient

        # Projection
        x = torch.clamp(x, nodes[index].coordLower, nodes[index].coordUpper)

        upperBound = torch.min(self.network(x) @ queryCoefficient)
        return upperBound