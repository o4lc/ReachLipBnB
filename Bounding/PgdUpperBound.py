import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import no_grad
from torch.autograd.functional import jacobian


class PgdUpperBound:
    def __init__(self, network, numberOfInitializationPoints, numberOfPgdSteps, pgdStepSize,
                 inputSpaceDimension, device):
        self.network = network
        self.pgdNumberOfInitializations = numberOfInitializationPoints
        self.pgdIterNum = numberOfPgdSteps
        self.inputDimension = inputSpaceDimension
        self.device = device
        self.pgdStepSize = pgdStepSize

    def upperBound(self, indices, nodes, queryCoefficient):
        upperBounds = []
        if self.pgdIterNum == 0:
            for index in indices:
                upperBounds.append(self.upperBoundViaRandomPoints(index, nodes, queryCoefficient))
        else:
            for index in indices:
                upperBounds.append(self.upperBoundPerIndexWithPgd(index, nodes, queryCoefficient))
        return upperBounds

    def upperBoundViaRandomPoints(self, index, nodes, queryCoefficient):
        x = (nodes[index].coordUpper - nodes[index].coordLower) \
             * torch.rand(self.pgdNumberOfInitializations, self.inputDimension, device=self.device) \
             + nodes[index].coordLower

        upperBound = torch.min(self.network(x) @ queryCoefficient)
        return upperBound

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

        # Projection
        x = torch.clamp(x, nodes[index].coordLower, nodes[index].coordUpper)

        upperBound = torch.min(self.network(x) @ queryCoefficient)
        return upperBound