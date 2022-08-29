import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA

torch.set_printoptions(precision=8)

def main():

    eps = .001
    verbose = 0
    virtualBranching = False
    numberOfVirtualBranches = 4,
    maxSearchDepthLipschitzBound = 10,
    normToUseLipschitz = 2
    useTwoNormDilation = False
    useSdpForLipschitzCalculation = True
    lipschitzSdpSolverVerbose = False
    finalHorizon = 1

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # Temporary
    device=torch.device("cpu")
    print(device)
    print(' ')

    # pathToStateDictionary = "Networks/randomNetwork.pth"
    pathToStateDictionary = "Networks/randomNetwork2.pth"
    # pathToStateDictionary = "Networks/randomNetwork3.pth"
    # pathToStateDictionary = "Networks/trainedNetwork1.pth"
    # pathToStateDictionary = "Networks/RobotArmStateDict2-5-2.pth"
    # pathToStateDictionary = "Networks/Test3-5-3.pth"
    # pathToStateDictionary = "Networks/ACASXU.pth"
    # pathToStateDictionary = "Networks/mnist_3_50.pth"

    network = NeuralNetwork(pathToStateDictionary)


    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(device)
    
    # The intial HyperRectangule
    lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    upperCoordinate = torch.Tensor([1., 1.]).to(device)

    startTime = time.time()

    previousRotationMatrix = np.eye(dim, dim)
    previousRotationBias = np.zeros(dim)

    for iteration in range(finalHorizon):
        inputData = (upperCoordinate - lowerCoordinate) * torch.rand(1000, dim, device=device) \
                + lowerCoordinate

        inputData = Variable(inputData, requires_grad=False)
        with no_grad():
            imageData = network.forward(inputData)


        pca = PCA()
        pcaData = pca.fit_transform(imageData)

        data_mean = pca.mean_
        data_comp = pca.components_

        # print(data_comp[0] @ data_comp[1])

        plt.figure()
        plt.scatter(imageData[:, 0], imageData[:, 1])
        plt.arrow(data_mean[0], data_mean[1], data_comp[0, 0] / 1000, data_comp[0, 1] / 1000, width=0.00003)
        plt.arrow(data_mean[0], data_mean[1], data_comp[1, 0] / 1000, data_comp[1, 1] / 1000, width=0.00003)
        
        pcaDirections = []
        for direction in data_comp:
            # Rows are the components!
            pcaDirections.append(direction)
            pcaDirections.append(-direction)
        pcaDirections = torch.Tensor(np.array(pcaDirections))
        calculatedLowerBoundsforpcaDirections = torch.Tensor(np.zeros(len(pcaDirections)))
        
        for i in range(len(pcaDirections)):
            c = pcaDirections[i]
            print('** Solving with coefficient =', c)
            BB = BranchAndBound(upperCoordinate, lowerCoordinate, verbose=verbose, inputDimension=dim, eps=eps, network=network,
                                queryCoefficient=c, device=device, nodeBranchingFactor=2, branchNodeNum=512,
                                scoreFunction='length',
                                pgdIterNum=0, pgdNumberOfInitializations=2, pgdStepSize=0.5, virtualBranching=virtualBranching,
                                numberOfVirtualBranches=numberOfVirtualBranches,
                                maxSearchDepthLipschitzBound=maxSearchDepthLipschitzBound,
                                normToUseLipschitz=normToUseLipschitz, useTwoNormDilation=useTwoNormDilation,
                                useSdpForLipschitzCalculation=useSdpForLipschitzCalculation,
                                lipschitzSdpSolverVerbose=lipschitzSdpSolverVerbose,
                                initialGD=False, rotationMatrix=torch.from_numpy(previousRotationMatrix).float().to(device), 
                                rotationConstant=torch.from_numpy(previousRotationBias).float().to(device)
                                )
            lowerBound, upperBound, space_left = BB.run()
            calculatedLowerBoundsforpcaDirections[i] = lowerBound
            print(' ')
            print('Best lower/upper bounds are:', lowerBound, '->' ,upperBound)

        
        previousRotationMatrix = data_comp
        previousRotationBias = data_mean

        # print(pcaDirections)
        # print(calculatedLowerBoundsforpcaDirections )
        calculatedLowerBoundsforpcaDirections = -calculatedLowerBoundsforpcaDirections
        directionMultipliers = np.zeros(pcaDirections.shape[0])
        upperCoordinateMultiplier = np.ones_like(upperCoordinate)
        for i, component in enumerate(data_comp):
            winningIndex = np.argmax(abs(component))
            if component[winningIndex] >0 :
                upperCoordinateMultiplier[i] = 1
                upperCoordinate[i] = calculatedLowerBoundsforpcaDirections[2 * i]
                lowerCoordinate[i] = calculatedLowerBoundsforpcaDirections[2 * i + 1]
            else:
                upperCoordinateMultiplier[i] = -1
                upperCoordinate[i] = calculatedLowerBoundsforpcaDirections[2 * i + 1]
                lowerCoordinate[i] = calculatedLowerBoundsforpcaDirections[2 * i]
        calculatedLowerBoundsforpcaDirections = calculatedLowerBoundsforpcaDirections * directionMultipliers

        for i in range(len(upperCoordinate)):
            x0 = np.array([torch.min(imageData[:, 0]).numpy(), torch.max(imageData[:, 0]).numpy()])
            c = -upperCoordinateMultiplier[i] * data_comp[i]
            y0 = ( upperCoordinate[i] - c[0] * x0)/c[1]

            plt.plot(x0, y0)

            y0 = (lowerCoordinate[i] - c[0] * x0)/c[1]

            plt.plot(x0, y0)

        # plt.gca().set_aspect('equal', adjustable='box')

        # x1 = np.array([torch.min(imageData[:, 0]).numpy(), torch.max(imageData[:, 0]).numpy()])
        # c = data_comp[1]
        # y1 = (-upperCoordinate[1] - c[0] * x1)/c[1]

        # plt.plot(x1, y1)

        # c = data_comp[1]
        # y1 = (lowerCoordinate[1] - c[0] * x1)/c[1]

        # plt.plot(x1, y1)


        plt.show()

            


    endTime = time.time()
    
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)


if __name__ == '__main__':
    main()