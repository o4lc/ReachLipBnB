from unicodedata import decimal
import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA

torch.set_printoptions(precision=8)

def main():

    eps = .01
    verbose = 0
    virtualBranching = False
    numberOfVirtualBranches = 4,
    maxSearchDepthLipschitzBound = 10,
    normToUseLipschitz = 2
    useTwoNormDilation = False
    useSdpForLipschitzCalculation = True
    lipschitzSdpSolverVerbose = False

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

    lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    upperCoordinate = torch.Tensor([1., 1.]).to(device)
    c = torch.Tensor([1., 2.]).to(device)

    startTime = time.time()
    
    inputData = (upperCoordinate - lowerCoordinate) * torch.rand(100, dim, device=device) \
             + lowerCoordinate

    inputData = Variable(inputData, requires_grad=False)
    with no_grad():
        imageData = network.forward(inputData)

    pca = PCA()
    pcaData = pca.fit_transform(imageData)

    data_mean = pca.mean_
    data_comp = pca.components_

    pcaDirections = []
    for direction in data_comp:
        pcaDirections.append(direction)
        pcaDirections.append(-direction)
    pcaDirections = torch.Tensor(np.array(pcaDirections))
    
    for c in pcaDirections:
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
                            initialGD = False
                            )
        lowerBound, upperBound, space_left = BB.run()

        print(' ')
        print('Best lower/upper bounds are:', lowerBound, '->' ,upperBound)


    endTime = time.time()
    
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)



if __name__ == '__main__':
    main()