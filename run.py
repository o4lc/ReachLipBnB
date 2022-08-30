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
    finalHorizon = 3
    initialGD = False

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # Temporary
    device = torch.device("cpu")
    print(device)
    print(' ')

    # fileName = "randomNetwork.pth"
    # fileName = "randomNetwork2.pth"
    # fileName = "randomNetwork3.pth"
    fileName = "trainedNetwork1.pth"
    # fileName = "RobotArmStateDict2-5-2.pth"
    # fileName = "Test3-5-3.pth"
    # fileName = "ACASXU.pth"
    # fileName = "mnist_3_50.pth"

    pathToStateDictionary = "Networks/" + fileName
    network = NeuralNetwork(pathToStateDictionary)


    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(device)
    
    # The intial HyperRectangule
    lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    upperCoordinate = torch.Tensor([1., 1.]).to(device)

    # lowerCoordinate = torch.Tensor([torch.pi / 3, torch.pi / 3, torch.pi / 3]).to(device)
    # upperCoordinate = torch.Tensor([2 * torch.pi / 3, 2 * torch.pi / 3, 2 * torch.pi / 3]).to(device)
    # if "ACAS" in pathToStateDictionary or "mnist" in pathToStateDictionary:
    #     lowerCoordinate = torch.Tensor([-2. / 2560] * dim).to(device)
    #     upperCoordinate = torch.Tensor([2. / 2560] * dim).to(device)
    #     # lowerCoordinate = torch.Tensor([-1.] * dim).to(device)
    #     # upperCoordinate = torch.Tensor([1.] * dim).to(device)
    #     # c = torch.ones(outputDim).to(device)
    #     c = torch.zeros(outputDim, dtype=torch.float).to(device)
    #     if "mnist" in pathToStateDictionary:
    #         df = pd.read_csv("mnistTestData.csv")
    #         testImage = torch.Tensor(df.loc[0].to_numpy()[1:] / 255.).to(device)
    #         lowerCoordinate += testImage
    #         upperCoordinate += testImage
    #         testLabel = df.loc[0].label
    #         c[testLabel] = 1.
    #         try:
    #             c[testLabel + 1] = -1
    #         except:
    #             c[testLabel - 1] = -1
    #     else:
    #         c[0] = 1.
    #         c[1] = -1

    startTime = time.time()


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
        plt.scatter(imageData[:, 0], imageData[:, 1], marker='.')
        plt.arrow(data_mean[0], data_mean[1], data_comp[0, 0] / 10000, data_comp[0, 1] / 10000, width=0.000003)
        plt.arrow(data_mean[0], data_mean[1], data_comp[1, 0] / 10000, data_comp[1, 1] / 10000, width=0.000003)
        
        pcaDirections = []
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

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
                                initialGD=initialGD
                                )
            lowerBound, upperBound, space_left = BB.run()
            calculatedLowerBoundsforpcaDirections[i] = lowerBound
            print(' ')
            print('Best lower/upper bounds are:', lowerBound, '->' ,upperBound)

        rotation = nn.Linear(dim, dim)
        rotation.weight = torch.nn.parameter.Parameter(torch.linalg.inv(torch.from_numpy(data_comp).float().to(device)))
        rotation.bias = torch.nn.parameter.Parameter(torch.from_numpy(data_mean).float().to(device))
        network.rotation = rotation

        for i, component in enumerate(data_comp):
            upperCoordinate[i] = -calculatedLowerBoundsforpcaDirections[2 * i]
            lowerCoordinate[i] = calculatedLowerBoundsforpcaDirections[2 * i + 1]

        x0 = np.array([torch.min(imageData[:, 0]).numpy() - eps, torch.max(imageData[:, 0]).numpy() + eps])
        for i in range(len(upperCoordinate)):
            c = data_comp[i]
            y0 = (upperCoordinate[i] - c[0] * x0)/c[1]
            plt.plot(x0, y0)

            y0 = (lowerCoordinate[i] - c[0] * x0)/c[1]
            plt.plot(x0, y0)

        print(upperCoordinate)
        print(lowerCoordinate)
        plt.axis("equal")

        plt.savefig("reachabilityPics/" + fileName + "Iteration" + str(iteration) + ".png")
        plt.show()


            


    endTime = time.time()
    
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)


if __name__ == '__main__':
    main()