import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
import pandas as pd
from sklearn.decomposition import PCA
import copy

torch.set_printoptions(precision=8)

def main():

    eps = 0.01
    verbose = 0
    verboseMultiHorizon = 1
    verboseEssential = 0
    scoreFunction = 'worstLowerBound'
    virtualBranching = False
    numberOfVirtualBranches = 4
    maxSearchDepthLipschitzBound = 10
    normToUseLipschitz = 2
    useTwoNormDilation = False
    useSdpForLipschitzCalculation = True
    lipschitzSdpSolverVerbose = False
    finalHorizon = 5
    initialGD = False
    performMultiStepSingleHorizon = False
    plotProjectionsOfHigherDims = True

    if finalHorizon > 1 and performMultiStepSingleHorizon and\
            (normToUseLipschitz != 2 or not useSdpForLipschitzCalculation):
        raise ValueError

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
    # fileName = "trainedNetwork1.pth"
    # fileName = "doubleIntegrator.pth"
    fileName = "doubleIntegrator_reachlp.pth"
    # fileName = "quadRotor5.pth"
    # fileName = "quadRotorv2.0.pth"
    # fileName = "RobotArmStateDict2-50-2.pth"
    # fileName = "Test3-5-3.pth"
    # fileName = "ACASXU.pth"
    # fileName = "mnist_3_50.pth"

    A = None
    B = None
    c = None
    if fileName == "doubleIntegrator.pth" or fileName == "doubleIntegrator_reachlp.pth":
        A = torch.Tensor([[1, 1], [0, 1]])
        B = torch.Tensor([[0.5], [1]])
        c = torch.Tensor([0])

        # lowerCoordinate = torch.Tensor([1., 1.5]).to(device)
        # upperCoordinate = torch.Tensor([2., 2.5]).to(device)

        lowerCoordinate = torch.Tensor([2.5, -0.25]).to(device)
        upperCoordinate = torch.Tensor([3., 0.25]).to(device)

    elif fileName == "quadRotor.pth" or fileName == "quadRotorv2.0.pth":
        dt = 0.1
        A = torch.Tensor([  [0., 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

        A = torch.eye(len(A)) + A * dt
        B = torch.Tensor([  [ 0. ,  0. ,  0. ],
                            [ 0. ,  0. ,  0. ],
                            [ 0. ,  0. ,  0. ],
                            [ 9.8,  0. ,  0. ],
                            [ 0. , -9.8,  0. ],
                            [ 0. ,  0. ,  1. ]])
        B = B * dt

        c = torch.Tensor([0, 0, 0, 0, 0, -9.8])
        c = c * dt

        # lowerCoordinate = torch.Tensor([4.6975, 4.6975, 2.9975, 0.9499, -0.0001, -0.0001]).to(device)
        # upperCoordinate = torch.Tensor([4.7025, 4.7025 ,3.0025, 0.9501,  0.0001,  0.0001 ]).to(device)
        lowerCoordinate = torch.Tensor([4.6, 4.6, 2.9, 0.93, -0.001, -0.001]).to(device)
        upperCoordinate = torch.Tensor([4.8, 4.9, 3.1, 0.96, 0.001, 0.001]).to(device)



    pathToStateDictionary = "Networks/" + fileName
    network = NeuralNetwork(pathToStateDictionary, A, B, c)
    horizonForLipschitz = 1
    originalNetwork = None
    if performMultiStepSingleHorizon:
        originalNetwork = copy.deepcopy(network)
        horizonForLipschitz = finalHorizon
        network.repetition = finalHorizon
        # repeatNetwork(network, finalHorizon)
        finalHorizon = 1

    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(device)
    # The intial HyperRectangule
    # lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    # upperCoordinate = torch.Tensor([1., 1.]).to(device)



    # lowerCoordinate = torch.Tensor([torch.pi / 3, torch.pi / 3]).to(device)
    # upperCoordinate = torch.Tensor([2 * torch.pi / 3, 2 * torch.pi / 3]).to(device)
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

    inputData = (upperCoordinate - lowerCoordinate) * torch.rand(1000, dim, device=device) \
                                                        + lowerCoordinate
    if verboseMultiHorizon:
        # fig = plt.figure()
        fig, ax = plt.subplots()
        plt.scatter(inputData[:, 0], inputData[:, 1], marker='.', label='Initial', alpha=0.5)



    startTime = time.time()

    for iteration in range(finalHorizon):

        inputDataVariable = Variable(inputData, requires_grad=False)
        with no_grad():
            imageData = network.forward(inputDataVariable)

        
        pca = PCA()
        pcaData = pca.fit_transform(imageData)

        data_mean = pca.mean_
        data_comp = pca.components_
        data_sd = np.sqrt(pca.explained_variance_)

        inputData = torch.from_numpy(data_comp @ (imageData.cpu().numpy() - data_mean).T).T.float()
        # print(np.linalg.norm(data_comp, 2, 1))

        if verboseMultiHorizon:
            # plt.figure()
            plt.scatter(imageData[:, 0], imageData[:, 1], marker='.', label='Horizon ' + str(iteration + 1), alpha=0.5)
            # plt.arrow(data_mean[0], data_mean[1], data_comp[0, 0] / 10000, data_comp[0, 1] / 10000, width=0.000003)
            # plt.arrow(data_mean[0], data_mean[1], data_comp[1, 0] / 10000, data_comp[1, 1] / 10000, width=0.000003)
        
        pcaDirections = []
        for direction in data_comp:
            pcaDirections.append(-direction)
            pcaDirections.append(direction)

        numberOfInitialDirections = len(pcaDirections)
        indexToStartReadingBoundsForPlotting = 0
        plottingDirections = pcaDirections
        if plotProjectionsOfHigherDims:
            indexToStartReadingBoundsForPlotting = len(pcaDirections)
            projectedImageData = imageData.clone()
            projectedImageData[:, 2:] = 0
            pca2 = PCA()
            _ = pca2.fit_transform(projectedImageData)
            plottingDirections = pca2.components_
            for direction in plottingDirections[:2]:
                pcaDirections.append(-direction)
                pcaDirections.append(direction)

        pcaDirections = torch.Tensor(np.array(pcaDirections))
        calculatedLowerBoundsforpcaDirections = torch.Tensor(np.zeros(len(pcaDirections)))
        

        for i in range(len(pcaDirections)):
            previousLipschitzCalculations = []
            if i % 2 == 1 and torch.allclose(pcaDirections[i], -pcaDirections[i - 1]):
                previousLipschitzCalculations = BB.lowerBoundClass.calculatedLipschitzConstants
            c = pcaDirections[i]
            print('** Solving Horizon: ', iteration, 'dimension: ', i)

            BB = BranchAndBound(upperCoordinate, lowerCoordinate, verbose=verbose, verboseEssential=verboseEssential, inputDimension=dim,
                                eps=eps, network=network, queryCoefficient=c, device=device, nodeBranchingFactor=2, branchNodeNum=512,
                                scoreFunction=scoreFunction,
                                pgdIterNum=0, pgdNumberOfInitializations=2, pgdStepSize=0.5, virtualBranching=virtualBranching,
                                numberOfVirtualBranches=numberOfVirtualBranches,
                                maxSearchDepthLipschitzBound=maxSearchDepthLipschitzBound,
                                normToUseLipschitz=normToUseLipschitz, useTwoNormDilation=useTwoNormDilation,
                                useSdpForLipschitzCalculation=useSdpForLipschitzCalculation,
                                lipschitzSdpSolverVerbose=lipschitzSdpSolverVerbose,
                                initialGD=initialGD,
                                previousLipschitzCalculations=previousLipschitzCalculations,
                                originalNetwork=originalNetwork,
                                horizonForLipschitz=horizonForLipschitz
                                )
            lowerBound, upperBound, space_left = BB.run()
            calculatedLowerBoundsforpcaDirections[i] = lowerBound
            print(' ')
            print('Best lower/upper bounds are:', lowerBound, '->' ,upperBound)

        rotation = nn.Linear(dim, dim)
        rotation.weight = torch.nn.parameter.Parameter(torch.linalg.inv(torch.from_numpy(data_comp).float().to(device)))
        rotation.bias = torch.nn.parameter.Parameter(torch.from_numpy(data_mean).float().to(device))
        network.rotation = rotation

        centers = []
        for i, component in enumerate(data_comp):
            u = -calculatedLowerBoundsforpcaDirections[2 * i]
            l = calculatedLowerBoundsforpcaDirections[2 * i + 1]
            # center = (u + l) / 2
            center = component @ data_mean
            centers.append(center)
            upperCoordinate[i] = u - center
            lowerCoordinate[i] = l - center



        xx = np.array([[torch.min(imageData[:, i]).numpy() - eps,
                        torch.max(imageData[:, i]).numpy() + eps] for i in range(1, len(imageData[0]))])

            # if verboseMultiHorizon:
            #     for i in range(len(data_comp)):
            #         c = data_comp[i]
            #         if c[0] != 0:
            #             yy = (upperCoordinate[i] + centers[i] - c[1:] @ xx)/c[0]
            #             plt.plot(yy, xx[0], '--',  c='grey')

            #             yy = (lowerCoordinate[i] + centers[i] - c[1:] @ xx)/c[0]
            #             plt.plot(yy, xx[0], '--', c='grey')
            #         else:
            #             raise



        if verboseMultiHorizon:
            AA = -np.array(pcaDirections[indexToStartReadingBoundsForPlotting:])
            AA = AA[:, :2]
            print(AA)
            bb = []
            for i in range(indexToStartReadingBoundsForPlotting, len(calculatedLowerBoundsforpcaDirections)):
                bb.append(-calculatedLowerBoundsforpcaDirections[i])

            bb = np.array(bb)
            # if dim == 2:
            pltp = polytope.Polytope(AA, bb)
            print(pltp)
            # plt.figure()
            ax = pltp.plot(ax, alpha = 0.1, color='grey', edgecolor='black')
            ax.set_xlim([0, 5])
            ax.set_ylim([-4, 5])

            plt.axis("equal")
            leg1 = plt.legend()
            plt.title("Double Integrator")
            plt.xlabel('x0')
            plt.ylabel('x1')






    if verboseMultiHorizon:
        if fileName == "doubleIntegrator_reachlp.pth":
            reachlp = np.array([
                # [[2.5, 3], [-0.25, 0.25]],
            [[ 1.90837383, 2.75 ],
            [-1.125, -0.70422709]],

            [[1.0081799, 1.8305043],
            [-1.10589671, -0.80364925]],

            [[ 0.33328745,  0.94537741],
            [-0.76938218, -0.41314635]],

            [[-0.06750171, 0.46302059],
            [-0.47266394, -0.07047667]],

            [[-0.32873616,  0.38155359],
            [-0.30535603,  0.09282264]]
            ])
            for i in range(len(reachlp)):
                currHorizon = reachlp[i]
                rectangle = patches.Rectangle((currHorizon[0][0], currHorizon[1][0]),
                                currHorizon[0][1] - currHorizon[0][0],
                                currHorizon[1][1] - currHorizon[1][0],
                                edgecolor='b', facecolor='none', linewidth=2, alpha=1)
                x = ax.add_patch(rectangle)

        # plt.hold(True)
        custom_lines = [Line2D([0], [0], color='b', lw=2),
                            Line2D([0], [0], color='red', lw=2, linestyle='--')]

        # leg2 = plt.legend([custom_lines], ["s", 'ss'], loc=4)
        plt.gca().add_artist(leg1)

        ax.legend(custom_lines, ['ReachLP', 'ReachLipSDP'], loc=4)
        plt.savefig("reachabilityPics/" + fileName + "Iteration" + str(iteration) + ".png")


        # plt.legend([x], ['a'])
        # plt.legend()
        plt.show()


    endTime = time.time()
    
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)


def repeatNetwork(network, horizon):
    # originalNetworkLength = len(network.Linear)
    # for j in range(horizon - 1):
    #     for i in range(originalNetworkLength):
    #         if type(network.Linear[i]) == nn.modules.activation.ReLU:
    #             network.Linear.append(nn.ReLU())
    #         else:
    #             s1, s0 = network.Linear[i].weight.shape
    #             network.Linear.append(nn.Linear(s0, s1))
    #             network.Linear[-1].weight = nn.Parameter(network.Linear[i].weight.detach().clone())
    #             network.Linear[-1].bias = nn.Parameter(network.Linear[i].bias.detach().clone())
    numberOfRepeats = horizon - 1
    originalNetworkLength = len(network.Linear)

    w0 = network.Linear[0].weight.detach().clone()
    wLast = network.Linear[originalNetworkLength - 1].weight.detach().clone()
    b0 = network.Linear[0].bias.detach().clone()
    bLast = network.Linear[originalNetworkLength - 1].bias.detach().clone().unsqueeze(1)
    weight = w0 @ wLast
    bias = (w0 @ bLast).squeeze() + b0

    for j in range(numberOfRepeats):
        for i in range(originalNetworkLength):
            if type(network.Linear[i]) == nn.modules.activation.ReLU:
                network.Linear.append(nn.ReLU())
            else:
                if i == 0:

                    network.Linear[-1] = nn.Linear(weight.shape[1], weight.shape[0])
                    network.Linear[-1].weight = nn.Parameter(weight)
                    network.Linear[-1].bias = nn.Parameter(bias)

                elif i < originalNetworkLength - 1 or j < numberOfRepeats - 1:
                    s1, s0 = network.Linear[i].weight.shape
                    network.Linear.append(nn.Linear(s0, s1))
                    network.Linear[-1].weight = nn.Parameter(network.Linear[i].weight.detach().clone())
                    network.Linear[-1].bias = nn.Parameter(network.Linear[i].bias.detach().clone())

    network.Linear.append(nn.Linear(wLast.shape[1], wLast.shape[0]))
    network.Linear[-1].weight = nn.Parameter(wLast)
    network.Linear[-1].bias = nn.Parameter(bLast.squeeze())


if __name__ == '__main__':
    main()
    plt.show()