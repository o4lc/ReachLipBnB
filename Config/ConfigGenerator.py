import json
import torch


def main():
    fileName = "quadRotorV2.0"
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
    finalHorizon = 1
    initialGD = False
    performMultiStepSingleHorizon = False
    plotProjectionsOfHigherDims = True
    minimalPCA = True

    # A = None
    # B = None
    # c = None
    # dt = None
    # if fileName == "doubleIntegrator.pth" or fileName == "doubleIntegrator_reachlp.pth":
    #     A = torch.Tensor([[1, 1], [0, 1]])
    #     B = torch.Tensor([[0.5], [1]])
    #     c = torch.Tensor([0])
    #
    #     # lowerCoordinate = torch.Tensor([1., 1.5])
    #     # upperCoordinate = torch.Tensor([2., 2.5])
    #
    #     lowerCoordinate = torch.Tensor([2.5, -0.25])
    #     upperCoordinate = torch.Tensor([3., 0.25])
    #
    # elif fileName == "quadRotor.pth" or fileName == "quadRotorv2.0.pth":
    dt = 0.1
    A = torch.Tensor([[0., 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])

    A = torch.eye(len(A)) + A * dt
    B = torch.Tensor([[0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.],
                      [9.8, 0., 0.],
                      [0., -9.8, 0.],
                      [0., 0., 1.]])
    B = B * dt

    c = torch.Tensor([0, 0, 0, 0, 0, -9.8])
    c = c * dt

    lowerCoordinate = torch.Tensor([4.69, 4.69, 2.9, 0.94, -0.001, -0.001])
    upperCoordinate = torch.Tensor([4.71, 4.71, 3.1, 0.95, 0.001, 0.001])
    # lowerCoordinate = torch.Tensor([4.6, 4.6, 2.9, 0.93, -0.001, -0.001])
    # upperCoordinate = torch.Tensor([4.8, 4.9, 3.1, 0.96, 0.001, 0.001])

    # elif fileName == "RobotArmStateDict2-50-2.pth":
    #     lowerCoordinate = torch.Tensor([np.pi / 3., np.pi / 3.])
    #     upperCoordinate = torch.Tensor([2 * np.pi / 3., 2 * np.pi / 3.])
    #
    #     minimalPCA = False

    pathToStateDictionary = "Networks/" + fileName + ".pth"

    configDictionary = {
        "eps":eps,
        "verbose": verbose,
        "verboseMultiHorizon": verboseMultiHorizon,
        "verboseEssential": verboseEssential,
        "scoreFunction": scoreFunction,
        "virtualBranching": virtualBranching,
        "numberOfVirtualBranches": numberOfVirtualBranches,
        "maxSearchDepthLipschitzBound": maxSearchDepthLipschitzBound,
        "normToUseLipschitz": normToUseLipschitz,
        "useTwoNormDilation": useTwoNormDilation,
        "useSdpForLipschitzCalculation": useSdpForLipschitzCalculation,
        "lipschitzSdpSolverVerbose": lipschitzSdpSolverVerbose,
        "finalHorizon": finalHorizon,
        "initialGD": initialGD,
        "performMultiStepSingleHorizon": performMultiStepSingleHorizon,
        "plotProjectionsOfHigherDims": plotProjectionsOfHigherDims,
        "minimalPCA": minimalPCA,
        "dt": dt,
        "A": A,
        "B": B,
        "c": c,
        "lowerCoordinate": lowerCoordinate,
        "upperCoordinate": upperCoordinate,
        "pathToStateDictionary": pathToStateDictionary
    }

    with open(fileName + ".json", 'w') as jsonFile:
        json.dump(configDictionary, jsonFile)

if __name__ == '__main__':
    main()

