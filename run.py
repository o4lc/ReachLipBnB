from unicodedata import decimal
import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork
torch.set_printoptions(precision=8)

def main():

    eps = .001
    verbose = 0
    virtualBranching = True

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # Temporary
    device=torch.device("cpu")
    print(device)
    print(' ')

    # pathToStateDictionary = "Networks/randomNetwork.pth"
    # pathToStateDictionary = "Networks/randomNetwork2.pth"
    # pathToStateDictionary = "Networks/trainedNetwork1.pth"
    # pathToStateDictionary = "Networks/RobotArmStateDict2-5-2.pth"
    # pathToStateDictionary = "Networks/Test3-5-3.pth"
    # pathToStateDictionary = "Networks/ACASXU.pth"
    pathToStateDictionary = "Networks/mnist_3_50.pth"
    network = NeuralNetwork(pathToStateDictionary)

    # @TODO: Input dimension and Output dimension are not necessarily the same!
    dim = network.Linear[0].weight.shape[1]
    outputDim = network.Linear[-1].weight.shape[0]
    network.to(device)

    # lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    # upperCoordinate = torch.Tensor([1., 1.]).to(device)
    # lowerCoordinate = torch.Tensor([torch.pi / 3, torch.pi / 3, torch.pi / 3]).to(device)
    # upperCoordinate = torch.Tensor([2 * torch.pi / 3, 2 * torch.pi / 3, 2 * torch.pi / 3]).to(device)
    # c = torch.Tensor([1., 1., 1.]).to(device)
    lowerCoordinate = torch.Tensor([-2./2560000] * dim).to(device)
    upperCoordinate = torch.Tensor([2./2560000] * dim).to(device)
    c = torch.ones(outputDim).to(device)

    startTime = time.time()
    BB = BranchAndBound(upperCoordinate, lowerCoordinate, verbose=verbose, inputDimension=dim, eps=eps, network=network,
                        queryCoefficient=c, device=device, nodeBranchingFactor=4, branchNodeNum=512,
                        scoreFunction='length',
                        pgdIterNum=0, pgdNumberOfInitializations=4, pgdStepSize=0.5, virtualBranching=virtualBranching)
    lowerBound, upperBound, space_left = BB.run()
    endTime = time.time()

    if verbose:
        print(BB)
    
    print(' ')
    print('Best lower/upper bounds are:', lowerBound, '->' ,upperBound)
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)


if __name__ == '__main__':
    main()