import torch

from packages import *

from BranchAndBound import BranchAndBound
from NeuralNetwork import NeuralNetwork

def main():

    eps = .001
    verbose = 0

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    # device=torch.device("cpu")
    # pathToStateDictionary = "Networks/randomNetwork.pth"
    pathToStateDictionary = "Networks/randomNetwork2.pth"
    network = NeuralNetwork(pathToStateDictionary)
    dim = network.Linear[0].weight.shape[1]

    network.to(device)

    lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    upperCoordinate = torch.Tensor([1., 1.]).to(device)
    c = torch.Tensor([1., 2]).to(device)

    startTime = time.time()
    BB = BranchAndBound(upperCoordinate, lowerCoordinate, verbose=verbose, inputDimension=dim, eps=eps, network=network,
                        queryCoefficient=c, device=device, nodeBranchingFactor=2, scoreFunction='length', pgdIterNum=0)
    lowerBound, upperBound, space_left = BB.run()
    endTime = time.time()

    if verbose:
        print(BB)

    print('Best lower/upper bounds are:', lowerBound, '->' ,upperBound)
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)
    

if __name__ == '__main__':
    main()