import torch

from packages import *

from Branch_Bound_Class import Branch_Bound
from NN_Class import NN

def main():
    dim = 2
    eps = 1
    verbose = 0

    if torch.cuda.is_available():
        device=torch.device("cuda", 0)
    else:
        device=torch.device("cpu")

    # device=torch.device("cpu")
    network = NN(dim)

    # torch.save(network.state_dict(), "randomNetwork.pth")
    network.load("./Networks/randomNetwork.pth")
    network.to(device)

    lowerCoordinate = torch.Tensor([-1., -1.]).to(device)
    upperCoordinate = torch.Tensor([1., 1.]).to(device)
    c = torch.Tensor([1., 2]).to(device)

    startTime = time.time()
    BB = Branch_Bound(upperCoordinate, lowerCoordinate, verbose=verbose, dim=dim, eps=eps, network=network,
                      queryCoefficient=c, device=device, branch_constant=2, scoreFunction='length', pgdIterNum=1)
    LB, UB, space_left = BB.run()
    endTime = time.time()

    if verbose:
        print(BB)

    print('Best lower/upper bounds are:', LB, '->' ,UB)
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)
    

if __name__ == '__main__':
    main()