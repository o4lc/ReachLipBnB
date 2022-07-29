import torch

from packages import *

from Branch_Bound_Class import Branch_Bound
from NN_Class import NN

def main():
    dim = 2
    eps = 60
    verbose = 1

    if torch.cuda.is_available():
        device=torch.device("cuda", 0)
    else:
        device=torch.device("cpu", 0)


    network = NN(dim)

    # torch.save(network.state_dict(), "randomNetwork.pth")
    network.load("randomNetwork.pth")
    # network.to(device)

    lowerCoordinate = [-100.1, -100.1]
    upperCoordinate = [100.1, 100.1]
    c = torch.Tensor([1, 2]).float()

    startTime = time.time()
    BB = Branch_Bound(upperCoordinate, lowerCoordinate, verbose=verbose, dim=dim, eps=eps, network=network,
                      queryCoefficient=c, device=device, branch_constant=8, scoreFunction='volume')
    LB, UB, space_left = BB.run()
    endTime = time.time()

    if verbose:
        print(BB)

    print('Best lower/upper bounds are:', LB, '->' ,UB)
    print('The algorithm took (s):', endTime - startTime, 'with eps =', eps)
    

if __name__ == '__main__':
    main()