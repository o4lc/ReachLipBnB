from packages import *

from Branch_Bound_Class import Branch_Bound
from NN_Class import NN

dim = 2
eps = 0.01

Net = NN().float()

l = [-100.1, -100.1]
u = [100.1, 100.1]
# c = torch.randn(2).float()
c = torch.Tensor([1, 2]).float()

print('c =', c)

BB = Branch_Bound(u, l, verbose=1, dim=dim, eps=eps, network=Net, queryCoefficient=c)
LB, UB, space_left = BB.run()
print('Best lower/upper bounds are:', LB, '->' ,UB)
print(BB)