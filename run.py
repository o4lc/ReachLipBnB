from packages import *

from Branch_Bound_Class import Branch_Bound
from utilities import plot_space

dim = 2
eps = 0.1



l = [0.1, 0.1]
u = [10.1, 10.1]

BB = Branch_Bound(u, l, verbose=1, dim=dim, eps=eps)
LB, UB, space_left = BB.run()
print('Best upper/lower bounds are:', LB, '->' ,UB)
print(BB)