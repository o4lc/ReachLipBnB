import numpy as np
import cvxpy as cp

import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import no_grad
from torch.autograd.functional import jacobian
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, patches