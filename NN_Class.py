from packages import *

class NN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Linear = nn.Sequential( 
                nn.Linear(dim, 10), 
                nn.ReLU(),
                nn.Linear(10, dim),
        )

    def load(self, path):
        stateDict = torch.load(path)
        self.load_state_dict(stateDict)
    # @TODO
    # def train(self):

    def forward(self, x):
        output = self.Linear(x.float())
        return output