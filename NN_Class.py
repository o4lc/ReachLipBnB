from packages import *

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Sequential( 
                nn.Linear(2, 10), 
                nn.ReLU(),
                nn.Linear(10, 2),
        )

    # @TODO
    # def train(self):

    def forward(self, x):
        output = self.Linear(x.float())
        return output