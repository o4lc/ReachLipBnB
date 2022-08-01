from packages import *

class NN(nn.Module):
    def __init__(self, path):
        super().__init__()
        stateDictionary = torch.load(path)
        layers = []
        for keyEntry in stateDictionary:
            if "weight" in keyEntry:
                layers.append(nn.Linear(stateDictionary[keyEntry].shape[1], stateDictionary[keyEntry].shape[0]))
                layers.append(nn.ReLU())
        layers.pop()
        self.Linear = nn.Sequential(
            *layers
        )
        self.load_state_dict(stateDictionary)

    def load(self, path):
        stateDict = torch.load(path)
        self.load_state_dict(stateDict)

    # @TODO
    # def train(self):

    def forward(self, x):
        return self.Linear(x)
