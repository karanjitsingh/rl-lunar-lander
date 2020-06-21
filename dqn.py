import torch.nn as nn
import torch.nn.functional as F
import gym

class Net(nn.Module):

    def __init__(self, layers):
        super(Net, self).__init__()

        llen = len(layers)

        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], layers[3])


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x