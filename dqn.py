import torch.nn as nn
import torch.nn.functional as F
import gym



class Net(nn.Module):

    def __init__(self, layers):
        super(Net, self).__init__()

        llen = len(layers)

        self.fc = []

        for i in range(len(layers)-1):
            self.fc.append(nn.Linear(layers[i],layers[i+1]))

    def forward(self, x):

        for i in range(len(self.fc)):
            x = F.relu(self.fc[i](x))

        return x