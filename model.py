import torch
import torch.nn as nn
import torch.optim as optim
from dqn import Net
from gym import Env
import os
from replay import ReplayMemory, Transition

device = torch.device("cpu")


class Model(object):

    def __init__(self, env: Env, config, net = None):
        super().__init__()

        self.config = config

        hiddenLayers = config.HiddenLayers

        self.GAMMA = config.training.Gamma
        self.ALPHA = config.training.Alpha
        self.EPSILON = config.training.Epsilon
        self.EDECAY = config.training.EpsilonDecay
        self.BATCH_SIZE = config.training.BatchSize

        self.env = env

        state = env.reset()
        nnLayers =  [len(state)] + hiddenLayers + [env.action_space.n]

        self.net = Net(nnLayers)

    def __optimize(self, memory: ReplayMemory):
        if self.BATCH_SIZE > len(memory):
            return

        transitions = memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        

    def train(self, num_episodes):
    
    # def save(self, path):
    #     torch.save(self.net, )

    def output(self, input):
        return self.net.forward(input)


mse = nn.MSELoss()

output = net(output)
target = torch.randn(10)
target = target.view(1, -1)

loss = mse(output, target)

print(loss)



# create your optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = mse(output, target)
loss.backward()
optimizer.step()    # Does the update


def optimize():
    if len