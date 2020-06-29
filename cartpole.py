import gym
import model
import random
import torch
import gym
import numpy

# torch.manual_seed(436111087444500)

env = gym.make('CartPole-v0').unwrapped

config = model.ModelConfig()

config.Training.Gamma = 0.9
config.Training.Alpha = 0.00019
config.Training.Epsilon = [0.9, 0.05]
config.Training.EpsilonDecay = 200
config.Training.BatchSize = 30
config.Training.MemorySize = 200
config.Training.MemoryInitFill = 0.2
config.Training.TargetUpdate = 10

config.HiddenLayers = [150,100]


m = model.Model(env, config)

m.train(300, render=True)
