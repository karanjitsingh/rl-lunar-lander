import gym
import model
import random
import torch
import gym
import numpy

torch.manual_seed(0)

env = gym.make('CartPole-v0').unwrapped

config = model.ModelConfig()

config.Training.Gamma = 0.999
config.Training.Alpha = 0.000022
config.Training.Epsilon = [0.99, 0.05]
config.Training.EpsilonDecay = 400
config.Training.BatchSize = 64
config.Training.MemorySize = 100000

config.HiddenLayers = [150,100]


m = model.Model(env, config)

m.train(500, render=True)

m.play(episodes=100)