import gym
import model
import random
import torch
import gym
import numpy


env = gym.make('CartPole-v0').unwrapped

config = model.ModelConfig()

config.Training.Gamma = 0.999
config.Training.Alpha = 0.00001
config.Training.Epsilon = [0.9, 0.05]
config.Training.EpsilonDecay = 400
config.Training.BatchSize = 128
config.Training.MemorySize = 100000
config.Training.MemoryInitFill = 0.3

config.HiddenLayers = [150,100]


m = model.Model(env, config)

m.train(300, render=False)

m.play(episodes=100)