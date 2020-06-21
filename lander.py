import gym
import model

env = gym.make('LunarLander-v2').unwrapped

config = model.ModelConfig()

m = model.Model(env, config)

m.train(2000)