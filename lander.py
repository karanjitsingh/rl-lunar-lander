import gym
import model
import sys
import utils.breaker as breaker
from pprint import pprint

env = gym.make('LunarLander-v2').unwrapped

render = False
if len(sys.argv) > 1 and sys.argv[1] == "render":
    render = True

config = model.ModelConfig()

config.Training.Gamma = 0.99
config.Training.Alpha = 0.0001
config.Training.Epsilon = [0.9, 0.05]
config.Training.EpsilonDecay = 100
config.Training.BatchSize = 128
config.Training.MemorySize = 10000
config.Training.MemoryInitFill = 0.2
config.Training.TargetUpdate = 10
config.Training.EpisodeLimit = 750
config.HiddenLayers = [100,100]
config.Description = "epsilon graph"


def onBreak():
    m.setSafeBreak()

breaker.setBreakHandle(onBreak)

m = model.Model(env, config)
m.train(3000, render=render)



props, path =  m.saveModel(suffix=".3000")
pprint(props)

input("\nPress Enter to continue...")
m.play(episodes=5)