import gym
import model
import sys
import utils.breaker as breaker
from pprint import pprint
import json
import io

env = gym.make('LunarLander-v2').unwrapped

render = False

if len(sys.argv) > 2 and sys.argv[2] == "render":
    render = True

rawconfig = dict()

with open(sys.argv[1]) as file:
    rawconfig = json.loads(file.read())

config = model.ModelConfig()

if "TorchSeed" in rawconfig.keys():
    config.Training.TorchSeed = rawconfig['TorchSeed']

config.Training.Gamma = rawconfig['Gamma']
config.Training.Alpha = rawconfig['Alpha']
config.Training.Epsilon = rawconfig['Epsilon']
config.Training.EpsilonDecay = rawconfig['EpsilonDecay']
config.Training.BatchSize = rawconfig['BatchSize']
config.Training.MemorySize = rawconfig['MemorySize']
config.Training.MemoryInitFill = rawconfig['MemoryInitFill']
config.Training.TargetUpdate = rawconfig['TargetUpdate']
config.Training.EpisodeLimit = rawconfig['EpisodeLimit']
config.HiddenLayers = rawconfig['HiddenLayers']
config.Description = rawconfig['Description']
config.PunishLimit = rawconfig['PunishLimit']

episodes = 3000

if( "Episodes" in rawconfig.keys()):
    episodes = rawconfig['Episodes']


def onBreak():
    m.setSafeBreak()

breaker.setBreakHandle(onBreak)

m = model.Model(env, config)
m.train(episodes, render=render)



props, path =  m.saveModel(suffix="." + str(episodes))
pprint(props)

input("\nPress Enter to continue...")
m.play(episodes=5)