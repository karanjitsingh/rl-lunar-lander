import os
import random
import time
import math

import torch
from torch.functional import F
import torch.nn as nn
import torch.optim as optim
from gym import Env

from dqn import Net
from replay import ReplayMemory, Transition

from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()
device = torch.device("cpu")


# Model config
class ModelConfig(object):

    # Training config
    class TrainingConfig(object):
        def __init__(self, Gamma=0.99, Alpha=0.01, Epsilon=[0.9,0.05], EpsilonDecay=200, BatchSize = 128, MemorySize = 100000, MemoryInitFill = 0.1, TorchSeed = torch.seed()):
            super().__init__()
            self.Gamma = Gamma
            self.Alpha = Alpha
            self.Epsilon = Epsilon
            self.EpsilonDecay = EpsilonDecay
            self.BatchSize = BatchSize
            self.MemorySize = MemorySize
            self.MemoryInitFill = MemoryInitFill
            self.TorchSeed = str(TorchSeed)

    # Default config
    def __init__(self, hiddenLayers = [150, 100], trainingConfig: TrainingConfig = TrainingConfig()):
        super().__init__()
        self.HiddenLayers = hiddenLayers
        self.Training = trainingConfig


class Model(object):
    # Create a new model based on Env & config
    def __init__(self, env: Env, config: ModelConfig):
        super().__init__()

        self.config = config

        hiddenLayers = config.HiddenLayers

        self.GAMMA = config.Training.Gamma
        self.ALPHA = config.Training.Alpha
        self.EPSILON_START = config.Training.Epsilon[0]
        self.EPSILON_END = config.Training.Epsilon[1]
        self.EDECAY = config.Training.EpsilonDecay
        self.BATCH_SIZE = config.Training.BatchSize
        self.MEMORY_SIZE = config.Training.MemorySize
        self.MEMORY_INIT_FILL = config.Training.MemoryInitFill

        self.env = env

        self.TrainingEpisodes = 0
        self.TrainingTime = 0

        state = env.reset()
        nnLayers =  [len(state)] + hiddenLayers + [env.action_space.n]

        self.net = Net(nnLayers)

        d = config.__dict__
        d['Training'] = d['Training'].__dict__
        
        d['Training']['Epsilon'] = str(d['Training']['Epsilon'])
        d['HiddenLayers'] = str(d['HiddenLayers'])

        d = dict(d, **d['Training'])

        del d['Training']
        
        writer.add_hparams(d, {})


    # Format seconds to mm:ss
    def __formatTime(self, time):
        minutes, seconds = divmod(time, 60)
        return "{:0>2}:{:05.2f}".format(int(minutes),seconds)

    
    # Get the output from the current model
    def output(self, input):
        return self.net.forward(torch.tensor(input))


    # Play an episode with current model
    def play(self, episodes = 10, printValues = False):
        env = self.env


        for i in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = torch.argmax(self.net.forward(torch.tensor(state).float())).item()
                state, reward, done, _ = self.env.step(action)

                if printValues:
                    print(state, reward, done)

                env.render()

        env.close()


    # Optimize the model, apply gradient descent
    def __optimize(self, memory: ReplayMemory, optimizer: optim.Optimizer, steps):
        if self.BATCH_SIZE > len(memory):
            return

        transitions = memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()


        # clip gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()

        writer.add_scalar('Loss', loss.item(), steps)


    def __selectAction(self, state, eps_threshold):
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                return self.net(state).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], device=device, dtype=torch.long)  


    def __fillMemory(self, memory: ReplayMemory, fill):
        
        steps = 0
        env = self.env

        while steps < fill* memory.capacity:
            
            state = torch.tensor([env.reset()]).float()
            done = False

            while not done:
              
                # always random
                action = self.__selectAction(state, 1)
                
                next_state, reward, done, _ = env.step(action.item())

                if not done:
                    next_state = torch.tensor([next_state]).float()
                else:
                    next_state = None

                memory.push(state, action, next_state, torch.tensor([reward]).float())
                steps += 1

                state = next_state


    # Train the current model
    def train(self, num_episodes, render = False):
        startTick = time.time()

        env = self.env
        memory = ReplayMemory(self.MEMORY_SIZE)

        self.__fillMemory(memory, self.MEMORY_INIT_FILL)


        optimizer = optim.SGD(list(self.net.parameters()), lr=self.ALPHA)

        steps_done = 0

        eps_threshold = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * math.exp(-1. * steps_done / self.EDECAY)
        avg_reward = 0

        for i in range(num_episodes):
            cum_reward = 0
            steps = 0
        
            state = torch.tensor([env.reset()]).float()
            done = False

            while not done:
                eps_threshold = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * math.exp(-1. * steps_done / self.EDECAY)

                action = self.__selectAction(state, eps_threshold)
                next_state, reward, done, _ = env.step(action.item())

                if not done:
                    next_state = torch.tensor([next_state]).float()
                else:
                    next_state = None

                memory.push(state, action, next_state, torch.tensor([reward]).float())

                state = next_state
                steps += 1
                steps_done += 1
                cum_reward += reward

                self.__optimize(memory, optimizer, steps_done)


                if render:
                    env.render()
        
            avg_reward += cum_reward

            writer.add_scalar('Steps', steps, i)
            writer.add_scalar('Reward', cum_reward, i)
            writer.add_scalar('Time', time.time() - startTick, i)
            writer.add_scalar('Epsilon', eps_threshold, i)

            print("{i}\t: Reward: {r}\t Steps: {s}\t AvgReward: {ar:.2f}\t\t{t}".format(i = str(i+1), r = cum_reward, s = steps, ar = avg_reward / (i+1), t = self.__formatTime(time.time() - startTick)))
        
        if render:
            env.close()
    
        elapsed = time.time()-startTick
        self.TrainingTime = elapsed
        self.TrainingEpisodes = num_episodes

        print("{n} Episodes, Time Elapsed {t}".format(n=str(num_episodes), t=self.__formatTime(elapsed)))
