import ast
import json
import math
import os
import random
import time
from pprint import pprint

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import Env
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter

from dqn import Net
from replay import ReplayMemory, Transition

device = torch.device("cpu")


# Model config
class ModelConfig(object):

    # Training config
    class TrainingConfig(object):
        def __init__(self, Gamma=0.99, Alpha=0.01, Epsilon=[0.9,0.05], EpsilonDecay=200, BatchSize = 128, MemorySize = 100000, MemoryInitFill = 0.1, TargetUpdate = 10, EpisodeLimit = 750, PunishLimit = 0, TorchSeed = torch.seed()):
            super().__init__()
            self.Gamma = Gamma
            self.Alpha = Alpha
            self.Epsilon = Epsilon
            self.EpsilonDecay = EpsilonDecay
            self.BatchSize = BatchSize
            self.MemorySize = MemorySize
            self.MemoryInitFill = MemoryInitFill
            self.TorchSeed = str(TorchSeed)
            self.TargetUpdate = TargetUpdate
            self.EpisodeLimit = EpisodeLimit
            self.PunishLimit = PunishLimit

    # Default config
    def __init__(self, hiddenLayers = [150, 100], trainingConfig: TrainingConfig = TrainingConfig(), description = ""):
        super().__init__()
        self.HiddenLayers = hiddenLayers
        self.Training = trainingConfig
        self.Description = description


class Model(object):
    # Create a new model based on Env & config
    def __init__(self, env: Env, config: ModelConfig):
        super().__init__()

        self.config = config
        self.writer = SummaryWriter()

        hiddenLayers = config.HiddenLayers

        self.safeBreak = False

        self.GAMMA = config.Training.Gamma
        self.ALPHA = config.Training.Alpha
        self.EPSILON_START = config.Training.Epsilon[0]
        self.EPSILON_END = config.Training.Epsilon[1]
        self.EDECAY = config.Training.EpsilonDecay
        self.BATCH_SIZE = config.Training.BatchSize
        self.MEMORY_SIZE = config.Training.MemorySize
        self.MEMORY_INIT_FILL = config.Training.MemoryInitFill
        self.TARGET_UPDATE = config.Training.TargetUpdate
        self.EPISODE_LIMIT = config.Training.EpisodeLimit
        self.PUNISH_LIMIT = config.Training.PunishLimit

        self.env = env

        self.TrainingEpisodes = 0
        self.TrainingTime = 0

        state = env.reset()
        nnLayers =  [len(state)] + hiddenLayers + [env.action_space.n]

        self.net = Net(nnLayers).to(device)
        self.target_net = Net(nnLayers).to(device)

        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.modelprops = config.__dict__
        self.modelprops['Training'] = self.modelprops['Training'].__dict__
        
        self.modelprops['Training']['Epsilon'] = str(self.modelprops['Training']['Epsilon'])
        self.modelprops['HiddenLayers'] = str(self.modelprops['HiddenLayers'])

        self.modelprops['EnvName'] = env.unwrapped.spec.id

        self.modelprops = dict(self.modelprops, **self.modelprops['Training'])

        del self.modelprops['Training']
        
        self.writer.add_hparams(self.modelprops, {})

        self.writer.add_text("Description", config.Description)
        self.writer.add_text("Params", json.dumps(self.modelprops, indent=2))

        self.modelprops['SummaryLogs'] = self.writer.log_dir


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
            raise "batch size greater than capacity"

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
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()


        # clip gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()

        lossvalue = loss.item()
        return lossvalue

    
    # Optimize the model, apply gradient descent
    def __optimize_nomem(self, transition: Transition, optimizer: optim.Optimizer, steps, onpolicy=False):
        
        state_action_value = self.net(transition.state).gather(1, transition.action)
        next_state_value = torch.tensor(0) if transition.next_state is None else self.net(transition.state).max(1)[0]

        action = None if transition.next_state is None else torch.tensor([[torch.argmax(self.net(transition.state))]])

        # Compute the expected Q values
        expected_state_action_value = (next_state_value * self.GAMMA) + transition.reward

        # Compute Huber loss
        loss = F.mse_loss(state_action_value, expected_state_action_value)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()


        # clip gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()

        lossvalue = loss.item()

        if onpolicy:
            return lossvalue, action
        return lossvalue

    
    # Optimize the model, apply gradient descent
    def __optimize_sarsa(self, transition: Transition, optimizer: optim.Optimizer, next_action, steps, onpolicy=False):
        
        state_action_value = self.net(transition.state).gather(1, transition.action)
        next_state_value = torch.tensor(0) if transition.next_state is None else self.net(transition.next_state).gather(1, next_action)

        # Compute the expected Q values
        expected_state_action_value = (next_state_value * self.GAMMA) + transition.reward

        # Compute Huber loss
        loss = F.mse_loss(state_action_value, expected_state_action_value)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()


        # clip gradients
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()

        lossvalue = loss.item()

        return lossvalue


    def __getPolicyAction(self, state):
        with torch.no_grad():
            return self.net(state).max(1)[1].view(1,1)


    def __getRandomAction(self, state, eps_threshold, defaultAction):
        sample = random.random()

        if sample <= eps_threshold:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], device=device, dtype=torch.long)

        return defaultAction
        

    def __selectAction(self, state, eps_threshold):
        sample = random.random()

        if sample > eps_threshold:
            return self.__getPolicyAction(state)
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


    def get_epsilon(self, steps):
        return self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * math.exp(-1. * steps / self.EDECAY)

    # Train the current model
    def train(self, num_episodes, render = False):
        startTick = time.time()

        env = self.env
        memory = ReplayMemory(self.MEMORY_SIZE)
        self.safeBreak = False

        self.__fillMemory(memory, self.MEMORY_INIT_FILL)


        optimizer = optim.Adam(list(self.net.parameters()), lr=self.ALPHA)

        steps_done = 0

        eps_threshold = self.EPSILON_START
        avg_reward = 0
        max_reward = 0

        for i in range(num_episodes):

            if i != 0 and i % 300 == 0:
                self.saveModel(suffix="." + str(i))

            if (self.safeBreak):
                print("Safely breaking training loop, Episodes: {i} of {n}".format(i=i, n=num_episodes))
                break

            cum_reward = 0
            steps = 0
            losssum = 0
        
            state = torch.tensor([env.reset()]).float()
            done = False


            while not done and (self.EPISODE_LIMIT == 0 or steps < self.EPISODE_LIMIT):
                eps_threshold = self.get_epsilon(steps_done)
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

                l = self.__optimize(memory, optimizer, steps_done)
                losssum += l

                if render:
                    env.render()

            if steps >= self.EPISODE_LIMIT and self.EPISODE_LIMIT != 0 and self.PUNISH_LIMIT > 0:
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__selectAction(state, eps_threshold)
                _, reward, done, _ = env.step(action.item())

                next_state = None

                if not done:
                    reward = -1 * self.PUNISH_LIMIT

                memory.push(state, action, next_state, torch.tensor([reward]).float())

                steps += 1
                steps_done += 1
                cum_reward += reward

                losssum += self.__optimize(memory, optimizer, steps_done)

                if render:
                    env.render()


        
            avg_reward += cum_reward
            max_reward = cum_reward if cum_reward > max_reward else cum_reward

            self.writer.add_scalar('LossAvg', losssum/steps, i)
            self.writer.add_scalar('Steps', steps, i)
            self.writer.add_scalar('Reward', cum_reward, i)
            self.writer.add_scalar('Epsilon', eps_threshold, i)

            print("{i}\t: Reward: {r}\t Steps: {s}\t AvgReward: {ar:.2f}\t\t{t}\t{d}\t{desc}".format(i = str(i+1), r = cum_reward, s = steps, ar = avg_reward / (i+1), t = self.__formatTime(time.time() - startTick), d = self.writer.log_dir, desc = self.config.Description))
        
            if i % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.net.state_dict())
        if render:
            env.close()
    
        elapsed = time.time()-startTick
        self.TrainingTime = elapsed
        self.TrainingEpisodes = num_episodes

        print("{n} Episodes, Time Elapsed {t}".format(n=str(num_episodes), t=self.__formatTime(elapsed)))


    
    # Train the current model
    def train_nomem(self, num_episodes, render = False):
        startTick = time.time()

        env = self.env
        self.safeBreak = False

        optimizer = optim.Adam(list(self.net.parameters()), lr=self.ALPHA)

        steps_done = 0

        eps_threshold = self.EPSILON_START
        avg_reward = 0
        max_reward = 0

        for i in range(num_episodes):

            if i != 0 and i % 300 == 0:
                self.saveModel(suffix="." + str(i))

            if (self.safeBreak):
                print("Safely breaking training loop, Episodes: {i} of {n}".format(i=i, n=num_episodes))
                break

            cum_reward = 0
            steps = 0
            losssum = 0
        
            state = torch.tensor([env.reset()]).float()
            done = False
            next_action = torch.tensor(0)

            while not done and (self.EPISODE_LIMIT == 0 or steps < self.EPISODE_LIMIT):
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__getRandomAction(state, eps_threshold)
                next_state, reward, done, _ = env.step(action.item())

                if not done:
                    next_state = torch.tensor([next_state]).float()
                else:
                    next_state = None

                steps += 1
                steps_done += 1
                cum_reward += reward

                losssum += self.__optimize_nomem(Transition(state, action, next_state, torch.tensor([reward]).float()), optimizer, steps_done)

                state = next_state

                if render:
                    env.render()

            if steps >= self.EPISODE_LIMIT and self.EPISODE_LIMIT != 0 and self.PUNISH_LIMIT > 0:
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__selectAction(state, eps_threshold)
                _, reward, done, _ = env.step(action.item())

                next_state = None

                if not done:
                    reward = -1 * self.PUNISH_LIMIT

                steps += 1
                steps_done += 1
                cum_reward += reward

                losssum += self.__optimize_nomem(Transition(state, action, next_state, torch.tensor([reward]).float()), optimizer, steps_done)

                if render:
                    env.render()

            avg_reward += cum_reward
            max_reward = cum_reward if cum_reward > max_reward else cum_reward

            self.writer.add_scalar('LossAvg', losssum/steps, i)
            self.writer.add_scalar('Steps', steps, i)
            self.writer.add_scalar('Reward', cum_reward, i)
            self.writer.add_scalar('Epsilon', eps_threshold, i)

            print("{i}\t: Reward: {r}\t Steps: {s}\t AvgReward: {ar:.2f}\t\t{t}\t{d}\t{desc}".format(i = str(i+1), r = cum_reward, s = steps, ar = avg_reward / (i+1), t = self.__formatTime(time.time() - startTick), d = self.writer.log_dir, desc = self.config.Description))
        
            if i % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.net.state_dict())
        if render:
            env.close()
    
        elapsed = time.time()-startTick
        self.TrainingTime = elapsed
        self.TrainingEpisodes = num_episodes

        print("{n} Episodes, Time Elapsed {t}".format(n=str(num_episodes), t=self.__formatTime(elapsed)))

    
    # Train the current model
    def train_onpolicy(self, num_episodes, render = False):
        startTick = time.time()

        env = self.env
        self.safeBreak = False

        optimizer = optim.Adam(list(self.net.parameters()), lr=self.ALPHA)

        steps_done = 0

        eps_threshold = self.EPSILON_START
        avg_reward = 0
        max_reward = 0

        for i in range(num_episodes):

            if i != 0 and i % 300 == 0:
                self.saveModel(suffix="." + str(i))

            if (self.safeBreak):
                print("Safely breaking training loop, Episodes: {i} of {n}".format(i=i, n=num_episodes))
                break

            cum_reward = 0
            steps = 0
            losssum = 0
        
            state = torch.tensor([env.reset()]).float()
            done = False

            while not done and (self.EPISODE_LIMIT == 0 or steps < self.EPISODE_LIMIT):
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__selectAction(state, eps_threshold)
                next_state, reward, done, _ = env.step(action.item())

                if not done:
                    next_state = torch.tensor([next_state]).float()
                else:
                    next_state = None

                steps += 1
                steps_done += 1
                cum_reward += reward

                losssum += self.__optimize_nomem(Transition(state, action, next_state, torch.tensor([reward]).float()), optimizer, steps_done)

                state = next_state

                if render:
                    env.render()

            if steps >= self.EPISODE_LIMIT and self.EPISODE_LIMIT != 0 and self.PUNISH_LIMIT > 0:
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__selectAction(state, eps_threshold)
                _, reward, done, _ = env.step(action.item())

                next_state = None

                if not done:
                    reward = -1 * self.PUNISH_LIMIT

                steps += 1
                steps_done += 1
                cum_reward += reward

                losssum += self.__optimize_nomem(Transition(state, action, next_state, torch.tensor([reward]).float()), optimizer, steps_done)

                if render:
                    env.render()

            avg_reward += cum_reward
            max_reward = cum_reward if cum_reward > max_reward else cum_reward

            self.writer.add_scalar('LossAvg', losssum/steps, i)
            self.writer.add_scalar('Steps', steps, i)
            self.writer.add_scalar('Reward', cum_reward, i)
            self.writer.add_scalar('Epsilon', eps_threshold, i)

            print("{i}\t: Reward: {r}\t Steps: {s}\t AvgReward: {ar:.2f}\t\t{t}\t{d}\t{desc}".format(i = str(i+1), r = cum_reward, s = steps, ar = avg_reward / (i+1), t = self.__formatTime(time.time() - startTick), d = self.writer.log_dir, desc = self.config.Description))
        
        if render:
            env.close()
    
        elapsed = time.time()-startTick
        self.TrainingTime = elapsed
        self.TrainingEpisodes = num_episodes

        print("{n} Episodes, Time Elapsed {t}".format(n=str(num_episodes), t=self.__formatTime(elapsed)))


    # Train the current model
    def train_onpolicy(self, num_episodes, render = False):
        startTick = time.time()

        env = self.env
        self.safeBreak = False

        optimizer = optim.Adam(list(self.net.parameters()), lr=self.ALPHA)

        steps_done = 0

        eps_threshold = self.EPSILON_START
        avg_reward = 0
        max_reward = 0

        for i in range(num_episodes):

            if i != 0 and i % 300 == 0:
                self.saveModel(suffix="." + str(i))

            if (self.safeBreak):
                print("Safely breaking training loop, Episodes: {i} of {n}".format(i=i, n=num_episodes))
                break

            cum_reward = 0
            steps = 0
            losssum = 0
        
            state = torch.tensor([env.reset()]).float()
            done = False
            next_action = self.__selectAction(state, eps_threshold)

            while not done and (self.EPISODE_LIMIT == 0 or steps < self.EPISODE_LIMIT):
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__getRandomAction(state, eps_threshold, next_action)
                next_state, reward, done, _ = env.step(action.item())

                if not done:
                    next_state = torch.tensor([next_state]).float()
                    next_action = self.__selectAction(next_state, eps_threshold)
                else:
                    next_state = None



                steps += 1
                steps_done += 1
                cum_reward += reward

                losssum += self.__optimize_sarsa(Transition(state, action, next_state, torch.tensor([reward]).float()), optimizer, next_action, steps_done, onpolicy=True)

                action = next_action
                state = next_state

                if render:
                    env.render()

            if steps >= self.EPISODE_LIMIT and self.EPISODE_LIMIT != 0 and self.PUNISH_LIMIT > 0:
                eps_threshold = self.get_epsilon(steps_done)
                action = self.__getRandomAction(state, eps_threshold, next_action)
                _, reward, done, _ = env.step(action.item())

                next_state = None

                if not done:
                    reward = -1 * self.PUNISH_LIMIT

                steps += 1
                steps_done += 1
                cum_reward += reward

                next_action = self.select_action(next_state, eps_threshold)

                losssum += self.__optimize_sarsa(Transition(state, action, next_state, torch.tensor([reward]).float()), optimizer, next_action, steps_done, onpolicy=True)

                if render:
                    env.render()

            avg_reward += cum_reward
            max_reward = cum_reward if cum_reward > max_reward else cum_reward

            self.writer.add_scalar('LossAvg', losssum/steps, i)
            self.writer.add_scalar('Steps', steps, i)
            self.writer.add_scalar('Reward', cum_reward, i)
            self.writer.add_scalar('Epsilon', eps_threshold, i)

            print("{i}\t: Reward: {r}\t Steps: {s}\t AvgReward: {ar:.2f}\t\t{t}\t{d}\t{desc}".format(i = str(i+1), r = cum_reward, s = steps, ar = avg_reward / (i+1), t = self.__formatTime(time.time() - startTick), d = self.writer.log_dir, desc = self.config.Description))
        
            if i % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.net.state_dict())
        if render:
            env.close()
    
        elapsed = time.time()-startTick
        self.TrainingTime = elapsed
        self.TrainingEpisodes = num_episodes

        print("{n} Episodes, Time Elapsed {t}".format(n=str(num_episodes), t=self.__formatTime(elapsed)))


    # Safely break training loop:
    def setSafeBreak(self):
        self.safeBreak = True


    def saveModel(self, suffix=""):

        summaryname = self.modelprops['SummaryLogs'].split('\\')
        if len(summaryname) > 1:
            summaryname = summaryname[-1]
        else:
            summaryname = summaryname.split('/')
            summaryname = summaryname[-1]        

        if not os.path.isdir('./models/' + summaryname):
            os.mkdir('./models/' + summaryname)

        path = "./models/{name}/{name}{suffix}.model".format(name = summaryname, suffix = suffix)

        props = dict.copy(self.modelprops)

        state_dict = self.net.state_dict()
        props["state_dict"] = state_dict

        torch.save(props, path)

        print("Saved model to " + path)

        del props['state_dict']
        return props, path


class TrainedModel(object):

    def __init__(self, path):
        super().__init__()

        self.model = torch.load(path)

        envName = self.model['EnvName']
        self.env = gym.make(envName)

        state = self.env.reset()
        hiddenLayers = ast.literal_eval(self.model['HiddenLayers'])
        nnLayers =  [len(state)] + hiddenLayers + [self.env.action_space.n]
        self.net = Net(nnLayers).to(device)

        self.net.load_state_dict(self.model['state_dict'])

    # Play an episode with current model
    def play(self, render=True):
        env = self.env

        state = env.reset()
        done = False

        cum_reward = 0
        steps = 0

        while not done:
            action = torch.argmax(self.net.forward(torch.tensor(state).float())).item()
            state, reward, done, _ = self.env.step(action)
            cum_reward += reward
            steps += 1
            if (render):
                env.render()

        return cum_reward, steps
