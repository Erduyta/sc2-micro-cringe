import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from sc2.ids.unit_typeid import UnitTypeId
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from bot_model import ReplayMemory, Transition, DQN
from IPython import display


class Agent():
    def __init__(self) -> None:
        self.device = torch.device("cpu")  # cuda additional tests
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.n_actions = 5
        self.n_observations = 170

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, 'bot_from_not_sc2.pth')
        self.policy_net.load_state_dict(torch.load(file_name, map_location=self.device))
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []
        self.record_duration = 0

        # eemmm..?
        self.is_ipython = 'inline' in matplotlib.get_backend()
        self.num_episodes = 105
        # link to botai on_step
        self.reset = True
        self.runstep = 0
        self.next_action = None  # aa for init dunnow never should be used
        self.action = None
        self.tensor_state = None

    def select_action(self, state, botai):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def main_ich(self, botai):
        # every return we trigger bot update in a way
        while True:
            if self.runstep == 0:
                self.reset = True
                self.runstep = 1
                return None
            if self.runstep == 1:
                self.tensor_state = torch.tensor(botai.state_bot, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.runstep = 2
            if self.runstep == 2:  # cycle for one round
                self.action = self.select_action(self.tensor_state, botai)
                self.next_action = self.action.item()
                self.runstep = 3  # make action
                return None
            if self.runstep == 3:  # made action
                # observation, reward, terminated, truncated = env.step(action.item())
                observation = botai.state_bot
                reward = botai.reward
                terminated = botai.terminated
                truncated = botai.truncated
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(self.tensor_state, self.action, next_state, reward)
                self.tensor_state = next_state
                # self.optimize_model()
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                if done:
                    self.episode_durations.append(botai.life_exp)
                    if self.record_duration <= botai.life_exp + len(self.episode_durations):
                        self.record_duration = botai.life_exp + len(self.episode_durations)
                        # self.policy_net.save('bot_from_not_sc2.pth')
                    self.plot_durations()
                    self.runstep = 0
                    return None
                self.runstep = 2
