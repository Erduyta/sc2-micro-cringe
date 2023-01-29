import torch
from game import Field
from collections import deque
import random
import numpy as np
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, name) -> None:
        self.name = name
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(9, 128, 9)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # model, trainer

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_state(self, game: Field):
        return game.field.flatten()

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, action, reward, next_state, done in mini_sample:
        #   self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: exploration/expoitation
        self.epsilon = 80 - self.n_games
        final_move = np.zeros([3, 3])
        if random.randint(0, 200) < self.epsilon:
            move1 = random.randint(0, 2)
            move2 = random.randint(0, 2)
            final_move[move1, move2] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # input_state ==> [ Q(input_state, a1), Q(input_state, a2), ... ]
            move = torch.argmax(prediction).item()
            final_move[move//3, move%3] = 1
        return final_move


def train():
    count_games = 400
    track_score = 0
    record = -100
    plot_scores = []
    plot_records = []
    agent = Agent(0)
    game = Field()
    fig, axs = plt.subplots(2)
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
        # perfom move and get new state
        reward, done, score = game.game_step(agent.name, final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            track_score += score
            if agent.n_games % count_games == 0:
                agent.train_long_memory()
                print(track_score, record)
                if track_score > record:
                    record = track_score
                    agent.model.save()
                plot_scores.append(track_score)
                plot_records.append(record)
                track_score = 0
                x_coord = np.linspace(0, agent.n_games, agent.n_games//count_games)
                # print(x_coord)
                axs[0].cla()
                axs[1].cla()
                axs[0].plot(x_coord, plot_scores)
                axs[1].plot(x_coord, plot_records)
                plt.pause(0.05)


if __name__ == "__main__":
    train()
