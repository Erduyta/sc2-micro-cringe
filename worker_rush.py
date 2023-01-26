from sc2 import maps
from sc2.bot_ai import BotAI
from matplotlib import pyplot as plt
from sc2.data import Difficulty, Race
from sc2.main import run_game
import random
import torch
from collections import deque
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
import numpy as np
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(441, 1024, 121)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.track_score = 0
        self.record = -100
        # model, trainer

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_state(self, current_state):
        return current_state.flatten()

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
        final_move = np.zeros([11, 11])
        if random.randint(0, 200) < self.epsilon:
            move1 = random.randint(0, 10)
            move2 = random.randint(0, 10)
            final_move[move1, move2] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_torch = torch.argmax(prediction).item()
            final_move[move_torch//11, move_torch%11] = 1
        return final_move


class TerranNoobgam(BotAI):
    def __init__(self) -> None:
        super().__init__()
        self.current_state = np.zeros([21, 21])  # actual state
        self.action = np.zeros([11, 11])
        self.action[5, 5] = 1
        self.reward = 0
        self.remember_our_hp = 1
        self.remember_enemy_hp = 4
        self.done = False
        self.agent = Agent()
        self.state_old = None  # remeber between cycles
        self.state_new = None  # remeber between cycles
        self.score = 0
        self.record = -1
        self.first_iter_of_round = True
        self.fig, self.axs = plt.subplots(2)
        self.plot_scores = []
        self.plot_records = []

    async def on_step(self, iteration):
        self.score = 0
        if iteration != 0:
            if not self.units(UnitTypeId.HELLION).exists or not self.enemy_units(UnitTypeId.ZERGLING).exists:
                self.done = True
                enemy_left_hp = 0
                for enemy_left in self.enemy_units(UnitTypeId.ZERGLING):
                    enemy_left_hp += enemy_left.health_percentage
                our_hp_left = 0
                if self.units(UnitTypeId.HELLION).exists:
                    our_hp_left = self.units(UnitTypeId.HELLION).first.health_percentage
                self.reward = (4 - enemy_left_hp + our_hp_left*2)*2  # fin reward
                self.score = self.reward
            # skip on first iteration of the round
            if not self.first_iter_of_round:
                self.state_new = self.agent.get_state(self.current_state)  # new state
                self.agent.train_short_memory(self.state_old, self.action, self.reward, self.state_new, self.done)  # stm
                self.agent.remember(self.state_old, self.action, self.reward, self.state_new, self.done)  # remember
            self.first_iter_of_round = False
            # await self.chat_send('end')
            # break cycle
            if not self.done:
                self.state_old = self.agent.get_state(self.current_state)
                self.action = self.agent.get_action(self.state_old)
                # state
                # -> action
                # action -> sc2
                # action
            if not self.done:
                self.score = 0  # score
                unit = self.units(UnitTypeId.HELLION).first
                enemy_units = self.enemy_units(UnitTypeId.ZERGLING)
                if_move = True
                place = np.argwhere(self.action == 1)[0]
                for enemy_unit in enemy_units:
                    x = enemy_unit.position.x - unit.position.x
                    y = enemy_unit.position.y - unit.position.y
                    if abs(x - place[0] + 5) < 1 and abs(y - place[1] + 5):
                        unit.attack(enemy_unit)
                        if_move = False
                        break
                if if_move:
                    move_to = Point2((place[0] + unit.position.x - 5, place[1] + unit.position.y - 5))
                    unit.move(move_to)
                cd = 0
                if unit.weapon_cooldown > 0:
                    cd = 1
                self.current_state = np.zeros([21, 21])
                current_enemy_hp = 0
                current_our_hp = unit.health_percentage  # 0-1
                for enemy_unit in enemy_units:
                    current_enemy_hp += enemy_unit.health_percentage
                    x = round(enemy_unit.position.x - unit.position.x)
                    y = round(enemy_unit.position.y - unit.position.y)
                    if 10 >= x >= -10 and 10 >= y >= -10:
                        self.current_state[x+10][y+10] = 1
                self.current_state[10, 10] = cd
                self.reward = (current_our_hp - self.remember_our_hp)*2 + (current_enemy_hp - self.remember_enemy_hp)/4  # short term reward

            if self.done:  # reset game
                self.agent.n_games += 1
                self.agent.train_long_memory()
                if self.score > self.record:
                    self.record = self.score
                    self.agent.model.save()
                self.plot_scores.append(self.score)
                self.plot_records.append(self.record)
                print(self.score, self.record)
                await self.chat_send('end')
                self.done = False
                self.first_iter_of_round = True
                self.reward = 0
                x_coord = np.linspace(0, self.agent.n_games, self.agent.n_games)
                # print(x_coord)
                self.axs[0].cla()
                self.axs[1].cla()
                self.axs[0].plot(x_coord, self.plot_scores)
                self.axs[1].plot(x_coord, self.plot_records)
                plt.pause(0.05)


        # state
        # -> action
        # action -> sc2
        # new state


def main():
    run_game(
        maps.get("ai-bot-erduyta"),
        [Bot(Race.Terran, TerranNoobgam()), Computer(Race.Zerg, Difficulty.Medium)],
        realtime=False,
        save_replay_as="WorkerRush.SC2Replay",
    )


if __name__ == "__main__":
    main()