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
LR = 0.0008


class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(441, 512, 121)
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
        self.big_score = 0

    async def on_step(self, iteration):
        self.score = 0
        # plt.imshow(self.current_state)
        # plt.pause(0.001)
        if iteration != 0:
            if not self.units(UnitTypeId.HELLION).exists or not self.enemy_units(UnitTypeId.ZERGLING).exists:
                self.done = True
                our_hp_left = 0 # 0
                if self.units(UnitTypeId.HELLION).exists:
                    our_hp_left = self.units(UnitTypeId.HELLION).first.health_percentage*3 + 1 # 1-4
                self.reward = 2 - len(self.enemy_units(UnitTypeId.ZERGLING)) + our_hp_left  # fin reward
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
                place = np.argwhere(self.action == 1)[0]
                if place[0] == 5 and place[1] == 5:
                    unit.attack(unit.position)
                else:
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
                    x = int(enemy_unit.position.x - unit.position.x)
                    y = int(enemy_unit.position.y - unit.position.y)
                    if 10 >= x >= -10 and 10 >= y >= -10:
                        self.current_state[x+10][y+10] = 1
                self.current_state[10, 10] = cd
                for i in range(21):
                    for j in range(21):
                        coord1 = i - 10 + unit.position.x
                        coord2 = j - 10 + unit.position.y
                        if coord1 < 2 or coord1 > 46 or coord2 < 2 or coord2 > 38:
                            self.current_state[i, j] = 2
                        elif self.game_info.terrain_height.__getitem__((int(coord1), int(coord2))) > 210:
                            # print(self.game_info.terrain_height.__getitem__((round(coord1), round(coord2))))
                            self.current_state[i, j] = 2
                summm = np.sum(self.current_state[3:8, 3:8])
                if summm > 4:
                    self.reward -= 0.5
                self.reward += ((current_our_hp - self.remember_our_hp)/1.5 + (current_enemy_hp - self.remember_enemy_hp)/2)/5  # short term reward
                # distnt = unit.distance_to(enemy_units.center)
                # self.reward = 0
                # if distnt < 5:
                #     self.reward = distnt/5

            if self.done:  # reset game
                self.agent.n_games += 1
                self.big_score += self.score
                if self.agent.n_games % 10 == 0:
                    self.agent.train_long_memory()
                    if self.big_score > self.record:
                        self.record = self.big_score
                        self.agent.model.save()
                    self.plot_scores.append(self.big_score/10)
                    self.plot_records.append(self.record/10)
                    print(self.big_score, self.record)
                    self.big_score = 0
                x_coord = np.linspace(0, self.agent.n_games//10*10, self.agent.n_games//10)
                self.axs[0].cla()
                self.axs[1].cla()
                self.axs[0].plot(x_coord, self.plot_scores)
                self.axs[1].plot(x_coord, self.plot_records)
                plt.pause(0.001)
                await self.chat_send('end')
                self.done = False
                self.first_iter_of_round = True
                self.reward = 0
                


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