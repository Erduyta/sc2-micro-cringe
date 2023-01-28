from __future__ import annotations

import logging
from typing import Optional
import numpy
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

logging.basicConfig()
logging.root.setLevel(logging.INFO)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0008


dict_fo_moves = {
    0: (-2, -1),
    1: (-2, 0),
    2: (-2, 1),
    3: (-1, 2),
    4: (0, 2),
    5: (1, 2),
    6: (-1, -2),
    7: (0, -2),
    8: (1, -2),
    9: (2, -1),
    10: (2, 0),
    11: (2, 1),
    12: (0, 0),
}


class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(625, 1024, 13)
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
        # for state, action, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(
            self,
            state: numpy.ndarray,
            action: numpy.ndarray,
            reward: float,
            next_state: numpy.ndarray,
            done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: numpy.ndarray):
        # random moves: exploration/expoitation
        self.epsilon = 201 - self.n_games
        final_move = np.zeros([13])
        if random.randint(0, 200) < self.epsilon:
            move1 = random.randint(0, 12)
            final_move[move1] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_torch = torch.argmax(prediction).item()
            final_move[move_torch] = 1
        return final_move


def get_state_from_botai(botai: BotAI) -> numpy.ndarray:
    # throws if hellion is not found
    units = botai.units(UnitTypeId.HELLION)
    current_state = np.zeros([25, 25])
    if not units:
        return current_state
    unit = units.first
    enemy_units = botai.enemy_units(UnitTypeId.ZERGLING)
    for enemy_unit in enemy_units:
        x = round((enemy_unit.position.x - unit.position.x)*2)
        y = round((enemy_unit.position.y - unit.position.y)*2)
        if 12 >= x >= -12 and 12 >= y >= -12:
            current_state[x + 12][y + 12] = 1
    cd = unit.weapon_cooldown/10
    current_state[12, 12] = cd
    for i in range(25):
        for j in range(25):
            coord1 = (i - 12)/2 + unit.position.x
            coord2 = (j - 12)/2 + unit.position.y
            if coord1 < 2 or coord1 > 45 or coord2 < 2 or coord2 > 37:
                current_state[i, j] = -1
            elif botai.game_info.terrain_height[(round(coord1), round(coord2))] > 210:
                # print(self.game_info.terrain_height.__getitem__((round(coord1), round(coord2))))
                current_state[i, j] = -1
    # plt.imshow(current_state, vmin=-1, vmax=4)
    # plt.pause(0.001)
    return current_state


class RememberedState:
    def __init__(self, raw_state, state, agent_hp, enemy_hp, tick, done_reward):
        # input state
        self.raw_state: numpy.ndarray = raw_state
        self.state: numpy.ndarray = state

        self.agent_hp: int = agent_hp
        self.enemy_hp: int = enemy_hp

        self.tick: int = tick

        # when defined means the round is over
        self.done_reward: Optional[float] = done_reward

    def short_term_reward(self, botai: TerranNoobgam):
        # self.start_tick, self.state_old, self.action
        reward = 0
        if self.done_reward:
            return self.done_reward
        # punish bot for moving into a wall
        unit = botai.units(UnitTypeId.HELLION)
        if unit.exists:
            unit = unit.first
            place = np.argwhere(botai.action == 1)[0]
            if place != 12:
                coords_delta = dict_fo_moves[place[0]]
                move_to = (int(coords_delta[0] + unit.position.x), int(coords_delta[1] + unit.position.y))
                if botai.game_info.terrain_height[move_to] > 210:
                    reward -= 10
            elif botai.state_old.raw_state[12, 12] > 0:
                reward -= 10
        # punish bot for losing health and encourage attacking enemies

        reward += (self.agent_hp - botai.state_old.agent_hp) * 5
        if botai.state_old.enemy_hp - self.enemy_hp:
            reward += (botai.state_old.enemy_hp - self.enemy_hp)*2
        reward += (botai.state_old.enemy_hp - self.enemy_hp)

        # reward += (self.tick - other.tick) * (self.tick - start_tick) / 1000

        return reward

    @staticmethod
    def create_from_agent(tick, botai: BotAI):
        state = get_state_from_botai(botai)
        units = botai.units(UnitTypeId.HELLION)
        if units:
            unit = botai.units(UnitTypeId.HELLION).first
            our_hp = unit.health_percentage
        else:
            our_hp = 0
        enemy_units = botai.enemy_units(UnitTypeId.ZERGLING)
        done_reward: Optional[float] = None
        if not units or not enemy_units.exists:
            our_hp_left = 0  # 0
            if units:
                our_hp_left = our_hp * 10  # 4-7
            done_reward = our_hp_left + 20 - 10*len(enemy_units)

        return RememberedState(
            raw_state=state,
            state=state.flatten(),
            agent_hp=our_hp,
            enemy_hp=sum(map(lambda e: e.health_percentage, enemy_units)),
            tick=tick,
            done_reward=done_reward,
        )


class TerranNoobgam(BotAI):
    def __init__(self) -> None:
        super().__init__()
        self.current_state = np.zeros([25, 25])  # actual state
        self.action = np.zeros([13])
        self.reward = 0
        self.done = False
        self.agent = Agent()
        self.state_old: Optional[RememberedState] = None  # remeber between cycles
        self.state_new: Optional[RememberedState] = None  # remeber between cycles
        self.score = 0
        self.record = -1000
        self.fig, self.axs = plt.subplots(2)
        self.plot_scores = []
        self.plot_records = []
        self.big_score = 0
        self.start_tick = 0

        self.skip_iterations = 5

    def round_cleanup(self):
        self.skip_iterations = 0
        self.done = False
        self.reward = 0
        self.state_old = None
        self.state_new = None

    async def reset_game(self):
        self.agent.n_games += 1
        # print(iteration)
        self.big_score += self.score
        if self.agent.n_games % 10 == 0:
            self.agent.train_long_memory()
            if self.big_score >= self.record:
                self.record = self.big_score
                self.agent.model.save()
            self.plot_scores.append(self.big_score / 10)
            self.plot_records.append(self.record / 10)
            print(self.big_score / 10, self.record / 10)
            self.big_score = 0
        self.score = 0
        await self.chat_send('end')
        self.round_cleanup()
        await self.update_graphs()

    async def update_graphs(self):
        x_coord = np.linspace(0, self.agent.n_games // 10 * 10 - 10, self.agent.n_games // 10)
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[0].plot(x_coord, self.plot_scores)
        self.axs[1].plot(x_coord, self.plot_records)
        plt.pause(0.001)

    async def on_step(self, iteration):
        if self.skip_iterations:
            self.skip_iterations -= 1
            return
        # plt.imshow(self.current_state)
        # plt.pause(0.001)
        if iteration != 0:

            self.state_new = RememberedState.create_from_agent(
                iteration,
                self
            )
            # skip on first iteration of the round
            if not self.state_old:
                self.start_tick = iteration
            else:
                self.reward = self.state_new.short_term_reward(self)
                self.agent.train_short_memory(self.state_old.state, self.action, self.reward, self.state_new.state, self.done)  # stm
                self.agent.remember(self.state_old.state, self.action, self.reward, self.state_new.state, self.done)  # remember


            self.state_old = RememberedState.create_from_agent(
                iteration,
                self
            )
            self.score += self.reward
            logging.info(f"Reward: {self.reward}, Score: {self.score}")

            if self.state_new.done_reward is None:
                self.action = self.agent.get_action(self.state_old.state)
                unit = self.units(UnitTypeId.HELLION).first

                place = np.argwhere(self.action == 1)[0]
                if place[0] == 12:
                    self.skip_iterations = 4
                    unit.attack(unit.position)  # TODO aa -> move
                    logging.info(f'attacked clicked on {unit.position}')
                else:
                    coords_delta = dict_fo_moves[place[0]]
                    move_to = Point2((coords_delta[0] + unit.position.x, coords_delta[1] + unit.position.y))
                    unit.move(move_to)
                    logging.info(f'moved to {move_to.x} {move_to.y}, from {unit.position.x} {unit.position.y}')
            else:
                await self.reset_game()


def main():
    run_game(
        maps.get("ai-bot-erduyta"),
        [Bot(Race.Terran, TerranNoobgam()), Computer(Race.Zerg, Difficulty.Medium)],
        realtime=False,
        save_replay_as="WorkerRush.SC2Replay",
    )


if __name__ == "__main__":
    main()