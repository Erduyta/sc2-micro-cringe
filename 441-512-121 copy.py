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
}


class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(169, 512, 13)
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
    current_state = np.zeros([13, 13])
    if not units:
        return current_state
    unit = units.first
    enemy_units = botai.enemy_units(UnitTypeId.ZERGLING)
    for enemy_unit in enemy_units:
        x = int(enemy_unit.position.x - unit.position.x)
        y = int(enemy_unit.position.y - unit.position.y)
        if 6 >= x >= -6 and 6 >= y >= -6:
            current_state[x + 6][y + 6] = 1
    cd = unit.weapon_cooldown
    current_state[6, 6] = cd
    for i in range(13):
        for j in range(13):
            coord1 = i - 6 + unit.position.x
            coord2 = j - 6 + unit.position.y
            if coord1 < 2 or coord1 > 46 or coord2 < 2 or coord2 > 38:
                current_state[i, j] = 2
            elif botai.game_info.terrain_height[(int(coord1), int(coord2))] > 210:
                # print(self.game_info.terrain_height.__getitem__((round(coord1), round(coord2))))
                current_state[i, j] = 2
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

    def short_term_reward(self, start_tick: int, other: RememberedState):
        reward = 0
        if self.done_reward:
            return self.done_reward

        # punish bot for staying close to walls
        summ = np.sum(self.raw_state[5:8, 5:8]) - self.raw_state[6, 6]
        if summ > 3:
            reward -= 1
        # punish bot for losing health and encourage attacking enemies
        reward += (self.agent_hp - other.agent_hp) * 2 + (
                other.enemy_hp - self.enemy_hp
        )

        reward += (self.tick - other.tick) * (self.tick - start_tick) / 1000

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
            our_hp_left = -10  # 0
            if units:
                our_hp_left = our_hp * 10  # 4-7
            done_reward = our_hp_left

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
        self.current_state = np.zeros([13, 13])  # actual state
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
            if self.big_score > self.record:
                self.record = self.big_score
                self.agent.model.save()
            self.plot_scores.append(self.big_score / 10)
            self.plot_records.append(self.record / 10)
            print(self.big_score / 10, self.record / 10)
            self.big_score = 0
        self.score = 0
        x_coord = np.linspace(0, self.agent.n_games // 10 * 10 - 10, self.agent.n_games // 10)
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[0].plot(x_coord, self.plot_scores)
        self.axs[1].plot(x_coord, self.plot_records)
        plt.pause(0.001)
        await self.chat_send('end')
        self.round_cleanup()

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
                self.agent.train_short_memory(self.state_old.state, self.action, self.reward, self.state_new.state, self.done)  # stm
                self.agent.remember(self.state_old.state, self.action, self.reward, self.state_new.state, self.done)  # remember
                self.reward = self.state_new.short_term_reward(self.start_tick, self.state_old)

            self.state_old = RememberedState.create_from_agent(
                iteration,
                self
            )
            self.score += self.reward
            logging.info(f"Reward: {self.reward}")
            logging.info(f"Score: {self.score}")

            if not self.state_new.done_reward:
                self.action = self.agent.get_action(self.state_old.state)
                unit = self.units(UnitTypeId.HELLION).first

                place = np.argwhere(self.action == 1)[0]
                if place[0] == 12:
                    self.skip_iterations = 4
                    unit.attack(unit.position)  # TODO aa -> move
                else:
                    coords_delta = dict_fo_moves[place[0]]
                    move_to = Point2((coords_delta[0] + unit.position.x, coords_delta[1] + unit.position.y))
                    unit.move(move_to)
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