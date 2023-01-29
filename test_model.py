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
import os
from model import Linear_QNet, QTrainer
from .learning import RememberedState, Agent, get_state_from_botai


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


class Agent2(Agent):

    def __init__(self) -> None:
        super().__init__()
        model_folder_path = './model'
        file_name = 'model_good.pth'
        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()


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
        await self.chat_send('end')
        self.round_cleanup()

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

            self.state_old = RememberedState.create_from_agent(
                iteration,
                self
            )
            self.score += self.reward
            logging.info(f"Reward: {self.reward}, Score: {self.score}")

            if self.state_old.done_reward is None:
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
        realtime=True,
        save_replay_as="WorkerRush.SC2Replay",
    )


if __name__ == "__main__":
    main()