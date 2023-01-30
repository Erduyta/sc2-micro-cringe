# action_space_n -> length of action space
# reset -> state
# action_space_sample -> action space (i) random
# step(action_i) -> state, reward, terminated, truncated

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
from bot_main import Agent
from sc2.position import Point2

logging.basicConfig()
logging.root.setLevel(logging.INFO)
move_diction = {
    0: Point2((0, 1)),
    1: Point2((0, -1)),
    2: Point2((1, 0)),
    3: Point2((-1, 0)),
}

class ErduytaTerran(BotAI):
    def __init__(self) -> None:
        super().__init__()
        self.action_space_n = 5
        self.observation_space_n = 169  # [13, 13]
        self.state_bot = np.zeros([170])  # TODO init state 13x13 + 1 (cd)
        self.agent = None
        self.life_exp = 0
        self.health_remeber = 1
        self.reward = 0
        self.truncated = False
        self.terminated = False
        self.framerate_drop = 4  # for aa at least

    def action_space_sample(self):
        num_i = random.randint(0, self.action_space_n-1)
        return num_i

    async def reset_game(self):
        self.state_bot = np.zeros([170])  # TODO init state
        self.reward = 0
        self.life_exp = 0
        self.health_remeber = 1
        self.truncated = False
        self.terminated = False
        # reset reward, reset score, life expectance
        await self.chat_send('end')

    # TODO update_state
    async def update_state(self, iteration):
        if iteration == 0:
            self.state_bot = np.zeros([170]).flatten()
            return None
        # create state
        units = self.units(UnitTypeId.HELLION)
        vision = np.zeros([13, 13])
        if not units:
            self.state_bot = np.zeros([170]).flatten()
            return None
        unit = units.first
        enemy_units = self.enemy_units(UnitTypeId.ZERGLING)
        for enemy_unit in enemy_units:
            x = round(enemy_unit.position.x - unit.position.x)
            y = round(enemy_unit.position.y - unit.position.y)
            if 6 >= x >= -6 and 6 >= y >= -6:
                vision[x + 6][y + 6] = 1
        for i in range(13):
            for j in range(13):
                coord1 = (i - 6) + unit.position.x
                coord2 = (j - 6) + unit.position.y
                if coord1 < 2 or coord1 > 45 or coord2 < 2 or coord2 > 37:
                    vision[i, j] = -1
                elif self.game_info.terrain_height[(round(coord1), round(coord2))] > 210:
                    # print(self.game_info.terrain_height.__getitem__((round(coord1), round(coord2))))
                    vision[i, j] = -1
        cd = unit.weapon_cooldown/10
        self.state_bot = np.append(vision.flatten(), cd)
        return None

    async def on_step(self, iteration):
        await self.update_state(iteration=iteration)
        if iteration == 0:
            self.agent = Agent()
            return None
            # start main function
            # split main function?
        self.agent.main_ich(self)
        if self.agent.runstep == 1:
            self.agent.reset = False
            await self.reset_game()
            return None
        if self.agent.runstep == 3:
            # make action
            # reward = +0.1/step -2/clickwall -1/health +10/lived enough
            self.reward = 0.1
            self.life_exp += 1
            units = self.units(UnitTypeId.HELLION)
            if not units.exists:
                self.terminated = True
                self.reward -= 1  # death
                return None  # state gets later, reward, termnated=1, truncated=0
            unit = units.first
            # lost hp
            self.reward -= (self.health_remeber - unit.health_percentage)*10
            self.health_remeber = unit.health_percentage
            if self.agent.next_action == 4:
                unit.attack(unit.position)
            else:
                unit.move(unit.position.offset(move_diction[self.agent.next_action]))
            if self.life_exp > 400:
                self.truncated = True
                self.reward += 10
            return None


def main():
    run_game(
        maps.get("ai-bot-erduyta"),
        [Bot(Race.Terran, ErduytaTerran()), Computer(Race.Zerg, Difficulty.Medium)],
        realtime=False,
        save_replay_as="WorkerRush.SC2Replay",
    )


if __name__ == "__main__":
    main()
