
"""
    Preserves replay buffer, evaluates model actions, maps ingame state to input
"""
from collections import deque
from typing import Optional

import numpy
import numpy as np
from s2clientprotocol.common_pb2 import Point2D
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2

from empty_model import EmptyModel
from generic_model import GenericModel

MAX_MEMORY = 100_000


class RememberedState:
    def __init__(self, state, agent_hp, enemy_hp, tick, done_reward):
        # input state
        self.state: numpy.ndarray = state

        self.agent_hp: int = agent_hp
        self.enemy_hp: int = enemy_hp

        self.tick: int = tick

        # when defined means the round is over
        self.done_reward: Optional[float] = done_reward

    def calculate_reward(self, other: 'RememberedState', action: Optional[numpy.ndarray]):
        reward = 0
        if self.done_reward:
            return self.done_reward
        # punish bot for losing health and encourage attacking enemies

        reward += (self.agent_hp - other.agent_hp) * 5

        if other.enemy_hp - self.enemy_hp:
            reward += (other.enemy_hp - self.enemy_hp) * 2
        reward += (other.enemy_hp - self.enemy_hp)
        reward += 0.1

        return reward


class StarcraftFrameWork(BotAI):
    def __init__(self, model: GenericModel) -> None:
        self.old_state: Optional[RememberedState] = None
        self.last_action: Optional[numpy.ndarray] = None
        self.model: GenericModel = model
        self.skip_iterations: int = 1

    def extract_state(self) -> numpy.ndarray:
        units = self.units(UnitTypeId.HELLION)
        current_state = np.zeros([25, 25])
        if not units:
            return current_state
        unit = units.first
        enemy_units = self.enemy_units(UnitTypeId.ZERGLING)
        for enemy_unit in enemy_units:
            x = round((enemy_unit.position.x - unit.position.x) * 2)
            y = round((enemy_unit.position.y - unit.position.y) * 2)
            if 12 >= x >= -12 and 12 >= y >= -12:
                current_state[x + 12][y + 12] = 1
        cd = unit.weapon_cooldown / 10
        current_state[12, 12] = cd
        for i in range(25):
            for j in range(25):
                coord1 = (i - 12) / 2 + unit.position.x
                coord2 = (j - 12) / 2 + unit.position.y
                if coord1 < 2 or coord1 > 45 or coord2 < 2 or coord2 > 37:
                    current_state[i, j] = -1
                elif self.game_info.terrain_height[(round(coord1), round(coord2))] > 210:
                    current_state[i, j] = -1
        return current_state

    def create_from_agent(self, tick):
        state = self.extract_state()
        units = self.units(UnitTypeId.HELLION)
        if units:
            unit = self.units(UnitTypeId.HELLION).first
            our_hp = unit.health_percentage
        else:
            our_hp = 0
        enemy_units = self.enemy_units(UnitTypeId.ZERGLING)
        done_reward: Optional[float] = None
        if not units or not enemy_units.exists:
            our_hp_left = 0  # 0
            if units:
                our_hp_left = our_hp * 10  # 4-7
            done_reward = our_hp_left + 20 - 10 * len(enemy_units)

        return RememberedState(
            state=state.flatten(),
            agent_hp=our_hp,
            enemy_hp=sum(map(lambda e: e.health_percentage, enemy_units)),
            tick=tick,
            done_reward=done_reward,
        )

    async def reset_game(self):
        self.old_state = None
        self.last_action = None
        await self.chat_send('end')
        self.skip_iterations = 1

    def act(self, action: numpy.ndarray):
        units = self.units(UnitTypeId.HELLION)
        if units.exists:
            unit = units.first
            # [0..1)
            x = action[0] * 2 - 1
            y = action[1] * 2 - 1
            unit.move(Point2((unit.position.x + x, unit.position.y + y)))

    async def on_step(self, iteration: int):
        if self.skip_iterations:
            self.skip_iterations -= 1
            return

        state = self.create_from_agent(iteration)
        if self.old_state is not None:
            self.model.consume_memory(
                self.old_state.state,
                self.last_action,
                state.calculate_reward(self.old_state, self.last_action),
                state.state
            )

        if state.done_reward is not None:
            await self.reset_game()

        self.last_action = self.model.decide_action(state.state)
        self.old_state = state
        self.act(self.last_action)


def main():
    run_game(
        maps.get("ai-bot-erduyta"),
        [Bot(Race.Terran, StarcraftFrameWork(EmptyModel())), Computer(Race.Zerg, Difficulty.Medium)],
        realtime=False,
        save_replay_as="WorkerRush.SC2Replay",
    )


if __name__ == "__main__":
    main()