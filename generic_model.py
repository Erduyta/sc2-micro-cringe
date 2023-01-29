from abc import ABC, abstractmethod

import numpy


class GenericModel(ABC):
    @abstractmethod
    def consume_memory(self, old_state: numpy.ndarray, action: numpy.ndarray, reward: float,
                       new_state: numpy.ndarray) -> None:
        pass

    @abstractmethod
    def decide_action(self, state: numpy.ndarray) -> numpy.ndarray:
        pass
