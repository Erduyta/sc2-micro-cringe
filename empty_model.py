import numpy

from generic_model import GenericModel


class EmptyModel(GenericModel):
    def consume_memory(self, old_state: numpy.ndarray, action: numpy.ndarray, reward: float,
                       new_state: numpy.ndarray) -> None:
        pass

    def decide_action(self, state: numpy.ndarray) -> numpy.ndarray:
        return numpy.array([0.5, 0.5])
