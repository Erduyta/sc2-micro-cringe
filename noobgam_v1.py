import numpy
import torch
from torch import nn
import torch.nn.functional as F

from generic_model import GenericModel
from model import QTrainer


class NoobgamV1Internal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class NoobgamV1(GenericModel):

    def __init__(self):
        self.model = NoobgamV1Internal(625, 1024, 2)
        self.trainer = (self.model, 0.001, 0.9)

    def consume_memory(self,
                       old_state: numpy.ndarray,
                       action: numpy.ndarray,
                       reward: float,
                       new_state: numpy.ndarray,

                       ) -> None:
        pass

    def decide_action(self, state: numpy.ndarray) -> numpy.ndarray:
        return self.model(state)
