import torch
from torch import optim, nn


class SCAdam:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done: bool):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        q_new = reward[0]

        # # this doesn't work for continuous spaces
        pass

        # if not done:4
        #     Q_new = reward[idx] + self.gamma * torch.max(
        #         self.model(next_state[idx]) # input_state ==> [ Q(input_state, a1), Q(input_state, a2), ... ]
        #         # output_state
        #     )
        #
        # target[idx][torch.argmax(action[idx]).item()] = Q_new
        #
        # # 2: Q_new(s, a) = r + y * max(next_predicted Q value) -> only do this if not done
        # self.optimizer.zero_grad()
        # loss = self.criterion(target, pred)
        # loss.backward()
        #
        # self.optimizer.step()