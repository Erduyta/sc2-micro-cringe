import numpy as np
import random
import sc2


class Field:
    def __init__(self) -> None:
        self.field = np.zeros([3, 3])
        self.done = False
        self.reward = 0
        self.score = 0

    def reset(self):
        self.field = np.zeros([3, 3])
        self.done = False
        self.reward = 0
        self.score = 0

    def game_opponent_step_result(self):
        score = 0
        if self.done:
            score = -self.score
        return score

    def rand_play(self, player):
        possibilities = np.argwhere(self.field == 0)
        #print(possibilities)
        inidex = random.randint(0, len(possibilities)-1)
        move = possibilities[inidex]
        #print(move)
        self.field[move[0], move[1]] = 2
        self.done = False
        if self.check_end():
            self.done = True
            self.score = 0
        if self.check_win():
            self.done = True
            if player + 1 == self.check_win():
                self.reward = 1
                self.score = 1
            else:
                self.score = -1
                self.reward = -1

    def game_step(self, player, action):
        """player = 0,1  action - [3, 3]"""
        place = np.argwhere(action == 1)[0]
        #print(place)
        #print(self.field[place])
        self.reward = 0
        if self.field[place[0], place[1]] == 0:
            self.field[place[0], place[1]] = player + 1
        else:
            self.reward = -2
            self.done = True
            return self.reward, self.done, -1
        self.done = False
        if self.check_end():
            self.done = True
            self.score = 0
        if self.check_win():
            self.done = True
            if player + 1 == self.check_win():
                self.reward = 1
                self.score = 1
            else:
                self.score = -1
                self.reward = -1
        if not self.done:
            self.rand_play(player)
        return self.reward, self.done, self.score            

    def check_win(self):
        """zero = no, else returns player 1 or 2"""
        for i in range(3):
            if self.field[i, 0] == self.field[i, 1] == self.field[i, 2] and self.field[i, 0] != 0:
                return self.field[i, 0]
        for i in range(3):
            if self.field[0, i] == self.field[1, i] == self.field[2, i] and self.field[1, i] != 0:
                return self.field[0, i]
        if self.field[0, 0] == self.field[1, 1] == self.field[2, 2] and self.field[1, 1] != 0:
            return self.field[0, 0]
        if self.field[2, 0] == self.field[1, 1] == self.field[0, 2] and self.field[1, 1] != 0:
            return self.field[1, 1]
        return 0

    def check_end(self):
        if len(np.where(self.field == 0)[0]) == 0:
            return True
        return False


if __name__ == '__main__':
    in_game = Field()
    while True:
        x = int(input())
        y = int(input())
        final_move = np.zeros([3, 3])
        final_move[x, y] = 1
        print(in_game.game_step(0, final_move))
        print(in_game.field)
