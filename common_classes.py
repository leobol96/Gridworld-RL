from enum import Enum


class Cell(object):

    class Action(Enum):
        UP = 'up'
        DOWN = 'down'
        LEFT = 'left'
        RIGHT = 'right'

    def __init__(self, reward: float, row, col, is_wall=False, q_a=None, v_pi=0, terminal=False):
        if q_a is None:
            q_a = {self.Action.UP.value: 0, self.Action.DOWN.value: 0, self.Action.LEFT.value: 0,
                   self.Action.RIGHT.value: 0}

        self.reward = reward
        self.terminal = terminal
        self.q_a = q_a
        self.v_pi = v_pi
        self.wall = is_wall
        self.row = row
        self.col = col

    def __hash__(self):
        return int(str(self.col) + str(self.row))


class Position(object):

    def __init__(self, col, row):
        self.col = col
        self.row = row


