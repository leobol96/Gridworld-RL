from enum import Enum


class Box(object):

    class Action(Enum):
        UP = 'up'
        DOWN = 'down'
        LEFT = 'left'
        RIGHT = 'right'

    def __init__(self, reward: float, is_wall=False, q_a=None, v_pi=0, terminal=False):
        if q_a is None:
            q_a = {self.Action.UP.value: 0, self.Action.DOWN.value: 0, self.Action.LEFT.value: 0,
                   self.Action.RIGHT.value: 0}

        self.reward = reward
        self.terminal = terminal
        self.q_a = q_a
        self.v_pi = v_pi
        self.wall = is_wall
