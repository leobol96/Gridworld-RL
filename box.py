from enum import Enum


class Box(object):

    class Direction(Enum):
        UP = 'up'
        DOWN = 'down'
        LEFT = 'left'
        RIGHT = 'right'

    def __init__(self, reward: float, is_wall=False, q_a=None, v_pi=0, terminal=False):
        if q_a is None:
            q_a = {self.Direction.UP.value: 0, self.Direction.DOWN.value: 0, self.Direction.LEFT.value: 0,
                   self.Direction.RIGHT.value: 0}

        self.reward = reward
        self.terminal = terminal
        self.q_a = q_a
        self.v_pi = v_pi
        self.wall = is_wall

    def get_max_q(self, value_type):
        max_value = None
        for possible_direction in [*self.q_a]:
            if max_value is None:
                max_value = self.q_a[possible_direction]
                max_q = possible_direction
            else:
                if self.q_a[possible_direction] > max_value:
                    max_value = self.q_a[possible_direction]
                    max_q = possible_direction
        if value_type == 'action':
            return max_q
        else:
            return max_value
