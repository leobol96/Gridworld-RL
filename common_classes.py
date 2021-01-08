from enum import Enum


class Cell(object):
    """
    This class represents the core of the Gridworld. Each world is made of n X n cells.
    """
    class Action(Enum):
        """
        Enumeration for the possible actions
        """
        UP = 'up'
        DOWN = 'down'
        LEFT = 'left'
        RIGHT = 'right'

    def __init__(self, reward: float, row, col, is_wall=False, q_a=None, v_pi=0, terminal=False):
        """
        Constructor method of the cell class
        Args:
            reward: Reward of the state
            row: row of the object in the world matrix
            col: column of the object in the world matrix
            is_wall: True if the cell is a wall. This is optional the default is false.
            q_a: Python dictionary represents the actions and their values. They are set to 0 as default.
            v_pi: Integer that represent the v values of the state. As default is set to 0.
            terminal: True if the state is a terminal.
        """
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
        """
        Numeric hash
        Returns: return the col + row of the state
        """
        return int(str(self.col) + str(self.row))

    def string_hash(self):
        """
        String hash
        Returns: return the col + row of the state
        """
        return str(self.col) + str(self.row)


class Position(object):
    """
    This class represents the position of an object in the world.
    It has two variables col that is the column value, and row that is the row value
    """
    def __init__(self, col, row):
        self.col = col
        self.row = row


