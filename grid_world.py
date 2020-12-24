import random

from box import Box
import matplotlib.pyplot as plt
import common_functions

from position import Position


class GridWord(object):
    """
    This class represent the gridworld.
    """

    def __init__(self, name, height, width, r_nt=0):

        self.name = name
        self.episode = 1
        self.step = 1
        self.rewards_for_step = []
        self.rewards_for_episode = []
        self.step_for_episode = []
        self.current_position = Position(0, 0)
        self.world = []
        for col in range(width):
            tmp = []
            for row in range(height):
                # Define a Box and for all the box remove the actions that bring the agents outside of the world
                box = Box(r_nt)
                if row == 0:
                    box.q_a.pop(Box.Action.LEFT.value)
                if row == width - 1:
                    box.q_a.pop(Box.Action.RIGHT.value)
                if col == 0:
                    box.q_a.pop(Box.Action.UP.value)
                if col == height - 1:
                    box.q_a.pop(Box.Action.DOWN.value)
                tmp.append(box)
            self.world.append(tmp)

    def set_terminal_state(self, row: int, col: int, reward: float) -> None:
        """
        This method is used to set terminal states inside the GridWorld.
        Args:
            row: Row of the terminal state
            col: Column of the terminal state
            reward: Reward getting arriving in that terminal state
        """
        self.world[row][col].reward = reward
        self.world[row][col].terminal = True
        self.world[row][col].wall = False

    def get_current_state(self):
        return self.world[self.current_position.col][self.current_position.row]

    def set_wall(self, walls: list) -> None:
        """
        Method used to set the walls inside the gridworld.
        Args:
            walls: List containing positions (x,y)
        """
        for wall in walls:
            self.world[wall[0]][wall[1]].wall = True

    def action_e_greedy(self, epsilon) -> str:
        """
        This method select the next action following the E-greedy paradigm
        Args:
            epsilon: Epsilon to use in the e-greedy function
        Returns: Action to take
        """
        epsilon = epsilon * 100
        q_current_state = self.world[self.current_position.col][self.current_position.row].q_a
        possible_action = [*q_current_state]
        value = random.choices(['random', 'greedy'], weights=[epsilon, 100 - epsilon], k=1)

        # Remove all the actions were a wall is found
        for action in possible_action[:]:
            if self.action_against_wall(action):
                possible_action.remove(action)

        # Choose greedy between the possible actions
        if 'greedy' in value:
            # take maximum
            action = self.world[self.current_position.col][self.current_position.row].get_max_q(value_type='action')
            if action not in possible_action: action = random.choice(possible_action)
        else:
            action = random.choice(possible_action)
        return action

    def get_next_state(self, action):
        """
        This method return the next position of the agent given a action to take
        Args:
            action: Action to take
        Returns: Position of the next state
        """
        col = self.current_position.col
        row = self.current_position.row

        if action == Box.Action.DOWN.value:
            col = col + 1
        elif action == Box.Action.UP.value:
            col = col - 1
        elif action == Box.Action.RIGHT.value:
            row = row + 1
        elif action == Box.Action.LEFT.value:
            row = row - 1
        return [col, row]

    def action_against_wall(self, action) -> bool:
        """
        This method return if the next block is a wall
        Args:
            action: Action to take
        Returns: True if the next block in the world is a wall
        """
        col, row = self.get_next_state(action=action)
        if self.world[col][row].wall:
            return True

    def update_sarsa(self, s, s_first, action, action_first, alpha, discount_factor):
        s.q_a[action] = s.q_a[action] + alpha * (
                    s_first.reward + discount_factor * (s_first.q_a[action_first]) - s.q_a[action])
        self.rewards_for_step.append(s_first.reward)
        self.step += 1

    def update_q_learning(self, action, alpha, discount_factor):
        """
        Method to update the Q_values using the Q_learning Algorithm
        Args:
            action: action took from the Epsilon greedy method
            alpha: Alpha constant
            discount_factor: Discount factor
        """
        # Update Q
        col, row = self.get_next_state(action=action)
        self.world[self.current_position.col][self.current_position.row].q_a[action] = \
            self.world[self.current_position.col][self.current_position.row].q_a[action] + alpha * (
                    self.world[col][row].reward + (
                    discount_factor * self.world[col][row].get_max_q(value_type='value')) -
                    self.world[
                        self.current_position.col][
                        self.current_position.row].q_a[action])
        self.rewards_for_step.append(self.world[col][row].reward)
        self.step += 1

    def update_state_q_learning(self, action):
        """
        This method update the state of the agent
        Args:
            action: action took from the agent
        """
        # Update current state
        col, row = self.get_next_state(action=action)
        if self.world[col][row].terminal:
            self.current_position.col = 0
            self.current_position.row = 0
            sum_reward = sum(self.rewards_for_step)
            self.rewards_for_episode.append(sum_reward)
            self.step_for_episode.append(self.step)
            print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
            self.rewards_for_step = []
            self.step = 0
            self.episode = self.episode + 1
        else:
            self.current_position.col = col
            self.current_position.row = row

    def restart_episode(self):
        self.current_position.col = 0
        self.current_position.row = 0
        sum_reward = sum(self.rewards_for_step)
        self.rewards_for_episode.append(sum_reward)
        self.step_for_episode.append(self.step)
        print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
        self.rewards_for_step = []
        self.step = 0
        self.episode = self.episode + 1


if __name__ == '__main__':
    n_episode = 100
    epsilon = 0.2
    alpha = 0.60
    discount_episode = 1

    gridworld = GridWord(name='Q-Learning', height=9, width=9, r_nt=-1)
    gridworld.set_terminal_state(row=8, col=8, reward=50)
    gridworld.set_terminal_state(row=6, col=6, reward=-50)
    gridworld.set_wall([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                        [2, 6], [3, 6], [4, 6], [5, 6], [2, 6],
                        [7, 1], [7, 2], [7, 3], [7, 4]])
    while gridworld.episode <= n_episode:
        action = gridworld.action_e_greedy(epsilon)
        gridworld.update_q_learning(action=action, alpha=alpha, discount_factor=discount_episode)
        gridworld.update_state_q_learning(action)

    gridworld2 = GridWord(name='SARSA', height=9, width=9, r_nt=-1)
    gridworld2.set_terminal_state(row=8, col=8, reward=50)
    gridworld2.set_terminal_state(row=6, col=6, reward=-50)
    gridworld2.set_wall([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                         [2, 6], [3, 6], [4, 6], [5, 6], [2, 6],
                         [7, 1], [7, 2], [7, 3], [7, 4]])

    s = gridworld2.get_current_state()
    action = gridworld2.action_e_greedy(epsilon)
    while gridworld2.episode <= n_episode:
        gridworld2.current_position.col, gridworld2.current_position.row = gridworld2.get_next_state(action)
        s_first = gridworld2.get_current_state()
        action_first = gridworld2.action_e_greedy(epsilon)
        gridworld2.update_sarsa(s=s, s_first=s_first, action=action, action_first=action_first, alpha=alpha,
                                discount_factor=discount_episode)
        action = action_first
        s = s_first

        if s.terminal:
            gridworld2.restart_episode()
            s = gridworld2.get_current_state()
            action = gridworld2.action_e_greedy(epsilon)

    common_functions.plot_world(worlds=[gridworld, gridworld2], variable='q_a')
    common_functions.plot_total_reward_step(world_qlearning=gridworld, world_sarsa=gridworld2)
