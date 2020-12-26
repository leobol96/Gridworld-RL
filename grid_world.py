import random

import common_functions
from box import Box
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
        """
        Get the current state in world
        Returns: Current state
        """
        return self.world[self.current_position.col][self.current_position.row]

    def set_wall(self, walls: list) -> None:
        """
        Method used to set the walls inside the gridworld.
        Args:
            walls: List containing positions (x,y)
        """
        for wall in walls:
            self.world[wall[0]][wall[1]].wall = True

    def action_e_greedy(self, current_state, epsilon) -> str:
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
        for a in possible_action[:]:
            if self.action_against_wall(a):
                possible_action.remove(a)

        # Choose greedy between the possible actions
        if 'greedy' in value:
            return self.get_max_q(current_state=current_state, value_type='action')
        else:
            return random.choice(possible_action)

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

    def get_max_q(self, current_state, value_type):
        """
        Return the maximum value q for the state s
        Args:
            current_state: actual state in the world
            value_type: VALUE || ACTION. With value on will get the value of q(a).
            Otherwise it will get the Action corresponding to the maximum value of q(a)
        Returns:
        """
        max_value = None
        for possible_action in [*current_state.q_a]:
            # Check if the next action is a wall
            if not self.action_against_wall(possible_action):
                if max_value is None:
                    max_value = current_state.q_a[possible_action]
                    max_q = possible_action
                else:
                    if current_state.q_a[possible_action] > max_value:
                        max_value = current_state.q_a[possible_action]
                        max_q = possible_action
        if value_type == 'action':
            return max_q
        else:
            return max_value

    def update_q_value(self, s, s_first, action, action_first, alpha, discount_factor):
        """
        Function to update the value of q(a)
        Args:
            s: State S
            s_first:  State S'
            action: Action A
            action_first: Action A'
            alpha: Learning rate
            discount_factor: Discount factor
        """
        if self.name == 'SARSA':
            s.q_a[action] = s.q_a[action] + alpha * (
                    s_first.reward + discount_factor * (s_first.q_a[action_first]) - s.q_a[action])
        elif self.name == 'Q-Learning':
            s.q_a[action] = s.q_a[action] + alpha * (
                    s_first.reward + discount_factor * (self.get_max_q(current_state=s_first, value_type='value')) -
                    s.q_a[action])

        self.rewards_for_step.append(s_first.reward)
        self.step += 1

    def restart_episode(self):
        """
        This method restarts the episode in position (0,0) and all the counters.
        """
        self.current_position.col = 0
        self.current_position.row = 0
        sum_reward = sum(self.rewards_for_step)
        self.rewards_for_episode.append(sum_reward)
        self.step_for_episode.append(self.step)
        self.rewards_for_step = []
        self.step = 0
        print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
        self.episode = self.episode + 1

    def sarsa_algorithm(self, epsilon, alpha, discount_factor):
        s = self.get_current_state()
        action = self.action_e_greedy(current_state=s, epsilon=epsilon)
        while self.episode <= n_episode:
            self.current_position.col, self.current_position.row = self.get_next_state(action)
            s_first = self.get_current_state()
            action_first = self.action_e_greedy(current_state=s_first, epsilon=epsilon)
            self.update_q_value(s=s, s_first=s_first, action=action, action_first=action_first, alpha=alpha,
                                discount_factor=discount_factor)
            action = action_first
            s = s_first

            if s.terminal:
                self.restart_episode()
                s = self.get_current_state()
                action = self.action_e_greedy(current_state=s, epsilon=epsilon)

    def q_learning_algorithm(self, epsilon, alpha, discount_factor):
        s = self.get_current_state()
        while self.episode <= n_episode:
            action = self.action_e_greedy(current_state=s, epsilon=epsilon)
            self.current_position.col, self.current_position.row = self.get_next_state(
                action)
            s_first = self.get_current_state()
            self.update_q_value(s=s, s_first=s_first, action=action, action_first=None, alpha=alpha,
                                discount_factor=discount_factor)
            s = s_first

            if s.terminal:
                self.restart_episode()
                s = self.get_current_state()


if __name__ == '__main__':
    n_episode = 100
    epsilon = 0.01
    alpha = 0.95
    discount_episode = 1
    height = 9
    width = 9
    walls = [
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
        [2, 6], [3, 6], [4, 6], [5, 6], [2, 6],
        [7, 1], [7, 2], [7, 3], [7, 4]]

    # Monte-Carlo evaluation
    monte_carlo_world = GridWord(name='Monte Carlo Evaluation', height=height, width=width, r_nt=-1)
    monte_carlo_world.set_terminal_state(row=8, col=8, reward=50)
    monte_carlo_world.set_terminal_state(row=6, col=5, reward=-50)
    monte_carlo_world.set_wall(walls=walls)

    returns = {}
    G = 0
    while monte_carlo_world.episode <= n_episode:
        s = monte_carlo_world.get_current_state()
        hash_position = str(monte_carlo_world.current_position.col) + str(monte_carlo_world.current_position.row)
        action = monte_carlo_world.action_e_greedy(current_state=s, epsilon=1)
        monte_carlo_world.current_position.col, monte_carlo_world.current_position.row = monte_carlo_world.get_next_state(
            action)
        s_first = monte_carlo_world.get_current_state()
        G = discount_episode * G + s_first.reward
        if hash_position not in returns:
            returns[hash_position] = [G]
        else:
            returns[hash_position].append(G)
            s.v_pi = round(sum(returns[hash_position]) / len(returns[hash_position]), 2)

        s = s_first

        if s.terminal:
            G = 0
            monte_carlo_world.restart_episode()
            s = monte_carlo_world.get_current_state()
            action = monte_carlo_world.action_e_greedy(current_state=s, epsilon=1)

    common_functions.plot_world(worlds=[monte_carlo_world], variable='v_pi')

    # Q-Learning
    q_learning_world = GridWord(name='Q-Learning', height=height, width=width, r_nt=-1)
    q_learning_world.set_terminal_state(row=8, col=8, reward=50)
    q_learning_world.set_terminal_state(row=6, col=5, reward=-50)
    q_learning_world.set_wall(walls=walls)
    q_learning_world.q_learning_algorithm(alpha=alpha, epsilon=epsilon, discount_factor=discount_episode)

    # Sarsa
    sarsa_world = GridWord(name='SARSA', height=height, width=width, r_nt=-1)
    sarsa_world.set_terminal_state(row=8, col=8, reward=50)
    sarsa_world.set_terminal_state(row=6, col=5, reward=-50)
    sarsa_world.set_wall(walls=walls)
    sarsa_world.sarsa_algorithm(alpha=alpha, epsilon=epsilon, discount_factor=discount_episode)

    # Graphs
    common_functions.plot_world(worlds=[q_learning_world, sarsa_world], variable='q_a')
    common_functions.plot_total_reward_step(world_qlearning=q_learning_world, world_sarsa=sarsa_world)
