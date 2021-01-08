import random

import common_functions
from common_classes import Cell
from common_classes import Position


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
                tmp.append(Cell(reward=r_nt, col=col, row=row))
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
        Get the current state in world considering the current position.
        Returns: Current state
        """
        return self.world[self.current_position.col][self.current_position.row]

    def set_wall(self, walls: list) -> None:
        """
        Method used to set the walls inside the gridworld.
        Args:
            walls: List containing positions (col,row)
        """
        for wall in walls:
            self.world[wall[0]][wall[1]].wall = True

    def action_e_greedy(self, current_state, epsilon, policy=None) -> str:
        """
        This method select the next action following the E-greedy paradigm
        Args:
            current_state: The current state in the gridworld
            epsilon: Epsilon to use in the e-greedy function
            policy: List of for integers (up, down, left, right). This parameter has been
        Returns: Action to take
        """
        epsilon = epsilon * 100
        q_current_state = self.world[self.current_position.col][self.current_position.row].q_a
        possible_action = [*q_current_state]

        # For monte carlo policy evaluation
        if policy is not None:
            return random.choices(possible_action, weights=[policy[0], policy[1], policy[2], policy[3]], k=1)[0]

        value = random.choices(['random', 'greedy'], weights=[epsilon, 100 - epsilon], k=1)

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

        if action == Cell.Action.DOWN.value:
            col = col + 1
        elif action == Cell.Action.UP.value:
            col = col - 1
        elif action == Cell.Action.RIGHT.value:
            row = row + 1
        elif action == Cell.Action.LEFT.value:
            row = row - 1

        # Walls or out of the world
        if (col < 0 or col > height - 1) or (row < 0 or row > width - 1) or self.world[col][row].wall:
            return [self.current_position.col, self.current_position.row]
        return [col, row]

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

    def random_position(self):
        """
        This method returns a random position that isn't neither a wall or a terminal state
        Returns: column, row of the random position
        """
        found_position = False
        while not found_position:
            col = random.randint(0, 8)
            row = random.randint(0, 8)
            if not self.world[col][row].wall and not self.world[col][row].terminal:
                found_position = True
        return col, row

    def update_q_value(self, s, s_first, action, action_first, alpha, discount_factor):
        """
        Function to update the value of q(a)
        Args:
            s: State S
            s_first:  State S'
            action: Action A
            action_first: Action A' Needed only for the Sarsa algorithm. Otherwise pass None.
            alpha: Learning rate
            discount_factor: Discount factor
        """
        if 'SARSA' in self.name:
            s.q_a[action] = s.q_a[action] + alpha * (
                    s_first.reward + discount_factor * (s_first.q_a[action_first]) - s.q_a[action])
        elif 'Q-Learning' in self.name:
            s.q_a[action] = s.q_a[action] + alpha * (
                    s_first.reward + discount_factor * (self.get_max_q(current_state=s_first, value_type='value')) -
                    s.q_a[action])

        self.rewards_for_step.append(s_first.reward)
        self.step += 1

    def restart_episode(self, random_start):
        """
        This method restarts the episode in position (0,0) and all the counters.
        random_start: True if it needed a random start
        """
        if random_start:
            self.current_position.col, self.current_position.row = self.random_position()
        else:
            self.current_position.col = 0
            self.current_position.row = 0

        sum_reward = sum(self.rewards_for_step)
        self.rewards_for_episode.append(sum_reward)
        self.step_for_episode.append(self.step)
        self.rewards_for_step = []
        self.step = 0
        print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
        self.episode = self.episode + 1

    def sarsa_algorithm(self, n_episode, epsilon, alpha, discount_factor, random_start):
        """
        Sarsa algorithm to find the optimal policy
        Args:
            n_episode: Number of episodes
            epsilon: Epsilon to use in e-greedy method
            alpha: Learning rate
            discount_factor: Discount factor gamma
        """
        print('START SARSA METHOD')
        epsilon_tmp = epsilon
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
                self.restart_episode(random_start=random_start)
                s = self.get_current_state()
                action = self.action_e_greedy(current_state=s, epsilon=epsilon)

                # Normal decrement of Epsilon and Alpha
                epsilon = epsilon_tmp * (1 - (self.episode / n_episode))
                # alpha = 1 - (self.episode / n_episode)

                # Last experiment that is possible to find in the Sarsa chapter
                # alpha = (2 * (self.episode / n_episode) - 1) ** 2
                # epsilon = (1 - (2 * (self.episode / n_episode) - 1) ** 2) * 0.8

    def q_learning_algorithm(self, n_episode, epsilon, alpha, discount_factor, random_start):
        """
        Q-learning algorithm to find the optimal policy
        Args:
            n_episode: Number of episodes
            epsilon: Epsilon to use in e-greedy method
            alpha: Learning rate
            discount_factor: Discount factor gamma
        """
        print('START Q-LEARNING METHOD')
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
                self.restart_episode(random_start)
                s = self.get_current_state()

                # Epsilon decrement to find the optimal policy
                epsilon = 1 - (self.episode / n_episode) ** 10

    def monte_carlo_evaluation(self, n_episode, epsilon, discount_factor, type_of_algorithm, random_start):
        """
        Q-learning algorithm to evaluate the state functions v
        Args:
            n_episode: Number of episodes
            epsilon: Epsilon to use in e-greedy method
            discount_factor: Discount factor gamma
            type_of_algorithm: Type of monte carlo algorithm. FV to use the First visit Monte Carlo evaluation. Other strings to Every visit Monte Carlo evaluation
        """
        print('START MC-EVALUATION METHOD')
        # Initialize:
        returns = {}

        # For every episode
        while self.episode <= n_episode:
            states = []
            states_hash = []
            rewards = []
            G = 0

            s = self.get_current_state()
            while not s.terminal:
                s = self.get_current_state()
                states_hash.append(s.string_hash())
                action = self.action_e_greedy(current_state=s, epsilon=epsilon, policy=[1 / 4, 1 / 4, 1 / 4, 1 / 4])
                self.current_position.col, self.current_position.row = self.get_next_state(
                    action)
                s_first = self.get_current_state()
                reward = s_first.reward
                rewards.append(reward)
                s = s_first
                self.rewards_for_step.append(s_first.reward)
                self.step += 1

            self.restart_episode(random_start=random_start)

            state_hash_copy = states_hash.copy()
            states_hash.reverse()
            rewards.reverse()

            # For each step
            for idx_step, step in enumerate(states_hash):
                G = discount_factor * G + rewards[idx_step]

                if type_of_algorithm == 'FV':
                    # First-visit
                    # If s doesn't appear in the S(t-1)
                    if states_hash[idx_step] not in state_hash_copy[:len(state_hash_copy) - (idx_step + 1)]:
                        if states_hash[idx_step] in returns:
                            returns[states_hash[idx_step]].append(G)
                        else:
                            returns[states_hash[idx_step]] = [G]
                        self.world[int(states_hash[idx_step][0])][int(states_hash[idx_step][1])].v_pi = round(
                            sum(returns[states_hash[idx_step]]) / len(returns[states_hash[idx_step]]), 1)
                else:
                    # Every-visit
                    if states_hash[idx_step] in returns:
                        returns[states_hash[idx_step]].append(G)
                    else:
                        returns[states_hash[idx_step]] = [G]
                    self.world[int(states_hash[idx_step][0])][int(states_hash[idx_step][1])].v_pi = round(
                        sum(returns[states_hash[idx_step]]) / len(returns[states_hash[idx_step]]), 1)


if __name__ == '__main__':
    height = 9
    width = 9
    walls = [
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
        [2, 6], [3, 6], [4, 6], [5, 6], [2, 6],
        [7, 1], [7, 2], [7, 3], [7, 4]]

    # Monte Carlo evaluation part
    monte_carlo_world = GridWord(name='Monte Carlo FV Random Evaluation', height=height, width=width, r_nt=-1)
    monte_carlo_world.set_terminal_state(row=8, col=8, reward=50)
    monte_carlo_world.set_terminal_state(row=6, col=5, reward=-50)
    monte_carlo_world.set_wall(walls=walls)
    monte_carlo_world.monte_carlo_evaluation(n_episode=500, epsilon=1, discount_factor=1, type_of_algorithm='FV',
                                             random_start=False)

    monte_carlo_world_random = GridWord(name='Monte Carlo EV Random Evaluation', height=height, width=width, r_nt=-1)
    monte_carlo_world_random.set_terminal_state(row=8, col=8, reward=50)
    monte_carlo_world_random.set_terminal_state(row=6, col=5, reward=-50)
    monte_carlo_world_random.set_wall(walls=walls)
    monte_carlo_world_random.monte_carlo_evaluation(n_episode=500, epsilon=1, discount_factor=1, type_of_algorithm='EV',
                                                    random_start=False)

    common_functions.plot_world(worlds=[monte_carlo_world, monte_carlo_world_random], variable='v_pi')

    # The Q learning and the Sarsa algorithm are set to find the optimal path in the less steps possible.
    # the lines to find the optimal policies are commented, if you want to change the training do it!

    # ----------------------------------------------------
    # Q-Learning
    q_learning_world = GridWord(name='Q-Learning', height=height, width=width, r_nt=-1)
    q_learning_world.set_terminal_state(row=8, col=8, reward=50)
    q_learning_world.set_terminal_state(row=6, col=5, reward=-50)
    q_learning_world.set_wall(walls=walls)
    # Optimal policy
    q_learning_world.q_learning_algorithm(n_episode=100, alpha=1, epsilon=1, discount_factor=1, random_start=False)
    # Optimal path
    # q_learning_world.q_learning_algorithm(n_episode=100, alpha=1, epsilon=0, discount_factor=1, random_start=False)
    common_functions.plot_world(worlds=[q_learning_world], variable='q_a')
    # ----------------------------------------------------
    # Q-Learning random
    q_learning_world_random = GridWord(name='Q-Learning random', height=height, width=width, r_nt=-1)
    q_learning_world_random.set_terminal_state(row=8, col=8, reward=50)
    q_learning_world_random.set_terminal_state(row=6, col=5, reward=-50)
    q_learning_world_random.set_wall(walls=walls)
    # Optimal policy
    q_learning_world_random.q_learning_algorithm(n_episode=100, alpha=1, epsilon=1, discount_factor=1, random_start=True)
    # Optimal path
    #q_learning_world_random.q_learning_algorithm(n_episode=100, alpha=1, epsilon=0, discount_factor=1,random_start=True)
    common_functions.plot_world(worlds=[q_learning_world_random], variable='q_a')
    common_functions.plot_total_reward_step(
        worlds=[q_learning_world, q_learning_world_random])
    # ----------------------------------------------------
    # Sarsa
    sarsa_world = GridWord(name='SARSA', height=height, width=width, r_nt=-1)
    sarsa_world.set_terminal_state(row=8, col=8, reward=50)
    sarsa_world.set_terminal_state(row=6, col=5, reward=-50)
    sarsa_world.set_wall(walls=walls)
    # Optimal policy
    sarsa_world.sarsa_algorithm(n_episode=10000, alpha=1, epsilon=0.00001, discount_factor=1, random_start=False)
    # Optimal path
    # sarsa_world.sarsa_algorithm(n_episode=100, alpha=1, epsilon=0, discount_factor=1, random_start=False)
    common_functions.plot_world(worlds=[sarsa_world], variable='q_a')
    # ----------------------------------------------------
    # Sarsa Random
    sarsa_world_random = GridWord(name='SARSA Random', height=height, width=width, r_nt=-1)
    sarsa_world_random.set_terminal_state(row=8, col=8, reward=50)
    sarsa_world_random.set_terminal_state(row=6, col=5, reward=-50)
    sarsa_world_random.set_wall(walls=walls)
    # Optimal policy
    sarsa_world_random.sarsa_algorithm(n_episode=10000, alpha=1, epsilon=0.00001, discount_factor=1, random_start=True)
    # Optimal path
    # sarsa_world_random.sarsa_algorithm(n_episode=100, alpha=1, epsilon=0, discount_factor=1, random_start=True)
    common_functions.plot_world(worlds=[sarsa_world_random], variable='q_a')
    common_functions.plot_total_reward_step(
        worlds=[sarsa_world, sarsa_world_random])

    # Plot all trainings
    common_functions.plot_total_reward_step(
        worlds=[sarsa_world, sarsa_world_random, q_learning_world, q_learning_world_random])
