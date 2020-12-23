import random

from box import Box
import matplotlib.pyplot as plt


class GridWord(object):
    """
    This class represent the gridworld.
    """

    def __init__(self, height, width, r_nt=0):

        self.episode = 1
        self.step = 1
        self.rewards_for_step = []
        self.rewards_for_episode = []
        self.step_for_episode = []
        self.current_state = [0, 0]
        self.world = []
        for col in range(width):
            tmp = []
            for row in range(height):
                # Define a Box and for all the box remove the directions that bring the agents outside of the world
                box = Box(r_nt)
                if row == 0:
                    box.q_a.pop(Box.Direction.LEFT.value)
                if row == width - 1:
                    box.q_a.pop(Box.Direction.RIGHT.value)
                if col == 0:
                    box.q_a.pop(Box.Direction.UP.value)
                if col == height - 1:
                    box.q_a.pop(Box.Direction.DOWN.value)
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

    def set_wall(self, walls: list) -> None:
        """
        Method used to set the walls inside the gridworld.
        Args:
            walls: List containing positions (x,y)
        """
        for wall in walls:
            self.world[wall[0]][wall[1]].wall = True

    def plot_world(self, variable: str = 'v_pi'):
        """
        Method used to to plot the world
        Args:
            variable: Type of variable visualized in the plot
        """
        fig, ax = plt.subplots()
        matrix_tmp = []
        matrix_tmp_a = []
        for col in self.world:
            tmp = []
            tmp_a = []
            for row in col:
                if variable == 'v_pi':
                    tmp.append(row.v_pi)
                elif variable == 'wall':
                    tmp.append(int(row.wall))
                elif variable == 'reward':
                    tmp.append(row.reward)
                elif variable == 'q_a':
                    tmp_a.append(row.get_max_q('action'))
                    tmp.append(round(row.get_max_q('value'), 1))
                elif variable == 'terminal':
                    tmp.append(int(row.terminal))
            matrix_tmp.append(tmp)
            matrix_tmp_a.append(tmp_a)

        ax.matshow(matrix_tmp, cmap=plt.cm.Greens)
        plt.title(variable)

        if variable in ['v_pi', 'reward', 'q_a']:
            for idx_col, col in enumerate(matrix_tmp):
                for idx_row, row in enumerate(col):
                    if variable == 'q_a':
                        ax.text(idx_row, idx_col, str(row), va='bottom', ha='center')
                        ax.text(idx_row, idx_col, matrix_tmp_a[idx_col][idx_row], va='top', ha='center')
                    else:
                        ax.text(idx_col, idx_row, str(row), va='center', ha='center')
        plt.show()

    def action_e_greedy(self, epsilon):
        epsilon = epsilon * 100
        q_current_state = self.world[self.current_state[0]][self.current_state[1]].q_a
        possible_direction = [*q_current_state]
        value = random.choices(['random', 'greedy'], weights=[epsilon, 100 - epsilon], k=1)

        # Remove all the directions were a wall is found
        for direction in possible_direction[:]:
            if self.direction_against_wall(direction):
                possible_direction.remove(direction)

        # Choose greedy between the possible directions
        if 'greedy' in value:
            # take maximum
            action = self.world[self.current_state[0]][self.current_state[1]].get_max_q(value_type='action')
            if action not in possible_direction: action = random.choice(possible_direction)
        else:
            action = random.choice(possible_direction)
        return action

    def get_next_state(self, direction):
        col = self.current_state[0]
        row = self.current_state[1]

        if direction == Box.Direction.DOWN.value:
            col = col + 1
        elif direction == Box.Direction.UP.value:
            col = col - 1
        elif direction == Box.Direction.RIGHT.value:
            row = row + 1
        elif direction == Box.Direction.LEFT.value:
            row = row - 1
        return [col, row]

    def direction_against_wall(self, direction):
        col, row = self.get_next_state(direction=direction)
        if self.world[col][row].wall:
            return True

    def update_sarsa(self, direction):
        """
        Method to update the Q_values using the SARSA Algorithm
        Args:
            direction:
        Returns:
        """
        pass

    def update_q_learning(self, direction, alpha, discount_factor):
        """
        Method to update the Q_values using the Q_learning Algorithm
        Args:
            direction: Direction took from the Epsilon greedy method
            alpha: Alpha constant
            discount_factor: Discount factor
        """
        # Update Q
        col, row = self.get_next_state(direction=direction)
        self.world[self.current_state[0]][self.current_state[1]].q_a[direction] = \
            self.world[self.current_state[0]][self.current_state[1]].q_a[direction] + alpha * (
                        self.world[col][row].reward + (
                        discount_factor * self.world[col][row].get_max_q(value_type='value')) -
                        self.world[
                            self.current_state[0]][
                            self.current_state[
                                1]].q_a[direction])
        self.rewards_for_step.append(self.world[col][row].reward)
        self.step += 1
        # Update current state
        if self.world[col][row].terminal:
            self.current_state[0] = 0
            self.current_state[1] = 0
            sum_reward = sum(self.rewards_for_step)
            self.rewards_for_episode.append(sum_reward)
            self.step_for_episode.append(self.step)
            print('Episode: ', self.episode, ' Cumulative reward of: ', sum_reward)
            self.rewards_for_step = []
            self.step = 0
            self.episode = self.episode + 1
        else:
            self.current_state[0] = col
            self.current_state[1] = row

    def plot_total_reward_step(self):
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(self.rewards_for_episode)
        axs[0].set_title('Total reward for each episode')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total reward')
        axs[1].plot(self.step_for_episode)
        axs[1].set_xlabel('Episode')
        axs[1].set_title('Number of step for episode')
        axs[1].set_ylabel('Steps')
        fig.suptitle('Reward and Step per Episode', fontsize=16)
        plt.show()


if __name__ == '__main__':
    gridworld = GridWord(height=9, width=9, r_nt=-1)
    gridworld.set_terminal_state(row=8, col=8, reward=50)
    gridworld.set_terminal_state(row=6, col=6, reward=-50)
    gridworld.set_wall([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                        [2, 6], [3, 6], [4, 6], [5, 6], [2, 6],
                        [7, 1], [7, 2], [7, 3], [7, 4]])
    while gridworld.episode <= 50:
        action = gridworld.action_e_greedy(0.01)
        gridworld.update_q_learning(direction=action, alpha=0.99, discount_factor=1)
    gridworld.plot_world('q_a')
    gridworld.plot_total_reward_step()
