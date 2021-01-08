import matplotlib.pyplot as plt


def plot_total_reward_step(worlds: list):
    """
    This method plot the graphs for the total amount of rewards and the total amount of steps for each episode.
    Args:
        worlds: List of worlds to draw in the graph
    """
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    for world in worlds:
        axs[0].plot(world.rewards_for_episode, label=world.name)
    axs[0].set_title('Total reward for each episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total reward')
    axs[0].legend()

    for world in worlds:
        axs[1].plot(world.step_for_episode, label=world.name)
    axs[1].set_xlabel('Episode')
    axs[1].set_title('Number of steps for episode')
    axs[1].set_ylabel('Steps')
    axs[1].legend()
    fig.suptitle('Reward and Steps per Episode', fontsize=16)
    plt.show()


def plot_world(worlds: list, variable: str = 'v_pi'):
    """
    This method plots the heatmap of the world. There are different logics for different variables.
    Args:
        worlds: Worlds to plot
        variable:
        -   'v_pi' for state values
        -   'q_a' for state-action values
        -   'wall' to print the wall
        -   'reward' to print the rewards for each state
        -   'terminal' to print the terminal states
    Returns:
    """
    fig, ax = plt.subplots(1, len(worlds))
    fig.suptitle('Heatmap for ' + variable, fontsize=16)
    for idx_world, world in enumerate(worlds):
        matrix_tmp = []
        matrix_tmp_a = []
        for col in world.world:
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
                    tmp_a.append(world.get_max_q(current_state=row, value_type='action'))
                    tmp.append(round(world.get_max_q(current_state=row, value_type='value'), 1))
                elif variable == 'terminal':
                    tmp.append(int(row.terminal))
            matrix_tmp.append(tmp)
            matrix_tmp_a.append(tmp_a)

        for k in matrix_tmp:
            print(k)

        if len(worlds) > 1:
            ax[idx_world].matshow(matrix_tmp, cmap=plt.cm.Greens)
            ax[idx_world].set_title(world.name + ' ' + variable)
        else:
            ax.matshow(matrix_tmp)
            ax.set_title(world.name + ' ' + variable)

        if variable in ['v_pi', 'reward', 'q_a']:
            for idx_col, col in enumerate(matrix_tmp):
                for idx_row, row in enumerate(col):
                    if variable == 'q_a':
                        if not world.world[idx_col][idx_row].wall and not world.world[idx_col][idx_row].terminal:
                            if len(worlds) > 1:
                                ax[idx_world].text(idx_row, idx_col, str(row), va='bottom', ha='center')
                                ax[idx_world].text(idx_row, idx_col, matrix_tmp_a[idx_col][idx_row], va='top', ha='center')
                            else:
                                ax.text(idx_row, idx_col, str(row), va='bottom', ha='center')
                                ax.text(idx_row, idx_col, matrix_tmp_a[idx_col][idx_row], va='top', ha='center')
                    else:
                        if not world.world[idx_col][idx_row].wall and not world.world[idx_col][idx_row].terminal:
                            if len(worlds) > 1:
                                ax[idx_world].text(idx_row, idx_col, str(row), va='center', ha='center')
                            else:
                                ax.text(idx_row, idx_col, str(row), va='center', ha='center')
    plt.show()
