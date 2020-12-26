import matplotlib.pyplot as plt


def plot_total_reward_step(world_qlearning, world_sarsa):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(world_qlearning.rewards_for_episode, label=world_qlearning.name)
    axs[0].plot(world_sarsa.rewards_for_episode, label=world_sarsa.name)
    axs[0].set_title('Total reward for each episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total reward')
    axs[0].legend()
    axs[1].plot(world_qlearning.step_for_episode, label=world_qlearning.name)
    axs[1].plot(world_sarsa.step_for_episode, label=world_sarsa.name)
    axs[1].set_xlabel('Episode')
    axs[1].set_title('Number of step for episode')
    axs[1].set_ylabel('Steps')
    axs[1].legend()
    fig.suptitle('Reward and Step per Episode', fontsize=16)
    plt.show()


def plot_world(worlds: list, variable: str = 'v_pi'):
    fig, ax = plt.subplots(1, len(worlds))
    fig.suptitle('Coloured MAP for ' + variable + ' variable', fontsize=16)
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
                        if len(worlds) > 1:
                            ax[idx_world].text(idx_row, idx_col, str(row), va='bottom', ha='center')
                            ax[idx_world].text(idx_row, idx_col, matrix_tmp_a[idx_col][idx_row], va='top', ha='center')
                        else:
                            ax.text(idx_row, idx_col, str(row), va='bottom', ha='center')
                            ax.text(idx_row, idx_col, matrix_tmp_a[idx_col][idx_row], va='top', ha='center')
                    else:
                        if len(worlds) > 1:
                            ax[idx_world].text(idx_row, idx_col, str(row), va='center', ha='center')
                        else:
                            ax.text(idx_row, idx_col, str(row), va='center', ha='center')
    plt.show()
