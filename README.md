# Gridworld-RL
Solve the Gridworld game with Reinforcement Learning (SARSA and Q-learning)

## Game 

A GridWorld is a matrix game. Every cell in the matrix can be non-terminal state, a terminal state or wall. An agent who tries to play this game should start from a state and try to reach the maximum reward possible. 
The agent has to discover the world, so positions of walls and terminal states are not known. If the agent wants to take any action that brings it against a wall, it will remain stationary. If the agent wants to move in a terminal state, the game will end, and a new episode will be restarted. 
To be more precise, in the Sarsa and the Q-learning sections, regular versions of the game will be compared to the game's random versions. The regular version is when the agent reached the terminal state, and it restarts from the position (0,0). Instead, the random version of the game is when the agent concludes an episode, and it restarts from a random state that is neither a terminal state nor a wall.

## Data Structure

1. The **Position** class represents a position of an object in the matrix.<br><br>
   **Class variables**:
   -  row : the row of the object in the matrix
   -	col : the column of the object in the matrix

2.  The **Cell** class represents the basic component of the Gridworld matrix.  <br><br>
  **Class variables**:
    -	reward : the reward got from the agent that passed from the cell
    -	terminal : a boolean variable that is True if the cell is a terminal state
    -	q_a: a python dictionary containing the actions states value for each action 
  e.g. { ‘up’ : 1, ‘down’ : 0.6, ‘right’: ‘20’, left : ‘10’}
    -	v_pi: the state value.
    -	wall: a boolean variable that is True if the state is a wall
    -	row : the row of the object in the matrix
    -	col : the column of the object in the matrix

3)	The **GridWorld** class represents the game and contains the Monte Carlo evaluation, the Sarsa and the Q-Learning algorithms. <br><br>
  **Class variables**: 
  
    -	name : name of the instance of the game, e.g. Q-Learning game.
    -	episode : number of the current episode
    -	step: number of the current step. This variable is reset during the restart of the game
    -	reward_for_step : list of rewards got from the agent during the steps on an episode
    -	reward_for_episode : list of the total rewards got from the agent during each episode
    -	step_for_episode : number of steps did from the agent for each episode
    -	current_position : position of the agent in the world
    -	world : matrix of cells that represents the Gridworld.


At the start of the game, the world is initialized. Height, width, rewards for non-terminal states, walls, positions and rewards for terminal states must be given. At this point, the software will create a matrix [height x width] made of  cells that are all initialized with state and state-action values to 0. Later, according to the Monte Carlo First-Visit evaluation, the Monte Carlo every-visit evaluation, the Sarsa + greedification or the Q-learning the corresponding procedure will be executed. 

However, please note that to run different algorithms or to run a different version of the same algorithm, e.g. regular or random, new instances of the Gridworld class must be created.

## Monte Carlo Policy Evaluation

The Monte Carlo Evaluation algorithm used to evaluate the policy is standard, no specific tricks have been taken. The hyper-parameters were set as follows.

- Number of episodes: 1000. This number has been chosen, making different tries and searching for a good result and a relatively low number of episodes. 
- Policy : ¼ for every action
- Discount factor (gamma): the discount factor is set to 1. In this way, the agent cares about future rewards as the immediate ones. In this way the agent should not mind about the non-terminal rewards, but instead  it should aim at searching the final big prize of +50.

### Results

The heatmaps show the results obtained with the Monte Carlo evaluation First-Visit and the Monte Carlo evaluation Every-Visit. The first two maps show the FV and EV algorithms in their regular version. It is possible to see that there are small differences. In fact, the top left corner of the FV version has slightly smaller values than the EV version. On the contrary, on the bottom right, the FV version has slightly bigger values than the EV version. 

<p align="center">
  <img src="/doc/images/Monte_carlo_eval_EV_FV_regular.png">
</p>

Running the algorithms with the random versions, little changes are visible. In general, the values are lower than those seen above.

<p align="center">
  <img src="/doc/images/Monte_carlo_eval_EV_FV_random.png">
</p>

From what is known in theory, these two algorithms converge both to v_\pi(s) as the number of visits s goes to infinity. Hence, further experiments have been carried out.
First, the number of episodes was increased to 100.000. From this larger number, it is expected to get two identical maps. In fact, as possible to see in figure 11, the two results are very close.

<p align="center">
  <img src="/doc/images/Monte_carlo_eval_EV_FV_random_longrun.png">
</p>

# Sarsa + Greedification to compute the Optimal policy

## Results

As it can be seen with the first training where epsilon is equal to 0.00001, only the random algorithm reached the optimal policy. The reason is that Sarsa + greedification, being an On-policy algorithm, uses the same policy to train and make decisions. The very low epsilon parameter, initialized to 0. 00001, makes the agent exploit instead of exploring. For this reason, the regular case partially explores the central states of the Gridworld. 

On the contrary, the second algorithm's randomity ensures that the agent will reach and explore as necessary every point of the world, even though the agent exploits instead of exploring. In this case, due to the algorithm's random nature, dozens of tests were performed to ensure that each cell was correctly valued. In many cases, the policy was not optimal for a few cells.


<p align="center">
  <img src="/doc/images/sarsa_comparation_0_0001.png">
</p>

<p align="center">
  <img src="/doc/images/sarsa_comparation_graphs_0_00001.png">
</p>


Many attempts have been made to find an optimal policy with the regular version. It comes naturally to increase the epsilon in order to explore as necessary the middle of the matrix. However, if the epsilon is over-increased, the agent will never find the optimal route and so the optimal policy.  
Furthermore, as one can see from the following images, the increase of the epsilon hyperparameter decreases the random sarsa algorithm's performance, preventing him from obtaining the optimal policy.


<p align="center">
  <img src="/doc/images/sarsa_comparation_0_2.png">
</p>

<p align="center">
  <img src="/doc/images/sarsa_comparation_graphs_0_2.png">
</p>

## Q-learning to compute the Optimal policy and a comparison with the Sarsa approach

The Q-learning algorithm used to search the optimal policy is standard, but a small trick to reach the optimal policy earlier was taken. Thanks to how the Q-learning algorithm was designed, it always converges to the optimal policy, even though the actions with the best state-action values are not taken. For this reason, a random policy was used, exploration was performed during the training and only at the end a greedy policy was chosen.

The other hyperparameter were set as follows:

-	Number of episodes: 100. This number is sufficient to obtain the optimal policy.
-	Discount factor (gamma): the discount factor is set to 1. In this way, the agent cares about future rewards as the immediate ones. In this way the agent should not mind about the non-terminal rewards, but instead  it should aim at searching the final big prize of +50.
-	Alpha: the learning rate was set to 1. This high learning rate allows to quickly get precise results.  

### Results

As expected, at the start and during the training, the agent explores all the grid. Every time it chooses random action without being greedy. In fact, in figure 23 it is possible to see how the total rewards remain very negative, and the total number of steps still very high. Only at the end, the total rewards start increasing, reaching the maximum possible rewards (35) and the total number of steps start decreasing to the minimum number of steps (15).
In this case, there is no significant difference between the regular and the random version. The heatmaps show how the state action values and the optimal policy are the same. Instead, a small difference can be seen from the graphs of the training. The random version, unlike the regular one, often ends earlier and reaches a higher reward.

<p align="center">
  <img src="/doc/images/q_learning_comparation.png">
</p>

<p align="center">
  <img src="/doc/images/q_learning_comparation_graphs.png">
</p>

The difference between Sarsa and Q-learning is evident. From the experiments' results, the first method is slower to converge into the optimal policy. Many times, using the Sarsa regular version, a decent policy cannot even be obtained, it tends to get stuck in optimal minimum. In the optimal cases, the Sarsa random version reaches the optimal policy but taking more time than the Q-learning method. 
Finally, an experiment was made to understand which algorithm could reach the highest reward within the lowest number of steps.

After many tests, the Q-learning algorithm always reached the best rewards earlier than the Sarsa one. Both reach a reasonable reward in about 25 steps, and they stabilize in about 35 steps.

<p align="center">
  <img src="/doc/images/Optimal_path_graph.png">
</p>

## Installation
Open the terminal in the current folder and digit:


```sh 
pip install -r requirements.txt 
```

```sh 
python grid_world.py
```
