import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import gc

from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Flatten, Conv2D, InputLayer
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam
import keras.backend as K

from collections import deque
import math
import json
from mini_pacman import PacmanGame, test, random_strategy, naive_strategy


# Create a deep neural network that takes byte vector as an input and produces Q-values for state-action pairs.

# ## Creating a DQN-model using Keras
# 
# The following model is of the same general type applied to the cartPole problem.
# 
# Use vanilla multi-layer dense network with relu activations which computes Q-values $Q(s,a)$ for all states $s$ and actions $a$ (with some discount factor $\gamma$).
# This neural network denoted by $Q(s\ |\ \theta)$ takes current state as an input and produces a vector of q-values for all 9 possible actions. Vector $\theta$ corresponds to all trainable parameters.

def create_dqn_model(input_shape, nb_actions, dense_layers, dense_units):
    model = Sequential()
    for i in range(dense_layers):
        if i==0:
            model.add(Dense(units=dense_units, activation='relu',input_shape=input_shape))
        else:
            model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

# Plot the architecture of the network saved as *online_DenseNetwork.png*. (To see the plot log into iLykei.com, then rerun this cell).
# 
# ![Model plot](https://ilykei.com/api/fileProxy/documents%2FAdvanced%20Machine%20Learning%2FReinforced%20Learning%2Fonline_DenseNetwork.png)

# This network is used to explore states and rewards of Markov decision process according to an $\varepsilon$-greedy exploration strategy:

# determine best action from q_values # incorporated possible actions
def epsilon_greedy(q_values, epsilon, possible_actions):
    if random.random() < epsilon:
        return random.choice(possible_actions)
        #return random.randrange(n_outputs)  # random action
    else:
        return possible_actions[np.argmax(q_values[possible_actions])]
        #return np.argmax(q_values)          # q-optimal action

# Compile online-network with Adam optimizer, mean squared error loss and `mean_q` metric, which measures the maximum of predicted q-values averaged over samples from minibatch (we expect it to increase during training process).
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

# step decay
def step_decay(epoch,lr):
    # initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 500000
    lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
# exponential decay
def exp_decay(epoch,lr):
    # initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 500000
    lrate = 0.5*lr * (1+drop**(epoch/epochs_drop)) #lr*drop^(epoch/epochs_drop)
    return lrate

# represent current state as a vector of features given obs
def get_state(obs):
    v = []
    x,y = obs['player']
    v.append(x)
    v.append(y)
    for x, y in obs['monsters']:
        v.append(x)
        v.append(y)
    for x, y in obs['diamonds']:
        v.append(x)
        v.append(y)
    for x, y in obs['walls']:
        v.append(x)
        v.append(y)
    return np.asarray(v)

def train_dqn():
    with open('test_params.json', 'r') as file:
        read_params = json.load(file)
    game_params = read_params['params']
    env = PacmanGame(**game_params)
    obs = env.reset() # restart game:
    state = get_state(obs)
    
    # Create a network using specific input shape and action space size. We call this network *online*.
    input_shape = state.shape
    nb_actions = 9
    dense_layers = 5
    dense_units = 256

    online_network = create_dqn_model(input_shape, nb_actions, dense_layers, dense_units)
    # Online network stores explored information in a *replay memory*, a double-ended queue (deque).
    replay_memory_maxlen = 1000000
    replay_memory = deque([], maxlen=replay_memory_maxlen)
    # So, online network explores the game using $\varepsilon$-greedy strategy and saves experienced transitions in replay memory. 
    # 
    # In order to produce Q-values for $\varepsilon$-greedy strategy, following the proposal of the [original paper by Google DeepMind](https://www.nature.com/articles/nature14236), use another network, called *target network*, to calculate "ground-truth" target for the online network. *Target network*, has the same architecture as online network and is not going to be trained. Instead, weights from the online network are periodically copied to target network.
    target_network = clone_model(online_network)
    target_network.set_weights(online_network.get_weights())


    # ## Training DQN
    # First, define hyperparameters (Do not forget to change them before moving to cluster):
    name = 'MiniPacman_DQN'  # used in naming files (weights, logs, etc)
    n_steps = 1000000      # total number of training steps (= n_epochs)
    warmup = 1000          # start training after warmup iterations
    training_interval = 4  # period (in actions) between training steps
    save_steps = int(n_steps/10)  # period (in training steps) between storing weights to file
    copy_steps = 200 #100      # period (in training steps) between updating target_network weights
    gamma = 0.99 #0.9      # discount rate
    skip_start = 0 #90        # skip the start of every game (it's just freezing time before game starts)
    batch_size = 32   # size of minibatch that is taken randomly from replay memory every training step
    double_dqn = True     # whether to use Double-DQN approach or simple DQN (see above)
    # eps-greedy parameters: we slowly decrease epsilon from eps_max to eps_min in eps_decay_steps
    eps_max = 1.0
    eps_min = 0.01 #0.05
    eps_decay_steps = int(n_steps/2) #int(n_steps/2)

    learning_rate = 0.00025
    decay_rate=learning_rate/n_steps

    online_network.compile(optimizer=Adam(learning_rate), loss='mse', metrics=[mean_q])


    # Create folder for logs and trained weights:
    if not os.path.exists(name):
        os.makedirs(name)
        
    weights_folder = os.path.join(name, 'weights')
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)


    # Use standard callbacks:
    csv_logger = CSVLogger(os.path.join(name, 'log.csv'), append=True, separator=';')
    tensorboard = TensorBoard(log_dir=os.path.join(name, 'tensorboard'), write_graph=False, write_images=False)
    
    # set up learning rate scheduler based on epoch
    lrate = LearningRateScheduler(step_decay)


    # Next chunk of code explores the game, trains online network and periodically copies weights to target network as explained above.
    # counters:
    step = 0          # training step counter (= epoch counter)
    iteration = 0     # frames counter
    episodes = 0      # game episodes counter
    # done = True       # indicator that env needs to be reset

    episode_scores = []  # collect total scores in this list and log it later

    while step < n_steps:
        if obs['end_game']:  # game over, restart it
            obs = env.reset()
            # score = 0  # reset score for current episode
            for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
                obs = env.make_action(4)
                # score += obs['reward']
            state = get_state(obs)
            episodes += 1

        # Online network evaluates what to do
        iteration += 1
        q_values = online_network.predict(np.array([state]))[0]  # calculate q-values using online network
        # select epsilon (which linearly decreases over training steps):
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
        action = epsilon_greedy(q_values, epsilon, obs['possible_actions'])
        # Play:
        obs = env.make_action(action)
        # score += obs['reward']
        if obs['end_game']:
            episode_scores.append(obs['total_score'])
        next_state = get_state(obs)
        # Let's memorize what just happened
        replay_memory.append((state, action, obs['reward'], next_state, obs['end_game'], obs['possible_actions']))
        state = next_state

        if iteration >= warmup and iteration % training_interval == 0:
            # learning branch
            step += 1
            minibatch = random.sample(replay_memory, batch_size)
            replay_state = np.array([x[0] for x in minibatch])
            replay_action = np.array([x[1] for x in minibatch])
            replay_rewards = np.array([x[2] for x in minibatch])
            replay_next_state = np.array([x[3] for x in minibatch])
            replay_done = np.array([x[4] for x in minibatch], dtype=int)
            replay_possible_actions = np.array([x[5] for x in minibatch]) # incorporated possible actions

            # calculate targets (see above for details) # incorporated possible actions
            if double_dqn == False:
                # DQN
                # target_for_action = replay_rewards + (1-replay_done) * gamma * np.amax(target_network.predict(replay_next_state), axis=1)
                qs = target_network.predict(replay_next_state)
                q_max_possible = np.zeros((batch_size,))
                for i in range(batch_size):
                    q_max_possible[i] = np.amax(qs[i, replay_possible_actions[i]])
                target_for_action = replay_rewards + (1-replay_done) * gamma * q_max_possible
            else:
                # Double DQN
                # best_actions = np.argmax(online_network.predict(replay_next_state), axis=1)
                qs = online_network.predict(replay_next_state)
                best_actions = np.zeros((batch_size,), dtype=int)
                for i in range(batch_size):
                    best_actions[i] = replay_possible_actions[i][np.argmax(qs[i, replay_possible_actions[i]])]
                target_for_action = replay_rewards + (1-replay_done) * gamma * target_network.predict(replay_next_state)[np.arange(batch_size), best_actions]

            target = online_network.predict(replay_state)  # targets coincide with predictions ...
            target[np.arange(batch_size), replay_action] = target_for_action  #...except for targets with actions from replay
            
            # Train online network
            online_network.fit(replay_state, target, epochs=step, verbose=2, initial_epoch=step-1,
                               callbacks=[csv_logger, tensorboard, lrate])

            # Periodically copy online network weights to target network
            if step % copy_steps == 0:
                target_network.set_weights(online_network.get_weights())
            # And save weights
            if step % save_steps == 0:
                online_network.save_weights(os.path.join(weights_folder, 'weights_{}.h5f'.format(step)))
                gc.collect()  # also clean the garbage
    # Save last weights:
    online_network.save_weights(os.path.join(weights_folder, 'weights_last.h5f'))

    # save model
    online_network.save(os.path.join(weights_folder, 'ram_model_4kk.h5f'))

    # Dump all scores to txt-file
    with open(os.path.join(name, 'episode_scores.txt'), 'w') as file:
        for item in episode_scores:
            file.write("{}\n".format(item))

    return online_network


def dqn_strategy(obs, model):
    # model = load_model('MiniPacman_DQN/weights/ram_model_4kk.h5f', compile=False)
    state = get_state(obs)
    q_values = model.predict(np.array([state]))[0]
    eps = 0.001
    action = epsilon_greedy(q_values, eps, obs['possible_actions'])
    return action

# run mini_pacman test with specified parameters
if __name__ == '__main__':
    if os.path.exists('MiniPacman_DQN/weights/ram_model_4kk.h5f'):
        online_network = load_model('MiniPacman_DQN/weights/ram_model_4kk.h5f', compile=False)
    elif os.path.exists('MiniPacman_DQN/weights'):
        if len(os.listdir(path='MiniPacman_DQN/weights/')) != 0:
            list_of_files = glob.glob('MiniPacman_DQN/weights/weights_*')
            lastest_file = max(list_of_files, key=os.path.getctime)
            online_network = create_dqn_model((32,), 9, 5, 256)
            online_network.load_weights(lastest_file)
        else:
            online_network = train_dqn()
    else:
        online_network = train_dqn()
    # test(strategy=random_strategy, model=online_network, log_file='test_pacman_random_log.json')
    # test(strategy=naive_strategy, model=online_network, log_file='test_pacman_naive_log.json')
    test(strategy=dqn_strategy, model=online_network, log_file='test_pacman_dqn_log.json', render=True)






