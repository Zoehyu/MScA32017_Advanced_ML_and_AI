#!/usr/bin/env python
# coding: utf-8

# # iLykei Lecture Series
# 
# # Advanced Machine Learning and Artificial Intelligence
# 
# # Reinforcement Learning
# 
# ## Notebook 6: Learning Ms. Pac-Man with DQN
# 
# ## Yuri Balasanov, Mihail Tselishchev, &copy; iLykei 2018
# 
# ##### Main text: Hands-On Machine Learning with Scikit-Learn and TensorFlow, Aurelien Geron, &copy; Aurelien Geron 2017, O'Reilly Media, Inc

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import gc

from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Conv2D, InputLayer
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam
import keras.backend as K

import gym

plt.rcParams['figure.figsize'] = (9, 9)


# # Deep Q-Learning of MS. Pac-Man with Keras
# 
# This notebook shows how to implement a deep neural network approach to train an agent to play Ms.Pac-Man Atari game.
# 
# 
# ## Explore the game
# 
# Use [Gym](https://gym.openai.com/) toolkit that provides both game environment and also a convenient renderer of the game.
# 
# Create an environment.
env = gym.make("MsPacman-ram-v0")

# ### Observation
# 
# In this environment, observation (i.e. current state) is the RAM of the Atari machine, namely a vector of 128 bytes:
obs = env.reset()

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


# Create a network using specific input shape and action space size. We call this network *online*.
input_shape = obs.shape
nb_actions = env.action_space.n  # 9
dense_layers = 5
dense_units = 256

online_network = create_dqn_model(input_shape, nb_actions, dense_layers, dense_units)
# online_network.summary()
# from keras.utils import plot_model
# plot_model(online_network, to_file='online_DenseNetwork.png',show_shapes=True,show_layer_names=True)


# Plot the architecture of the network saved as *online_DenseNetwork.png*. (To see the plot log into iLykei.com, then rerun this cell).
# 
# ![Model plot](https://ilykei.com/api/fileProxy/documents%2FAdvanced%20Machine%20Learning%2FReinforced%20Learning%2Fonline_DenseNetwork.png)

# This network is used to explore states and rewards of Markov decision process according to an $\varepsilon$-greedy exploration strategy:

def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)          # q-optimal action


# Online network stores explored information in a *replay memory*, a double-ended queue (deque).
from collections import deque

replay_memory_maxlen = 1000000
replay_memory = deque([], maxlen=replay_memory_maxlen)


# So, online network explores the game using $\varepsilon$-greedy strategy and saves experienced transitions in replay memory. 
# 
# In order to produce Q-values for $\varepsilon$-greedy strategy, following the proposal of the [original paper by Google DeepMind](https://www.nature.com/articles/nature14236), use another network, called *target network*, to calculate "ground-truth" target for the online network. *Target network*, has the same architecture as online network and is not going to be trained. Instead, weights from the online network are periodically copied to target network.
target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())


# The target network uses past experience in the form of randomly selected records of the replay memory to predict targets for the online network: 
# 
# - Select a random minibatch from replay memory containing tuples $(\text{state},\text{action},\text{reward},\text{next_state})$
# 
# - For every tuple $(\text{state},\text{action},\text{reward},\text{next_state})$ from minibatch Q-value function $Q(\text{state},\text{action}\ |\ \theta_{\text{online}})$ is trained on predictions of $Q(\text{next_state}, a\ |\ \theta_\text{target})$ according to Bellman-type equation: 
# 
# $$y_\text{target} = \text{reward} + \gamma \cdot \max_a Q(\text{next_state}, a\ |\ \theta_\text{target})$$
# if the game continues and $$ y_\text{target} = \text{reward}$$ if the game has ended. 
# 
# Note that at this step predictions are made by the target network. This helps preventing situations when online network simultaneously predicts values and creates targets, which might potentially lead to instability of training process.
# 
# - For each record in the minibatch targets need to be calculated for only one specific $\text{action}$ output of online network. It is important to ignore all other outputs during optimization (calculating gradients). So, predictions for every record in the minibatch are calculated by online network first, then the values corresponding to the actually selected action are replaced with ones predicted by target network. 
# 
# ## Double DQN
# 
# Approach proposed in the previous section is called **DQN**-approach. 
# 
# DQN approach is very powerful and allows to train agents in very complex, very multidimentional environments.
# 
# However, [it is known](https://arxiv.org/abs/1509.06461) to overestimate q-values under certain conditions. 
# 
# Alternative approach proposed in the [same paper](https://arxiv.org/abs/1509.06461) is called **Double DQN**. 
# 
# Instead of taking action that maximizes q-value for target network, they pick an action that maximizes q-value for online network as an optimal one:
# 
# $$y_\text{target} = \text{reward} + \gamma \cdot Q\left(\text{next_state}, \arg\max_a Q\left(\text{next_state},a\ |\ \theta_\text{online}\right)\ |\ \theta_\text{target}\right).$$
# 

# ## Training DQN
# 
# First, define hyperparameters (Do not forget to change them before moving to cluster):
name = 'MsPacman_DQN'  # used in naming files (weights, logs, etc)
n_steps = 1000000      # total number of training steps (= n_epochs)
warmup = 1000          # start training after warmup iterations
training_interval = 4  # period (in actions) between training steps
save_steps = int(n_steps/10)  # period (in training steps) between storing weights to file
copy_steps = 200 #100      # period (in training steps) between updating target_network weights
gamma = 0.99 #0.9      # discount rate
skip_start = 90        # skip the start of every game (it's just freezing time before game starts)
batch_size = 32 #64  # size of minibatch that is taken randomly from replay memory every training step
double_dqn = True     # whether to use Double-DQN approach or simple DQN (see above)
# eps-greedy parameters: we slowly decrease epsilon from eps_max to eps_min in eps_decay_steps
eps_max = 1.0
eps_min = 0.01 #0.05
eps_decay_steps = int(n_steps/2)

learning_rate = 0.00025
decay_rate=learning_rate/n_steps

# Compile online-network with Adam optimizer, mean squared error loss and `mean_q` metric, which measures the maximum of predicted q-values averaged over samples from minibatch (we expect it to increase during training process).
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

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
# early stopping if val loss stops decreasing
# early_stopping = EarlyStopping(monitor='mean_q', patience=200, verbose=1, mode='max')
# reduce learning rate when val loss plateaus
# scheduler = ReduceLROnPlateau(monitor='mean_q', factor=0.1, patience=20, verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=1e-10)

import math
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
# set up learning rate scheduler based on epoch
lrate = LearningRateScheduler(step_decay)


# Next chunk of code explores the game, trains online network and periodically copies weights to target network as explained above.
# counters:
step = 0          # training step counter (= epoch counter)
iteration = 0     # frames counter
episodes = 0      # game episodes counter
done = True       # indicator that env needs to be reset

episode_scores = []  # collect total scores in this list and log it later

while step < n_steps:
    if done:  # game over, restart it
        obs = env.reset()
        score = 0  # reset score for current episode
        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
            obs, reward, done, info = env.step(0)
            score += reward
        state = obs
        episodes += 1

    # Online network evaluates what to do
    iteration += 1
    q_values = online_network.predict(np.array([state]))[0]  # calculate q-values using online network
    # select epsilon (which linearly decreases over training steps):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, nb_actions)
    # Play:
    obs, reward, done, info = env.step(action)
    score += reward
    if done:
        episode_scores.append(score)
    next_state = obs
    # Let's memorize what just happened
    replay_memory.append((state, action, reward, next_state, done))
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

        # calculate targets (see above for details)
        if double_dqn == False:
            # DQN
            target_for_action = replay_rewards + (1-replay_done) * gamma * np.amax(target_network.predict(replay_next_state), axis=1)
        else:
            # Double DQN
            best_actions = np.argmax(online_network.predict(replay_next_state), axis=1)
            target_for_action = replay_rewards + (1-replay_done) * gamma * target_network.predict(replay_next_state)[np.arange(batch_size), best_actions]

        target = online_network.predict(replay_state)  # targets coincide with predictions ...
        target[np.arange(batch_size), replay_action] = target_for_action  #...except for targets with actions from replay
        
        # Train online network
        online_network.fit(replay_state, target, epochs=step, verbose=2, initial_epoch=step-1,
                           callbacks=[csv_logger, tensorboard])

        # Periodically copy online network weights to target network
        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())
        # And save weights
        if step % save_steps == 0:
            online_network.save_weights(os.path.join(weights_folder, 'weights_{}.h5f'.format(step)))
            gc.collect()  # also clean the garbage


# Save last weights:
online_network.save_weights(os.path.join(weights_folder, 'weights_last.h5f'))


# Dump all scores to txt-file
with open(os.path.join(name, 'episode_scores.txt'), 'w') as file:
    for item in episode_scores:
        file.write("{}\n".format(item))

# print(episode_scores)


# Don't forget to check TensorBoard for fancy statistics on loss and metrics using in terminal
# 
# `tensorboard --logdir=tensorboard`
# 
# after navigating to the folder containing the created folder `tensorboard`: 
# Then visit http://localhost:6006/

# ## Testing model
# 
# Finally, create a function to evalutate the trained network. 
# Note that we still using $\varepsilon$-greedy strategy here to prevent an agent from getting stuck. 
# `test_dqn` returns a list with scores for the specified number of games.

def test_dqn(env, n_games, model, nb_actions, skip_start, eps=0.05, render=False, sleep_time=0.01):
    scores = []
    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)
            obs, reward, done, info = env.step(0)
            score += reward
        while not done:
            state = obs
            q_values = model.predict(np.array([state]))[0]
            action = epsilon_greedy(q_values, eps, nb_actions)
            obs, reward, done, info = env.step(action)
            score += reward
            if render:
                env.render()
                time.sleep(sleep_time)
                if done:
                    time.sleep(1)
        scores.append(score)
    return scores


scores = test_dqn(env, n_games=5, online_network, nb_actions, skip_start, eps=0.001, render=False)
print(scores)

env.close()


# Results are pretty poor since the training was too short. 
# 
# Try to train DQN on a cluster. You might want to adjust some hyperparameters (increase `n_steps`, `warmup`, `copy_steps` and `eps_decay_steps`; gradually decrease learning rate during training, select appropriate `batch_size` to fit gpu memory, adjust `gamma`, switch on double dqn apporach and so on). 
# 
# You can even try to make the network deeper and/or use more than one observation as an input of neural network. For instance, using few consecutive game observations would definetely improve the results since they contain some helpful information such as monsters directions, etc. Turning off TensorBoard callback on a cluster would be a good idea too.

# # save model
# online_network.save(os.path.join(weights_folder, 'ram_model_4kk.h5f'))




