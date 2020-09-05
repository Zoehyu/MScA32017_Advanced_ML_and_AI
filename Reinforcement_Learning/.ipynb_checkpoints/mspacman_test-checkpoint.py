{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# iLykei Lecture Series\n",
    "\n",
    "# Advanced Machine Learning and Artificial Intelligence (MScA 32017)\n",
    "\n",
    "# Pac-Man Competition for Human-Machine Teams \n",
    "\n",
    "### Y.Balasanov, M. Tselishchev, &copy; iLykei 2018\n",
    "\n",
    "## Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import numpy as np\n",
    "import gym"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Load trained model (which was previously saved by `model.save()`-method) for online network:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Flatten, Conv2D\n",
    "def create_dqn_model(input_shape, nb_actions, dense_layers, dense_units):\n",
    "    model = Sequential()\n",
    "    for i in range(dense_layers):\n",
    "        if i==0:\n",
    "            model.add(Dense(units=dense_units, activation='relu',input_shape=input_shape))\n",
    "        else:\n",
    "            model.add(Dense(units=dense_units, activation='relu'))\n",
    "    model.add(Dense(nb_actions, activation='linear'))\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12\n",
      "11\n",
      "MiniPacman_DQN/weights/weights_100000.h5f\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import glob\n",
    "# os.path.exists('MiniPacman_DQN/weights/weights_*.h5f')\n",
    "# files = os.listdir(path='MiniPacman_DQN/weights/')\n",
    "list_of_files = glob.glob('MiniPacman_DQN/weights/weights_*')\n",
    "print(len(os.listdir(path='MiniPacman_DQN/weights/')))\n",
    "print(len(list_of_files))\n",
    "print(max(list_of_files, key=os.path.getctime))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# if ram_model_4kk.h5f was saved using model.save\n",
    "from keras.models import load_model\n",
    "online_network = load_model('MiniPacman_DQN/weights/ram_model_4kk.h5f', compile=False)\n",
    "\n",
    "# elseif weights were saved using savwe_weights\n",
    "online_network = create_dqn_model((32,), 9, 5, 256)\n",
    "online_network.load_weights('MiniPacman_DQN/weights/weights_last.h5f')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Define $\\varepsilon$-greedy strategy (using small $\\varepsilon$):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def epsilon_greedy(q_values, epsilon, n_outputs):\n",
    "    if random.random() < epsilon:\n",
    "        return random.randrange(n_outputs)  # random action\n",
    "    else:\n",
    "        return np.argmax(q_values)          # q-optimal action"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing model\n",
    "\n",
    "Define a function to evalutate the trained network. \n",
    "Note that we still using $\\varepsilon$-greedy strategy here to prevent an agent from getting stuck. \n",
    "`test_dqn` returns a list with scores for specific number of games."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_dqn(n_games, model, nb_actions=9, skip_start=90, eps=0.05, render=False, sleep_time=0.01):\n",
    "    env = gym.make(\"MsPacman-ram-v0\")\n",
    "    scores = []\n",
    "    for i in range(n_games):\n",
    "        obs = env.reset()\n",
    "        score = 0\n",
    "        done = False\n",
    "        for skip in range(skip_start):  # skip the start of each game (it's just freezing time before game starts)\n",
    "            obs, reward, done, info = env.step(0)\n",
    "            score += reward\n",
    "        while not done:\n",
    "            state = obs\n",
    "            q_values = model.predict(np.array([state]))[0]\n",
    "            action = epsilon_greedy(q_values, eps, nb_actions)\n",
    "            obs, reward, done, info = env.step(action)\n",
    "            score += reward\n",
    "            if render:\n",
    "                env.render()\n",
    "                time.sleep(sleep_time)\n",
    "                if done:\n",
    "                    time.sleep(1)\n",
    "        scores.append(score)\n",
    "        # print('{}/{}: {}'.format(i+1, n_games, score))\n",
    "        env.close()\n",
    "    return scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Collecting scores\n",
    "\n",
    "Run 100 games without rendering and collect necessary statistics for final score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Mean score:  448.8\n",
      "\n",
      "Max score:  3250.0\n",
      "\n",
      "Fifth percentile:  862.0\n",
      "\n",
      "Percentiles:\n",
      "[80.0, 250.0, 330.0, 450.0, 3250.0]\n"
     ]
    }
   ],
   "source": [
    "ngames = 100\n",
    "eps = 0.05\n",
    "render = False\n",
    "\n",
    "scores = test_dqn(ngames, online_network, eps=eps, render=render)\n",
    "\n",
    "print('\\nMean score: ', np.mean(scores))\n",
    "print('\\nMax score: ', np.max(scores))\n",
    "print('\\nFifth percentile: ',np.percentile(scores, 95))\n",
    "print('\\nPercentiles:')\n",
    "print([ np.percentile(scores, p) for p in [0, 25, 50, 75, 100] ])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Rendering\n",
    "\n",
    "Play 3 more times with rendering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Mean score:  320.0\n",
      "\n",
      "Max score:  540.0\n",
      "\n",
      "Percentiles:\n",
      "[100.0, 250.0, 260.0, 450.0, 540.0]\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "ngames = 5\n",
    "eps = 0.05\n",
    "render = True\n",
    "\n",
    "scores = test_dqn(ngames, online_network, eps=eps, render=render)\n",
    "\n",
    "print('\\nMean score: ', np.mean(scores))\n",
    "print('\\nMax score: ', np.max(scores))\n",
    "print('\\nPercentiles:')\n",
    "print([ np.percentile(scores, p) for p in [0, 25, 50, 75, 100] ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
