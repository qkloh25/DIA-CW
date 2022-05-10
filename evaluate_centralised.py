from enviroments import *
from buffers import *
from agents import *

import pygame
import numpy as np
import time
import math
import torch
import json


nb_agents = 2
best_score = 0
# Create the environment, agent, memory buffers, displaying screen.
screen = pygame.display.set_mode((WINDOW_PIXELS, WINDOW_PIXELS))
env  = GridworldMultiAgentv35(nb_agents = nb_agents, screen=screen, nb_steps=25, seed = 3)
state_dim = env.observation_space.shape[0]
agent = DQN_agent_centralised(input_dim=state_dim, output_dim=25)
agent.load_checkpoint()

# Variables for training loops
N_GAMES = 10000
PRINT_INTERVAL = 1000
score_history = []
total_steps = 0
render = True
# Training loops
for i in range(N_GAMES):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(obs, eps_threshold=0)
        obs_, reward, done, info = env.step(action)
        if render:
            env.render()
            time.sleep(0.5) # to slow down the action for the video
        obs = obs_
        score += reward
        total_steps += 1
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print('episode', i, 'average score {:.1f}'.format(avg_score))

with open("DQN_centralised_evaluate","w") as fp:
    json.dump(score_history, fp)
