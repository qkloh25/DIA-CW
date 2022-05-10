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
env  = GridworldMultiAgentv35(nb_agents = nb_agents, screen=screen, nb_steps=50, alpha = 10)
state_dim = env.observation_space.shape[0]
agent1 = MAA2C_agent(input_dim=state_dim, output_dim = 5, n_agents=2, chkpt_dir = "tmp/MAA2C_agent_1/")
agent2 = MAA2C_agent(input_dim=state_dim, output_dim = 5, n_agents=2,chkpt_dir = "tmp/MAA2C_agent_2/")
memory = ReplayMemory(state_dim = state_dim, action_size=10)



# Variables for training loops
N_GAMES = 1_000_000
PRINT_INTERVAL = 1000
score_history = []
total_steps = 0
# Training loops
for i in range(N_GAMES):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act1 = agent1.choose_action(obs)
        act2 = agent2.choose_action(obs)
        action1 = np.argmax(act1)
        action2 = np.argmax(act2)
        action = action1*5 + action2
        obs_, reward, done, info = env.step(action)
        memory.save(obs, np.concatenate((act1, act1)), obs_, reward, done)

        if i % (PRINT_INTERVAL*10) == 0:
            env.render()
            time.sleep(0.2) # to slow down the action for the video

        if memory.ready():
            states, actions, states_, rewards, dones  = memory.sample()
            states_ = torch.tensor(states_, dtype=torch.float).to(agent1.actor.device)
            common_new_actions = np.concatenate((agent1.actor.forward(states_).detach().numpy(), agent2.actor.forward(states_).detach().numpy()), axis = 1)

            states = torch.tensor(states_, dtype=torch.float).to(agent1.actor.device)
            common_actions = np.concatenate((agent1.actor.forward(states_).detach().numpy(), agent2.actor.forward(states_).detach().numpy()), axis = 1)
            
            
            agent1.learn(memory, common_new_actions, common_actions)
            agent2.learn(memory, common_new_actions, common_actions)
            agent1.critic.optimizer.step()
            agent2.critic.optimizer.step()
            agent1.actor.optimizer.step()
            agent2.actor.optimizer.step()
            agent1.update_network_parameters()
            agent2.update_network_parameters()


        obs = obs_
        score += reward
        total_steps += 1

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        print("......saving checkpoint......")
        agent1.save_checkpoint()
        agent2.save_checkpoint()
        best_score = avg_score
    if i % PRINT_INTERVAL == 0 and i > 0:
        print('episode', i, 'average score {:.1f}'.format(avg_score))


with open("MAA2C_train_score1","w") as fp:
    json.dump(score_history, fp)
