import numpy as np
import random

class ReplayMemory(object):
    def __init__(self, state_dim, action_size, mem_size = 50000, batch_size = 1024):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.cntr = 0
        self.states = np.zeros((mem_size, state_dim))
        self.actions = np.zeros((mem_size, action_size))
        self.states_ = np.zeros((mem_size, state_dim))
        self.rewards = np.zeros((mem_size, 1))
        self.dones = np.zeros((mem_size, 1))
        

    def save(self, state, action, state_, reward, done):
        index = self.cntr % self.mem_size
        self.states[index] = np.array(state)
        self.actions[index] = action
        self.states_[index] = np.array(state_)
        self.rewards[index] = reward
        self.dones[index] = done
        self.cntr += 1
        

    def sample(self, batch_size = 1024):
        max_mem = min(self.cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return self.states[batch], self.actions[batch], self.states_[batch], self.rewards[batch], self.dones[batch]

    def ready(self):
        return self.cntr % self.batch_size == 0

    def __len__(self):
        return len(self.memory)