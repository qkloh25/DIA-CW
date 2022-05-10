import os
from numpy import asfarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, internal_nodes, output_dim, optimizer = None, lr = 0.001, name ="DQN", chkpt_dir="tmp/DQN"):
        super(DQN, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        # Initilizing layers, internal_nodes is expected to have at least one element.
        self.fc1 = nn.Linear(input_dim, internal_nodes[0])
        if len(internal_nodes)>1:
            for i in range(len(internal_nodes)-1):
                self.linears = nn.ModuleList([nn.Linear(internal_nodes[i], internal_nodes[i+1])])
        self.head = nn.Linear(internal_nodes[-1], output_dim)

        # Initilizing optimizers
        if optimizer:
            self.optimizer = optimizer
        else:    
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initilizing the computing device.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.linears:
            for  linear in self.linears:
                x = F.relu(linear(x))
        x = self.head(x)

        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir, lr = 0.001):
        super(CriticNetwork, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims+n_actions*n_agents, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir, lr=0.001):
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = torch.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))
