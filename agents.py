import torch
import numpy as np
import torch.nn as nn
import math
from networks import *
from buffers import *

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5_000_000

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class DQN_agent_centralised:
    def __init__(self, input_dim, output_dim, chkpt_dir = "tmp/DQN_agent_centralised/", internal_nodes = [32,16], lr = 0.001, gamma = 0.99, tau = 0.01):
        self.policy = DQN(input_dim = input_dim, internal_nodes=internal_nodes, output_dim=output_dim, lr = lr, name = "policy", chkpt_dir = chkpt_dir)
        self.policy.apply(init_weights)
        self.target_policy = DQN(input_dim = input_dim, internal_nodes=internal_nodes, output_dim=output_dim, name = "target", chkpt_dir = chkpt_dir)
        self.gamma = gamma
        self.tau = tau

        self.update_network_parameters(tau = 1)

    def choose_action(self, observation, eps_threshold=0.1):
        input = torch.tensor(np.array([observation]), dtype=torch.float).to(self.policy.device)
        q_value = self.policy.forward(input)
        q_value = q_value.detach().cpu().numpy()[0]

        if random.random() < eps_threshold:
            return random.randint(0,len(q_value)-1)
        
        return np.argmax(q_value)

    def update_network_parameters(self, tau=None): 
        if tau is None:
            tau = self.tau

        target_policy_params = self.target_policy.named_parameters()
        policy_params = self.policy.named_parameters()

        target_policy_state_dict = dict(target_policy_params)
        policy_state_dict = dict(policy_params)
        for name in policy_state_dict:
            policy_state_dict[name] = tau*policy_state_dict[name].clone() + \
                    (1-tau)*target_policy_state_dict[name].clone()

        self.target_policy.load_state_dict(policy_state_dict)

    def learn(self, memory):
        if not memory.ready():
            return
        states, actions, states_, rewards, dones  = memory.sample()

        device = self.policy.device
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # Compute Targets
        target_q_values = self.target_policy(states_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0].detach()
        expected_q_values = max_target_q_values * self.gamma * (1-dones) + rewards

        # Compute Loss
        criterion = nn.MSELoss()
        q_values = self.policy(states)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions)
        loss = criterion(action_q_values, expected_q_values)

        self.policy.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()
        self.update_network_parameters()


    def save_checkpoint(self):
        self.policy.save_checkpoint()
        self.target_policy.save_checkpoint()

    def load_checkpoint(self):
        self.policy.load_checkpoint()
        self.target_policy.load_checkpoint()

class MAA2C_agent:
    def __init__(self, input_dim, output_dim, n_agents, chkpt_dir = "tmp/DQN_agent_centralised/", internal_nodes = [32,16], lr = 0.001, gamma = 0.99, tau = 0.01):
        self.gamma = gamma
        self.tau = tau
        self.output_dim = output_dim

        self.actor = ActorNetwork(input_dims=input_dim, fc1_dims = internal_nodes[0], fc2_dims = internal_nodes[1], n_actions = output_dim, 
                                  chkpt_dir=chkpt_dir,  name='actor')
        self.actor.apply(init_weights)
                                  
        self.critic = CriticNetwork(input_dims=input_dim,  fc1_dims = internal_nodes[0], fc2_dims= internal_nodes[1], n_agents = n_agents, n_actions= output_dim, 
                            chkpt_dir=chkpt_dir, name='critic')
        self.critic.apply(init_weights)
                    
        self.target_actor = ActorNetwork(input_dims=input_dim, fc1_dims = internal_nodes[0], fc2_dims = internal_nodes[1], n_actions = output_dim, 
                                  chkpt_dir=chkpt_dir,  name='target_actor')

        self.target_critic = CriticNetwork(input_dims=input_dim,  fc1_dims = internal_nodes[0], fc2_dims= internal_nodes[1], n_agents = n_agents, n_actions= output_dim, 
                            chkpt_dir=chkpt_dir, name='target_critic')

        self.update_network_parameters(tau = 1)

    def choose_action(self, observation, eps_threshold=0.1):
        input = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)
        act = self.actor.forward(input)
        act = act.detach().cpu().numpy()[0]

        if random.random() < eps_threshold:
            return torch.rand(self.output_dim).numpy()
        
        return act

    def update_network_parameters(self, tau=None): 
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*target_actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*target_critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def learn(self, memory, common_new_actions, common_actions):
        if not memory.ready():
            return
        states, actions, states_, rewards, dones  = memory.sample()

        device = self.actor.device
        common_actions = torch.tensor(common_actions, dtype=torch.float).to(device)
        common_new_actions = torch.tensor(common_new_actions, dtype=torch.float).to(device)
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)
        
        target_q_values = self.target_critic(states_, common_new_actions)
        expected_q_values = target_q_values * self.gamma * (1-dones) + rewards

        # Compute Loss
        criterion = nn.MSELoss()
        q_values = self.critic(states, common_actions)
        critic_loss = criterion(q_values, expected_q_values)

        self.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)

        actor_loss = -torch.mean(self.critic.forward(states, common_actions))
        actor_loss.backward(retain_graph=True)
        

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


