from collections import namedtuple, deque
from model import Actor, Critic

import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.       # L2 weight decay

N_LEARN_UPDATES = 10
N_LEARN_TIMESTEPS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, n_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(random_seed)
        
        #Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        #Critic Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        #Noise Process
        self.noise = OUNoise((n_agents, action_size), random_seed)
        
        #Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def step(self, state, action, reward, next_state, done, timestep):
        
        #Save Memory
        for state, action, reward, next_state, done in zip(state, action, reward, next_state, done):
            self.memory.add(state, action, reward, next_state, done)
            
        if timestep % N_LEARN_TIMESTEPS != 0:
            return
        
        #IF enough samples in memory
        if len(self.memory) > BATCH_SIZE:
            for i in range(N_LEARN_UPDATES):
                #Load sample of tuples from memory
                experiences = self.memory.sample()

                #Learn from a randomly selected sample
                self.learn(experiences, GAMMA)
            
    def act(self, state, add_noise=True):
        
        
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            action=self.actor_local(state).cpu().data.numpy()
        
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
            
        #Return action    
        return np.clip(action, -1, 1)
    
    def reset(self):
        
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        #Get predicted actions + Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        #Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        #Critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        #Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        #Actor Loss
        actions_pred = self.actor_local(states)
        
        #Negative sign for gradient ascent
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        #Minimize Loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau) * target_param.data)
    
class OUNoise:
    #Ornstein-Uhlenbeck process
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        #Initialize parameters and noise process
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        #Reset the internal state (= noise) to mean (mu)
        self.state = copy.copy(self.mu)

    def sample(self):
        #Update internal state and return it as a noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

#Fixed-size buffer to store experience tuples
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        #Initialize a ReplayBuffer object
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        #Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        #Randomly sample a batch of experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        #Return the current size of internal memory
        return len(self.memory)