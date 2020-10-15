#Importing Packages
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#Helper function which initializes the hidden layer weights, preventing them from exploding or vanishing
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

#Actor model
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        #Initializing 3 hidden layers, taking in the state size as input and outputting the action size
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
        #Resetting the parameters that were initialized above by calling the helper function 'hidden_init'
        self.reset_parameters()
    
    #Initialization function
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    #Does a forward pass through the Actor model
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        #Making use of a tanh activation function to return a value between -1 and 1
        return F.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        #Initializing hidden layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.reset_parameters()
       
    #Initialization function
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        
        #Concatenating the action to get the value between the state and action pair
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        
        #Returns a single value output
        return self.fc3(x)