from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class Value(nn.Module):
    '''
    Learns a parametrized critic network that takes in state and returns v(s),
    which is the expected future reward from the state
    '''
    def __init__(self, hidden_size, input_size = 4, output_size = 1) -> None:
        super(Value, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()

    
    def forward(self, state: torch.Tensor):
        out1 = self.relu(self.layer1(state))
        v_s = self.layer2(out1)
        return v_s
    



def compute_gain(traj, gamma = 0.99):
    '''
    Compute gain and add it in place as the last element to each element in trajectories
    returns overwritten trajectory
    '''

    rewards = [x[3] for x in traj]
    T = len(rewards)
    G = 0

    for t in range(T-1, -1, -1):
        #Gt = rt + gamma * Gt+1
        G = rewards[t] + gamma * G
        traj[t].append(G)
    return traj





#average reward per step as a function of the number of episodes used for training
#will need to track total_reward and total_steps
def avg_reward_per_step(traj, total_reward, total_steps):

    ep_reward = sum(x[3] for x in traj)
    total_reward += ep_reward
    total_steps += len(traj)
    avg_r_per_step = total_reward / total_steps

    #if we want to plot curve, we can add the avg_r_per_step of each episode to a list
    return avg_r_per_step, total_reward, total_steps
