from typing import Any
from utils import *
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from collections import deque, namedtuple
import random


class Policy(nn.Module):
    '''
    Learns a parametrized actor network that takes in state (x_t, y_t, xdot_t, ydot_t) and returns mu_x, mu_y
    means of 2 gaussian's. We sample these gaussians to obtain action a_t = (u_x, u_y)
    '''
    def __init__(self, hidden_size, input_size = 4, output_size = 2, var = 0.1, var_decay = 0.99) -> None:
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.var = var
        self.gamma = var_decay
    
    def forward(self, state: torch.tensor):

        out1 = F.leaky_relu(self.layer1(state))
        out2 = F.tanh(self.layer2(out1))
        return out2
    
    def decay_variance(self):
        self.var *= self.gamma
    
    
    def sample(self, state: torch.tensor) -> torch.Tensor:
        '''
        This method sample's an action given a state.
        Does this by computing a predicted mean mu for each action DoF and samples
        From a gaussian using the mean and it's own set variance

        returns tensor of shape (2, action.shape[0]): first row is the action at each DoF, 2nd row is the log prob of those actions
        '''
        mu = self.forward(state)
        distr = dist.Normal(mu, self.var)

        action = torch.clip(distr.sample(), -1, 1).clone()
        # print(f'action: {action}, {action.shape}') DEBUG PRINT

        logprobs = distr.log_prob(action)
        # print(logprobs) DEBUG PRINT

        return torch.stack((action, logprobs))


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'logprob', 'gain'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, bs):
        return random.sample(self.memory, bs)

    def __len__(self):
        return len(self.memory)
    
    def add_trajectory(self, trajectory):
        '''
        Read in a whole trajectory into replay memory AFTER we compute the gain at each step
        '''
        for t in trajectory:
            s,a,s_n,r,lp,g = t
            self.push(s, a, s_n, r, lp, g)



def reward(state: torch.tensor, collided: bool=False) -> float:
    r = 0
    dist_to_goal = torch.norm(state - goal).item()
    if state.dist(goal) <= epsilon:
        r+=1
    
    if collided:
        r-= collision_penalty

    return r

def rollout(policy: Policy, max_len = 200):
    '''
    Given a policy:
     - Sample s_0 uniformly in C_free
     - set a finite horizon in order to avoid trajectory getting too long
     - At each timestep we store: (s_t, a_t, s_t+1, r_t) and logprob(a|s)

     return list of lists where each elt = [s_t, a_t, s_t+1, r_t, logprob(a_t|s_t)]
    '''
    # Sample first state uniformly s.t. it's not colliding
    s0 = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)

    traj, terminal_s = [], False
    # We define terminal state as the state where we are within epsilon of goal
    
    s_t, coll = s0, False
    while len(traj) < max_len and not terminal_s:
        # Collect reward given state and sample an action given state (and log prob)
        r_t = reward(s_t, collided=coll)
        a_t, logprob_a_t = policy.sample(s_t).unbind(0)

        # transition based on environment, current state and action
        s_t1, coll = transition(s_t, a_t)

        # Collect this trajectory
        traj.append([s_t, a_t, s_t1, r_t, logprob_a_t])

        # Determine if this upcoming state is a terminal state
        if torch.norm(s_t - goal).item() < epsilon: 
            terminal_s = True

        # reset current state
        s_t = s_t1

    return traj

if __name__ == '__main__':
    pi = Policy(12).float()

    state = torch.tensor([0.,0.,0.,0.])

    mu = pi(state)
    
    print(f'here is our mean prediction: {mu}')

    sampled = pi.sample(state)

    print(f'here is our sampled shape {sampled.shape} and here are the values {sampled}')

    logprobs = sampled[1,:].sum()

    logprobs.backward()

    traj = rollout(pi)
    print(f'Length of trajectory: {len(traj)}\n first two steps: {traj[:2]}')

    