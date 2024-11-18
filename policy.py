from typing import Any
from utils import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Policy(nn.Module):
    '''
    Learns a parametrized actor network that takes in state and returns mu_x, mu_y
    means of 2 gaussian's. We sample these gaussians to obtain action a_t = (u_x, u_y)
    '''
    def __init__(self, hidden_size, var, input_size = 2, output_size = 4, var_decay = 0.99) -> None:
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.var = var
        self.gamma = var_decay
    
    def forward(self, state: torch.Tensor):
        # Ensure State.to_tensor() is called before passing it to forward
        out1 = F.leaky_relu(self.layer1(state))
        out2 = F.tanh(self.layer2(out1))
        return out2
    
    def decay_variance(self):
        self.var *= self.gamma
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)






def reward(state: State, collided: bool =False) -> float:
    r = 0
    if state.dist(goal) <= epsilon:
        r+=1
    
    if collided:
        r-= collision_penalty

    return r

def rollout(policy):
    # Sample first state uniformly s.t. it's not colliding
    s0 = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)
