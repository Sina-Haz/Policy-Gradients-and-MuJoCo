from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class Value(nn.Module):
    '''
    Learns a parametrized critic network that takes in state and returns v(s),
    which is the expected future reward from the state
    '''
    def __init__(self, hidden_size, input_size = 2, output_size = 1) -> None:
        super(Value, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    
    def forward(self, state: torch.Tensor):
        out1 = F.leaky_relu(self.layer1(state))
        v_s = self.layer2(out1)
        return v_s
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

