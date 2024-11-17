import torch 
from dataclasses import dataclass
import mujoco
import mujoco_viewer as mjv
from typing import Union, Tuple


@dataclass
class State:
    x:float
    y:float
    xdot:float
    ydot:float

    def to_tensor(self) -> torch.Tensor:
        return torch.Tensor([self.x, self.y, self.xdot, self.ydot])
    
    def dist(self, other: 'State') -> float:
        s1, s2 = self.to_tensor(), other.to_tensor()
        return torch.linalg.norm(s1 - s2)
    
    @classmethod
    def from_tensor(data: torch.Tensor) -> 'State':
        return State(data[0].item(), data[1].item(), data[2].item(), data[3].item())

# Global Parameters
gamma = 0.99
model = mujoco.MjModel.from_xml_path('nav.xml')
data = mujoco.MjData(model)
dt = 0.1
epsilon = 0.1
goal = State(0.9, 0, 0, 0)
collision_penalty = 1e-3 # add a small collision penalty so it learns that its a bad behavior
sample_bounds = [[-.2, 1.1], [-.36, .36]]


def sample_state(bounds: torch.Tensor)->State:
    '''
    Uniformly samples: 
        - q within bounds  
        - qdot is zero
    '''
    # Bounds have the shape (q, 2) -> each row is for a q_i has lower and upper bound for q_i
    q = torch.uniform(low=bounds[:, 0], high=bounds[:, 1])

    # Return qdot from a standard normal population with same shape as q
    qdot = torch.zeros_like(q)

    return State.from_tensor(torch.cat((q,qdot)))


def sample_non_colliding(sampler_fn, collision_checker, sample_bounds):
    '''
    A generic function that takes in a sampler and collision checker as function pointers and continues sampling with
    The sampler until it gets to a non-colliding state.

    sampler_fn should only need bounds as argument (everything else either keyword argument or not provided here)

    collision checker should take in output of sampler function and return False if no collision, True if collision
    '''
    while True:
        sample = sampler_fn(sample_bounds)
        # If no collision from sample -> return this
        if not collision_checker(sample):
            break
    
    return sample


def is_colliding(state: State) -> bool:
    # Set the position to the state
    data.qpos[:] = state.to_tensor()[:2].numpy()
    data.qvel[:] = [0,0]

    mujoco.mj_forward(model, data) # step the simulation to update collision data

    return data.ncon == 0


def transition(state: State, action: torch.Tensor) -> Tuple[State, bool]:
    # Set mujoco state and control
    t = state.to_tensor().numpy()
    q, qdot = t[:2], t[2:]
    data.qpos[:] = q
    data.qvel[:] = qdot
    data.ctrl = action.numpy()
    data.time = 0

    collides = False

    while data.time < dt:
        mujoco.mj_step(model, data)

        if data.ncon > 0:
            collides = True
        
    q, qdot = torch.Tensor(data.qpos.copy()), torch.Tensor(data.qvel.copy())
    new = State.from_tensor(torch.cat(q, qdot))

    return new, collides

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






    