import numpy as np
import mujoco
import mujoco_viewer as mjv
from typing import Union, Tuple

class Node:
    def __init__(self, q: np.array,  qdot:np.array = np.array([0, 0]), parent=None) -> None:
        self.q = q
        self.qdot = qdot
        self.parent = parent
        self.ctrl = None

    
    def __repr__(self) -> str:
        return f'Node(q = {self.q}, qdot = {self.qdot}, ctrl = {self.ctrl})'

    def __eq__(self, other):
        return np.array_equal(self.q, other.q) and np.array_equal(self.qdot, other.qdot) and (self.parent == other.parent) and np.array_equal(self.ctrl, other.ctrl)


def weighted_euclidean_distance(x1: Node, x2: Node, pw: float = 1, vw: float = 0.1) -> float:
    '''
    Computes weighted euclidean distance between nodes x1 and x2 with pw being the position weight and vw the velocity weight
    Since for this application we mainly want to reach goal region, don't care as much about velocity and weigh position more
    '''
    pos_diff = np.linalg.norm(x2.q - x1.q)
    v_diff = np.linalg.norm(x2.qdot - x1.qdot)
    weighted_dist = np.sqrt(pw*pos_diff**2 + vw*v_diff**2)
    return weighted_dist



def ctrl_effort_distance(x1: Node, x2: Node) -> float:
    raise NotImplementedError

def sample_state(bounds: np.array)->Node:
    '''
    Uniformly samples: 
        - configuration within bounds  
        - qdot from standard normal
        - Returns a node with this data and uninitialized parent
    '''
    # Bounds have the shape (q, 2) -> each row is for a q_i has lower and upper bound for q_i
    q = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])

    # Return qdot from a standard normal population with same shape as q
    qdot = np.random.normal(size=q.shape)

    return Node(q=q, qdot=qdot)


def sample_non_colliding(sampler_fn, collision_checker, sample_bounds):
    '''
    A generic function that takes in a sampler and collision checker as function pointers and continues sampling with
    The sampler until it gets to a non-colliding state.

    sampler_fn should only need bounds as argument (everything else either keyword argument or not provided here)

    collision checker should take in output of sampler function and return True if no collision, False if collision
    '''
    while True:
        sample = sampler_fn(sample_bounds)
        # If no collision from sample -> return this
        if collision_checker(sample):
            break
    
    return sample

def sample_ctrl(ctrl_lim)->np.array:
    '''
    Uniformly samples an fx, fy control within the defined control limits. Returns as a vector of shape (2,)
    '''
    return np.random.uniform(ctrl_lim[0], ctrl_lim[1], size=(2,))



    