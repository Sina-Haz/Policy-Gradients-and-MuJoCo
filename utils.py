import torch 
from dataclasses import dataclass
import mujoco
import mujoco_viewer as mjv
from typing import Union, Tuple
import numpy as np

np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)


# Global Parameters
gamma = 0.99
model = mujoco.MjModel.from_xml_path('nav.xml')
data = mujoco.MjData(model)
dt = 0.1
epsilon = 0.1
goal = torch.Tensor([0.9, 0, 0, 0])
collision_penalty = 1e-3 # add a small collision penalty so it learns that its a bad behavior
sample_bounds = torch.Tensor([[-.2, 1.1], [-.36, .36]])


def sample_state(bounds: torch.Tensor) -> torch.tensor:
    """
    Uniformly samples:
        - q within bounds
        - qdot is zero
    """
    # Convert bounds to NumPy array
    bounds_np = bounds.numpy()

    # Sample q within bounds
    q = np.random.uniform(low=bounds_np[:, 0], high=bounds_np[:, 1])

    # Initialize qdot as zeros with the same shape as q
    qdot = np.zeros_like(q)

    # Combine q and qdot into a single tensor
    q_qdot = torch.tensor(np.concatenate((q, qdot))).float()
    return q_qdot


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


def is_colliding(state: torch.tensor) -> bool:
    global model, data
    # Set the position to the state
    data.qpos[:] = state[:2].numpy()
    data.qvel[:] = np.array([0,0])

    mujoco.mj_forward(model, data) # step the simulation to update collision data

    return data.ncon == 0


def transition(state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    global model, data
    # Set mujoco state and control
    t = state.numpy()
    q, qdot = t[:2], t[2:]
    data.qpos[:] = q.copy()
    data.qvel[:] = qdot.copy()
    mujoco.mj_forward(model, data)  # Ensure the state is initialized correctly
    data.time = 0

    timestep = model.opt.timestep  # Get the model's timestep
    num_steps = int(dt / timestep)  # Calculate number of steps needed

    collides = False

    # print(f"Before q={data.qpos}, qdot={data.qvel}, ctrl={data.ctrl}") DEBUG 
    # Define noisy control clipped w/in limits:
    ctrl = np.clip(action.detach().numpy().copy() + np.random.normal([0, 0], [0.1, 0.1]), -1, 1)

    for _ in range(num_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if data.ncon > 0:
            collides = True
        
    # print(f"After: q={data.qpos}, qdot={data.qvel}\n")

    q, qdot = torch.Tensor(data.qpos.copy()), torch.Tensor(data.qvel.copy())
    new = torch.cat((q, qdot)).float()

    return new, collides








    