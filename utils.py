import torch 
from policy import Policy
import mujoco
import mujoco_viewer as mjv
from typing import Tuple
import numpy as np


np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)


# Global Parameters
gamma = 0.99
model = mujoco.MjModel.from_xml_path('nav.xml')
data = mujoco.MjData(model)
dt = 0.1
epsilon = 0.2
goal = np.array([0.9, 0, 0, 0])
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


def transition(state: np.ndarray, action: np.ndarray, noise = 0.01) -> Tuple[torch.Tensor, bool]:
    '''
    Ensure state and action are numpy arrays!
    '''
    global model, data

    q, qdot = state[:2], state[2:]
    data.qpos[:] = q.copy()
    data.qvel[:] = qdot.copy()
    mujoco.mj_forward(model, data)  # Ensure the state is initialized correctly
    data.time = 0

    timestep = model.opt.timestep  # Get the model's timestep
    num_steps = int(dt / timestep)  # Calculate number of steps needed

    collides = False

    # print(f"Before q={data.qpos}, qdot={data.qvel}, ctrl={data.ctrl}") DEBUG 
    # Define noisy control clipped w/in limits:
    noise = np.random.normal([0,0], [noise, noise])
    ctrl = action + noise

    for _ in range(num_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if data.ncon > 0:
            collides = True
        
    # print(f"After: q={data.qpos}, qdot={data.qvel}\n")

    q, qdot = torch.Tensor(data.qpos.copy()), torch.Tensor(data.qvel.copy())
    new = torch.cat((q, qdot)).float()

    return new, collides




def init_viewer() -> mjv.MujocoViewer:
    # Global viewer for rendering our policy in action
    viewer = mjv.MujocoViewer(model, data)
    # Set camera parameters for bird's-eye view
    viewer.cam.azimuth = 0  # Horizontal angle
    viewer.cam.elevation = -60  # Directly above, looking down
    viewer.cam.distance = 2  # Distance from the ground; adjust as needed
    viewer.cam.lookat[:] = [0.2, 0, 0]  # Center of the scene (adjust if needed)

    return viewer


def visualize_policy(policy: Policy, max_len = 75):
    '''
    Given a policy:
     - Sample s_0 uniformly in C_free
     - set a finite horizon in order to avoid trajectory getting too long
     - Render instead of storing the trajectory

     return list of lists where each elt = [s_t, a_t, s_t+1, r_t, logprob(a_t|s_t)]
    '''
    global model, data
    viewer = init_viewer()

    # Sample first state uniformly s.t. it's not colliding
    s0 = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)

    terminal_s = False

    state, steps = s0, 0

    timestep = model.opt.timestep  # Get the model's timestep
    num_steps = int(dt / timestep)  # Calculate number of steps needed

    while not terminal_s and steps < max_len:
        # Sample an action
        action, _ = policy.sample(state)
        
        # Set mujoco state and control
        t = state.numpy()
        q, qdot = t[:2], t[2:]
        data.qpos[:] = q.copy()
        data.qvel[:] = qdot.copy()
        mujoco.mj_forward(model, data)  # Ensure the state is initialized correctly
        data.time = 0

        # Define noisy control clipped w/in limits:
        ctrl = np.clip(action.detach().numpy().copy() + np.random.normal([0, 0], [0.1, 0.1]), -1, 1)

        for _ in range(num_steps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            viewer.render()
        
        # Get new state
        q, qdot = torch.Tensor(data.qpos.copy()), torch.Tensor(data.qvel.copy())
        state = torch.cat((q, qdot)).float()

        # increment number of steps
        steps+=1
    
    viewer.close()


def evaluate_policy(pi, trials = 100, maxsteps = 100, epsilon=0.15):
    '''
    Evaluates policy pi by doing trials number of rollouts each with maxsteps number of possible steps per rollout.

    We return the success rate, average reward per step, and average reward per episode
    '''
    successes = 0
    sum_reward = 0
    total_steps = 0

    with torch.no_grad():
        for _ in range(trials):
            state = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)
            done = False
            collided = False
            steps = 0

            while not done and steps < maxsteps:
                action, _ = pi.sample(state)
                next_state, collided = transition(state, action)
                r_next = reward(next_state, collided) # Should I do r_t+1 or r_t?
                done = (torch.norm(state - goal).item() < epsilon)

                # Update state, return, and step count
                state = next_state
                steps +=1
                sum_reward += r_next

                if r_next >= 1 - collision_penalty: 
                    successes += 1
                    done = True
            
            total_steps += steps
    
    success_rate = float(successes) / trials
    avg_reward_per_step = sum_reward / total_steps
    avg_reward_per_episode = sum_reward / trials

    return success_rate, avg_reward_per_step, avg_reward_per_episode


def reward(state: torch.tensor, collided: bool=False) -> float:
    r = 0
    if isinstance(state, np.ndarray):
        state = torch.tensor(state)
    dist_to_goal = torch.norm(state - goal).item()
    if dist_to_goal <= epsilon:
        r+=1
    
    if collided:
        r-= collision_penalty
    return r


def dense_reward(state, collided, scale = 0.1, collision_penalty = 0) -> float:
    '''
    Gives a negative scaled reward based on distance to goal (i.e. as it gets farther it's a more negative reward).
    Gives a much higher collision penalty than our sparse reward

    Assigns infinite sum reward upon getting within epsilon of goal
    '''
    r = 0
    dist_to_goal = np.linalg.norm(state - goal).item()

    # Reward for being within epsilon of the goal
    if dist_to_goal <= epsilon:
        r += (1 / (1 - gamma))  # Infinite sum reward
    else:
        # Dense reward based on distance to goal
        r += 1 / (1 + (dist_to_goal*scale))

    # Penalty for collisions
    if collided:
        r -= collision_penalty

    return r




    