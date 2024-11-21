from utils import *
from value import *
from policy import *
from typing import List
import torch.optim as optim
import torch.nn.functional as F
import mujoco_viewer as mjv

# Global parameters of our vanilla policy gradient (and possibly other algorithms)
policy_hidden, value_hidden = 10, 10
num_trajectories = 3
replay_capacity = 500
value_bs, policy_bs = 12, 16
v_lr = 1e-3
p_lr = 1e-4

torch.autograd.set_detect_anomaly(True)


def optimize_value(optimizer, V, states, gains):
    '''
    Train the value based on what it thinks the value of the state is vs. the actual gain (future reward)
    '''
    # Ensure states and gains are tensors
    states = torch.stack(states) # stack the states b/c its a tuple of tensors
    gains = torch.tensor(gains, dtype=torch.float32)

    # Predict the values for each state and squeeze them from 2D tensor (batch_size, 1) to -> (batch_size, )
    predicted_values = V(states).squeeze()

    # compute huber loss for moderate gradients in case there's a big mismatch b/w predicted and gain
    loss = F.huber_loss(predicted_values, gains)

    # Backward pass and update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.mean().item()


def optimize_policy(optimizer, V, state, gains, logprobs):
    '''
    Optimize the policy by using the following gradient:
    We weight the log probabilities of the actions it did by the advantage incurred by those actions

    We do this offline as our stochastic policy may predict a different action than it did online and if so
    we would have to use environment transitions to assess reward for these actions and then use bootstrapping to
    estimate Q^pi. Instead here we use gain as Q^pi and subtract it by value of next state to get 
    '''
    # Firstly we ensure everything is a tensor
    state = torch.stack(state) # stack the states b/c its a tuple of tensors
    gains = torch.tensor(gains, dtype=torch.float32)
    logprobs = torch.stack(logprobs)

    # Compute the advantage:
    advantage = gains - V(state).squeeze().clone()

    # Compute policy gradient as the sum of logprobs weighted by advantage
    pg = (logprobs * advantage.unsqueeze(1)).sum()

    # negate it so that we do gradient ascent on this gradient instead of descent
    neg_pg = -1 * pg

    # Backward pass and update step
    optimizer.zero_grad()
    neg_pg.backward()
    optimizer.step()

    return pg.mean().item()





def vanilla_pg(N, decay_variance = False):
    # Initialize random policy and value functions
    V = Value(value_hidden)
    
    if decay_variance: pi = Policy(policy_hidden, var = 0.5)
    else: pi = Policy(policy_hidden)

    V_optim = optim.Adam(V.parameters(), lr = v_lr)
    pi_optim = optim.Adam(pi.parameters(), lr = p_lr)
    mem = ReplayMemory(replay_capacity)

    for i in range(N):
        # Do a few rollouts with the current policy and store it in the memory buffer
        for j in range(num_trajectories):
            traj = rollout(pi)
            traj = compute_gain(traj)
            mem.add_trajectory(traj)
        
        # Get a batch sample of transitions to use for updating our value function
        val_trans = mem.sample(value_bs)
        val_batch = Transition(*zip(*val_trans)) # reshapes data as you will see below
        mvl = optimize_value(V_optim, V, val_batch.state, val_batch.gain)
        if i % 100 == 0: print('value loss:', mvl)

        # Get a batch sample of transitions to use for optimizing policy
        policy_trans = mem.sample(policy_bs)
        p_batch = Transition(*zip(*policy_trans))
        mpg = optimize_policy(pi_optim, V, p_batch.state, p_batch.gain, p_batch.logprob)
        if i % 100 == 0: print('policy gradient:', mpg)

        if decay_variance: pi.decay_variance()

    return pi, V

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
        action, _ = policy.sample(state).unbind(0)
        
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

def save_model(model, fpath):
    torch.save(model.state_dict(), fpath)

def load_model(model, fpath):
    model.load_state_dict(torch.load(fpath))
    model.eval()




pi, _ = vanilla_pg(N = 10000)

visualize_policy(pi)

save_model(pi, 'pi_N=10k.pth')



        

        
    


        

        
        



