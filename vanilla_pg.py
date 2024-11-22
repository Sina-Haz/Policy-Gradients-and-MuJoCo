from utils import *
from value import *
from policy import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import mujoco_viewer as mjv
import matplotlib.pyplot as plt

# Global parameters of our vanilla policy gradient (and possibly other algorithms)
policy_hidden, value_hidden = 12, 12
num_trajectories = 1
replay_capacity = 500
value_bs, policy_bs = 8, 32
v_lr = 1e-3
p_lr = 1e-3
state_shape = (4, )
action_shape = (2, )

torch.autograd.set_detect_anomaly(True)


def step_env(state, pi):
    '''
    Given a state step the environment by one timestep and return:
    current state, action, next_state, next_reward, logprob
    '''
    action, logprobs = pi.sample(state)
    next_state, collided = transition(state, action)
    next_reward = reward(state, collided)

    return state, action, next_state, next_reward, logprobs




def optimize_value(optimizer, V, states, gains):
    '''
    Train the value based on what it thinks the value of the state is vs. the actual gain (future reward)
    '''
    # # Ensure states and gains are tensors

    # Predict the values for each state and squeeze them from 2D tensor (batch_size, 1) to -> (batch_size, )
    predicted_values = V(states).squeeze()
    targets = gains.detach() # No gradient should flow through gains

    # compute huber loss for moderate gradients in case there's a big mismatch b/w predicted and gain
    criterion = nn.HuberLoss()
    loss = criterion(predicted_values, targets)


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
    # No gradient computation for Value network when optimizing policy
    with torch.no_grad():
        baseline = V(state).squeeze() # squeeze into 1D tensor

    # Compute the advantage:
    advantage = gains.detach() - baseline.detach()

    # Compute policy gradient as the sum of logprobs weighted by advantage
    pg = (logprobs * advantage.unsqueeze(1)).mean()

    # negate it so that we do gradient ascent on this gradient instead of descent
    neg_pg = -1 * pg

    # Backward pass and update step
    optimizer.zero_grad()
    neg_pg.backward()
    optimizer.step()

    return pg.mean().item()

# def optimize_actor_critic(actor_opt, critic_opt, V, state, gains, logprobs):
#     print(f'shapes, state: {state.shape}, gains: {gains.shape}, logprobs:{logprobs.shape}')
#     # First compute critic loss
#     pred = V(state)
#     targets = gains.detach()

#     # compute huber loss for moderate gradients in case there's a big mismatch b/w predicted and gain
#     criterion = nn.HuberLoss()
#     critic_loss = criterion(pred, targets)

#     # Optimize the critic
#     critic_opt.zero_grad()
#     critic_loss.backward()
#     critic_opt.step()

#     # compute advantage to weight with logprobs
#     advantage = (gains - pred).detach()
#     print(advantage.shape)
#     actor_loss = (-logprobs * advantage).sum()
#     actor_opt.zero_grad()
#     actor_loss.backward()
#     actor_opt.step()

#     return critic_loss, actor_loss


def vanilla_pg(N, decay_variance = False, nvb = 6, npb = 2, actor_fpath = 'td_models/actor.pth', critic_fpath = 'td_models/critic.pth'):
    # Initialize random policy and value functions
    V = Value(value_hidden)
    
    if decay_variance: pi = Policy(policy_hidden, var = 0.5)
    else: pi = Policy(policy_hidden)

    V_optim = optim.Adam(V.parameters(), lr = v_lr)
    pi_optim = optim.Adam(pi.parameters(), lr = p_lr)
    mem = ReplayMemory(maxlen=replay_capacity)

    for i in range(N):
        # Do a few rollouts with the current policy and store it in the memory buffer
        for _ in range(num_trajectories):
            traj = rollout(pi)
            traj = compute_gain(traj)
            mem.add_trajectory(traj)
        
        # vloader, ploader = DataLoader(mem, batch_size=value_bs, shuffle=True), DataLoader(mem, batch_size=policy_bs, shuffle=True)
        # loader = DataLoader(mem, batch_size=value_bs, shuffle=True)
  
        # Optimize value function using a limited number of batches from DataLoader
        for batch_idx, batch in enumerate(vloader):
            if batch_idx >= nvb:
                break
            states = batch['state']
            gains = batch['gains']
            V_loss = optimize_value(V_optim, V, states, gains)
        

        # PROBLEM HAPPENING HERE:
        # Use the same limited batches for policy updates
        for batch_idx, batch in enumerate(ploader):
            if batch_idx >= npb:
                break
            states = batch['state']
            gains = batch['gains']
            logprobs = batch['logprobs']
            pi_loss = optimize_policy(pi_optim, V, states, gains, logprobs)

        if decay_variance: pi.decay_variance()
        if i % 100 ==0: 
            print('value loss:', V_loss)
            print('policy gradient:', pi_loss)
            torch.save(pi.state_dict(), actor_fpath)
            torch.save(V.state_dict(), critic_fpath)

    return pi, V

def policy_gradient_mc(V, pi, episodes, max_steps=200, actor_fpath='actor_mc.pth', critic_fpath='critic_mc.pth', decay=False):
    """
    Trains an actor and critic using the A2C algorithm with Monte Carlo estimation for advantage.
    Advantage = (future discounted reward - value).
    """
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    optimizer_critic = optim.AdamW(V.parameters(), lr=v_lr)

    for episode in range(1, episodes + 1):
        # Sample initial state and initialize episode return
        state = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)
        ep_return = 0
        done = False
        step_count = 0
        collided = False

        # Stores rewards and states for Monte Carlo return calculation
        episode_rewards = []
        episode_states = []
        episode_logprobs = []

        while not done and step_count < max_steps:
            # Given current state compute current reward, action, next state, and terminal condition
            state, action, next_state, next_reward, logprob = step_env(state, pi)

            # Record state, reward, and logprobs for MC estimation
            episode_states.append(state)
            episode_rewards.append(next_reward)
            episode_logprobs.append(logprob)

            # Update state, step count, and episode return
            state = next_state
            step_count += 1

        # Calculate future discounted rewards for the episode
        returns = deque()
        future_reward = 0
        for r in reversed(episode_rewards):
            future_reward = r + gamma * future_reward
            returns.appendleft(future_reward)
        returns = torch.tensor(returns)

        # Update actor and critic
        for t in range(len(episode_states)):
            state = episode_states[t]
            G_t = returns[t]
            value = V(state)

            # Compute advantage using Monte Carlo method
            advantage = G_t - value

            # Update critic using Huber loss
            critic_loss = F.smooth_l1_loss(value, G_t.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Update actor
            logprobs = episode_logprobs[t]
            actor_loss = (-logprobs * advantage.detach()).sum()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

        # Print episode statistics
        if episode % 100 == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = eval_policy(pi, trials=50)
            print(f'Episode: {episode}, Success rate: {success_rate}, '
                  f'average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')
            if decay:
                pi.decay_variance()

    # Save models
    torch.save(pi.state_dict(), actor_fpath)
    torch.save(V.state_dict(), critic_fpath)



def policy_gradient_td(V, pi, episodes, max_steps = 200, actor_fpath = 'actor.pth', critic_fpath = 'critic.pth', decay = False):
    '''
    Unlike the above function vanilla policy gradient which uses a monte-carlo method to estimate the advantage and
    to update the value function. This function is completely online and uses SGD (instead of offline and BGD) to
    update the actor and critic.

    Returns some statistics on the actor loss, critic loss, and returns
    '''
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    optimizer_critic = optim.AdamW(V.parameters(), lr=v_lr)

    for episode in range(1, episodes+1):
        # Sample initial state and initialize episode return
        state = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)
        ep_return = 0
        done = False
        step_count = 0
        collided = False

        while not done and step_count < max_steps:
            # Given current state compute current reward, action, next state, and terminal condition
            action, logprobs = pi.sample(state)
            next_state, collided = transition(state, action)
            r_next = reward(next_state, collided) # Should I do r_t+1 or r_t?
            done = (torch.norm(state - goal).item() < epsilon)

            # Compute the TD target (bootstrapping)
            value = V(state)
            next_value = V(next_state)
            td_target = r_next + gamma * next_value * (1 - done)

            # Based on td_target (basically q-value) and current estimated value, we estimate advantage
            advantage = td_target - value

            # We update the critic with HuberLoss
            critic_loss = F.smooth_l1_loss(value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # We update the actor based on logprobs weighted by advantage
            actor_loss = (- logprobs * advantage.detach()).sum()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Update state, return, and step count
            state = next_state
            step_count +=1
            ep_return += r_next

        # Print episode statistics
        if episode % 100 == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = eval_policy(pi, trials=50)
            print(f'Episode: {episode}, Success rate: {success_rate}, average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')
            # print(f"Episode {episode}: Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Return: {ep_return}, Steps: {step_count}")
            if decay: pi.decay_variance()

    torch.save(pi.state_dict(), actor_fpath)
    torch.save(V.state_dict(), critic_fpath)



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


def eval_policy(pi, trials = 100, maxsteps = 100):
    '''
    Evaluates policy pi by doing trials number of rollouts each with maxsteps number of possible steps per rollout.

    We return the success rate, average reward per step, and average reward per episode
    '''
    successes = 0
    sum_reward = 0
    total_steps = 0

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

            if r_next >= 1 - collision_penalty: successes += 1
        
        total_steps += steps
    
    success_rate = float(successes) / trials
    avg_reward_per_step = sum_reward / total_steps
    avg_reward_per_episode = sum_reward / trials

    return success_rate, avg_reward_per_step, avg_reward_per_episode








if __name__ == '__main__':
    V, pi = Value(value_hidden), Policy(policy_hidden, var=0.1, var_decay=.975)
    # V.load_state_dict(torch.load('models/critic.pth'))
    # pi.load_state_dict(torch.load('models/actor.pth'))
    # V.eval()
    # pi.eval()
    # stats = policy_gradient_td(V, pi, 5000, actor_fpath='models/actor_dv.pth', critic_fpath='models/critic_dv.pth', max_steps=100, decay=True)
    # visualize_policy(pi)
    # success_rate, avg_reward_per_step, avg_reward_per_episode = eval_policy(pi)
    # print(f'Success rate: {success_rate}, average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')

        

        
    


        

        
        



