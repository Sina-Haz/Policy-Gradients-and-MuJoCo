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
bs = 32
v_lr = 1e-3
p_lr = 1e-3
state_shape = (4, )
action_shape = (2, )

small_hidden = 6
large_hidden = 16

# torch.autograd.set_detect_anomaly(True)


def step_env(state, action, dense=False):
    '''
    Given a state and action pair take a step in the environment and return:
    next state, rewards, termination (i.e if stop condition reached)
    '''
    next_state, collided = transition(state, action)

    if dense: next_reward = dense_reward(state, collided)
    else: next_reward = reward(state, collided)

    done = bool((torch.norm(next_state - goal) < epsilon).item())

    return next_state, next_reward, done

def plot_training_stats(stats):
    episodes = range(50, 50 * len(stats['success_rate']) + 1, 50)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(episodes, stats['success_rate'], marker='o')
    plt.title('Success Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')

    plt.subplot(1, 3, 2)
    plt.plot(episodes, stats['average reward per step'], marker='o')
    plt.title('Average Reward Per Step')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.subplot(1, 3, 3)
    plt.plot(episodes, stats['average reward per episode'], marker='o')
    plt.title('Average Reward Per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.show()


def a2c_td(V, pi, episodes, max_steps = 100, actor_fpath = 'actor.pth', critic_fpath = 'critic.pth', decay = False, dense = False):
    '''
    Unlike the above function vanilla policy gradient which uses a monte-carlo method to estimate the advantage and
    to update the value function. This function is completely online and uses SGD (instead of offline and BGD) to
    update the actor and critic.

    Prints some evaluation metric on the policy every N number of episodes
    '''
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    optimizer_critic = optim.AdamW(V.parameters(), lr=v_lr)
    stats = {'success_rate':[], 'average reward per step':[], 'average reward per episode': []}


    for episode in range(1, episodes+1):
        # Sample initial state and initialize episode return
        state = sample_non_colliding(sampler_fn=sample_state, collision_checker=is_colliding, sample_bounds=sample_bounds)
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # Given current state compute current reward, action, next state, and terminal condition
            action, logprobs = pi.sample(state)
            next_state, r_next, done = step_env(state, action, dense=dense)

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
            actor_loss = (-logprobs * advantage.detach()).sum()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Update state, and step count
            state = next_state
            step_count +=1

        # Print episode statistics
        if episode % max_steps == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = evaluate_policy(pi, trials=50, maxsteps=max_steps)
            print(f'Episode: {episode}, Success rate: {success_rate}, average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')
            stats['success_rate'].append(success_rate); stats['average reward per step'].append(avg_reward_per_step); stats['average reward per episode'].append(avg_reward_per_episode)
            if decay: pi.decay_variance()
    plot_training_stats(stats)

    torch.save(pi.state_dict(), actor_fpath)
    torch.save(V.state_dict(), critic_fpath)


def a2c_mc(V, pi, episodes, max_steps = 100, actor_fpath = 'actor.pth', critic_fpath = 'critic.pth', decay = False, dense=False, lamda = 0):
    '''
    In this function we use monte-carlo advantage estimation for more stable/accurate learning. We also use batch gradient descent
    '''
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    optimizer_critic = optim.AdamW(V.parameters(), lr=v_lr)
    mem = ReplayMemory(maxlen=2*max_steps)
    stats = {'success_rate':[], 'average reward per step':[], 'average reward per episode': []}

    for e in range(episodes):
        state = sample_non_colliding(sample_state, is_colliding, sample_bounds)
        traj = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Sample an action and step the environment
            action = pi.select_action(state)
            next_state, reward, done = step_env(state, action, dense=dense)

            # Apped this transition to our trajectory
            traj.append([state, action, next_state, reward])
            steps += 1

        # Compute the gain at all steps and add this to our replay buffer
        traj = compute_gain(traj)
        mem.add_trajectory(traj)

        # Train our policy
        data = DataLoader(mem, batch_size=bs, shuffle=True)
        for i, batch in enumerate(data):
            if i > 10: break
            gains = batch['gains'].unsqueeze(1)
            states = batch['state']
            actions = batch['action']
            logprobs = pi.get_logprob(states, actions)
            rewards = batch['reward']
            next_state = batch['next_state']

            values = V(states)
            critic_loss = F.smooth_l1_loss(values, gains)
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            with torch.no_grad():
                advantage = (1 - lamda) * (gains - values) + lamda * (rewards[:, None] + gamma * V(next_state) - V(state))
            
            actor_loss = (-logprobs * advantage).sum()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
        
        if e % 50 == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = evaluate_policy(pi, trials=50, maxsteps=50)
            print(f'Episode: {e}, Success rate: {success_rate}, average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')
            stats['success_rate'].append(success_rate); stats['average reward per step'].append(avg_reward_per_step); stats['average reward per episode'].append(avg_reward_per_episode)
            if decay: pi.decay_variance()
        
    plot_training_stats(stats)
    torch.save(pi.state_dict(), actor_fpath)
    torch.save(V.state_dict(), critic_fpath)



def reinforce(pi, episodes, max_steps = 100, actor_fpath = 'actor.pth', critic_fpath = 'critic.pth', decay = False):
    '''
    In this function instead of using advantage estimation and training online we implement policy gradients using
    gains and implement the REINFORCE algorithm for on-policy optimization
    '''
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    running_avg_reward = 0
    alpha = 1e-2
    mem = ReplayMemory(maxlen=2*max_steps)
    stats = {'success_rate':[], 'average reward per step':[], 'average reward per episode': []}

    for e in range(episodes):
        state = sample_non_colliding(sample_state, is_colliding, sample_bounds)
        traj = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Sample an action and step the environment
            action = pi.select_action(state)
            next_state, reward, done = step_env(state, action)

            # Update running average of reward
            running_avg_reward = (1-alpha)*running_avg_reward + alpha * reward

            # Apped this transition to our trajectory
            traj.append([state, action, next_state, reward])
            steps += 1

        # Compute the gain at all steps and add this to our replay buffer
        traj = compute_gain(traj)
        mem.add_trajectory(traj)

        # Train our policy
        data = DataLoader(mem, batch_size=bs, shuffle=True)
        for i, batch in enumerate(data):
            if i > 10: break
            gains = batch['gains']
            states = batch['state']
            actions = batch['action']
            logprobs = pi.get_logprob(states, actions)

            weight = (gains - running_avg_reward).unsqueeze(1).detach()
            actor_loss = (-logprobs * weight).sum()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
        
        if e % 50 == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = evaluate_policy(pi, trials=50, maxsteps=50)
            print(f'Episode: {e}, Success rate: {success_rate}, average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')
            stats['success_rate'].append(success_rate); stats['average reward per step'].append(avg_reward_per_step); stats['average reward per episode'].append(avg_reward_per_episode)
            if decay: pi.decay_variance()
    plot_training_stats(stats)
    


def a2c_td_dv(V, pi, episodes, 
              max_steps = 100, 
              target_update = 25,
              actor_fpath = 'actor.pth', critic_fpath = 'critic.pth', decay = False, dense = False):
    
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    optimizer_critic = optim.AdamW(V.parameters(), lr=v_lr)

    # Update this one less frequently, use this to compute targets
    V_target = Value(value_hidden)
    V_target.load_state_dict(V.state_dict())
    V_target.eval()

    mem = ReplayMemory(maxlen=2*max_steps)
    stats = {'success_rate':[], 'average reward per step':[], 'average reward per episode': []}

    for episode in range(1, episodes+1):
        state = sample_non_colliding(sample_state, is_colliding, sample_bounds)
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Sample an action and step the environment
            action = pi.select_action(state)
            next_state, reward, done = step_env(state, action)
            steps+=1

            # Add this transition to memory
            mem.push(state, action, next_state, reward, 0)
        
        data = DataLoader(mem, batch_size=bs, shuffle=True)
        for i, batch in enumerate(data):
            if i > 10: break
            states = batch['state']
            actions = batch['action']
            logprobs = pi.get_logprob(states, actions)
            rewards = batch['reward']
            next_states = batch['next_state']

            value = V(states)
            td_target = rewards[:, None] + gamma * V_target(next_states).detach()

            # We update the critic with HuberLoss
            critic_loss = F.smooth_l1_loss(value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            with torch.no_grad():
                advantage = td_target - value

            # We update the actor based on logprobs weighted by advantage
            actor_loss = (-logprobs * advantage).sum()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

        if episode % target_update == 0:
            V_target.load_state_dict(V.state_dict())
            V_target.eval()
        
        if episode % 50 == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = evaluate_policy(pi, trials=50, maxsteps=50)
            print(f'Episode: {episode}, Success rate: {success_rate}, average reward per step: {avg_reward_per_step}, avg reward per episode {avg_reward_per_episode}')
            stats['success_rate'].append(success_rate); stats['average reward per step'].append(avg_reward_per_step); stats['average reward per episode'].append(avg_reward_per_episode)
            if decay: pi.decay_variance()
    plot_training_stats(stats)
        


def ppo(V, pi, episodes, 
        max_steps = 100, actor_fpath = 'actor.pth', critic_fpath = 'critic.pth', decay = False,
        dense=False, lamda = 0, epsilon = 0.2, n_epochs = 10, p_lr = 3e-4, v_lr = 1e-3, bs = 64, clip_grad = False):
    '''
    In this function we use monte-carlo advantage estimation for more stable/accurate learning. We also use batch gradient descent
    '''
    optimizer_actor = optim.AdamW(pi.parameters(), lr=p_lr)
    optimizer_critic = optim.AdamW(V.parameters(), lr=v_lr)
    mem = ReplayMemory(maxlen=3*max_steps, store_logprobs=True)
    stats = {'success_rate':[], 'average reward per step':[], 'average reward per episode': []}

    for e in range(episodes):
        state = sample_non_colliding(sample_state, is_colliding, sample_bounds)
        traj = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Sample an action and step the environment
            action, logprob = pi.sample(state)
            next_state, reward, done = step_env(state, action, dense=dense)

            # Apped this transition to our trajectory
            traj.append([state, action, next_state, reward, logprob])
            state = next_state
            steps += 1

        # Compute the gain at all steps and add this to our replay buffer
        traj = compute_gain(traj)
        mem.add_trajectory(traj)

        # Train our policy
        data = DataLoader(mem, batch_size=bs, shuffle=True)
        
        for _ in range(n_epochs):
            for batch in data:
                gains = batch['gains'].unsqueeze(1)
                states = batch['state']
                actions = batch['action']
                rewards = batch['reward']
                next_state = batch['next_state']
                old_logprobs = batch['logprobs']

                # Compute new logprobs
                logprobs = pi.get_logprob(states, actions)

                values = V(states)
                critic_loss = F.mse_loss(values, gains)
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                with torch.no_grad():
                    advantage = (1 - lamda) * (gains - values) + lamda * (rewards[:, None] + gamma * V(next_state) - V(state))
                    # Normalize advantage for better stability
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                
                # Compute importance sampling ratio and clip it
                ratio = torch.exp(logprobs - old_logprobs.detach())
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()

                # Optionally we clip actor gradients directly to ensure small updates to our policy
                if clip_grad: torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=1)
                optimizer_actor.step()
        
        if e % 50 == 0:
            success_rate, avg_reward_per_step, avg_reward_per_episode = evaluate_policy(pi, trials=50, maxsteps=max_steps)
            print(f'Episode: {e}, Success rate: {success_rate:.2f}, average reward per step: {avg_reward_per_step:.3f}, avg reward per episode {avg_reward_per_episode:.3f}')
            stats['success_rate'].append(success_rate); stats['average reward per step'].append(avg_reward_per_step); stats['average reward per episode'].append(avg_reward_per_episode)
            if decay: pi.decay_variance()
        
    plot_training_stats(stats)
    torch.save(pi.state_dict(), actor_fpath)
    torch.save(V.state_dict(), critic_fpath)





if __name__ == '__main__':

    # Code to run td a2c algorithm
    # V, pi = Value(value_hidden), Policy(policy_hidden, var=0.1, var_decay=.99, scale=0.25)
    # V.load_state_dict(torch.load('models/critic.pth'))
    # pi.load_state_dict(torch.load('models/actor.pth'))
    # V.eval()
    # pi.eval()
    # a2c_mc(V, pi, episodes = 5000, actor_fpath='td_models/actor_dense_lamda.pth', critic_fpath='td_models/critic_dense_lamda.pth', max_steps=75, lamda=0.2, dense=True)
    # visualize_policy(pi)

    # Reinforce algorithm
    # pi = Policy(policy_hidden)
    # reinforce(pi, episodes=10, max_steps=75)

    # # Code to run td a2c double variance algorithm
    # V, pi = Value(value_hidden), Policy(policy_hidden, scale=0.1)
    # a2c_td_dv(V, pi, episodes=5000, max_steps=75, dense=True)
    # visualize_policy(pi)
    # # Save the functions
    # torch.save(pi.state_dict(), 'td_models/actor_dv_dense.pth')
    # torch.save(V.state_dict(), 'td_models/critic_dv_dense.pth')

    # Code to run PPO
    V, pi = Value(large_hidden), Policy(large_hidden, var=0.5, var_decay=.975, scale=0.25)

    actor_file, critic_file = 'mc_models/actor_nonlinear_dense_reward.pth', 'mc_models/critic_nonlinear_dense_reward.pth'

    # ppo(V, pi, episodes = 2000, actor_fpath=actor_file, critic_fpath=critic_file, max_steps=100, dense=True, epsilon = 0.1, n_epochs=3, bs = 16, lamda=0.5)
    visualize_policy(pi, 500)




        
    


        

        
        



