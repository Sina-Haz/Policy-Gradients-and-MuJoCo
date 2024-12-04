from collections import namedtuple
from torch.distributions import Normal
import torch.nn as nn
import torch.optim as optim
from utils import *
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wandb

# Smallest possible epsilon value for nonzero division
eps = 1e-10

def step(state, action, reward_fn = reward, position_only = False):
    '''
    Given a state and action pair take a step in the environment and return:
    next state, rewards, termination (i.e if stop condition reached)
    '''
    state = state # Truncate state dim if it's greater than 4 b/c  
    next_state, collided = transition(state, action)

    next_reward = reward_fn(next_state, action_magnitude=np.linalg.norm(action))

    if position_only:
        done = bool((torch.norm(next_state[:2] - goal[:2]) < epsilon).item())
    else:
        done = bool((torch.norm(next_state - goal) < epsilon).item())

    return next_state, next_reward, done



class CustomEnv(gym.Env):
    def __init__(self, reward_fn) -> None:
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape = (2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.viewer = init_viewer()
        self.render_mode = 'human'
        self.reward = reward_fn

    def reset(self, *, seed=None, options=None):
        self.state = (sample_non_colliding(sample_state, is_colliding, sample_bounds)).numpy()
        return self.state, {}
    
    def step(self, action):
        '''
        Ensure action is numpy array!
        '''
        next_state, reward, done = step(self.state, action, reward_fn=self.reward)

        self.state = next_state.numpy()

        return self.state, reward, done, False, {}
    
    def render(self):
        if self.render_mode == 'human':
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()


class CustomEnv2(gym.Env):
    '''
    Same as custom env but now we expand observation space to include position of obstacle and goal
    '''
    def __init__(self, reward_fn, pos_only) -> None:
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape = (2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.viewer = init_viewer()
        self.render_mode = 'human'
        self.reward = reward_fn

        self.goal = goal
        self.obs = np.array([0.5, 0.0, 0, 0])
        self.pos_only = pos_only

    def reset(self, *, seed=None, options=None):
        # self.state = (sample_non_colliding(sample_state, is_colliding, sample_bounds)).numpy()
        self.state = np.array([0,0,0,0])

        # Compute relative dist to goal and obstacle
        goal_dist = self.goal - self.state
        obs_dist = self.obs - self.state

        self.state = np.concatenate([self.state, obs_dist, goal_dist])
        return self.state, {}
    
    def step(self, action):
        '''
        Ensure action is numpy array!
        '''
        next_state, reward, done = step(self.state[:4], action, reward_fn=self.reward, position_only=self.pos_only)

        self.state = next_state.numpy()

        # Compute relative dist to goal and obstacle
        goal_dist = self.goal - self.state
        obs_dist = self.obs - self.state

        self.state = np.concatenate([self.state, obs_dist, goal_dist])

        return self.state, reward, done, False, {}
    
    def render(self):
        if self.render_mode == 'human':
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, scale=1) -> None:
        super(ActorCritic, self).__init__()

        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, act_dim*2) # output mean and logstd of each 
        self.value_head = nn.Linear(hidden_dim, 1)

        self.saved_actions = []
        self.rewards = []
        self.act_dim = act_dim
        self.scale = scale

    def forward(self, state):
        '''
        Returns action distribution (normal) and state value
        '''
        x = F.sigmoid(self.layer1(state))
        
        # Get action:
        action_out = F.tanh(self.action_head(x)) * self.scale
        mean, log_var = action_out[:self.act_dim], action_out[self.act_dim:]
        log_var = torch.clamp(log_var, min=-20, max=2) # For numerical stability!
        std = torch.exp(0.5 * log_var)
        distr = Normal(mean, std)

        # Get Value
        value = self.value_head(x)

        return distr, value
    
    def act(self, state):
        state = torch.from_numpy(state).float()
        norm_distr, val = self(state)
        action = torch.clamp(norm_distr.sample(),-1, 1) # Makes sure action is within valid range
        lp = norm_distr.log_prob(action).sum(dim=-1)

        self.saved_actions.append(SavedAction(lp, val))
        return action.numpy()
    

    def visualize(self, env, maxsteps = 200):
        state, _ = env.reset()

        for _ in range(maxsteps):
            action = model.act(state)
            state, reward, done, _, _ = env.step(action)
            env.render()
            
            if done:
                break


# Initialize WandB project for keeping track of experiments
gamma = 0.95
lr = 1e-4
episodes = 1000
maxsteps = 100
hidden_dim = 128


# Initialize model and optimizer:
model = ActorCritic(obs_dim=4, act_dim=2, hidden_dim=hidden_dim, scale = 1)
optimizer = optim.Adam(model.parameters(), lr=lr)


def finish_episode(gamma = 0.99):
    '''
    Computes loss and optimizes w.r.t. global model and optimizer
    '''

    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    gains = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        gains.insert(0, R)

    gains = torch.tensor(gains)

    # Normalize gains
    if gains.numel() > 1:
        gains = (gains - gains.mean()) / (gains.std() + eps)
    else:
        gains = gains - gains.mean()  # Skip std normalization if only one element

    for (lp, val), g in zip(saved_actions, gains):
        adv = g - val.item()
        policy_losses.append(-lp * adv)
        value_losses.append(F.smooth_l1_loss(val, g.detach().unsqueeze(0)))
    
    optimizer.zero_grad()
    ploss, vloss = torch.stack(policy_losses).sum(), torch.stack(value_losses).sum()
    loss = ploss + vloss
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

    return ploss, vloss


def train(env, num_episodes = episodes, render=False, log_interval=25, maxsteps=maxsteps):
    running_reward = 0
    successes = 0
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(maxsteps): # Don't infinite loop while learning
            action = model.act(state)
            state, reward, done, _, _ = env.step(action)

            if render:
                env.render()
            
            model.rewards.append(reward)
            episode_reward += reward

            if done:
                successes += 1
                break
        
        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        success_rate = successes / episode

        # Optimize model
        policy_loss, value_loss = finish_episode(gamma=gamma)

        # Log to WandB
        wandb.log({
            "episode": episode,
            "running_reward": running_reward,
            "episode_reward": episode_reward,
            "success_rate":  success_rate,
            "policy_loss": policy_loss,
            "value_loss": value_loss
        })
        
        if episode % log_interval == 0: print(running_reward)


if __name__ == "__main__":
    fname = 'test_models/ac.test14.pth'
    env = CustomEnv(reward_fn=new_reward)

    # Load the model
    # model.load_state_dict(torch.load(f=fname, weights_only=True))

    wandb.init(project="rl-agent", config={
    "env": "CustomEnv",
    "gamma": gamma,
    "learning_rate": lr,
    "episodes": episodes,
    "reward_fn": 'Using new reward function',
    "maxsteps": maxsteps,
    "hidden_dim": hidden_dim,
    "seed": 543,
    "description": "Back to regular observations and scale"
    })

    # # Set random seeds for reproducibility
    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    
    # Train the agent
    train(num_episodes=episodes, env=env)
    torch.save(model.state_dict(), f=fname)

    # # Finish WandB run
    wandb.finish()


    for _ in range(5):
        model.visualize(env=env)





            




