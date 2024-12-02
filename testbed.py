from collections import namedtuple
from torch.distributions import Normal
from utils import *
from value import *
from vanilla_pg import visualize_policy, plot_training_stats
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wandb

# Smallest possible epsilon value for nonzero division
eps = 1e-10

def step(state, action, reward_fn = reward):
    '''
    Given a state and action pair take a step in the environment and return:
    next state, rewards, termination (i.e if stop condition reached)
    '''
    next_state, collided = transition(state, action)

    next_reward = reward_fn(next_state, collided)

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



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim) -> None:
        super(ActorCritic, self).__init__()

        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, act_dim*2) # output mean and logstd of each 
        self.value_head = nn.Linear(hidden_dim, 1)

        self.saved_actions = []
        self.rewards = []
        self.act_dim = act_dim

    def forward(self, state):
        '''
        Returns action distribution (normal) and state value
        '''
        x = F.relu(self.layer1(state))
        
        # Get action:
        action_out = self.action_head(x)
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
    

# Initialize WandB project for keeping track of experiments
wandb.init(project="rl-agent", config={
    "env": "CustomEnv",
    "gamma": 0.99,
    "learning_rate": 3e-3,
    "episodes": 1000,
    "reward_fn": 'dense and nonlinear, scale=0.1 and goal=1/(1-gamma)',
    "maxsteps": 100,
    "hidden_dim":56,
    "seed": 543
})

# Initialize model and optimizer:
model = ActorCritic(4, 2, 56)
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)


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
        
    if torch.isnan(gains).any():  # Debugging check
        print("NaNs in gains!")

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


def train(num_episodes = wandb.config.episodes, render=False, log_interval=10, maxsteps=wandb.config.maxsteps):
    # Create env:
    env = CustomEnv(reward_fn=dense_reward)

    running_reward = 0
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
                break
        
        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Optimize model
        policy_loss, value_loss = finish_episode(gamma=wandb.config.gamma)

        # Log to WandB
        wandb.log({
            "episode": episode,
            "running_reward": running_reward,
            "episode_reward": episode_reward,
            "success": done,
            "policy_loss": policy_loss,
            "value_loss": value_loss
        })
        
        if log_interval: print(running_reward)

    

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    
    # Train the agent
    train(num_episodes=wandb.config.episodes)
    
    # Finish WandB run
    wandb.finish()

            




