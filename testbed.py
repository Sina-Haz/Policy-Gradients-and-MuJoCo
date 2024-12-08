from collections import namedtuple, deque
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, obs_dim, act_dim, hidden_dim, scale=1, extra_layer=False) -> None:
        super(ActorCritic, self).__init__()

        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        if extra_layer: self.layer2 = nn.Linear(hidden_dim, hidden_dim)
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
        x = F.relu(self.layer1(state))
        if self.layer2: x = F.relu(self.layer2(x))
        
        # Get action:
        action_out = F.tanh(self.action_head(x)) * self.scale
        mean, log_var = action_out[:self.act_dim], action_out[self.act_dim:]
        log_var = torch.clamp(log_var, min=-5, max=2) # For numerical stability!
        std = torch.exp(0.5 * log_var)
        distr = Normal(mean, std)

        # Get Value
        value = self.value_head(x)

        return distr, value
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        norm_distr, val = self(state)
        action = torch.clamp(norm_distr.sample(),-1, 1) # Makes sure action is within valid range
        lp = norm_distr.log_prob(action).sum(dim=-1)

        self.saved_actions.append(SavedAction(lp, val))
        return action.cpu().numpy()
    

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
lr = 1e-3
maxsteps = 100
hidden_dim = 256
episodes = 500_000


# Initialize model and optimizer:
model = ActorCritic(obs_dim=4, act_dim=2, hidden_dim=hidden_dim, scale = 0.25, extra_layer=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)


def finish_episode(gamma = gamma):
    '''
    Computes loss and optimizes w.r.t. global model and optimizer
    '''

    R = 0
    saved_actions = model.saved_actions
    gains = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        gains.insert(0, R)

    gains = torch.tensor(gains).to(device)

    # Normalize gains
    if gains.numel() > 1:
        gains = (gains - gains.mean()) / (gains.std() + eps)
    else:
        gains = gains - gains.mean()  # Skip std normalization if only one element

    log_probs = torch.stack([action.log_prob for action in saved_actions])
    vals = torch.stack([action.value for action in saved_actions])

    advantage = gains - vals.squeeze()
    ploss = (-log_probs * advantage.detach()).sum()
    vloss = F.smooth_l1_loss(vals, gains.unsqueeze(1))

    optimizer.zero_grad()
    loss = ploss + vloss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler: scheduler.step()

    del model.rewards[:]
    del model.saved_actions[:]

    return ploss, vloss


def train(env, num_episodes = episodes, render=False, log_interval=50, maxsteps=maxsteps, path=None):
    running_reward = 0
    successes = 0
    recent_successes = deque(maxlen=log_interval*4)


    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        success=False

        for t in range(maxsteps): # Don't infinite loop while learning
            action = model.act(state)
            state, reward, done, _, _ = env.step(action)

            if render:
                env.render()
            
            model.rewards.append(reward)
            episode_reward += reward

            if done:
                success=True
                successes += 1
                break
        
        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Update total and recent success rate:
        total_success_rate = successes / episode
        recent_successes.append(1 if success else 0)
        success_rate_recent = sum(recent_successes) / len(recent_successes)


        # Optimize model
        policy_loss, value_loss = finish_episode(gamma=gamma)
        
        if episode % log_interval == 0:
            print(running_reward)
            # Log to WandB
            wandb.log({
                "episode": episode,
                "running_reward": running_reward,
                "episode_reward": episode_reward,
                "total success_rate":  total_success_rate,
                "recent success rate": success_rate_recent,
                "policy_loss": policy_loss,
                "value_loss": value_loss
            })
        # Save the model every 1000 episodes
        if episode % 1000 == 0:
             torch.save(model.state_dict(), f=path or f'model_episode:{episode}.pth')

        if success_rate_recent > 0.70 and episode > 1000:
            torch.save(model.state_dict(), f=f'accurate_ac.pth')



if __name__ == "__main__":
    fname = 'ac_3layer.pth'
    env = CustomEnv(reward_fn=new_reward)
    # print('about to initialize wandb and train')

    # Load the model
    # model.load_state_dict(torch.load(f=fname, weights_only=True, map_location=torch.device('cpu')))

    wandb.init(project="rl-agent", config={
    "env": "CustomEnv",
    "gamma": gamma,
    "learning_rate": lr,
    "episodes": episodes,
    "reward_fn": 'Using new reward function',
    "maxsteps": maxsteps,
    "hidden_dim": hidden_dim,
    "seed": 543,
    "description": "Scaled down actions, added an extra layer for complexity, "
    })

    # Set random seeds for reproducibility
    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    
    # Train the agent
    train(num_episodes=episodes, env=env, path = fname)
    torch.save(model.state_dict(), f=fname)

    # Finish WandB run
    wandb.finish()


    # for _ in range(5):
    #    model.visualize(env=env)





            




