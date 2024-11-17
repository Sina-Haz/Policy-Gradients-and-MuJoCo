### The Project
#### Problem Formulation (MDP)
Actions: 
 - $a_t = (f_{x, t} ,f_{y,t})$ force in x, y direction. Each force within \[-1, 1] newtons

State: 
 - $s_t = (x_t, y_t, \dot x_t, \dot y_t,)$, position and velocity of robot
At beggining of episode initial state obtained by uniformly sampling non-colliding positions w/ 0 velocity in the workspace

Transition: 
- we move with 1kg mass according to mujoco transitions (so to respect collisions)
- Our actions are subtracted by some independent gaussian noise ~ N(0, 0.1)
- Use discretized timestep with dt = 0.1

Reward Function:
 - 1 if L2 norm of current and goal state is within some epsilon
 - Otherwise it's 0

Discount $\gamma = 0.99$

Free Parameters:
 - Goal position (goal velocity is 0)
 - epsilon for distance b/w goal and current state
 - Can add some small negative reward for colliding

#### Task
 - Use small Actor Neural Network that predicts actions as Gaussians 
 - 2nd small Critic Network that predicts $V_w^\pi(s_t)$
 - Implement Vanilla Policy Gradient Algoirthm
 - Report avg reward/step as function of # episodes for training (learning curve)

#### What's Necessary:
Utils:
 - State class
 - transition model (using mujoco)
 - state sampling
 - Rollout - returns a full trajectory (up to X steps)
 - Compute gain $G_t$ over trajectory (linearly go through and sum from end to beginning)
 - Want to use pytorch or something to come up with NN class (could be policy or value specific)

### Vanilla PG Algorithm:
Initialize policy params $\theta$, value params $w$ ([[Value Function Estimation|Estimated Value Function]])
for N iterations:
 - Collect set of trajectories by executing the current policy
 - At each timestep in each trajectory compute the gain $G_t = Q^\pi(s,a)$ 
	 - can also bootstrap via $Q^\pi(s_t,a_t) \approx r_t + \gamma V_w(s_{t+1})$
- Subtract by baseline to get [[Advantage Functions|Advantage]] $\hat A_t = Q^\pi(s_t,a_t) - V_w(s_t)$ 
- Refit $V_w$ by minimizing $||V_w(s_t) - G_t||^2$
- Update policy using estimate $\hat g$ which is sum of terms $\nabla_\theta \log \pi(a_t|s_t ; \theta)\hat A_t$ 