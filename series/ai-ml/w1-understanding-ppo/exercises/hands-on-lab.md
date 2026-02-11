# Hands-On Lab: Understanding PPO Through Practice

Welcome to the hands-on exercises! In this lab, you'll implement, experiment with, and debug PPO step-by-step. Each exercise builds on the previous one, taking you from basic concepts to a fully working PPO agent.

**During Workshop:** Focus on Exercises 1-3 (20 minutes)  
**Complete at Home:** Exercises 4-6 (additional 60-90 minutes)

---

## Prerequisites

Before starting, ensure you have:
- ✅ Python 3.9 or 3.10 installed
- ✅ Conda or virtualenv set up
- ✅ Completed the environment setup (see `../scripts/lab-setup.sh`)
- ✅ Reviewed the workshop content material

---

## Lab Setup

### 1. Activate Your Environment

```bash
# If using conda
conda activate ppo-workshop

# If using venv
source ppo-workshop/bin/activate  # Linux/Mac
# or
ppo-workshop\Scripts\activate  # Windows
```

### 2. Verify Installation

```bash
python -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

Expected output:
```
Gymnasium version: 0.29.x
PyTorch version: 2.1.x
```

### 3. Navigate to Exercise Directory

```bash
cd series/ai-ml/w1-understanding-ppo/exercises
```

---

## Exercise 1: Understanding the Environment (15 minutes)

**Goal:** Get familiar with the Lunar Lander environment and its dynamics.

### Task 1.1: Explore the Environment

Create a file `exercise1_explore_env.py`:

```python
import gymnasium as gym
import numpy as np

# Create the Lunar Lander environment
env = gym.make('LunarLander-v2', render_mode='human')

# Print environment details
print("=" * 50)
print("LUNAR LANDER ENVIRONMENT")
print("=" * 50)
print(f"Observation space: {env.observation_space}")
print(f"Observation dimension: {env.observation_space.shape[0]}")
print(f"Action space: {env.action_space}")
print(f"Number of actions: {env.action_space.n}")
print("=" * 50)

# Reset and print initial state
state, info = env.reset(seed=42)
print("\nInitial state:")
print(f"  Position (x, y): ({state[0]:.3f}, {state[1]:.3f})")
print(f"  Velocity (vx, vy): ({state[2]:.3f}, {state[3]:.3f})")
print(f"  Angle: {state[4]:.3f} radians")
print(f"  Angular velocity: {state[5]:.3f}")
print(f"  Leg 1 contact: {bool(state[6])}")
print(f"  Leg 2 contact: {bool(state[7])}")

# Run random agent for one episode
print("\n" + "=" * 50)
print("RUNNING RANDOM AGENT...")
print("=" * 50)

state, _ = env.reset()
total_reward = 0
steps = 0
done = False

while not done and steps < 1000:
    # Take random action
    action = env.action_space.sample()
    
    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    total_reward += reward
    steps += 1
    state = next_state

print(f"\nEpisode finished!")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Steps taken: {steps}")
print(f"  Success: {'Yes!' if total_reward > 200 else 'No'}")

env.close()
```

**Run it:**
```bash
python exercise1_explore_env.py
```

**Questions to Answer:**
1. What are the 8 dimensions of the state space?
2. What do the 4 actions represent?
3. What's the typical total reward for a random policy?
4. How many steps does a random agent usually last?

**💡 Tip:** Run it multiple times with different seeds to see variation!

---

### Task 1.2: Collect Episode Statistics

Create `exercise1_statistics.py`:

```python
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def run_random_episodes(num_episodes=100):
    """Run random policy and collect statistics"""
    env = gym.make('LunarLander-v2')
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(num_episodes), desc="Running episodes"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    env.close()
    
    return episode_rewards, episode_lengths

# Collect data
print("Collecting statistics from random policy...")
rewards, lengths = run_random_episodes(100)

# Compute statistics
print("\n" + "=" * 50)
print("RANDOM POLICY STATISTICS (100 episodes)")
print("=" * 50)
print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"Min reward: {np.min(rewards):.2f}")
print(f"Max reward: {np.max(rewards):.2f}")
print(f"Average episode length: {np.mean(lengths):.1f} steps")
print(f"Success rate (reward > 200): {np.sum(np.array(rewards) > 200) / len(rewards) * 100:.1f}%")
print("=" * 50)

# TODO: Add visualization (matplotlib)
# Plot histogram of rewards
```

**Run it:**
```bash
python exercise1_statistics.py
```

**Expected Output:**
- Average reward: around -200 to -150
- Success rate: close to 0%
- This is your baseline to beat!

---

## Exercise 2: Build the Actor-Critic Network (20 minutes)

**Goal:** Implement the neural network architecture for PPO.

### Task 2.1: Implement the Network

Create `exercise2_network.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """
    EXERCISE: Complete the Actor-Critic network implementation
    
    This network has:
    - Shared feature extraction layers
    - Actor head (outputs action probabilities)
    - Critic head (outputs state value)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # TODO: Implement shared layers
        # Hint: Use nn.Sequential with Linear layers and ReLU activations
        # Architecture: state_dim -> hidden_dim -> hidden_dim
        self.shared = nn.Sequential(
            # YOUR CODE HERE
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # TODO: Implement actor head
        # Hint: Linear layer from hidden_dim to action_dim
        self.actor = nn.Linear(hidden_dim, action_dim)  # YOUR CODE HERE
        
        # TODO: Implement critic head
        # Hint: Linear layer from hidden_dim to 1 (single value output)
        self.critic = nn.Linear(hidden_dim, 1)  # YOUR CODE HERE
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            action_logits: Tensor of shape (batch_size, action_dim)
            state_value: Tensor of shape (batch_size, 1)
        """
        # TODO: Pass state through shared layers
        features = self.shared(state)  # YOUR CODE HERE
        
        # TODO: Get actor output (action logits)
        action_logits = self.actor(features)  # YOUR CODE HERE
        
        # TODO: Get critic output (state value)
        state_value = self.critic(features)  # YOUR CODE HERE
        
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
            deterministic: If True, pick best action; if False, sample
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value estimate
        """
        action_logits, state_value = self.forward(state)
        
        # TODO: Convert logits to probabilities using softmax
        action_probs = F.softmax(action_logits, dim=-1)  # YOUR CODE HERE
        
        # TODO: Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)  # YOUR CODE HERE
        
        if deterministic:
            # TODO: Pick action with highest probability
            action = torch.argmax(action_probs, dim=-1)  # YOUR CODE HERE
        else:
            # TODO: Sample action from distribution
            action = dist.sample()  # YOUR CODE HERE
        
        # TODO: Compute log probability of the action
        log_prob = dist.log_prob(action)  # YOUR CODE HERE
        
        return action, log_prob, state_value


# Test the network
if __name__ == "__main__":
    # Create network
    state_dim = 8  # Lunar Lander state dimension
    action_dim = 4  # Lunar Lander action dimension
    network = ActorCritic(state_dim, action_dim)
    
    print("=" * 50)
    print("NETWORK ARCHITECTURE TEST")
    print("=" * 50)
    
    # Print network
    print("\nNetwork structure:")
    print(network)
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 32
    dummy_state = torch.randn(batch_size, state_dim)
    
    print(f"\nInput shape: {dummy_state.shape}")
    
    action_logits, state_value = network(dummy_state)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"State value shape: {state_value.shape}")
    
    # Test get_action
    action, log_prob, value = network.get_action(dummy_state)
    print(f"\nAction shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test deterministic action
    action_det, _, _ = network.get_action(dummy_state, deterministic=True)
    print(f"Deterministic action shape: {action_det.shape}")
    
    print("\n✓ All tests passed!")
```

**Run it:**
```bash
python exercise2_network.py
```

**Expected Output:**
- Total parameters: ~4,500
- All shape assertions should pass

---

### Task 2.2: Visualize Action Probabilities

Create `exercise2_visualize.py`:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from exercise2_network import ActorCritic

# Create network
network = ActorCritic(state_dim=8, action_dim=4)

# Create different states
states = torch.tensor([
    [0.0, 1.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],  # High and falling
    [0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0],  # Near ground
    [0.5, 1.0, -0.3, 0.0, 0.2, 0.1, 0.0, 0.0],  # Drifting left
], dtype=torch.float32)

action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']

# Get action probabilities
with torch.no_grad():
    action_logits, values = network(states)
    action_probs = torch.softmax(action_logits, dim=-1).numpy()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    ax.bar(action_names, action_probs[i])
    ax.set_ylabel('Probability')
    ax.set_title(f'State {i+1}\nValue: {values[i].item():.3f}')
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('action_probabilities.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'action_probabilities.png'")
plt.show()
```

**Run it:**
```bash
python exercise2_visualize.py
```

**Question:** Are the action probabilities uniform? Why or why not?

---

## Exercise 3: Implement Advantage Estimation (20 minutes)

**Goal:** Implement Generalized Advantage Estimation (GAE).

### Task 3.1: Compute GAE

Create `exercise3_gae.py`:

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    EXERCISE: Implement Generalized Advantage Estimation
    
    Formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    Where: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    Args:
        rewards: Array of rewards (length T)
        values: Array of state values (length T)
        dones: Array of done flags (length T)
        gamma: Discount factor
        lambda_: GAE parameter
    
    Returns:
        advantages: Array of advantages (length T)
        returns: Array of returns for value function (length T)
    """
    # TODO: Initialize advantages array
    advantages = np.zeros_like(rewards)  # YOUR CODE HERE
    
    last_advantage = 0
    last_value = 0
    
    # TODO: Loop backwards through time
    for t in reversed(range(len(rewards))):
        # TODO: Get next state value
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]  # YOUR CODE HERE
        
        # TODO: Compute TD error
        # δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]  # YOUR CODE HERE
        
        # TODO: Compute advantage using GAE
        # A_t = δ_t + (γλ)A_{t+1}
        advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage  # YOUR CODE HERE
        
        last_advantage = advantages[t]
        
        # Reset on episode boundary
        if dones[t]:
            last_advantage = 0
            last_value = 0
    
    # TODO: Compute returns (targets for value function)
    # Returns = advantages + values
    returns = advantages + values  # YOUR CODE HERE
    
    return advantages, returns


# Test the function
if __name__ == "__main__":
    print("=" * 50)
    print("TESTING GAE IMPLEMENTATION")
    print("=" * 50)
    
    # Simple test case
    rewards = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    values = np.array([5.0, 5.0, 5.0, 5.0, 0.0])
    dones = np.array([0, 0, 0, 0, 1])
    
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95)
    
    print("\nTest Case:")
    print(f"Rewards:    {rewards}")
    print(f"Values:     {values}")
    print(f"Dones:      {dones}")
    print(f"\nAdvantages: {advantages}")
    print(f"Returns:    {returns}")
    
    # Verification
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    # Check shapes
    assert advantages.shape == rewards.shape, "Advantages shape mismatch!"
    assert returns.shape == rewards.shape, "Returns shape mismatch!"
    
    # Check that returns = advantages + values
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-5)
    
    print("✓ Shape tests passed")
    print("✓ Returns = Advantages + Values")
    
    # Check GAE properties
    print(f"✓ Advantages sum: {np.sum(advantages):.3f}")
    print(f"✓ Returns mean: {np.mean(returns):.3f}")
    
    # Test with different gamma/lambda
    print("\n" + "=" * 50)
    print("TESTING DIFFERENT PARAMETERS")
    print("=" * 50)
    
    for gamma in [0.9, 0.99, 0.999]:
        for lambda_ in [0.9, 0.95, 1.0]:
            adv, ret = compute_gae(rewards, values, dones, gamma, lambda_)
            print(f"gamma={gamma:.3f}, lambda={lambda_:.2f} -> "
                  f"adv_mean={np.mean(adv):.3f}, ret_mean={np.mean(ret):.3f}")
```

**Run it:**
```bash
python exercise3_gae.py
```

**Questions:**
1. How does increasing `gamma` affect advantages?
2. How does increasing `lambda` affect advantages?
3. What happens when `lambda=1.0` vs `lambda=0.0`?

---

## Exercise 4: Implement PPO Update (30 minutes)

**Goal:** Implement the core PPO update step with clipping.

### Task 4.1: PPO Loss Function

Create `exercise4_ppo_loss.py`:

```python
import torch
import torch.nn.functional as F

def ppo_loss(policy, batch, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    EXERCISE: Implement PPO loss function
    
    Components:
    1. Clipped surrogate objective (actor loss)
    2. Value function loss (critic loss)
    3. Entropy bonus (exploration)
    
    Args:
        policy: Actor-Critic network
        batch: Dictionary with states, actions, advantages, returns, old_log_probs
        clip_epsilon: PPO clipping parameter
        value_coef: Weight for value loss
        entropy_coef: Weight for entropy bonus
    
    Returns:
        Dictionary with losses and metrics
    """
    # Convert batch to tensors
    states = torch.FloatTensor(batch['states'])
    actions = torch.LongTensor(batch['actions'])
    advantages = torch.FloatTensor(batch['advantages'])
    returns = torch.FloatTensor(batch['returns'])
    old_log_probs = torch.FloatTensor(batch['old_log_probs'])
    
    # TODO: Normalize advantages
    # Hint: advantages = (advantages - mean) / (std + 1e-8)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # YOUR CODE HERE
    
    # TODO: Forward pass through policy
    action_logits, state_values = policy(states)  # YOUR CODE HERE
    
    # TODO: Compute new action probabilities
    action_probs = F.softmax(action_logits, dim=-1)  # YOUR CODE HERE
    dist = torch.distributions.Categorical(action_probs)
    new_log_probs = dist.log_prob(actions)
    
    # TODO: Compute probability ratio
    # ratio = π_new / π_old = exp(log π_new - log π_old)
    ratio = torch.exp(new_log_probs - old_log_probs)  # YOUR CODE HERE
    
    # TODO: Compute clipped surrogate objective
    # surr1 = ratio * advantages (unclipped)
    # surr2 = clip(ratio, 1-ε, 1+ε) * advantages (clipped)
    # L^CLIP = E[min(surr1, surr2)]
    surr1 = ratio * advantages  # YOUR CODE HERE
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages  # YOUR CODE HERE
    actor_loss = -torch.min(surr1, surr2).mean()  # YOUR CODE HERE (note the negative!)
    
    # TODO: Compute value function loss
    # L^VF = MSE(V(s), returns)
    value_loss = F.mse_loss(state_values.squeeze(), returns)  # YOUR CODE HERE
    
    # TODO: Compute entropy bonus
    entropy = dist.entropy().mean()  # YOUR CODE HERE
    
    # TODO: Compute total loss
    # L = L^CLIP + c_1 * L^VF - c_2 * entropy
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy  # YOUR CODE HERE
    
    # Collect metrics
    metrics = {
        'loss': loss.item(),
        'actor_loss': actor_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'ratio_mean': ratio.mean().item(),
        'ratio_std': ratio.std().item(),
        'ratio_min': ratio.min().item(),
        'ratio_max': ratio.max().item(),
        'advantages_mean': advantages.mean().item(),
        'advantages_std': advantages.std().item(),
    }
    
    return loss, metrics


# Test the loss function
if __name__ == "__main__":
    from exercise2_network import ActorCritic
    import numpy as np
    
    print("=" * 50)
    print("TESTING PPO LOSS FUNCTION")
    print("=" * 50)
    
    # Create dummy batch
    batch_size = 128
    batch = {
        'states': np.random.randn(batch_size, 8).astype(np.float32),
        'actions': np.random.randint(0, 4, size=batch_size),
        'advantages': np.random.randn(batch_size).astype(np.float32),
        'returns': np.random.randn(batch_size).astype(np.float32),
        'old_log_probs': np.random.randn(batch_size).astype(np.float32),
    }
    
    # Create policy
    policy = ActorCritic(state_dim=8, action_dim=4)
    
    # Compute loss
    loss, metrics = ppo_loss(policy, batch)
    
    print("\nLoss Components:")
    print(f"  Total loss: {metrics['loss']:.4f}")
    print(f"  Actor loss: {metrics['actor_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    
    print("\nRatio Statistics:")
    print(f"  Mean: {metrics['ratio_mean']:.4f}")
    print(f"  Std: {metrics['ratio_std']:.4f}")
    print(f"  Min: {metrics['ratio_min']:.4f}")
    print(f"  Max: {metrics['ratio_max']:.4f}")
    
    print("\nAdvantage Statistics:")
    print(f"  Mean: {metrics['advantages_mean']:.4f}")
    print(f"  Std: {metrics['advantages_std']:.4f}")
    
    # Test gradient flow
    print("\n" + "=" * 50)
    print("TESTING GRADIENT FLOW")
    print("=" * 50)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            print(f"  {name}: grad_norm = {grad_norm:.4f}")
    
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f"\nTotal gradient norm: {total_grad_norm:.4f}")
    
    optimizer.step()
    
    print("\n✓ All tests passed!")
```

**Run it:**
```bash
python exercise4_ppo_loss.py
```

---

## Exercise 5: Train Your First PPO Agent (30 minutes)

**Goal:** Put everything together and train a working PPO agent!

### Task 5.1: Complete Training Loop

Create `exercise5_train_ppo.py`:

```python
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from exercise2_network import ActorCritic
from exercise3_gae import compute_gae
from exercise4_ppo_loss import ppo_loss


def collect_rollouts(env, policy, num_steps=2048):
    """Collect experience from the environment"""
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    state, _ = env.reset()
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(num_steps):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value = policy.get_action(state_tensor)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # Store transition
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())
        log_probs.append(log_prob.item())
        
        current_episode_reward += reward
        
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            state, _ = env.reset()
        else:
            state = next_state
    
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'values': np.array(values),
        'old_log_probs': np.array(log_probs),
    }, episode_rewards


def train_ppo(env_name='LunarLander-v2',
              total_timesteps=500_000,
              batch_size=2048,
              num_epochs=10,
              learning_rate=3e-4,
              save_path='ppo_lunar_lander.pth'):
    """
    Main PPO training loop
    
    TODO: Complete the training loop
    """
    # Initialize environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize policy and optimizer
    policy = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training metrics
    all_episode_rewards = []
    training_losses = []
    
    num_updates = total_timesteps // batch_size
    
    print("=" * 70)
    print(f"TRAINING PPO ON {env_name}")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Batch size: {batch_size}")
    print(f"Number of updates: {num_updates}")
    print(f"Epochs per update: {num_epochs}")
    print("=" * 70)
    
    for update in tqdm(range(num_updates), desc="Training"):
        # TODO: Collect rollout data
        batch, episode_rewards = collect_rollouts(env, policy, num_steps=batch_size)
        all_episode_rewards.extend(episode_rewards)
        
        # TODO: Compute advantages using GAE
        advantages, returns = compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones']
        )
        
        batch['advantages'] = advantages
        batch['returns'] = returns
        
        # TODO: Perform multiple epochs of PPO updates
        for epoch in range(num_epochs):
            loss, metrics = ppo_loss(policy, batch)
            
            # TODO: Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
        
        training_losses.append(metrics['loss'])
        
        # Log progress
        if update % 10 == 0:
            recent_rewards = all_episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            print(f"\nUpdate {update}/{num_updates}")
            print(f"  Avg reward (last 50 episodes): {avg_reward:.2f}")
            print(f"  Actor loss: {metrics['actor_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Ratio mean: {metrics['ratio_mean']:.4f}")
    
    # Save model
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': all_episode_rewards,
    }, save_path)
    
    print(f"\n✓ Model saved to {save_path}")
    
    return policy, all_episode_rewards, training_losses


if __name__ == "__main__":
    # Train the agent
    policy, episode_rewards, losses = train_ppo(
        total_timesteps=500_000,  # Adjust based on your compute
        batch_size=2048,
        num_epochs=10,
        learning_rate=3e-4
    )
    
    # Plot training progress
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot episode rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode rewards')
    # Plot moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                    label=f'{window}-episode moving average', linewidth=2)
    axes[0].axhline(y=200, color='r', linestyle='--', label='Success threshold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training loss
    axes[1].plot(losses)
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('PPO Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Saved training plot to 'training_progress.png'")
    plt.show()
```

**Run it:**
```bash
python exercise5_train_ppo.py
```

**Expected Training Time:**
- CPU: 30-60 minutes
- GPU: 10-20 minutes

**Success Criteria:**
- Average reward > 200 (landing successfully)
- Consistent performance over last 100 episodes

---

## Exercise 6: Evaluate and Analyze (15 minutes)

**Goal:** Evaluate your trained agent and understand its behavior.

### Task 6.1: Evaluate Trained Agent

Create `exercise6_evaluate.py`:

```python
import gymnasium as gym
import torch
import numpy as np
from exercise2_network import ActorCritic

def evaluate_policy(policy, env_name='LunarLander-v2', num_episodes=100, render=False):
    """Evaluate a trained policy"""
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _ = policy.get_action(state_tensor, deterministic=True)
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if total_reward >= 200:
            success_count += 1
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / num_episodes * 100,
        'rewards': episode_rewards,
        'lengths': episode_lengths,
    }


if __name__ == "__main__":
    # Load trained model
    checkpoint = torch.load('ppo_lunar_lander.pth')
    
    policy = ActorCritic(state_dim=8, action_dim=4)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print("=" * 50)
    print("EVALUATING TRAINED POLICY")
    print("=" * 50)
    
    # Evaluate without rendering
    results = evaluate_policy(policy, num_episodes=100, render=False)
    
    print(f"\nResults over 100 episodes:")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min reward: {results['min_reward']:.2f}")
    print(f"  Max reward: {results['max_reward']:.2f}")
    print(f"  Mean episode length: {results['mean_length']:.1f}")
    print(f"  Success rate: {results['success_rate']:.1f}%")
    
    # Compare with random policy
    print("\n" + "=" * 50)
    print("COMPARISON WITH RANDOM POLICY")
    print("=" * 50)
    
    # Run a few episodes with rendering
    print("\nRunning 3 episodes with rendering...")
    input("Press Enter to continue...")
    
    for i in range(3):
        print(f"\nEpisode {i+1}/3")
        results = evaluate_policy(policy, num_episodes=1, render=True)
        print(f"  Reward: {results['mean_reward']:.2f}")
```

**Run it:**
```bash
python exercise6_evaluate.py
```

---

## Challenge Exercises (Optional)

### Challenge 1: Hyperparameter Tuning

Experiment with different hyperparameters:

```python
# Try these variations
configs = [
    {'learning_rate': 1e-4, 'clip_epsilon': 0.1},
    {'learning_rate': 3e-4, 'clip_epsilon': 0.2},  # Default
    {'learning_rate': 1e-3, 'clip_epsilon': 0.3},
]

# Compare their performance
```

### Challenge 2: Different Architectures

Modify the network:

```python
# Deeper network
self.shared = nn.Sequential(
    nn.Linear(state_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU()
)

# Or add Layer Normalization
```

### Challenge 3: Curriculum Learning

Train on easier environments first:

```python
# Start with CartPole-v1
# Then move to LunarLander-v2
# Finally try BipedalWalker-v3
```

### Challenge 4: Add TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/ppo_experiment')

# Log metrics
writer.add_scalar('reward/episode', reward, episode)
writer.add_scalar('loss/actor', actor_loss, update)
```

---

## Troubleshooting

### Common Issues

**Issue 1: Reward not improving**

```python
# Check your learning rate
# Try: 1e-4, 3e-4, 1e-3

# Check advantage normalization
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Check reward scaling
rewards = rewards / 100.0  # Scale rewards to [-1, 1] range
```

**Issue 2: Training is unstable**

```python
# Increase batch size
batch_size = 4096  # instead of 2048

# Reduce learning rate
learning_rate = 1e-4  # instead of 3e-4

# Check gradient clipping
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
```

**Issue 3: Entropy drops too quickly**

```python
# Increase entropy coefficient
entropy_coef = 0.02  # instead of 0.01
```

---

## Summary

Congratulations! You've completed the hands-on lab. You now have:

✅ Understanding of the Lunar Lander environment  
✅ Implemented Actor-Critic network  
✅ Computed GAE advantages  
✅ Built PPO loss function  
✅ Trained a working PPO agent  
✅ Evaluated and analyzed performance

### What You've Learned

- How to structure an RL training loop
- The importance of advantage estimation
- How clipping stabilizes training
- How to monitor and debug RL training
- Practical implementation of a state-of-the-art algorithm

### Next Steps

1. Try other Gymnasium environments
2. Implement continuous action PPO
3. Add recurrent layers (LSTM/GRU)
4. Explore multi-agent settings
5. Apply to your own problems!

---

**Questions?** Review the workshop content or ask in our Discord community!

**Share your results!** Tweet your trained agent with #TFDWorkshop #PPO

---

*Hands-On Lab: Understanding PPO*  
*AI & ML Workshop Series | TFDevs*  
*Last Updated: February 11, 2026*
