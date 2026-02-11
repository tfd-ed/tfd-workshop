#!/usr/bin/env python3
"""
PPO Lunar Lander - Complete Training with Visualization
Trains a PPO agent to solve LunarLander-v2 and generates:
- GIFs at regular intervals showing improvement
- Gradient analysis graphs
- Reward/loss/entropy plots
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path
from collections import deque
from datetime import datetime
import time

# Set up directories
OUTPUT_DIR = Path("outputs")
GIF_DIR = OUTPUT_DIR / "gifs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"

for dir_path in [OUTPUT_DIR, GIF_DIR, PLOT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PPO LUNAR LANDER - TRAINING WITH VISUALIZATION")
print("=" * 70)
print()


# ============================================================================
# Actor-Critic Network
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic network with shared feature extraction"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.shared(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value
    
    def get_action_and_value(self, state, action=None):
        action_logits, state_value = self.forward(state)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), state_value


# ============================================================================
# Memory Buffer
# ============================================================================

class RolloutBuffer:
    """Store experience for PPO training"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.dones)
        )


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # For tracking
        self.gradient_norms = []
        self.actor_grad_norms = []
        self.critic_grad_norms = []
    
    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, _, value = self.network.get_action_and_value(state)
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, buffer, n_epochs=10, batch_size=64):
        """Update policy using PPO"""
        states, actions, rewards, old_values, old_log_probs, dones = buffer.get()
        
        # Compute advantages and returns
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, old_values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        
        # Multiple epochs of optimization
        dataset_size = len(states)
        for epoch in range(n_epochs):
            # Mini-batch training
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipping objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norms before clipping
                total_norm = 0
                actor_norm = 0
                critic_norm = 0
                
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        
                        if 'actor' in name:
                            actor_norm += param_norm ** 2
                        elif 'critic' in name:
                            critic_norm += param_norm ** 2
                
                total_norm = np.sqrt(total_norm)
                actor_norm = np.sqrt(actor_norm)
                critic_norm = np.sqrt(critic_norm)
                
                self.gradient_norms.append(total_norm)
                self.actor_grad_norms.append(actor_norm)
                self.critic_grad_norms.append(critic_norm)
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                # Compute KL divergence (approximate)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    approx_kls.append(approx_kl)
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
            'gradient_norm': np.mean(self.gradient_norms[-len(policy_losses):])
        }


# ============================================================================
# Environment Wrappers for Visualization
# ============================================================================

class RecordEpisode:
    """Record episode frames for GIF generation"""
    
    def __init__(self, env):
        self.env = env
        self.frames = []
        
    def reset(self, **kwargs):
        self.frames = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        result = self.env.step(action)
        
        # Render and store frame
        frame = self.env.render()
        if frame is not None:
            self.frames.append(frame)
        
        return result
    
    def get_frames(self):
        return self.frames
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def save_episode_gif(frames, filename, fps=30):
    """Save episode frames as GIF"""
    if not frames:
        return
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    img = ax.imshow(frames[0])
    
    def update(frame_idx):
        img.set_array(frames[frame_idx])
        return [img]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000/fps, blit=True
    )
    
    anim.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"  Saved: {filename}")


def demo1_environment_basics():
    """Run a demo episode to test the environment"""
    print("\n" + "=" * 70)
    print("Testing Environment Setup")
    print("=" * 70)
    
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    state, _ = env.reset(seed=42)
    
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial state: {state[:4]}")
    
    env.close()
    print("✓ Environment test complete!")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_metrics(metrics_history, save_path):
    """Plot comprehensive training metrics"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    episodes = range(1, len(metrics_history['rewards']) + 1)
    
    # 1. Episode Rewards
    ax1 = fig.add_subplot(gs[0, :])
    rewards = metrics_history['rewards']
    ax1.plot(episodes, rewards, alpha=0.6, color='blue', label='Episode Reward')
    
    # Moving average
    window = min(50, len(rewards))
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), moving_avg, 
                color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.axhline(y=200, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Policy Loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(metrics_history['policy_loss'], color='orange')
    ax2.set_xlabel('Update')
    ax2.set_ylabel('Loss')
    ax2.set_title('Policy Loss', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Value Loss
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(metrics_history['value_loss'], color='purple')
    ax3.set_xlabel('Update')
    ax3.set_ylabel('Loss')
    ax3.set_title('Value Loss', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Entropy
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(metrics_history['entropy'], color='green')
    ax4.set_xlabel('Update')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Gradient Norms
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(metrics_history['gradient_norms'], alpha=0.7, label='Total')
    ax5.set_xlabel('Gradient Update')
    ax5.set_ylabel('Norm')
    ax5.set_title('Gradient Norms', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. Actor vs Critic Gradients
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(metrics_history['actor_grads'], alpha=0.7, label='Actor', color='blue')
    ax6.plot(metrics_history['critic_grads'], alpha=0.7, label='Critic', color='red')
    ax6.set_xlabel('Gradient Update')
    ax6.set_ylabel('Norm')
    ax6.set_title('Actor vs Critic Gradients', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # 7. Clip Fraction
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(metrics_history['clip_fraction'], color='brown')
    ax7.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax7.set_xlabel('Update')
    ax7.set_ylabel('Fraction')
    ax7.set_title('PPO Clip Fraction', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Approximate KL Divergence
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.plot(metrics_history['approx_kl'], color='teal')
    ax8.axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='Target KL')
    ax8.set_xlabel('Update')
    ax8.set_ylabel('KL Divergence')
    ax8.set_title('Approximate KL Divergence', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Episode Length
    ax9 = fig.add_subplot(gs[3, 1])
    ax9.plot(episodes, metrics_history['episode_lengths'], color='magenta', alpha=0.6)
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Steps')
    ax9.set_title('Episode Length', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # 10. Success Rate
    ax10 = fig.add_subplot(gs[3, 2])
    window = 100
    if len(rewards) >= window:
        success_rate = []
        for i in range(window, len(rewards) + 1):
            recent_rewards = rewards[i-window:i]
            success_count = sum(1 for r in recent_rewards if r >= 200)
            success_rate.append(success_count / window * 100)
        
        ax10.plot(range(window, len(rewards) + 1), success_rate, color='darkgreen')
        ax10.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% target')
        ax10.set_xlabel('Episode')
        ax10.set_ylabel('Success Rate (%)')
        ax10.set_title(f'Success Rate (Last {window} Episodes)', fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim([0, 100])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training plots: {save_path}")


def plot_gradient_interpolation(agent, save_path, n_samples=100):
    """Analyze gradient landscape through interpolation"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Gradient Analysis & Interpolation', fontsize=16, fontweight='bold')
    
    # Recent gradient norms
    recent_grads = agent.gradient_norms[-n_samples:] if len(agent.gradient_norms) >= n_samples else agent.gradient_norms
    
    # 1. Gradient norm distribution
    ax1 = axes[0, 0]
    ax1.hist(recent_grads, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(recent_grads), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(recent_grads):.4f}')
    ax1.set_xlabel('Gradient Norm')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Gradient Norm Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradient norms over time (zoomed recent)
    ax2 = axes[0, 1]
    ax2.plot(recent_grads, color='blue', alpha=0.6)
    ax2.axhline(agent.max_grad_norm, color='red', linestyle='--', 
                label=f'Clip threshold: {agent.max_grad_norm}')
    ax2.set_xlabel('Update Step (recent)')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Recent Gradient Norms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Actor vs Critic gradient comparison
    ax3 = axes[1, 0]
    recent_actor = agent.actor_grad_norms[-n_samples:] if len(agent.actor_grad_norms) >= n_samples else agent.actor_grad_norms
    recent_critic = agent.critic_grad_norms[-n_samples:] if len(agent.critic_grad_norms) >= n_samples else agent.critic_grad_norms
    
    ax3.scatter(recent_actor, recent_critic, alpha=0.5, s=20)
    ax3.plot([0, max(max(recent_actor), max(recent_critic))], 
             [0, max(max(recent_actor), max(recent_critic))], 
             'r--', alpha=0.5, label='Equal gradients')
    ax3.set_xlabel('Actor Gradient Norm')
    ax3.set_ylabel('Critic Gradient Norm')
    ax3.set_title('Actor vs Critic Gradient Magnitudes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gradient smoothness (interpolation)
    ax4 = axes[1, 1]
    if len(recent_grads) > 10:
        # Compute gradient of gradients (second derivative approximation)
        grad_changes = np.diff(recent_grads)
        smoothness = np.abs(grad_changes)
        
        ax4.plot(smoothness, color='purple', alpha=0.7, label='Gradient smoothness')
        ax4.axhline(np.mean(smoothness), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(smoothness):.4f}')
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('|∆ Gradient Norm|')
        ax4.set_title('Gradient Smoothness (Change Rate)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved gradient analysis: {save_path}")


# ============================================================================
# Training Loop
# ============================================================================

def train_ppo(
    total_episodes=1000,
    max_steps_per_episode=1000,
    update_frequency=2048,
    gif_interval=50,
    eval_episodes=5
):
    """Train PPO agent on Lunar Lander"""
    
    print("\n" + "=" * 70)
    print("TRAINING PPO AGENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Update frequency: {update_frequency} steps")
    print(f"  GIF generation interval: every {gif_interval} episodes")
    print(f"  Evaluation episodes: {eval_episodes}")
    print()
    
    # Create environment
    env = gym.make('LunarLander-v2')
    eval_env = gym.make('LunarLander-v2', render_mode='rgb_array')
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    # Tracking
    buffer = RolloutBuffer()
    metrics_history = {
        'rewards': [],
        'episode_lengths': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'approx_kl': [],
        'clip_fraction': [],
        'gradient_norms': [],
        'actor_grads': [],
        'critic_grads': []
    }
    
    # Training loop
    episode = 0
    total_steps = 0
    best_reward = -np.inf
    start_time = time.time()
    
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    print("Starting training...\n")
    
    while episode < total_episodes:
        # Collect experience
        action, log_prob, value = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        buffer.add(state, action, reward, value, log_prob, done)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        total_steps += 1
        
        # Episode ended
        if done or episode_length >= max_steps_per_episode:
            metrics_history['rewards'].append(episode_reward)
            metrics_history['episode_lengths'].append(episode_length)
            
            episode += 1
            
            # Print progress
            if episode % 10 == 0 or episode == 1:
                avg_reward = np.mean(metrics_history['rewards'][-100:])
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(100): {avg_reward:7.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Steps: {total_steps:6d}")
            
            # Generate GIF at intervals
            if episode % gif_interval == 0:
                print(f"\n  Generating GIF at episode {episode}...")
                record_env = RecordEpisode(eval_env)
                
                eval_reward = 0
                state_eval, _ = record_env.reset()
                
                for _ in range(max_steps_per_episode):
                    action, _, _ = agent.get_action(state_eval)
                    state_eval, reward, terminated, truncated, _ = record_env.step(action)
                    eval_reward += reward
                    
                    if terminated or truncated:
                        break
                
                gif_path = GIF_DIR / f"episode_{episode:04d}_reward_{eval_reward:.0f}.gif"
                save_episode_gif(record_env.get_frames(), gif_path)
                print(f"  Evaluation reward: {eval_reward:.2f}\n")
            
            # Reset for next episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        
        # Update policy
        if total_steps % update_frequency == 0 and len(buffer.states) > 0:
            print(f"\n  Performing policy update at step {total_steps}...")
            update_metrics = agent.update(buffer)
            
            metrics_history['policy_loss'].append(update_metrics['policy_loss'])
            metrics_history['value_loss'].append(update_metrics['value_loss'])
            metrics_history['entropy'].append(update_metrics['entropy'])
            metrics_history['approx_kl'].append(update_metrics['approx_kl'])
            metrics_history['clip_fraction'].append(update_metrics['clip_fraction'])
            metrics_history['gradient_norms'].append(update_metrics['gradient_norm'])
            
            # Store gradient details
            if len(agent.actor_grad_norms) > 0:
                metrics_history['actor_grads'].append(np.mean(agent.actor_grad_norms[-100:]))
                metrics_history['critic_grads'].append(np.mean(agent.critic_grad_norms[-100:]))
            
            print(f"    Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"    Value loss: {update_metrics['value_loss']:.4f}")
            print(f"    Entropy: {update_metrics['entropy']:.4f}")
            print(f"    KL divergence: {update_metrics['approx_kl']:.4f}")
            print(f"    Clip fraction: {update_metrics['clip_fraction']:.4f}\n")
            
            buffer.clear()
            
            # Save best model
            if metrics_history['rewards'] and np.mean(metrics_history['rewards'][-100:]) > best_reward:
                best_reward = np.mean(metrics_history['rewards'][-100:])
                torch.save(agent.network.state_dict(), MODEL_DIR / "best_model.pth")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Final average reward (last 100): {np.mean(metrics_history['rewards'][-100:]):.2f}")
    print(f"Best average reward: {best_reward:.2f}")
    
    # Save final model
    torch.save(agent.network.state_dict(), MODEL_DIR / "final_model.pth")
    print(f"\nModels saved to: {MODEL_DIR}")
    
    # Generate final visualizations
    print("\nGenerating final analysis plots...")
    plot_training_metrics(metrics_history, PLOT_DIR / "training_metrics.png")
    plot_gradient_interpolation(agent, PLOT_DIR / "gradient_analysis.png")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_rewards = []
    for i in range(eval_episodes):
        record_env = RecordEpisode(eval_env)
        state_eval, _ = record_env.reset()
        eval_reward = 0
        
        for _ in range(max_steps_per_episode):
            action, _, _ = agent.get_action(state_eval)
            state_eval, reward, terminated, truncated, _ = record_env.step(action)
            eval_reward += reward
            
            if terminated or truncated:
                break
        
        final_rewards.append(eval_reward)
        gif_path = GIF_DIR / f"final_eval_{i+1}_reward_{eval_reward:.0f}.gif"
        save_episode_gif(record_env.get_frames(), gif_path)
    
    print(f"\nFinal evaluation results:")
    print(f"  Mean reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
    print(f"  Min reward: {np.min(final_rewards):.2f}")
    print(f"  Max reward: {np.max(final_rewards):.2f}")
    
    env.close()
    eval_env.close()
    
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    print(f"  - GIFs: {GIF_DIR}")
    print(f"  - Plots: {PLOT_DIR}")
    print(f"  - Models: {MODEL_DIR}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    # Quick environment test
    demo1_environment_basics()
    
    # Start training
    train_ppo(
        total_episodes=1000,
        max_steps_per_episode=1000,
        update_frequency=2048,
        gif_interval=50,
        eval_episodes=5
    )
    
    print("\n" + "=" * 70)
    print("🎉 PPO TRAINING COMPLETE!")
    print("=" * 70)
    print("\nCheck the outputs/ directory for:")
    print("  📊 Training metrics and gradient analysis plots")
    print("  🎬 GIFs showing agent improvement over time")
    print("  💾 Trained model weights")


if __name__ == "__main__":
    main()

