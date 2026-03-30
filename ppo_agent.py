import json
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime

# Actor - Critic Network for PPO Agent
class ActorCriticNetwork(nn.Module):
    ''' Two hidden layer Actor - Critic network with 128 ReLU units each. '''
    def __init__(self, state_size, action_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value

# Rollout Buffer for PPO Agent
class RolloutBuffer:
    ''' Stores trajectories for PPO agent. '''
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)

# PPO Agent Implementation
class PPOAgent:
    def __init__(
        self,
        state_size = 8,
        action_size = 4,
        lr = 3e-4,
        gamma = 0.99,
        gae_lamda = 0.95,
        clip_epsilon = 0.2,
        entropy_coefficient = 0.02,
        value_coefficient = 0.5,
        max_grad_norm = 0.5,
        update_epochs = 10,
        batch_size = 64,
        rollout_length = 2048,
        seed = None
    ):
        self.gamma = gamma
        self.gae_lamda = gae_lamda
        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        
    def select_action(self, state):
        ''' Sample action from the policy and return action, log probability and value. '''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.network(state_tensor)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, next_value):
        ''' Compute generalized advantage estimation (GAE) for the collected trajectory. '''
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values
        
        advantages = []
        gae = 0
        
        # Start from the last step and move backwards
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lamda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        return advantages, returns
    
    def update(self, next_value):
        ''' Run PPO update using the collected trajectory in the buffer. '''
        advantages, returns = self.compute_gae(next_value)
        
        # Convert vuffer to tensors
        states = torch.FloatTensor(self.buffer.states).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss_value = 0
        num_updates = 0
        
        for epoch in range(self.update_epochs):
            # Create mini-batch indices
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                logits, value = self.network(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss_value += loss.item()
                num_updates += 1
                
        self.buffer.clear()
        return total_loss_value / max(num_updates, 1)
    
    def save(self, filepath):
        ''' Save model parameters and training state. '''
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        ''' Load model parameters and training state. '''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

''' Training loop for PPO agent. '''
def train_ppo_agent(
    total_timesteps = 750000,
    seed = 42,
    log_interval = 20,
    save_dir = 'results/ppo_agent'
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    
    agent = PPOAgent(seed=seed)
    
    # Logging variables
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    best_avg_reward = -float("inf")
    episodes_completed = 0
    total_steps = 0
    
    state, info = env.reset()
    
    print(f"Training PPO | seed={seed} | device={agent.device}")
    print(f"Total timesteps: {total_timesteps} | Rollout steps: {agent.rollout_length} | Update epochs: {agent.update_epochs}")
    print("-" * 60)
    
    while total_steps < total_timesteps:
        # Collect rollout
        for _ in range(agent.rollout_length):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.buffer.store(state, action, log_prob, reward, float(done), value)
            
            current_episode_reward += reward
            current_episode_length += 1
            total_steps += 1
            state = next_state
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episodes_completed += 1
                current_episode_reward = 0
                current_episode_length = 0
                state, info = env.reset()
                
            if total_steps >= total_timesteps:
                break
            
        # Compute next value for GAE
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            _, next_value = agent.network(next_state_tensor)
            next_value = next_value.item()
            
        # PPO update
        avg_loss = agent.update(next_value)
        
        # Logging
        if len(episode_rewards) >= log_interval:
            recent = episode_rewards[-log_interval:]
            average_reward = np.mean(recent)
            average_length = np.mean(episode_lengths[-log_interval:])
            print(
                f"Steps: {total_steps:7d} | "
                f"Episodes: {episodes_completed:4d} | "
                f"Avg Reward: {average_reward:8.2f} | "
                f"Avg Len: {average_length:6.1f} | "
                f"Loss: {avg_loss:.4f}"
            )
            
            # Save the best model based on average reward
            if average_reward > best_avg_reward:
                best_avg_reward = average_reward
                agent.save(os.path.join(save_dir, f"best_model_seed{seed}.pt"))
            
    # Save the final model at end of training
    agent.save(os.path.join(save_dir, f"final_model_seed{seed}.pt"))
    metrics = {
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "best_avg_reward": best_avg_reward,
        "total_episodes": episodes_completed,
        "total_steps": total_steps,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(save_dir, f"metrics_seed{seed}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    env.close()
    print(f"\nTraining Complete!. Best Avg Reward: {best_avg_reward:.2f}")
    print(f"Episodes completed: {episodes_completed} | Total steps: {total_steps}")
    return metrics

if __name__ == "__main__":
    seeds = [42, 100, 200, 300, 400]
    all_metrics = []
    
    for seed in seeds:
        print(f"\n=== Training with seed {seed} ===")
        m = train_ppo_agent(seed=seed, total_timesteps=750000)
        all_metrics.append(m)
        
    # Summary of results across seeds
    print(f"\nSummary across seeds:")
    best_avgs = [m["best_avg_reward"] for m in all_metrics]
    print(f"Best average rewards: {best_avgs}")
    print(f"Mean: {np.mean(best_avgs):.2f} ± {np.std(best_avgs):.2f}")
        
                
            