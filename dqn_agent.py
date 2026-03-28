import json
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from datetime import datetime

class QNetwork(nn.Module):
    '''Two hidden layer Q-network with 128 Relu units each.'''''
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
    
    def forward(self, x):
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype = np.float32),
            np.array(actions, dtype = np.int64),
            np.array(rewards, dtype = np.float32),
            np.array(next_states, dtype = np.float32),
            np.array(dones, dtype = np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(
        self,
        state_size = 8,
        action_size = 4,
        lr = 1e-3,
        gamma = 0.99,
        batch_size = 64,
        buffer_capacity = 100000,
        epsilon_start = 1.0,
        epsilon_end = 0.01,
        epsilon_decay_steps = 100000,
        target_update_freq = 1000,
        seed = None
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # epsilion-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def epsilon(self):
        '''Current epsilon (Linear decay)'''
        fraction = min(self.steps_done / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, state):
        '''Epsilon-greedy action selection'''
        if random.random() < self.epsilon():
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.q_network(state).argmax(dim=1).item()
    
    def update(self):
        '''Update Q-network using a batch from replay buffer'''
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            targert_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Copute loss
        loss = nn.MSELoss()(q_values, targert_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def step(self, state, action, reward, next_state, done):
        '''Store transition, update Q-network, and sync targt network peiodically'''
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1
        loss = self.update()
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss
    
    def save(self, filepath):
        '''Save model parameters'''
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, filepath)
    
    def load(self, filepath):
        '''Load model parameters'''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        
def train_dqn_agent(
    num_episodes = 1000,
    seed = 42,
    log_interval = 50,
    save_dir = 'results/dqn_agent'
    ):
    
    '''Train DQN agent and log results'''
    os.makedirs(save_dir, exist_ok = True)
    
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    
    agent = DQNAgent(seed=seed)
    
    #Logging variables
    episode_rewards = []
    episode_lengths = []
    losses = []
    best_avg = -float("inf")
    
    print(f"Training DQN | seed={seed} | device={agent.device}")
    
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if episode_loss:
            losses.append(np.mean(episode_loss))
            
        # Periodic logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Length: {avg_length:6.1f} | "
                f"Epsilon: {agent.epsilon():.3f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )
            
            # Save best model
            if avg_reward > best_avg:
                best_avg = avg_reward
                agent.save(os.path.join(save_dir, f"best_model_seed{seed}.pt"))
        
    # Save final model at end of training
    agent.save(os.path.join(save_dir, f"final_model_seed{seed}.pt"))
    metrics = {
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "best_avg_reward": best_avg,
        "total_episodes": agent.steps_done,
        "time": datetime.now().isoformat()
    }
    with open(os.path.join(save_dir, f"metrics_seed{seed}.json"), "w") as f:
        json.dump(metrics, f)
            
    env.close()
    return metrics
        
if __name__ == "__main__":
    # Train DQN agent across multiple seeds for robustness
    seeds = [42, 100, 200]
    all_metrics = []
    
    for s in seeds:
        print(f"\n=== Training with seed {s} ===")
        m = train_dqn_agent(num_episodes=1000, seed=s)
        all_metrics.append(m)
    
    print(f"\n Summary of result across seeds:")
    best_avgs = [m["best_avg_reward"] for m in all_metrics]
    print(f"Best avg rewards: {best_avgs}")
    print(f"Mean: {np.mean(best_avgs):.2f} ± {np.std(best_avgs):.2f}")