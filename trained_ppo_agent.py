import gymnasium as gym
import torch
import time
from ppo_agent import PPOAgent

def watch_trained_ppo_agent(filepath="results/ppo_agent/best_model_seed200.pt", num_episodes=5):
    '''
    Watch the trained PPO agent
    Args:
        filepath: Path to the saved model checkpoint
        num_epsodes: Number of episodes to watch
    '''
    
    # Load the trained agent
    agent = PPOAgent()
    agent.load(filepath)
    agent.network.eval() # Set to evaluation mode
    print(f"Loaded PPO agent from {filepath} | device: {agent.device}")
    
    # Create environment with visual rendering
    env = gym.make("LunarLander-v3", render_mode="human")
    
    print(f"Episodes to watch: {num_episodes}")
    print("The lander takes action based on the trained PPO agent")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action using the trained agent (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                logits, _ = agent.network(state_tensor)
                action = logits.argmax(dim=1).item()
            
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Delay to make it watchable
            time.sleep(0.01)
            
            if terminated or truncated:
                status = "Landed Successfully!" if terminated and total_reward > 200 else "Crashed!"
                print(f"Episode {episode}: {steps} steps | Reward: {total_reward:.2f} | Status: {status}")
                time.sleep(1.5)
                break
            
    env.close()
    
if __name__ == "__main__":
    # Change the filepath below to watch a different trained model
    # Change the number of episodes to watch more or fewer
    watch_trained_ppo_agent(filepath="results/ppo_agent/best_model_seed200.pt", num_episodes=5)