import gymnasium as gym
import torch
import time
from dqn_agent import DQNAgent

def watch_trained_dqn_agent(filepath="results/dqn-agent/best_model_seed200.pt", num_episodes=5):
    '''
    
    Watch the trained DQN agent
    Args:
        filepath: Path to the saved model checkpoint
        num_episodes: Number of episodes to watch
        
    '''
    # Load trained agent
    agent = DQNAgent()
    agent.load(filepath)
    print(f"Loaded DQn agent from {filepath} | device: {agent.device}")
    
    # Create environment with visual rendering
    env = gym.make("LunarLander-v3", render_mode="human")
    
    print(f"Episodes to watch: {num_episodes}")
    print("The lander takes actions based on the trained DQN agent")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action using the trained agent (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action = agent.q_network(state_tensor).argmax(1).item()
            
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Delay to make it watchable
            time.sleep(0.01)
            
            if terminated or truncated:
                status = "Landed Successfully!" if terminated and total_reward > 200 else "Crashed!"
                print(f"Episode {episode + 1} finished after {steps} steps | Total Reward: {total_reward:.2f} | Status: {status}")
                time.sleep(1.5)
                break
    
    env.close()

if __name__ == "__main__":
    # Change the filepath below to watch a different trained model
    # Change the number of episodes to watch mor or fewer
    watch_trained_dqn_agent(filepath="results/dqn_agent/best_model_seed200.pt", num_episodes=5)  
        
            
            
            