import gymnasium as gym
import time

def watch_random_lander(num_episodes=5):
    """
    Watch the lander with random actions (no training).
    
    Args:
        num_episodes: Number of episodes to watch
    """
    # Create environment with visual rendering
    env = gym.make("LunarLander-v3", render_mode="human")
    
    print(f"Episodes to watch: {num_episodes}")
    print("The lander takes RANDOM actions (not trained)")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Take random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Small delay to make it watchable
            time.sleep(0.01)
            
            if terminated or truncated:
                print(f"Finished: {steps} steps, Reward: {total_reward:.2f}")
                time.sleep(1.5)  # Pause between episodes
                break
    
    env.close()

if __name__ == "__main__":
    # Watch 5 episodes by default
    # Change the number below to watch more or fewer
    watch_random_lander(num_episodes=5)