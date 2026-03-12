import gymnasium as gym
import numpy as np

def test_env():
    env = gym.make("LunarLander-v3")

    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Actions: {env.action_space.n}")

    observation, info = env.reset(seed=42)
    print(f"\nInitial Observation: {observation}")
    print(f"Observation Shape: {observation.shape}")

    print("\nRunning test episode with random actions")
    total_reward = 0
    steps = 0

    for _ in range (100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            print(f"Episode ended after {steps} steps with total reward: {total_reward}")
            break

    env.close()
    print("Environment test completed successfully.")
    return True

if __name__ == "__main__":
    test_env()
