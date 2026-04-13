# Reinforcement Learning Agents for LunarLander

Comparative study of **DQN** and **PPO** on Gymnasium's `LunarLander-v3` environment.

## Results Summary
| Metric                       | DQN              | PPO                         |
|------------------------------|------------------|-----------------------------|
| Best Avg Reward (mean ± std) | 256.24 ± 9.63    | 282.11 ± 6.92               |
| Seeds                        | 3 (42, 100, 200) | 5 (42, 100, 200, 300, 400)  |
| Peak Single-Seed Reward      | 269.81           | 291.28                      |
| Training Duration            | 2,000 episodes   | 750,000 steps               |


---

## Setup

```bash
python -m venv LunarLanderEnv
source LunarLanderEnv/bin/activate       # Windows: LunarLanderEnv\Scripts\activate
pip install -r requirements.txt
```

> Requires Python 3.14.3

---

## Project Structure

```
.
├── test_env.py              # Verify environment observation/action spaces
├── random_base_agent.py     # Random-policy baseline with human rendering
├── dqn_agent.py             # DQN training script
├── ppo_agent.py             # PPO training script
├── trained_dqn_agent.py     # Watch a trained DQN agent
├── trained_ppo_agent.py     # Watch a trained PPO agent
├── plot_learning_curves.py  # Generate reward-vs-episode learning curve plots
├── compute_metrics.py       # Compute extended evaluation metrics
└── results/
    ├── dqn_agent/           # DQN model checkpoints and metrics JSON per seed
    ├── ppo_agent/           # PPO model checkpoints and metrics JSON per seed
    └── plots/               # All generated figures
```

---

## Reproducing Experiments

### 1. Verify the environment
```bash
python test_env.py
```

### 2. Run the random baseline
```bash
python random_base_agent.py
```

### 3. Train DQN (seeds 42, 100, 200)
```bash
python dqn_agent.py
```
Saves checkpoints and `metrics_seed<N>.json` to `results/dqn_agent/`.

### 4. Train PPO (seeds 42, 100, 200, 300, 400)
```bash
python ppo_agent.py
```
Saves checkpoints and `metrics_seed<N>.json` to `results/ppo_agent/`.

### 5. Watch trained agents
```bash
python trained_dqn_agent.py   # loads best_model_seed42.pt by default
python trained_ppo_agent.py   # loads best_model_seed42.pt by default
```

### 6. Generate learning curve plots
```bash
python plot_learning_curves.py
```
Outputs to `results/plots/`: `dqn_learning_curve.png`, `ppo_learning_curve.png`, `comparison_side_by_side.png`, `comparison_overlaid.png`, `individual_seeds.png`.

### 7. Compute extended metrics
```bash
python compute_metrics.py
```
Prints a comparison table (success rate, convergence speed, avg episode length) and saves `results/plots/metrics_comparison.png`.

---

## Hyperparameters

### DQN
| Parameter                 | Value                          |
|---------------------------|--------------------------------|
| Hidden layers             | 2 × 128 ReLU                   |
| Replay buffer capacity    | 100,000                        |
| Batch size                | 64                             |
| Learning rate             | 5e-4                           |
| Discount factor (γ)       | 0.99                           |
| Epsilon decay             | 1.0 → 0.01 over 150,000 steps  |
| Target network sync       | Every 1,000 steps              |
| Gradient clip norm        | 1.0                            |
| Training episodes         | 2,000                          |

### PPO
| Parameter              | Value                          |
|------------------------|--------------------------------|
| Hidden layers          | 2 × 128 ReLU (shared backbone) |
| Learning rate          | 3e-4                           |
| Discount factor (γ)    | 0.99                           |
| GAE lambda (λ)         | 0.95                           |
| Clip ratio (ε)         | 0.2 (range [0.8, 1.2])         |
| Entropy coefficient    | 0.02                           |
| Rollout length         | 2,048 steps                    |
| Epochs per rollout     | 10                             |
| Mini-batch size        | 64                             |
| Total timesteps        | 750,000                        |

