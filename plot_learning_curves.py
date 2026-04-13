"""
Learning curve visualizations for DQN and PPO agents on LunarLander-v3.

Generates:
  - Per-algorithm plots with mean reward and ±1 std shaded band
  - Side-by-side comparison plot
  - Smoothed curves using a 50-episode rolling window

Saves all figures to results/plots/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
RESULTS    = BASE_DIR / "results"
PLOTS_DIR  = RESULTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_WINDOW = 50          # rolling-average window for smoothing
SUCCESS_THRESHOLD = 200     # reward threshold for "success"

DQN_SEEDS = [42, 100, 200]
PPO_SEEDS = [42, 100, 200, 300, 400]

DQN_COLOR = "#2563EB"   # blue
PPO_COLOR = "#DC2626"   # red

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_metrics(algo: str, seeds: list[int]) -> dict:
    """Return dict with episode_rewards and episode_lengths per seed."""
    data = {}
    folder = RESULTS / f"{algo}_agent"
    for seed in seeds:
        path = folder / f"metrics_seed{seed}.json"
        with open(path) as f:
            raw = json.load(f)
        data[seed] = {
            "rewards":  np.array(raw["episode_rewards"]),
            "lengths":  np.array(raw["episode_lengths"]),
        }
    return data


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean; edges use shrinking windows."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def align_and_stack(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth each seed's rewards, truncate all to the shortest run,
    then return (episodes, mean, std) arrays.
    """
    smoothed = [rolling_mean(v["rewards"], SMOOTH_WINDOW) for v in data.values()]
    min_len  = min(len(s) for s in smoothed)
    matrix   = np.stack([s[:min_len] for s in smoothed])   # (n_seeds, episodes)
    episodes = np.arange(1, min_len + 1)
    return episodes, matrix.mean(axis=0), matrix.std(axis=0)


def shade_plot(ax, episodes, mean, std, color, label, alpha_band=0.20):
    ax.plot(episodes, mean, color=color, linewidth=2, label=label)
    ax.fill_between(episodes, mean - std, mean + std,
                    color=color, alpha=alpha_band, linewidth=0)


def style_axes(ax, title, xlabel="Episode", ylabel="Reward"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--",
               linewidth=1.0, label=f"Success threshold ({SUCCESS_THRESHOLD})")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dqn_data = load_metrics("dqn", DQN_SEEDS)
    ppo_data = load_metrics("ppo", PPO_SEEDS)

    dqn_ep, dqn_mean, dqn_std = align_and_stack(dqn_data)
    ppo_ep, ppo_mean, ppo_std = align_and_stack(ppo_data)

    # ── 1. DQN standalone ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    shade_plot(ax, dqn_ep, dqn_mean, dqn_std, DQN_COLOR,
               f"DQN mean (seeds: {DQN_SEEDS})")
    style_axes(ax, "DQN – Learning Curve (LunarLander-v3)")
    fig.tight_layout()
    out = PLOTS_DIR / "dqn_learning_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── 2. PPO standalone ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    shade_plot(ax, ppo_ep, ppo_mean, ppo_std, PPO_COLOR,
               f"PPO mean (seeds: {PPO_SEEDS})")
    style_axes(ax, "PPO – Learning Curve (LunarLander-v3)")
    fig.tight_layout()
    out = PLOTS_DIR / "ppo_learning_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── 3. Side-by-side comparison ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    shade_plot(axes[0], dqn_ep, dqn_mean, dqn_std, DQN_COLOR,
               f"DQN mean ± 1 std  (n={len(DQN_SEEDS)} seeds)")
    style_axes(axes[0], "DQN")

    shade_plot(axes[1], ppo_ep, ppo_mean, ppo_std, PPO_COLOR,
               f"PPO mean ± 1 std  (n={len(PPO_SEEDS)} seeds)")
    style_axes(axes[1], "PPO", ylabel="")

    fig.suptitle("DQN vs PPO – Learning Curves on LunarLander-v3",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "comparison_side_by_side.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── 4. Overlaid comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    shade_plot(ax, dqn_ep, dqn_mean, dqn_std, DQN_COLOR,
               f"DQN mean ± 1 std  (n={len(DQN_SEEDS)} seeds)")
    shade_plot(ax, ppo_ep, ppo_mean, ppo_std, PPO_COLOR,
               f"PPO mean ± 1 std  (n={len(PPO_SEEDS)} seeds)")

    style_axes(ax, "DQN vs PPO – Learning Curves on LunarLander-v3")
    fig.tight_layout()
    out = PLOTS_DIR / "comparison_overlaid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── 5. Individual seed traces (small multiples) ───────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, max(len(DQN_SEEDS), len(PPO_SEEDS)), figure=fig,
                            hspace=0.45, wspace=0.3)

    for col, (seed, v) in enumerate(dqn_data.items()):
        ax = fig.add_subplot(gs[0, col])
        smooth = rolling_mean(v["rewards"], SMOOTH_WINDOW)
        ep = np.arange(1, len(smooth) + 1)
        ax.plot(ep, v["rewards"], color=DQN_COLOR, alpha=0.18, linewidth=0.6)
        ax.plot(ep, smooth, color=DQN_COLOR, linewidth=2)
        ax.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", linewidth=0.9)
        ax.set_title(f"DQN seed={seed}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Reward", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)

    for col, (seed, v) in enumerate(ppo_data.items()):
        ax = fig.add_subplot(gs[1, col])
        smooth = rolling_mean(v["rewards"], SMOOTH_WINDOW)
        ep = np.arange(1, len(smooth) + 1)
        ax.plot(ep, v["rewards"], color=PPO_COLOR, alpha=0.18, linewidth=0.6)
        ax.plot(ep, smooth, color=PPO_COLOR, linewidth=2)
        ax.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", linewidth=0.9)
        ax.set_title(f"PPO seed={seed}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Reward", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Individual Seed Traces – DQN (top) and PPO (bottom)",
                 fontsize=13, fontweight="bold")
    out = PLOTS_DIR / "individual_seeds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
