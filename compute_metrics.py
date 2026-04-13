"""
Extended evaluation metrics for DQN and PPO agents on LunarLander-v3.

Computes per-seed and aggregate statistics:
  - Success rate       : % episodes with reward > 200
  - Convergence speed  : first episode where 100-ep rolling average >= 200
  - Average episode length: mean steps per episode (fuel-efficiency proxy)

Prints a formatted comparison table and saves a summary bar chart to
results/plots/metrics_comparison.png.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
RESULTS    = BASE_DIR / "results"
PLOTS_DIR  = RESULTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SUCCESS_THRESHOLD  = 200   # reward > this counts as a successful landing
CONVERGENCE_WINDOW = 100   # rolling window for convergence check
NOT_CONVERGED      = -1    # sentinel: agent never reached rolling avg >= 200

DQN_SEEDS = [42, 100, 200]
PPO_SEEDS = [42, 100, 200, 300, 400]

# ── Data loading ─────────────────────────────────────────────────────────────

def load_metrics(algo: str, seeds: list[int]) -> dict:
    folder = RESULTS / f"{algo}_agent"
    data   = {}
    for seed in seeds:
        with open(folder / f"metrics_seed{seed}.json") as f:
            raw = json.load(f)
        data[seed] = {
            "rewards": np.array(raw["episode_rewards"]),
            "lengths": np.array(raw["episode_lengths"]),
        }
    return data

# ── Metric helpers ───────────────────────────────────────────────────────────

def success_rate(rewards: np.ndarray) -> float:
    """Fraction of episodes where reward exceeded the success threshold."""
    return float((rewards > SUCCESS_THRESHOLD).mean()) * 100.0


def convergence_episode(rewards: np.ndarray, window: int = CONVERGENCE_WINDOW) -> int:
    """
    First episode index (1-based) at which the trailing rolling average
    of `window` episodes first reaches SUCCESS_THRESHOLD.
    Returns NOT_CONVERGED if it never happens.
    """
    cumsum = np.cumsum(rewards)
    for i in range(window - 1, len(rewards)):
        avg = (cumsum[i] - (cumsum[i - window] if i >= window else 0)) / window
        if avg >= SUCCESS_THRESHOLD:
            return i + 1   # 1-based episode number
    return NOT_CONVERGED


def avg_episode_length(lengths: np.ndarray) -> float:
    return float(lengths.mean())

# ── Per-algo summary ─────────────────────────────────────────────────────────

def summarise(algo: str, data: dict) -> dict:
    """Return per-seed rows and aggregate stats."""
    rows = {}
    for seed, v in data.items():
        rows[seed] = {
            "success_rate":   success_rate(v["rewards"]),
            "convergence_ep": convergence_episode(v["rewards"]),
            "avg_ep_length":  avg_episode_length(v["lengths"]),
        }

    sr_vals  = [r["success_rate"]   for r in rows.values()]
    ce_vals  = [r["convergence_ep"] for r in rows.values() if r["convergence_ep"] != NOT_CONVERGED]
    el_vals  = [r["avg_ep_length"]  for r in rows.values()]

    agg = {
        "success_rate_mean": np.mean(sr_vals),
        "success_rate_std":  np.std(sr_vals),
        "convergence_mean":  np.mean(ce_vals)   if ce_vals else float("nan"),
        "convergence_std":   np.std(ce_vals)    if ce_vals else float("nan"),
        "n_converged":       len(ce_vals),
        "n_seeds":           len(rows),
        "avg_len_mean":      np.mean(el_vals),
        "avg_len_std":       np.std(el_vals),
    }
    return {"algo": algo, "rows": rows, "agg": agg}

# ── Printing ─────────────────────────────────────────────────────────────────

def print_summary(s: dict):
    algo = s["algo"].upper()
    agg  = s["agg"]
    rows = s["rows"]

    print(f"\n{'═' * 62}")
    print(f"  {algo}")
    print(f"{'═' * 62}")
    print(f"  {'Seed':<8} {'Success Rate':>14} {'Convergence Ep':>16} {'Avg Ep Length':>15}")
    print(f"  {'-'*8} {'-'*14} {'-'*16} {'-'*15}")

    for seed, r in rows.items():
        conv = str(r["convergence_ep"]) if r["convergence_ep"] != NOT_CONVERGED else "not converged"
        print(f"  {seed:<8} {r['success_rate']:>13.1f}% {conv:>16} {r['avg_ep_length']:>14.1f}")

    print(f"  {'-'*8} {'-'*14} {'-'*16} {'-'*15}")
    conv_str = (
        f"{agg['convergence_mean']:.0f} ± {agg['convergence_std']:.0f} "
        f"({agg['n_converged']}/{agg['n_seeds']} seeds)"
        if not np.isnan(agg["convergence_mean"])
        else "none converged"
    )
    print(f"  {'Mean':<8} {agg['success_rate_mean']:>13.1f}% "
          f"{conv_str:>16}  {agg['avg_len_mean']:>13.1f}")
    print(f"  {'Std':<8} {agg['success_rate_std']:>13.1f}%"
          f"{' ':>18} {agg['avg_len_std']:>14.1f}")


def print_comparison(dqn: dict, ppo: dict):
    da, pa = dqn["agg"], ppo["agg"]

    def winner(dqn_val, ppo_val, higher_is_better=True):
        if np.isnan(dqn_val) or np.isnan(ppo_val):
            return "  —  "
        if higher_is_better:
            return "DQN >" if dqn_val > ppo_val else "PPO >"
        else:
            return "DQN <" if dqn_val < ppo_val else "PPO <"

    print(f"\n{'═' * 62}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'═' * 62}")
    print(f"  {'Metric':<30} {'DQN':>10} {'PPO':>10} {'Better':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")

    print(f"  {'Success Rate (mean %)' :<30} "
          f"{da['success_rate_mean']:>9.1f}% "
          f"{pa['success_rate_mean']:>9.1f}% "
          f"{winner(da['success_rate_mean'], pa['success_rate_mean']):>8}")

    dconv = da['convergence_mean'] if not np.isnan(da['convergence_mean']) else float('inf')
    pconv = pa['convergence_mean'] if not np.isnan(pa['convergence_mean']) else float('inf')
    conv_dqn = f"{da['convergence_mean']:.0f}" if not np.isnan(da['convergence_mean']) else "N/A"
    conv_ppo = f"{pa['convergence_mean']:.0f}" if not np.isnan(pa['convergence_mean']) else "N/A"
    print(f"  {'Convergence Speed (ep)' :<30} "
          f"{conv_dqn:>10} "
          f"{conv_ppo:>10} "
          f"{winner(dconv, pconv, higher_is_better=False):>8}")

    print(f"  {'Avg Episode Length (steps)':<30} "
          f"{da['avg_len_mean']:>10.1f} "
          f"{pa['avg_len_mean']:>10.1f} "
          f"{winner(da['avg_len_mean'], pa['avg_len_mean'], higher_is_better=False):>8}")

    print(f"\n  Notes:")
    print(f"    Success threshold : reward > {SUCCESS_THRESHOLD}")
    print(f"    Convergence window: {CONVERGENCE_WINDOW}-episode rolling average")
    print(f"    Shorter episodes  = better fuel efficiency")
    print(f"{'═' * 62}\n")

# ── Bar chart ────────────────────────────────────────────────────────────────

def plot_metrics_comparison(dqn: dict, ppo: dict):
    da, pa = dqn["agg"], ppo["agg"]

    metrics    = ["Success Rate (%)", "Convergence Speed\n(episodes)", "Avg Episode\nLength (steps)"]
    dqn_vals   = [da["success_rate_mean"], da["convergence_mean"], da["avg_len_mean"]]
    ppo_vals   = [pa["success_rate_mean"], pa["convergence_mean"], pa["avg_len_mean"]]
    dqn_errs   = [da["success_rate_std"],  da["convergence_std"],  da["avg_len_std"]]
    ppo_errs   = [pa["success_rate_std"],  pa["convergence_std"],  pa["avg_len_std"]]

    x     = np.arange(len(metrics))
    width = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors = {"DQN": "#2563EB", "PPO": "#DC2626"}

    for i, (ax, metric, dv, pv, de, pe) in enumerate(
        zip(axes, metrics, dqn_vals, ppo_vals, dqn_errs, ppo_errs)
    ):
        bars = ax.bar(
            [0, 1], [dv, pv],
            width=0.5,
            color=[colors["DQN"], colors["PPO"]],
            yerr=[de, pe],
            capsize=6,
            error_kw={"elinewidth": 1.5},
            alpha=0.85,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["DQN", "PPO"], fontsize=11)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        # value labels on bars
        for bar, val in zip(bars, [dv, pv]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(de, pe) * 0.05,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # annotate lower = better for convergence & length
        if i > 0:
            ax.set_ylabel("← lower is better", fontsize=9, color="gray")

    fig.suptitle("DQN vs PPO – Extended Evaluation Metrics (LunarLander-v3)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "metrics_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    dqn_data = load_metrics("dqn", DQN_SEEDS)
    ppo_data = load_metrics("ppo", PPO_SEEDS)

    dqn_summary = summarise("dqn", dqn_data)
    ppo_summary = summarise("ppo", ppo_data)

    print_summary(dqn_summary)
    print_summary(ppo_summary)
    print_comparison(dqn_summary, ppo_summary)

    plot_metrics_comparison(dqn_summary, ppo_summary)


if __name__ == "__main__":
    main()
