from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_win_rate_over_training(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["win_rate"],
        marker="o",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Win rate")
    plt.title("PPO win rate over training")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_mean_score_difference_over_training(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["mean_score_diff"],
        marker="o",
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean score difference")
    plt.title("Mean score difference during PPO training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_target_alignment_error_over_training(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["mean_final_target_alignment_error"],
        marker="o",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean final target alignment error [pixels]")
    plt.title("Target alignment error during PPO training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_target_alignment_reward_over_training(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["mean_final_target_alignment_reward"],
        marker="o",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean final target alignment reward")
    plt.title("Target alignment reward during PPO training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_successful_returns_and_decision_depth(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["mean_successful_agent_returns"],
        marker="o",
        label="Successful agent returns",
    )
    plt.plot(
        history_df["timesteps"],
        history_df["mean_steps"],
        marker="s",
        label="Decision steps",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode interaction count")
    plt.title("Multi-return predictive control during training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_final_speed_multiplier_over_training(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["mean_final_speed_multiplier"],
        marker="o",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean final speed multiplier")
    plt.title("Use of offensive rebound physics during training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_final_outgoing_speed_over_training(
    history_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["timesteps"],
        history_df["mean_final_outgoing_speed"],
        marker="o",
    )
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean final outgoing ball speed")
    plt.title("Outgoing shot speed during PPO training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_final_policy_win_rate_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = _prepare_output_path(output_path)

    plot_df = comparison_df.sort_values(
        "win_rate",
        ascending=False,
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        plot_df["policy"],
        plot_df["win_rate"],
    )
    plt.ylabel("Win rate")
    plt.title("Final policy comparison on held-out evaluation episodes")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_all_training_figures(
    history_df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = [
        plot_win_rate_over_training(
            history_df,
            output_dir / "ppo_win_rate_over_training.png",
        ),
        plot_mean_score_difference_over_training(
            history_df,
            output_dir / "ppo_mean_score_difference_over_training.png",
        ),
        plot_target_alignment_error_over_training(
            history_df,
            output_dir / "ppo_target_alignment_error_over_training.png",
        ),
        plot_target_alignment_reward_over_training(
            history_df,
            output_dir / "ppo_target_alignment_reward_over_training.png",
        ),
        plot_successful_returns_and_decision_depth(
            history_df,
            output_dir / "ppo_successful_returns_and_decision_depth.png",
        ),
        plot_final_speed_multiplier_over_training(
            history_df,
            output_dir / "ppo_final_speed_multiplier_over_training.png",
        ),
        plot_final_outgoing_speed_over_training(
            history_df,
            output_dir / "ppo_final_outgoing_speed_over_training.png",
        ),
    ]

    return saved_paths