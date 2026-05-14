from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.baselines.pong_baselines import run_all_baselines
from src.training.evaluate_policy import (
    evaluate_checkpoint_series,
    evaluate_saved_model,
)
from src.training.train_ppo import (
    BEST_MODEL_DIR,
    CHECKPOINT_DIR,
    FINAL_MODEL_DIR,
    MAX_EPISODE_DECISIONS,
    N_ACTION_BINS,
    TOTAL_TIMESTEPS,
    TRAINING_LOG_DIR,
)
from src.utils.plotting import (
    plot_final_policy_win_rate_comparison,
    save_all_training_figures,
)


N_FINAL_EVAL_EPISODES = 300
FINAL_EVAL_SEED_OFFSET = 100_000
CHECKPOINT_STEPS = list(
    range(50_000, TOTAL_TIMESTEPS + 1, 50_000)
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "evaluation_tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_analysis_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_trained_ppo_models() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    final_model_path = FINAL_MODEL_DIR / "ppo_final_model.zip"
    best_model_path = BEST_MODEL_DIR / "ppo_best_by_win_rate.zip"

    final_episode_df, final_summary_df = evaluate_saved_model(
        model_path=final_model_path,
        policy_name="ppo_final",
        n_episodes=N_FINAL_EVAL_EPISODES,
        seed_offset=FINAL_EVAL_SEED_OFFSET,
        deterministic=True,
        device="cpu",
        n_action_bins=N_ACTION_BINS,
        max_episode_decisions=MAX_EPISODE_DECISIONS,
    )

    final_episode_df.to_csv(
        TABLES_DIR / "ppo_final_episode_details.csv",
        index=False,
    )
    final_summary_df.to_csv(
        TABLES_DIR / "ppo_final_summary.csv",
        index=False,
    )

    best_episode_df = None
    best_summary_df = None

    if best_model_path.exists():
        best_episode_df, best_summary_df = evaluate_saved_model(
            model_path=best_model_path,
            policy_name="ppo_best_by_win_rate",
            n_episodes=N_FINAL_EVAL_EPISODES,
            seed_offset=FINAL_EVAL_SEED_OFFSET,
            deterministic=True,
            device="cpu",
            n_action_bins=N_ACTION_BINS,
            max_episode_decisions=MAX_EPISODE_DECISIONS,
        )

        best_episode_df.to_csv(
            TABLES_DIR / "ppo_best_episode_details.csv",
            index=False,
        )
        best_summary_df.to_csv(
            TABLES_DIR / "ppo_best_summary.csv",
            index=False,
        )

    return (
        final_episode_df,
        final_summary_df,
        best_episode_df,
        best_summary_df,
    )


def evaluate_training_checkpoints() -> pd.DataFrame:
    checkpoint_summary_df = evaluate_checkpoint_series(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_steps=CHECKPOINT_STEPS,
        checkpoint_prefix="ppo_checkpoint",
        n_episodes=N_FINAL_EVAL_EPISODES,
        seed_offset=FINAL_EVAL_SEED_OFFSET,
        deterministic=True,
        device="cpu",
        n_action_bins=N_ACTION_BINS,
        max_episode_decisions=MAX_EPISODE_DECISIONS,
    )

    checkpoint_summary_df.to_csv(
        TABLES_DIR / "ppo_checkpoint_summary.csv",
        index=False,
    )

    return checkpoint_summary_df


def evaluate_non_learning_baselines() -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    baseline_summary_df, baseline_details = run_all_baselines(
        n_episodes=N_FINAL_EVAL_EPISODES,
        max_episode_decisions=MAX_EPISODE_DECISIONS,
        seed_offset=FINAL_EVAL_SEED_OFFSET,
    )

    baseline_summary_df.to_csv(
        TABLES_DIR / "baseline_summary.csv",
        index=False,
    )

    for policy_name, detail_df in baseline_details.items():
        detail_df.to_csv(
            TABLES_DIR / f"baseline_{policy_name}_episode_details.csv",
            index=False,
        )

    return baseline_summary_df, baseline_details


def create_final_policy_comparison_table(
    baseline_summary_df: pd.DataFrame,
    final_summary_df: pd.DataFrame,
    best_summary_df: pd.DataFrame | None,
) -> pd.DataFrame:
    comparison_frames = [
        baseline_summary_df,
        final_summary_df,
    ]

    if best_summary_df is not None:
        comparison_frames.append(best_summary_df)

    comparison_df = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    comparison_columns = [
        "policy",
        "episodes",
        "win_rate",
        "loss_rate",
        "mean_reward",
        "mean_score_diff",
        "mean_steps",
        "mean_successful_agent_returns",
        "mean_final_target_alignment_error",
        "mean_final_target_alignment_reward",
    ]

    final_comparison_table = (
        comparison_df[comparison_columns]
        .sort_values("win_rate", ascending=False)
        .reset_index(drop=True)
    )

    final_comparison_table.to_csv(
        TABLES_DIR / "final_policy_comparison_table.csv",
        index=False,
    )

    return final_comparison_table


def create_training_figures() -> list[Path]:
    history_path = (
        TRAINING_LOG_DIR / "ppo_training_evaluation_history.csv"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            "Training evaluation history was not found. "
            f"Expected file: {history_path}"
        )

    history_df = pd.read_csv(history_path)

    history_df.to_csv(
        TABLES_DIR / "ppo_training_evaluation_history.csv",
        index=False,
    )

    return save_all_training_figures(
        history_df=history_df,
        output_dir=FIGURES_DIR,
    )


def main() -> None:
    ensure_analysis_directories()

    print("Evaluating final and best PPO models...")
    _, final_summary_df, _, best_summary_df = evaluate_trained_ppo_models()

    print("Evaluating PPO checkpoints...")
    checkpoint_summary_df = evaluate_training_checkpoints()

    print("Evaluating non-learning baselines...")
    baseline_summary_df, _ = evaluate_non_learning_baselines()

    print("Creating final policy comparison table...")
    final_comparison_table = create_final_policy_comparison_table(
        baseline_summary_df=baseline_summary_df,
        final_summary_df=final_summary_df,
        best_summary_df=best_summary_df,
    )

    print("Generating training figures...")
    saved_training_figures = create_training_figures()

    print("Generating final policy comparison figure...")
    comparison_figure_path = plot_final_policy_win_rate_comparison(
        final_comparison_table,
        FIGURES_DIR / "final_policy_win_rate_comparison.png",
    )

    print("\nFinal policy comparison:")
    print(final_comparison_table.to_string(index=False))

    print("\nCheckpoint summary:")
    if checkpoint_summary_df.empty:
        print("No checkpoint summaries were generated.")
    else:
        print(checkpoint_summary_df.to_string(index=False))

    print("\nSaved figures:")
    for path in saved_training_figures:
        print(path)

    print(comparison_figure_path)

    print("\nPost-training analysis completed.")


if __name__ == "__main__":
    main()
