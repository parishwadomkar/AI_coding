from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from src.training.evaluate_policy import evaluate_ppo_model
from src.utils.metrics import summarize_policy_results


class PeriodicPPOEvaluationCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 100,
        seed_offset: int = 50_000,
        best_model_dir: str | Path = "models/best_model",
        n_action_bins: int = 11,
        max_episode_decisions: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed_offset = seed_offset
        self.best_model_dir = Path(best_model_dir)
        self.n_action_bins = n_action_bins
        self.max_episode_decisions = max_episode_decisions
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_history: list[dict[str, Any]] = []

        self.best_win_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.best_timestep: int | None = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        eval_df = evaluate_ppo_model(
            model=self.model,
            n_episodes=self.n_eval_episodes,
            seed_offset=self.seed_offset,
            deterministic=True,
            render=False,
            n_action_bins=self.n_action_bins,
            max_episode_decisions=self.max_episode_decisions,
        )

        summary = summarize_policy_results(
            name="ppo_eval",
            df=eval_df,
        )

        summary["timesteps"] = int(self.num_timesteps)
        self.evaluation_history.append(summary)

        current_win_rate = summary["win_rate"]
        current_mean_reward = summary["mean_reward"]

        improved = (
            current_win_rate > self.best_win_rate
            or (
                current_win_rate == self.best_win_rate
                and current_mean_reward > self.best_mean_reward
            )
        )

        if improved:
            self.best_win_rate = current_win_rate
            self.best_mean_reward = current_mean_reward
            self.best_timestep = int(self.num_timesteps)

            best_model_path = (
                self.best_model_dir / "ppo_best_by_win_rate"
            )

            self.model.save(str(best_model_path))

            best_summary_df = pd.DataFrame([summary])
            best_summary_df.to_csv(
                self.best_model_dir / "ppo_best_by_win_rate_summary.csv",
                index=False,
            )

            if self.verbose:
                print(
                    f"[New best model at {self.num_timesteps:>7} steps] "
                    f"win_rate={current_win_rate:.3f}, "
                    f"mean_reward={current_mean_reward:.3f}"
                )

        if self.verbose:
            print(
                f"[Evaluation at {self.num_timesteps:>7} steps] "
                f"mean_reward={summary['mean_reward']:.3f}, "
                f"win_rate={summary['win_rate']:.3f}, "
                f"loss_rate={summary['loss_rate']:.3f}, "
                f"mean_steps={summary['mean_steps']:.2f}"
            )

        return True
    