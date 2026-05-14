from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.env.modified_pong_env import ModifiedPongEnv
from src.utils.metrics import summarize_policy_results


DEFAULT_N_ACTION_BINS = 11
DEFAULT_MAX_EPISODE_DECISIONS = 50


def make_eval_env(
    render: bool = False,
    n_action_bins: int = DEFAULT_N_ACTION_BINS,
    max_episode_decisions: int = DEFAULT_MAX_EPISODE_DECISIONS,
) -> ModifiedPongEnv:
    render_mode = "human" if render else None

    return ModifiedPongEnv(
        n_action_bins=n_action_bins,
        render_mode=render_mode,
        max_episode_decisions=max_episode_decisions,
    )


def evaluate_ppo_model(
    model: Any,
    n_episodes: int = 200,
    seed_offset: int = 20_000,
    deterministic: bool = True,
    render: bool = False,
    n_action_bins: int = DEFAULT_N_ACTION_BINS,
    max_episode_decisions: int = DEFAULT_MAX_EPISODE_DECISIONS,
) -> pd.DataFrame:
    env = make_eval_env(
        render=render,
        n_action_bins=n_action_bins,
        max_episode_decisions=max_episode_decisions,
    )

    records: list[dict[str, Any]] = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + episode)

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            action = int(np.asarray(action).item())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        records.append(
            {
                "episode": episode,
                "reward": total_reward,
                "win": 1 if info["right_score"] > info["left_score"] else 0,
                "loss": 1 if info["left_score"] > info["right_score"] else 0,
                "steps": steps,
                "right_score": info["right_score"],
                "left_score": info["left_score"],
                "score_diff": info["score_diff"],
                "frames": info["total_frames"],
                "rally_hits": info["rally_hits"],
                "decision_count": info["decision_count"],
                "truncated": truncated,
                "truncation_reason": info.get("truncation_reason", None),
                "last_right_hit_offset": info["last_right_hit_offset"],
                "last_right_speed_multiplier": info[
                    "last_right_speed_multiplier"
                ],
                "last_right_outgoing_speed": info[
                    "last_right_outgoing_speed"
                ],
                "final_ball_speed": info["current_ball_speed"],
                "last_left_outgoing_speed": info[
                    "last_left_outgoing_speed"
                ],
                "successful_agent_returns": info[
                    "successful_agent_returns"
                ],
                "last_selected_target_y": info["last_selected_target_y"],
                "last_predicted_return_y": info[
                    "last_predicted_return_y"
                ],
                "last_target_alignment_error": info[
                    "last_target_alignment_error"
                ],
                "last_target_alignment_reward": info[
                    "last_target_alignment_reward"
                ],
                "cumulative_target_alignment_reward": info[
                    "cumulative_target_alignment_reward"
                ],
            }
        )

    env.close()
    return pd.DataFrame(records)


def view_ppo_episodes(
    model: Any,
    n_episodes: int = 5,
    seed_offset: int = 200_000,
    deterministic: bool = True,
    n_action_bins: int = DEFAULT_N_ACTION_BINS,
    max_episode_decisions: int = DEFAULT_MAX_EPISODE_DECISIONS,
) -> pd.DataFrame:
    env = make_eval_env(
        render=True,
        n_action_bins=n_action_bins,
        max_episode_decisions=max_episode_decisions,
    )

    episode_summaries: list[dict[str, Any]] = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + episode)

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            action = int(np.asarray(action).item())

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        episode_summaries.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "rally_hits": info["rally_hits"],
                "right_score": info["right_score"],
                "left_score": info["left_score"],
                "truncated": truncated,
            }
        )

    env.close()
    return pd.DataFrame(episode_summaries)


def evaluate_saved_model(
    model_path: str | Path,
    policy_name: str,
    n_episodes: int = 300,
    seed_offset: int = 100_000,
    deterministic: bool = True,
    device: str = "cpu",
    n_action_bins: int = DEFAULT_N_ACTION_BINS,
    max_episode_decisions: int = DEFAULT_MAX_EPISODE_DECISIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved PPO model was not found: {model_path}")

    model = PPO.load(
        str(model_path),
        device=device,
    )

    episode_df = evaluate_ppo_model(
        model=model,
        n_episodes=n_episodes,
        seed_offset=seed_offset,
        deterministic=deterministic,
        render=False,
        n_action_bins=n_action_bins,
        max_episode_decisions=max_episode_decisions,
    )

    summary = summarize_policy_results(
        name=policy_name,
        df=episode_df,
    )

    summary_df = pd.DataFrame([summary])

    return episode_df, summary_df


def evaluate_checkpoint_series(
    checkpoint_dir: str | Path,
    checkpoint_steps: Iterable[int],
    checkpoint_prefix: str = "ppo_checkpoint",
    n_episodes: int = 300,
    seed_offset: int = 100_000,
    deterministic: bool = True,
    device: str = "cpu",
    n_action_bins: int = DEFAULT_N_ACTION_BINS,
    max_episode_decisions: int = DEFAULT_MAX_EPISODE_DECISIONS,
) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_summaries: list[dict[str, Any]] = []

    for step in checkpoint_steps:
        checkpoint_path = checkpoint_dir / (
            f"{checkpoint_prefix}_{step}_steps.zip"
        )

        if not checkpoint_path.exists():
            continue

        checkpoint_model = PPO.load(
            str(checkpoint_path),
            env=None,
            device=device,
        )

        checkpoint_eval_df = evaluate_ppo_model(
            model=checkpoint_model,
            n_episodes=n_episodes,
            seed_offset=seed_offset,
            deterministic=deterministic,
            render=False,
            n_action_bins=n_action_bins,
            max_episode_decisions=max_episode_decisions,
        )

        checkpoint_summary = summarize_policy_results(
            name=f"ppo_checkpoint_{step}",
            df=checkpoint_eval_df,
        )

        checkpoint_summary["checkpoint_step"] = step
        checkpoint_summaries.append(checkpoint_summary)

    return pd.DataFrame(checkpoint_summaries)
