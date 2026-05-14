from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.env.modified_pong_env import ModifiedPongEnv
from src.training.callbacks import PeriodicPPOEvaluationCallback


SEED = 2026
N_ACTION_BINS = 11
MAX_EPISODE_DECISIONS = 50

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
CHECKPOINT_FREQ = 50_000
N_EVAL_EPISODES_CALLBACK = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final"
BEST_MODEL_DIR = MODELS_DIR / "best_model"

LOGS_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
TRAINING_LOG_DIR = LOGS_DIR / "training_logs"


def ensure_project_directories() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)


def make_train_env(seed: int = SEED) -> Monitor:
    env = ModifiedPongEnv(
        n_action_bins=N_ACTION_BINS,
        render_mode=None,
        max_episode_decisions=MAX_EPISODE_DECISIONS,
    )

    env = Monitor(
        env,
        filename=str(TRAINING_LOG_DIR / "train_monitor"),
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


def validate_environment() -> None:
    raw_env_for_check = ModifiedPongEnv(
        n_action_bins=N_ACTION_BINS,
        render_mode=None,
        max_episode_decisions=MAX_EPISODE_DECISIONS,
    )

    check_env(raw_env_for_check, warn=True)
    raw_env_for_check.close()


def build_model(train_env: Monitor) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        verbose=1,
        device="cpu",
        tensorboard_log=str(TENSORBOARD_DIR),
    )


def build_callbacks() -> tuple[
    PeriodicPPOEvaluationCallback,
    CallbackList,
]:
    periodic_eval_callback = PeriodicPPOEvaluationCallback(
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES_CALLBACK,
        seed_offset=50_000,
        best_model_dir=BEST_MODEL_DIR,
        n_action_bins=N_ACTION_BINS,
        max_episode_decisions=MAX_EPISODE_DECISIONS,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_checkpoint",
    )

    callbacks = CallbackList(
        [
            periodic_eval_callback,
            checkpoint_callback,
        ]
    )

    return periodic_eval_callback, callbacks


def train_ppo() -> tuple[PPO, pd.DataFrame]:
    ensure_project_directories()
    set_all_seeds(SEED)
    validate_environment()

    train_env = make_train_env(seed=SEED)

    try:
        model = build_model(train_env)

        periodic_eval_callback, callbacks = build_callbacks()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
        )

        final_model_path = FINAL_MODEL_DIR / "ppo_final_model"
        model.save(str(final_model_path))

        evaluation_history_df = pd.DataFrame(
            periodic_eval_callback.evaluation_history
        )

        evaluation_history_df.to_csv(
            TRAINING_LOG_DIR / "ppo_training_evaluation_history.csv",
            index=False,
        )

        print(f"Saved final PPO model to: {final_model_path}.zip")
        print(
            "Saved periodic PPO evaluation history to: "
            f"{TRAINING_LOG_DIR / 'ppo_training_evaluation_history.csv'}"
        )

        return model, evaluation_history_df

    finally:
        train_env.close()


def main() -> None:
    train_ppo()


if __name__ == "__main__":
    main()
    