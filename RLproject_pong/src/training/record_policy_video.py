from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pygame
from stable_baselines3 import PPO

from src.env.modified_pong_env import (
    BG,
    FG,
    HEIGHT,
    WIDTH,
    ModifiedPongEnv,
    draw_background,
)


STATS_BAR_HEIGHT = 72
VIEWER_FONT_SIZE = 24
VIEWER_SMALL_FONT_SIZE = 20

DEFAULT_OUTPUT_FPS = 90
DEFAULT_VIDEO_CRF = 28
DEFAULT_VIDEO_PRESET = "medium"


class PPOVideoRecorderEnv(ModifiedPongEnv):
    def __init__(
        self,
        video_writer: Any,
        n_action_bins: int = 11,
        max_internal_frames: int = 5000,
        max_episode_decisions: int = 50,
        model_label: str = "Saved PPO policy",
    ):
        super().__init__(
            n_action_bins=n_action_bins,
            render_mode="human",
            max_internal_frames=max_internal_frames,
            max_episode_decisions=max_episode_decisions,
        )

        self.video_writer = video_writer
        self.model_label = model_label

        self.viewer_episode = 0
        self.viewer_total_episodes = 0
        self.viewer_wins = 0
        self.viewer_losses = 0
        self.viewer_current_steps = 0
        self.viewer_current_reward = 0.0
        self.viewer_last_action = None

        self.viewer_large_font = None
        self.viewer_small_font = None

    def update_viewer_stats(
        self,
        episode: int,
        total_episodes: int,
        wins: int,
        losses: int,
        current_steps: int,
        current_reward: float,
        last_action: int | None,
    ) -> None:
        self.viewer_episode = episode
        self.viewer_total_episodes = total_episodes
        self.viewer_wins = wins
        self.viewer_losses = losses
        self.viewer_current_steps = current_steps
        self.viewer_current_reward = current_reward
        self.viewer_last_action = last_action

    def _initialize_recording_surface(self) -> None:
        if self.screen is not None:
            return

        pygame.init()
        pygame.font.init()

        self.screen = pygame.Surface(
            (WIDTH, HEIGHT + STATS_BAR_HEIGHT)
        )

        self.font = pygame.font.SysFont("consolas", 60)

        self.viewer_large_font = pygame.font.SysFont(
            "consolas",
            VIEWER_FONT_SIZE,
        )

        self.viewer_small_font = pygame.font.SysFont(
            "consolas",
            VIEWER_SMALL_FONT_SIZE,
        )

    def _capture_rgb_frame(self) -> np.ndarray:
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        return np.ascontiguousarray(frame)

    def render(self) -> None:
        if self.render_mode != "human":
            return

        self._initialize_recording_surface()

        draw_background(self.screen)

        pygame.draw.rect(self.screen, FG, self.left.rect)
        pygame.draw.rect(self.screen, FG, self.right.rect)
        pygame.draw.rect(self.screen, FG, self.ball.rect)

        left_score_surface = self.font.render(
            str(self.viewer_losses),
            True,
            FG,
        )

        right_score_surface = self.font.render(
            str(self.viewer_wins),
            True,
            FG,
        )

        self.screen.blit(
            left_score_surface,
            (WIDTH // 2 - 120, 20),
        )

        self.screen.blit(
            right_score_surface,
            (WIDTH // 2 + 60, 20),
        )

        pygame.draw.rect(
            self.screen,
            BG,
            (0, HEIGHT, WIDTH, STATS_BAR_HEIGHT),
        )

        pygame.draw.line(
            self.screen,
            FG,
            (0, HEIGHT),
            (WIDTH, HEIGHT),
            width=2,
        )

        top_line = (
            f"{self.model_label} | "
            f"Episode {self.viewer_episode}/{self.viewer_total_episodes} | "
            f"Cumulative W-L: {self.viewer_wins}-{self.viewer_losses}"
        )

        action_text = (
            "—"
            if self.viewer_last_action is None
            else str(self.viewer_last_action)
        )

        bottom_line = (
            f"Current decision steps: {self.viewer_current_steps} | "
            f"Episode reward: {self.viewer_current_reward:.3f} | "
            f"Last action bin: {action_text} | "
            f"Successful returns: {self.successful_agent_returns}"
        )

        top_surface = self.viewer_large_font.render(
            top_line,
            True,
            FG,
        )

        bottom_surface = self.viewer_small_font.render(
            bottom_line,
            True,
            FG,
        )

        self.screen.blit(top_surface, (18, HEIGHT + 10))
        self.screen.blit(bottom_surface, (18, HEIGHT + 40))

        self.video_writer.append_data(
            self._capture_rgb_frame()
        )


def record_saved_ppo_gameplay(
    model_path: str | Path,
    output_path: str | Path,
    n_episodes: int = 50,
    seed_offset: int = 200_000,
    deterministic: bool = True,
    output_fps: int = DEFAULT_OUTPUT_FPS,
    n_action_bins: int = 11,
    max_episode_decisions: int = 50,
    device: str = "cpu",
    model_label: str = "Saved PPO policy",
    crf: int = DEFAULT_VIDEO_CRF,
    preset: str = DEFAULT_VIDEO_PRESET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved PPO model not found: {model_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = PPO.load(
        str(model_path),
        device=device,
    )

    writer = imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        mode="I",
        fps=output_fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=8,
        output_params=[
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
        ],
        ffmpeg_log_level="error",
    )

    env = PPOVideoRecorderEnv(
        video_writer=writer,
        n_action_bins=n_action_bins,
        max_episode_decisions=max_episode_decisions,
        model_label=model_label,
    )

    episode_records: list[dict[str, Any]] = []

    cumulative_wins = 0
    cumulative_losses = 0

    try:
        for episode_idx in range(n_episodes):
            env.update_viewer_stats(
                episode=episode_idx + 1,
                total_episodes=n_episodes,
                wins=cumulative_wins,
                losses=cumulative_losses,
                current_steps=0,
                current_reward=0.0,
                last_action=None,
            )

            obs, info = env.reset(
                seed=seed_offset + episode_idx
            )

            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            last_action = None

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                )

                action = int(np.asarray(action).item())
                last_action = action

                env.update_viewer_stats(
                    episode=episode_idx + 1,
                    total_episodes=n_episodes,
                    wins=cumulative_wins,
                    losses=cumulative_losses,
                    current_steps=steps,
                    current_reward=total_reward,
                    last_action=last_action,
                )

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1

                env.update_viewer_stats(
                    episode=episode_idx + 1,
                    total_episodes=n_episodes,
                    wins=cumulative_wins,
                    losses=cumulative_losses,
                    current_steps=steps,
                    current_reward=total_reward,
                    last_action=last_action,
                )

            win = (
                1
                if info["right_score"] > info["left_score"]
                else 0
            )

            loss = (
                1
                if info["left_score"] > info["right_score"]
                else 0
            )

            cumulative_wins += win
            cumulative_losses += loss

            episode_records.append(
                {
                    "episode": episode_idx + 1,
                    "reward": total_reward,
                    "win": win,
                    "loss": loss,
                    "steps": steps,
                    "right_score": info["right_score"],
                    "left_score": info["left_score"],
                    "score_diff": info["score_diff"],
                    "rally_hits": info["rally_hits"],
                    "successful_agent_returns": info[
                        "successful_agent_returns"
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
                    "truncated": truncated,
                }
            )

    finally:
        env.close()
        writer.close()

    episode_df = pd.DataFrame(episode_records)

    summary_df = pd.DataFrame(
        [
            {
                "episodes": len(episode_df),
                "win_rate": episode_df["win"].mean(),
                "loss_rate": episode_df["loss"].mean(),
                "mean_reward": episode_df["reward"].mean(),
                "mean_steps": episode_df["steps"].mean(),
                "mean_successful_agent_returns": episode_df[
                    "successful_agent_returns"
                ].mean(),
                "mean_alignment_error": episode_df[
                    "last_target_alignment_error"
                ].mean(),
            }
        ]
    )

    episode_csv_path = output_path.with_name(
        f"{output_path.stem}_episodes.csv"
    )

    summary_csv_path = output_path.with_name(
        f"{output_path.stem}_summary.csv"
    )

    episode_df.to_csv(
        episode_csv_path,
        index=False,
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    print(f"Saved MP4 video to: {output_path}")
    print(f"Saved episode-level video summary to: {episode_csv_path}")
    print(f"Saved aggregate video summary to: {summary_csv_path}")

    return episode_df, summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record saved PPO Pong gameplay to compact MP4."
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/final/ppo_final_model.zip"),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/videos/ppo_final_demo_50episodes_90fps.mp4"
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_OUTPUT_FPS,
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_VIDEO_CRF,
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_VIDEO_PRESET,
    )

    parser.add_argument(
        "--label",
        type=str,
        default="PPO final policy",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _, summary_df = record_saved_ppo_gameplay(
        model_path=args.model_path,
        output_path=args.output_path,
        n_episodes=args.episodes,
        seed_offset=args.seed_offset,
        output_fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        model_label=args.label,
    )

    print("\nVideo policy summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()