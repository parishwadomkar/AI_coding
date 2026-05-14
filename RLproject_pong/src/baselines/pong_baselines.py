import numpy as np
import pandas as pd

from src.env.modified_pong_env import (
    ModifiedPongEnv,
    HEIGHT,
    BORDER,
    PADDLE_H,
    predict_ball_y_at_next_right_return,
)

def action_center(env): return env.n_action_bins // 2
def action_random(env): return env.action_space.sample()
def action_top(env): return 0
def action_bottom(env):return env.n_action_bins - 1

def target_y_to_action(env, target_y):
    lo = BORDER + PADDLE_H / 2
    hi = HEIGHT - BORDER - PADDLE_H / 2
    target_y = np.clip(target_y, lo, hi)
    frac = (target_y - lo) / (hi - lo)
    action = int(round(frac * (env.n_action_bins - 1)))
    action = int(np.clip(action, 0, env.n_action_bins - 1))
    return action


def action_heuristic_predictive(env):
    predicted_y = predict_ball_y_at_next_right_return(env.ball)
    return target_y_to_action(env, predicted_y)


def evaluate_policy_function(
    policy_fn,
    n_episodes=100,
    seed_offset=0,
    render=False,
    max_episode_decisions=50,
):
    render_mode = "human" if render else None

    env = ModifiedPongEnv(
        n_action_bins=11,
        render_mode=render_mode,
        max_episode_decisions=max_episode_decisions,
    )
    records = []
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = policy_fn(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
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
                "last_right_speed_multiplier": info["last_right_speed_multiplier"],
                "last_right_outgoing_speed": info["last_right_outgoing_speed"],
                "final_ball_speed": info["current_ball_speed"],
                "last_left_outgoing_speed": info["last_left_outgoing_speed"],
                "successful_agent_returns": info["successful_agent_returns"],
                "last_selected_target_y": info["last_selected_target_y"],
                "last_predicted_return_y": info["last_predicted_return_y"],
                "last_target_alignment_error": info["last_target_alignment_error"],
                "last_target_alignment_reward": info["last_target_alignment_reward"],
                "cumulative_target_alignment_reward": info["cumulative_target_alignment_reward"],
            }
        )

    env.close()
    return pd.DataFrame(records)

def summarize_results(name, df):
    return {
        "policy": name,
        "episodes": len(df),
        "mean_reward": df["reward"].mean(),
        "win_rate": df["win"].mean(),
        "loss_rate": df["loss"].mean(),
        "mean_score_diff": df["score_diff"].mean(),
        "mean_steps": df["steps"].mean(),
        "mean_rally_hits": df["rally_hits"].mean(),
        "truncation_rate": df["truncated"].mean(),
        "mean_final_hit_offset": df["last_right_hit_offset"].mean(),
        "mean_final_speed_multiplier": df["last_right_speed_multiplier"].mean(),
        "mean_final_outgoing_speed": df["last_right_outgoing_speed"].mean(),
        "mean_final_ball_speed": df["final_ball_speed"].mean(),
        "mean_last_left_outgoing_speed": df["last_left_outgoing_speed"].mean(),
        "mean_successful_agent_returns": df["successful_agent_returns"].mean(),
        "mean_final_target_alignment_error": df["last_target_alignment_error"].mean(),
        "mean_final_target_alignment_reward": df["last_target_alignment_reward"].mean(),
        "mean_cumulative_target_alignment_reward": df["cumulative_target_alignment_reward"].mean(),
    }

def view_heuristic_episodes(
    n_episodes=5,
    max_episode_decisions=50,
    seed_offset=5000,
):
    env = ModifiedPongEnv(
        n_action_bins=11,
        render_mode="human",
        max_episode_decisions=max_episode_decisions,
    )

    episode_summaries = []
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action = action_heuristic_predictive(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
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

def run_all_baselines(
    n_episodes=300,
    max_episode_decisions=50,
    seed_offset=100_000,
):
    baselines = {
        "random": action_random,
        "center": action_center,
        "top": action_top,
        "bottom": action_bottom,
        "heuristic_predictive": action_heuristic_predictive,
    }

    all_summaries = []
    all_details = {}
    for name, policy_fn in baselines.items():
        print(f"Evaluating baseline: {name}")
        df = evaluate_policy_function(
            policy_fn=policy_fn,
            n_episodes=n_episodes,
            seed_offset=seed_offset,
            render=False,
            max_episode_decisions=max_episode_decisions,
        )
        all_details[name] = df
        all_summaries.append(summarize_results(name, df))
    summary_df = pd.DataFrame(all_summaries)
    return summary_df, all_details

# heuristic_view_df = view_heuristic_episodes(n_episodes=5)
# heuristic_view_df

if __name__ == "__main__":
    summary_df, all_details = run_all_baselines(
        n_episodes=300,
        max_episode_decisions=50,
        seed_offset=100_000,
    )
    print(summary_df)