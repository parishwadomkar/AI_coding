from __future__ import annotations
from typing import Any
import pandas as pd

_REQUIRED_SUMMARY_COLUMNS = {
    "reward",
    "win",
    "loss",
    "score_diff",
    "steps",
    "rally_hits",
    "truncated",
    "last_right_hit_offset",
    "last_right_speed_multiplier",
    "last_right_outgoing_speed",
    "final_ball_speed",
    "last_left_outgoing_speed",
    "successful_agent_returns",
    "last_target_alignment_error",
    "last_target_alignment_reward",
    "cumulative_target_alignment_reward",
}


def _mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean())


def summarize_policy_results(name: str, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("Cannot summarize an empty evaluation DataFrame.")

    missing_columns = sorted(_REQUIRED_SUMMARY_COLUMNS.difference(df.columns))
    if missing_columns:
        raise KeyError(
            "Evaluation DataFrame is missing required columns: "
            + ", ".join(missing_columns)
        )

    return {
        "policy": name,
        "episodes": int(len(df)),
        "mean_reward": _mean(df, "reward"),
        "win_rate": _mean(df, "win"),
        "loss_rate": _mean(df, "loss"),
        "mean_score_diff": _mean(df, "score_diff"),
        "mean_steps": _mean(df, "steps"),
        "mean_rally_hits": _mean(df, "rally_hits"),
        "truncation_rate": _mean(df, "truncated"),
        "mean_final_hit_offset": _mean(df, "last_right_hit_offset"),
        "mean_final_speed_multiplier": _mean(
            df, "last_right_speed_multiplier"
        ),
        "mean_final_outgoing_speed": _mean(
            df, "last_right_outgoing_speed"
        ),
        "mean_final_ball_speed": _mean(df, "final_ball_speed"),
        "mean_last_left_outgoing_speed": _mean(
            df, "last_left_outgoing_speed"
        ),
        "mean_successful_agent_returns": _mean(
            df, "successful_agent_returns"
        ),
        "mean_final_target_alignment_error": _mean(
            df, "last_target_alignment_error"
        ),
        "mean_final_target_alignment_reward": _mean(
            df, "last_target_alignment_reward"
        ),
        "mean_cumulative_target_alignment_reward": _mean(
            df, "cumulative_target_alignment_reward"
        ),
    }
