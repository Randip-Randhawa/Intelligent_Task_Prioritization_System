"""
utils/preprocessing.py

Feature engineering and preprocessing utilities.

These functions are intentionally separate from the model code so that
the same transformations can be applied consistently during training
and at inference time without duplicating logic.
"""

import numpy as np
import pandas as pd
from typing import List
from data.schema import Task
from datetime import datetime


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few interaction and ratio features to the training DataFrame.

    These capture relationships that are hard for a linear model to learn
    but which a Random Forest can still leverage:

    - urgency_x_difficulty: captures the combination of "close deadline AND hard"
      which is particularly stressful and should be highly ranked.
    - deadline_slack: how much buffer exists beyond the estimated work time.
      Negative slack means the student cannot realistically finish even if
      they use all available time.
    - priority_x_urgency: captures "important AND soon", useful for tasks that
      the student rated highly and are also coming up fast.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain: urgency, difficulty, time_pressure,
        estimated_hours, available_hours, priority.

    Returns
    -------
    pd.DataFrame with new columns added in place (copy returned).
    """
    df = df.copy()

    df["urgency_x_difficulty"] = df["urgency"] * (df["difficulty"] / 5.0)

    df["deadline_slack"] = df["available_hours"] - df["estimated_hours"]
    # Normalize slack: divide by max available hours so scale is consistent
    max_avail = df["available_hours"].max()
    if max_avail > 0:
        df["deadline_slack_norm"] = df["deadline_slack"] / max_avail
    else:
        df["deadline_slack_norm"] = 0.0

    df["priority_x_urgency"] = (df["priority"] / 5.0) * df["urgency"]

    return df


def validate_task(task: Task, reference_time=None) -> List[str]:
    """
    Check that a task's field values are within expected ranges.
    Returns a list of warning strings; empty list means all clear.

    This is lightweight input validation, not full schema enforcement.
    """
    warnings = []

    if not (1 <= task.difficulty <= 5):
        warnings.append(
            f"Task {task.task_id}: difficulty={task.difficulty} is outside [1, 5]"
        )
    if not (1 <= task.priority <= 5):
        warnings.append(
            f"Task {task.task_id}: priority={task.priority} is outside [1, 5]"
        )
    if task.estimated_hours <= 0:
        warnings.append(
            f"Task {task.task_id}: estimated_hours must be positive"
        )
    if task.available_hours <= 0:
        warnings.append(
            f"Task {task.task_id}: available_hours must be positive"
        )
    if task.is_overdue():
        warnings.append(
            f"Task {task.task_id}: deadline has already passed"
        )

    return warnings


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Print a brief descriptive summary of the training dataset.
    Useful for sanity-checking the generated data before training.
    """
    print("\n--- Dataset Summary ---")
    print(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Score range  : [{df['score'].min():.4f}, {df['score'].max():.4f}]")
    print(f"  Score mean   : {df['score'].mean():.4f}")
    print(f"  Score std    : {df['score'].std():.4f}")
    print(f"  Difficulty   : {dict(df['difficulty'].value_counts().sort_index())}")
    print(f"  Priority     : {dict(df['priority'].value_counts().sort_index())}")
    print()
