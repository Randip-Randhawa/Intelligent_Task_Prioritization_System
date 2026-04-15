"""
data/generator.py

Generates a synthetic dataset of study tasks for training and evaluation.

Design decisions:
- Deadlines follow an exponential distribution so that most tasks cluster
  in the near future (realistic student scenario).
- Difficulty and estimated hours are weakly correlated: harder tasks tend
  to take longer, but not perfectly.
- Available hours are sampled independently because real availability
  depends on the student's schedule, not the task's nature.
- The ground-truth priority score is the heuristic score plus small
  Gaussian noise. This simulates students who behave mostly rationally
  but not perfectly (they sometimes procrastinate or over-prioritize
  a subject they enjoy).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple

from data.schema import Task
from models.heuristic import HeuristicScorer


SUBJECTS = [
    "Mathematics",
    "Computer Science",
    "Physics",
    "English Literature",
    "History",
    "Statistics",
    "Economics",
]

TASK_NAMES = [
    "Assignment", "Project", "Essay", "Lab Report",
    "Problem Set", "Case Study", "Presentation", "Review",
]

_rng = np.random.default_rng(seed=42)


def generate_task_list(
    n_tasks: int = 10,
    reference_time: datetime = None,
    max_deadline_hours: float = 168.0,  # 7 days
) -> List[Task]:
    """
    Generate a list of Task objects for a single 'snapshot' scenario.
    Used in main.py to demonstrate the ranking system interactively.

    Parameters
    ----------
    n_tasks : int
        Number of tasks to generate.
    reference_time : datetime, optional
        The current moment; defaults to now.
    max_deadline_hours : float
        Maximum hours into the future a deadline can be set.
    """
    if reference_time is None:
        reference_time = datetime.now()

    tasks = []
    for i in range(n_tasks):
        subject = _rng.choice(SUBJECTS)
        task_type = _rng.choice(TASK_NAMES)
        name = f"{subject} {task_type} {i + 1}"

        # Exponential distribution: many near-future deadlines
        hours_to_deadline = float(
            _rng.exponential(scale=max_deadline_hours / 3)
        )
        hours_to_deadline = np.clip(hours_to_deadline, 1.0, max_deadline_hours)
        deadline = reference_time + timedelta(hours=hours_to_deadline)

        difficulty = int(_rng.integers(1, 6))
        priority = int(_rng.integers(1, 6))

        # Harder tasks tend to take longer, but there is scatter
        base_hours = 1.0 + (difficulty - 1) * 0.8
        estimated_hours = float(
            np.clip(_rng.normal(loc=base_hours, scale=0.5), 0.5, 10.0)
        )

        available_hours = float(_rng.uniform(1.0, hours_to_deadline * 0.9))

        tasks.append(Task(
            task_id=f"TASK_{i + 1:03d}",
            name=name,
            deadline=deadline,
            estimated_hours=round(estimated_hours, 1),
            difficulty=difficulty,
            priority=priority,
            available_hours=round(available_hours, 1),
            subject=subject,
        ))

    return tasks


def generate_training_dataset(
    n_samples: int = 2000,
    noise_std: float = 0.05,
    reference_time: datetime = None,
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame suitable for ML training and evaluation.

    The target variable 'score' is the heuristic score plus small noise.
    This represents the assumption that student behavior is approximately
    rational (follows urgency/difficulty logic) but not perfectly so.

    Parameters
    ----------
    n_samples : int
        Number of synthetic task records to create.
    noise_std : float
        Standard deviation of the noise added to heuristic scores.
        Keep this small so the ML model can learn the signal.
    reference_time : datetime, optional
        Used as the baseline "now" for computing deadline proximity.

    Returns
    -------
    pd.DataFrame with columns:
        hours_until_deadline, estimated_hours, difficulty, priority,
        available_hours, urgency, time_pressure, score (target)
    """
    if reference_time is None:
        reference_time = datetime.now()

    max_deadline_hours = 336.0  # 14 days as the maximum window

    rows = []

    # Generate raw features first, then compute derived ones
    hours_until_deadline = _rng.exponential(
        scale=max_deadline_hours / 3, size=n_samples
    )
    hours_until_deadline = np.clip(hours_until_deadline, 1.0, max_deadline_hours)

    difficulty = _rng.integers(1, 6, size=n_samples)

    # Estimated hours weakly correlated with difficulty
    base_hours = 1.0 + (difficulty - 1) * 0.8
    estimated_hours = np.clip(
        _rng.normal(loc=base_hours, scale=0.8), 0.5, 12.0
    )

    priority = _rng.integers(1, 6, size=n_samples)

    # Available hours: somewhere between 1 hour and the deadline
    # We must ensure high > low, so cap the upper bound at max(1.1, h * 0.95)
    available_hours = np.array([
        float(_rng.uniform(1.0, max(1.1, h * 0.95)))
        for h in hours_until_deadline
    ])

    scorer = HeuristicScorer()

    for i in range(n_samples):
        task = Task(
            task_id=f"SYN_{i:05d}",
            name=f"Synthetic Task {i}",
            deadline=reference_time + timedelta(hours=float(hours_until_deadline[i])),
            estimated_hours=float(estimated_hours[i]),
            difficulty=int(difficulty[i]),
            priority=int(priority[i]),
            available_hours=float(available_hours[i]),
        )

        heuristic_score = scorer.score_single(
            task,
            reference_time=reference_time,
            max_deadline_hours=max_deadline_hours,
        )

        # Add noise to simulate imperfect human behavior
        noisy_score = float(np.clip(
            heuristic_score + _rng.normal(0.0, noise_std),
            0.0, 1.0,
        ))

        rows.append({
            "hours_until_deadline": round(float(hours_until_deadline[i]), 2),
            "estimated_hours": round(float(estimated_hours[i]), 2),
            "difficulty": int(difficulty[i]),
            "priority": int(priority[i]),
            "available_hours": round(float(available_hours[i]), 2),
            # Pre-computed derived features (also used in ML pipeline)
            "urgency": round(
                1.0 - hours_until_deadline[i] / max_deadline_hours, 4
            ),
            "time_pressure": round(
                min(estimated_hours[i] / available_hours[i], 1.0), 4
            ),
            "score": round(noisy_score, 4),
        })

    return pd.DataFrame(rows)
