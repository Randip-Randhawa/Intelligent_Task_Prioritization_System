"""
models/heuristic.py

Phase 1: Rule-based weighted scoring system.

The scoring function is:

    S = w1 * urgency + w2 * difficulty_norm + w3 * priority_norm + w4 * time_pressure

All features are normalized to [0, 1] before weighting.

Weight rationale:
    w1 = 0.40  -- Urgency (deadline proximity) is the strongest signal.
                  Missing a deadline is the worst outcome for a student.
    w2 = 0.25  -- Difficulty matters because hard tasks need to be started
                  early; you cannot cram a 5-difficulty task the night before.
    w3 = 0.25  -- Priority reflects the student's own judgment of importance.
                  Equal weight to difficulty because self-assessment is valuable.
    w4 = 0.10  -- Time pressure catches the edge case where estimated work
                  nearly fills available time, even if deadline is not close.
                  Smaller weight because it partly overlaps with urgency.

The weights sum to 1.0, making S directly interpretable as a weighted average.
"""

from datetime import datetime
from typing import List, Tuple, Dict

from data.schema import Task


# Weights must sum to 1.0
W_URGENCY = 0.40
W_DIFFICULTY = 0.25
W_PRIORITY = 0.25
W_TIME_PRESSURE = 0.10

# Maximum scale values for normalization
DIFFICULTY_MAX = 5.0
PRIORITY_MAX = 5.0


class HeuristicScorer:
    """
    Scores and ranks tasks using a hand-crafted weighted formula.

    This is intentionally simple and explainable: every number that
    goes into the score can be traced back to a single feature.
    """

    def __init__(
        self,
        w_urgency: float = W_URGENCY,
        w_difficulty: float = W_DIFFICULTY,
        w_priority: float = W_PRIORITY,
        w_time_pressure: float = W_TIME_PRESSURE,
    ):
        if abs(w_urgency + w_difficulty + w_priority + w_time_pressure - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.w_urgency = w_urgency
        self.w_difficulty = w_difficulty
        self.w_priority = w_priority
        self.w_time_pressure = w_time_pressure

    def _compute_features(
        self,
        task: Task,
        reference_time: datetime,
        max_deadline_hours: float,
    ) -> Dict[str, float]:
        """
        Extract and normalize the four scoring features for one task.

        Returns a dict so callers can inspect individual feature values,
        which is important for explainability.
        """
        hours_left = task.hours_until_deadline(reference_time)

        # Urgency: 1 when deadline is now, 0 when deadline is at the max horizon
        # Clamped to [0, 1] -- overdue tasks get urgency = 1.0
        urgency = 1.0 - max(0.0, hours_left) / max_deadline_hours
        urgency = min(urgency, 1.0)

        # Difficulty normalized to [0, 1]
        difficulty_norm = (task.difficulty - 1) / (DIFFICULTY_MAX - 1)

        # Priority normalized to [0, 1]
        priority_norm = (task.priority - 1) / (PRIORITY_MAX - 1)

        # Time pressure: how much of available time does the task consume?
        # Cap at 1.0 so it does not dominate the score when available_hours is tiny
        if task.available_hours > 0:
            time_pressure = min(task.estimated_hours / task.available_hours, 1.0)
        else:
            time_pressure = 1.0

        return {
            "urgency": urgency,
            "difficulty_norm": difficulty_norm,
            "priority_norm": priority_norm,
            "time_pressure": time_pressure,
            "hours_left": hours_left,
        }

    def score_single(
        self,
        task: Task,
        reference_time: datetime = None,
        max_deadline_hours: float = 168.0,
    ) -> float:
        """
        Compute the heuristic priority score for a single task.

        Parameters
        ----------
        task : Task
            The task to score.
        reference_time : datetime, optional
            The current moment; defaults to now.
        max_deadline_hours : float
            The planning horizon in hours. Tasks with deadlines beyond this
            are treated as having near-zero urgency.

        Returns
        -------
        float in [0, 1], higher means more urgent/important.
        """
        if reference_time is None:
            reference_time = datetime.now()

        feats = self._compute_features(task, reference_time, max_deadline_hours)

        score = (
            self.w_urgency * feats["urgency"]
            + self.w_difficulty * feats["difficulty_norm"]
            + self.w_priority * feats["priority_norm"]
            + self.w_time_pressure * feats["time_pressure"]
        )

        return round(float(score), 4)

    def rank(
        self,
        tasks: List[Task],
        reference_time: datetime = None,
        max_deadline_hours: float = 168.0,
    ) -> List[Tuple[Task, float, Dict]]:
        """
        Score and rank all tasks, highest score first.

        Completed tasks are silently excluded.

        Parameters
        ----------
        tasks : List[Task]
        reference_time : datetime, optional
        max_deadline_hours : float

        Returns
        -------
        List of (Task, score, feature_dict) tuples, sorted descending by score.
        """
        if reference_time is None:
            reference_time = datetime.now()

        # Determine max_deadline_hours dynamically from the dataset if not specified
        # This ensures normalization is consistent across the batch
        active_tasks = [t for t in tasks if not t.completed]
        if not active_tasks:
            return []

        results = []
        for task in active_tasks:
            feats = self._compute_features(task, reference_time, max_deadline_hours)
            score = (
                self.w_urgency * feats["urgency"]
                + self.w_difficulty * feats["difficulty_norm"]
                + self.w_priority * feats["priority_norm"]
                + self.w_time_pressure * feats["time_pressure"]
            )
            results.append((task, round(float(score), 4), feats))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
