"""
models/ml_model.py

Phase 2: Machine Learning-based priority scorer.

Formulation: Regression
    We predict a continuous score in [0, 1] rather than a class label.
    This preserves within-class ordering, which matters for ranking.
    (If we classified as high/med/low, all "high" tasks would be tied.)

Model: Random Forest Regressor
    Chosen because:
    1. Handles non-linear interactions (e.g., high difficulty AND tight deadline
       is disproportionately urgent, not just additive).
    2. No need for manual feature scaling.
    3. Built-in feature importance for interpretability.
    4. Robust to the relatively small feature set we have.
    5. Does not require hyperparameter tuning to get reasonable results.

    We considered:
    - Linear Regression: too simple, misses feature interactions.
    - SVM Regression: harder to interpret, no feature importance.
    - Neural Network: overkill for 7 features; adds complexity without
      a clear benefit at this scale.

Features used (7 total):
    hours_until_deadline, estimated_hours, difficulty, priority,
    available_hours, urgency (derived), time_pressure (derived)

The two derived features are added because they capture the non-linear
relationships the raw features describe only partially.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from data.schema import Task


# These are the exact column names the model expects at inference time
FEATURE_COLUMNS = [
    "hours_until_deadline",
    "estimated_hours",
    "difficulty",
    "priority",
    "available_hours",
    "urgency",
    "time_pressure",
]

# Planning horizon used to compute urgency consistently between train and infer
MAX_DEADLINE_HOURS = 336.0


class MLScorer:
    """
    Trains a Random Forest on the synthetic dataset and uses it
    to score and rank tasks at inference time.

    The model is intentionally kept simple so that training, evaluation,
    and inference are all transparent and reproducible.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=8,          # limit depth to reduce overfitting on synthetic data
            min_samples_leaf=5,   # prevent memorizing individual noise points
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained = False
        self.feature_importances_: Optional[Dict[str, float]] = None

    def train(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train the model on the provided DataFrame and return evaluation metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the columns in FEATURE_COLUMNS plus a 'score' column.
        verbose : bool
            Print training summary if True.

        Returns
        -------
        dict with keys: mse, mae, r2, cv_r2_mean, cv_r2_std
        """
        X = df[FEATURE_COLUMNS].values
        y = df["score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on held-out test set
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 5-fold cross-validation on full data to get a more stable estimate
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring="r2"
        )

        # Store feature importances with names for display
        self.feature_importances_ = dict(
            zip(FEATURE_COLUMNS, self.model.feature_importances_)
        )

        metrics = {
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        if verbose:
            print("\n--- ML Model Training Results ---")
            print(f"  Training samples : {metrics['n_train']}")
            print(f"  Test samples     : {metrics['n_test']}")
            print(f"  Test MSE         : {metrics['mse']:.6f}")
            print(f"  Test MAE         : {metrics['mae']:.6f}")
            print(f"  Test R^2         : {metrics['r2']:.4f}")
            print(f"  CV R^2 (5-fold)  : {metrics['cv_r2_mean']:.4f} "
                  f"(+/- {metrics['cv_r2_std']:.4f})")
            print("\n  Feature Importances:")
            sorted_feats = sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1], reverse=True
            )
            for fname, imp in sorted_feats:
                bar = "#" * int(imp * 40)
                print(f"    {fname:<25} {imp:.4f}  {bar}")

        return metrics

    def _task_to_features(
        self,
        task: Task,
        reference_time: datetime,
    ) -> np.ndarray:
        """
        Convert a Task object into the feature vector the model expects.
        This must produce features in the same order as FEATURE_COLUMNS.
        """
        hours_left = task.hours_until_deadline(reference_time)
        hours_left_clamped = max(hours_left, 0.0)

        urgency = 1.0 - hours_left_clamped / MAX_DEADLINE_HOURS
        urgency = min(urgency, 1.0)

        if task.available_hours > 0:
            time_pressure = min(task.estimated_hours / task.available_hours, 1.0)
        else:
            time_pressure = 1.0

        return np.array([
            hours_left_clamped,
            task.estimated_hours,
            task.difficulty,
            task.priority,
            task.available_hours,
            urgency,
            time_pressure,
        ])

    def score_single(
        self,
        task: Task,
        reference_time: datetime = None,
    ) -> float:
        """
        Predict the priority score for a single task.

        Returns
        -------
        float in approximately [0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring.")
        if reference_time is None:
            reference_time = datetime.now()

        feats = self._task_to_features(task, reference_time)
        score = self.model.predict(feats.reshape(1, -1))[0]
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def rank(
        self,
        tasks: List[Task],
        reference_time: datetime = None,
    ) -> List[Tuple[Task, float]]:
        """
        Score all active tasks and return them sorted highest score first.

        Parameters
        ----------
        tasks : List[Task]
        reference_time : datetime, optional

        Returns
        -------
        List of (Task, score) tuples sorted descending by score.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before ranking.")
        if reference_time is None:
            reference_time = datetime.now()

        active = [t for t in tasks if not t.completed]
        if not active:
            return []

        feature_matrix = np.vstack([
            self._task_to_features(t, reference_time) for t in active
        ])
        scores = self.model.predict(feature_matrix)
        scores = np.clip(scores, 0.0, 1.0)

        results = [
            (task, round(float(score), 4))
            for task, score in zip(active, scores)
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
