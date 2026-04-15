# Intelligent Task Prioritization and Study Recommendation System

## Overview

This project implements a two-phase system for recommending which study tasks
a student should focus on at a given point in time. Phase 1 uses a hand-crafted
heuristic scoring function. Phase 2 trains a machine learning model on synthetic
data to learn the same (and richer) patterns automatically.

The goal is not to schedule tasks across a calendar -- it is purely to rank them
by urgency and importance so a student knows what to tackle next.

---

## Folder Structure

```
task_prioritization/
    data/
        generator.py        -- Synthetic dataset creation
        schema.py           -- Dataclass / type definitions for a Task
    models/
        heuristic.py        -- Phase 1: rule-based weighted scorer
        ml_model.py         -- Phase 2: ML-based ranker (Random Forest)
        evaluator.py        -- Compare heuristic vs ML rankings
    utils/
        preprocessing.py    -- Feature normalization and engineering
        display.py          -- Pretty-print ranked task lists
    main.py                 -- Entry point: runs both phases end-to-end
    requirements.txt
    README.md
```

---

## Phase 1: Heuristic Scoring

### Scoring Function

Each task receives a score S computed as:

```
S = w1 * urgency + w2 * difficulty + w3 * priority + w4 * time_pressure
```

Where each feature is normalized to [0, 1] before weighting.

**Features:**

- `urgency` = 1 - (hours_until_deadline / max_hours_in_dataset)
  High urgency when deadline is close.

- `difficulty` = task_difficulty / 5  (difficulty is rated 1-5)
  Harder tasks should be started earlier because they take more mental effort.

- `priority` = user_priority / 5  (user-assigned importance, 1-5)
  Directly reflects the student's own assessment.

- `time_pressure` = estimated_hours / available_hours (capped at 1.0)
  If a task needs 6 hours but you only have 8, that is tighter than
  a task needing 1 hour in the same 8 hours.

**Weights (w1=0.40, w2=0.25, w3=0.25, w4=0.10):**

Urgency dominates because a missed deadline has the worst consequence.
Difficulty and priority share equal weight -- both matter but neither
overrides the deadline signal. Time pressure gets a smaller weight
because it partly overlaps with urgency information already.

---

## Phase 2: Machine Learning Enhancement

### Formulation

This is treated as a **regression** problem: predict a continuous priority
score for each task. Classification (high/med/low) is simpler but throws
away ordering information inside each class, which matters for ranking.

### Model Choice: Random Forest Regressor

- Handles non-linear interactions between features (e.g., a hard task
  with a tight deadline is disproportionately urgent).
- Robust to scale differences without manual normalization.
- Provides feature importance, which aids interpretability.
- No hyperparameter sensitivity as extreme as SVM or gradient boosting.
- Appropriate complexity for an undergraduate ML context.

Linear Regression was considered but rejected because the interaction between
urgency and difficulty is multiplicative, not additive.

### Dataset

A synthetic dataset of 2000 tasks is generated with realistic distributions:
- Deadlines: exponential distribution (most tasks have near-future deadlines)
- Difficulty: discrete uniform over {1, 2, 3, 4, 5}
- Priority: discrete uniform over {1, 2, 3, 4, 5}
- Available hours: uniform over [1, 12]
- Estimated hours: correlated weakly with difficulty

The ground-truth score is the heuristic score plus Gaussian noise, which
simulates that real students behave roughly like the heuristic but not
perfectly. The ML model then tries to recover and generalize this pattern.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Assumptions and Limitations

1. The heuristic weights are hand-tuned and may not reflect every student's
   actual behavior. A survey-based calibration would improve them.

2. The synthetic dataset is derived from the heuristic itself, so the ML model
   cannot significantly outperform it -- the point is to show the pipeline
   and that the model successfully learns the pattern.

3. "Available hours" is treated as a static snapshot. In reality it changes
   throughout the day.

4. No calendar integration: the system does not block time or create schedules.

5. The ML model does not currently accept new user feedback to retrain online.

---

## Future Improvements

- Collect real student data via a simple logging app and replace synthetic data.
- Add a feedback loop: after a session, the student rates how well the ranking
  matched their needs, and this feeds back into model retraining.
- Extend to a classification head that explains *why* a task is high priority
  in natural language.
- Integrate calendar APIs to pull real deadlines automatically.
- Experiment with LightGBM or XGBoost for marginal accuracy gains on larger datasets.
