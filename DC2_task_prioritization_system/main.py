"""
main.py

Entry point for the Intelligent Task Prioritization and Study
Recommendation System.

Run this with:
    python main.py

What this script does:
1. Generates a realistic list of 10 demo tasks (the "student's current workload").
2. Phase 1: Scores and ranks them using the heuristic scorer.
3. Phase 2: Generates a 2000-sample training set, trains the ML model,
   evaluates it, and produces its own ranking of the same 10 tasks.
4. Compares the two rankings with rank correlation and top-k overlap.

The comparison is not meant to show that ML "wins" -- with synthetic data
derived from the heuristic, they should agree strongly. The point is to
demonstrate that the full pipeline (data -> train -> evaluate -> infer)
works correctly and that the ML model successfully learned the heuristic pattern.
"""

import sys
import os
from datetime import datetime

# Make sure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generator import generate_task_list, generate_training_dataset
from data.schema import Task
from models.heuristic import HeuristicScorer
from models.ml_model import MLScorer
from models.evaluator import full_comparison_report
from utils.preprocessing import validate_task, summarize_dataset
from utils.display import (
    print_heuristic_ranking,
    print_ml_ranking,
    print_comparison,
)


def run_phase1(tasks, reference_time):
    """Run the heuristic scorer and display results."""
    print("\n" + "#" * 70)
    print("  PHASE 1: Rule-Based Heuristic Scoring")
    print("#" * 70)

    scorer = HeuristicScorer()
    heuristic_ranked = scorer.rank(tasks, reference_time=reference_time)

    print_heuristic_ranking(
        heuristic_ranked,
        reference_time=reference_time,
        title="Heuristic Task Ranking (w1=0.40, w2=0.25, w3=0.25, w4=0.10)",
    )
    return heuristic_ranked


def run_phase2(tasks, reference_time):
    """Generate training data, train the ML model, and display results."""
    print("\n" + "#" * 70)
    print("  PHASE 2: Machine Learning Enhancement")
    print("#" * 70)

    print("\n[Step 1] Generating synthetic training dataset (2000 samples)...")
    train_df = generate_training_dataset(
        n_samples=2000,
        noise_std=0.05,
        reference_time=reference_time,
    )
    summarize_dataset(train_df)

    print("[Step 2] Training Random Forest Regressor...")
    ml_scorer = MLScorer(n_estimators=100, random_state=42)
    ml_scorer.train(train_df, verbose=True)

    print("\n[Step 3] Ranking the same demo tasks with the ML model...")
    ml_ranked = ml_scorer.rank(tasks, reference_time=reference_time)
    print_ml_ranking(
        ml_ranked,
        reference_time=reference_time,
        title="ML Model Task Ranking (Random Forest Regressor)",
    )
    return ml_ranked, ml_scorer


def run_comparison(heuristic_ranked, ml_ranked, k=3):
    """Compare heuristic and ML rankings."""
    print("\n" + "#" * 70)
    print("  EVALUATION: Heuristic vs. ML Ranking Comparison")
    print("#" * 70)

    comparison = full_comparison_report(heuristic_ranked, ml_ranked, k=k)
    print_comparison(comparison, k=k)
    return comparison


def validate_all_tasks(tasks, reference_time):
    """Print any warnings about invalid task inputs."""
    all_warnings = []
    for task in tasks:
        all_warnings.extend(validate_task(task))

    if all_warnings:
        print("\n[Validation Warnings]")
        for w in all_warnings:
            print(f"  ! {w}")
    else:
        print("\n[Validation] All task inputs look valid.")


def main():
    # Use a fixed reference time so the demo is reproducible
    # In production, replace with datetime.now()
    reference_time = datetime(2025, 6, 1, 9, 0, 0)  # 9 AM on June 1

    print("=" * 70)
    print("  Intelligent Task Prioritization & Study Recommendation System")
    print("  Reference time:", reference_time.strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)

    # Generate the demo task list
    print("\n[Setup] Generating demo workload (10 tasks)...")
    tasks = generate_task_list(n_tasks=10, reference_time=reference_time)

    print("\n  Tasks loaded:")
    for t in tasks:
        print(
            f"    {t.task_id} | {t.name:<38} | "
            f"Deadline in {t.hours_until_deadline(reference_time):.1f}h | "
            f"Diff:{t.difficulty} | Pri:{t.priority}"
        )

    validate_all_tasks(tasks, reference_time)

    # Phase 1
    heuristic_ranked = run_phase1(tasks, reference_time)

    # Phase 2
    ml_ranked, ml_scorer = run_phase2(tasks, reference_time)

    # Comparison
    run_comparison(heuristic_ranked, ml_ranked, k=3)

    print("\n" + "=" * 70)
    print("  Run complete. See README.md for interpretation of results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
