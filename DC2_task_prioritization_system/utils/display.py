"""
utils/display.py

Pretty-printing utilities for ranked task lists and comparison tables.

Keeping display logic separate from business logic means we can later
swap this out for a web UI or CLI without touching the scoring code.
"""

from datetime import datetime
from typing import List, Tuple, Dict

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from data.schema import Task


def format_hours(hours: float) -> str:
    """Human-readable hours string, e.g. '2.5h' or '3d 4h'."""
    if hours < 0:
        return "OVERDUE"
    if hours < 24:
        return f"{hours:.1f}h"
    days = int(hours // 24)
    remaining_h = hours % 24
    return f"{days}d {remaining_h:.0f}h"


def print_heuristic_ranking(
    ranked: List[Tuple[Task, float, Dict]],
    reference_time: datetime = None,
    title: str = "Heuristic Task Ranking",
) -> None:
    """
    Print the heuristic ranking with feature breakdowns.

    The breakdown shows each component score so the student can see
    exactly why one task outranked another. This is the key advantage
    of the heuristic over black-box methods.

    Parameters
    ----------
    ranked : output of HeuristicScorer.rank()
    reference_time : used for computing hours left display
    title : section header
    """
    if reference_time is None:
        reference_time = datetime.now()

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    rows = []
    for rank_pos, (task, score, feats) in enumerate(ranked, start=1):
        hours_left = feats.get("hours_left", task.hours_until_deadline(reference_time))
        rows.append([
            rank_pos,
            task.task_id,
            task.name[:32],
            f"{score:.3f}",
            f"{feats['urgency']:.3f}",
            f"{feats['difficulty_norm']:.3f}",
            f"{feats['priority_norm']:.3f}",
            f"{feats['time_pressure']:.3f}",
            format_hours(hours_left),
        ])

    headers = [
        "Rank", "ID", "Task Name",
        "Score", "Urgency", "Difficulty", "Priority", "TimePres", "TimeLeft"
    ]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        # Fallback if tabulate is not installed
        header_line = "  ".join(f"{h:<12}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print("  ".join(f"{str(v):<12}" for v in row))

    # Print a brief explanation for the top task
    if ranked:
        top_task, top_score, top_feats = ranked[0]
        print(f"\n  [Recommendation] Focus on: {top_task.name}")
        print(f"  Score: {top_score:.3f} -- ", end="")
        dominant = max(
            [("urgency", top_feats["urgency"] * 0.40),
             ("difficulty", top_feats["difficulty_norm"] * 0.25),
             ("priority", top_feats["priority_norm"] * 0.25),
             ("time pressure", top_feats["time_pressure"] * 0.10)],
            key=lambda x: x[1]
        )
        print(f"primary driver: {dominant[0]} (weighted contribution: {dominant[1]:.3f})")


def print_ml_ranking(
    ranked: List[Tuple[Task, float]],
    reference_time: datetime = None,
    title: str = "ML Model Task Ranking",
) -> None:
    """
    Print the ML-based ranking. Simpler display since we do not have
    per-feature breakdowns from a black-box model in the same way.

    Parameters
    ----------
    ranked : output of MLScorer.rank()
    """
    if reference_time is None:
        reference_time = datetime.now()

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    rows = []
    for rank_pos, (task, score) in enumerate(ranked, start=1):
        hours_left = task.hours_until_deadline(reference_time)
        rows.append([
            rank_pos,
            task.task_id,
            task.name[:32],
            f"{score:.3f}",
            task.difficulty,
            task.priority,
            f"{task.estimated_hours:.1f}h",
            format_hours(hours_left),
        ])

    headers = ["Rank", "ID", "Task Name", "ML Score",
               "Difficulty", "Priority", "Est.Hours", "TimeLeft"]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        header_line = "  ".join(f"{h:<12}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print("  ".join(f"{str(v):<12}" for v in row))


def print_comparison(comparison_dict: Dict, k: int = 3) -> None:
    """
    Print the heuristic vs ML comparison metrics in a readable format.
    """
    print(f"\n{'=' * 70}")
    print("  Heuristic vs. ML Comparison Metrics")
    print(f"{'=' * 70}")

    rho = comparison_dict.get("spearman_rho", float("nan"))
    overlap = comparison_dict.get(f"top_{k}_overlap", float("nan"))
    mad = comparison_dict.get("mean_abs_diff", float("nan"))
    maxd = comparison_dict.get("max_abs_diff", float("nan"))

    print(f"  Spearman rank correlation   : {rho:.4f}")
    print(f"    (1.0 = identical ordering, 0.0 = no relationship)")
    print(f"  Top-{k} task overlap          : {overlap:.4f}")
    print(f"    (1.0 = same top-{k}, 0.0 = completely different)")
    print(f"  Mean absolute score diff    : {mad:.4f}")
    print(f"  Max absolute score diff     : {maxd:.4f}")

    print(f"\n  Interpretation:")
    if rho > 0.85:
        print(f"  The ML model has learned the heuristic pattern very well.")
        print(f"  Rankings are nearly identical, which is expected because")
        print(f"  the training labels were derived from the heuristic.")
    elif rho > 0.60:
        print(f"  The ML model captures the main signal but diverges on")
        print(f"  some tasks -- likely where noise in training data had effect.")
    else:
        print(f"  Significant disagreement: check feature engineering and data.")
