"""
models/evaluator.py

Compares heuristic and ML rankings on the same set of tasks.

We use two comparison metrics:
1. Rank correlation (Spearman's rho): measures how similarly the two methods
   order the tasks. Values near 1.0 mean they agree, near -1.0 means they
   completely disagree, 0 means no relationship.

2. Top-k overlap: what fraction of the top-k tasks from the heuristic
   also appear in the top-k from ML? This is practically useful because
   students care most about whether the top priorities agree.

We also compute score-level agreement: on the same task list, how different
are the raw scores assigned by each method?
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


def spearman_rank_correlation(
    heuristic_ranking: List[Tuple],  # list of (task, score, feats)
    ml_ranking: List[Tuple],         # list of (task, score)
) -> float:
    """
    Compute Spearman's rho between heuristic and ML task orderings.

    Parameters
    ----------
    heuristic_ranking : output of HeuristicScorer.rank()
    ml_ranking        : output of MLScorer.rank()

    Returns
    -------
    float: Spearman rho in [-1, 1]
    """
    # Extract task IDs in the order each method ranks them
    heuristic_order = [item[0].task_id for item in heuristic_ranking]
    ml_order = [item[0].task_id for item in ml_ranking]

    # Assign integer ranks based on position (1 = highest priority)
    heuristic_ranks = {tid: rank for rank, tid in enumerate(heuristic_order)}
    ml_ranks = {tid: rank for rank, tid in enumerate(ml_order)}

    # Build parallel rank arrays for all tasks that appear in both
    common_ids = [
        tid for tid in heuristic_order if tid in ml_ranks
    ]

    if len(common_ids) < 2:
        return float("nan")

    h_vals = np.array([heuristic_ranks[tid] for tid in common_ids])
    m_vals = np.array([ml_ranks[tid] for tid in common_ids])

    rho, _ = stats.spearmanr(h_vals, m_vals)
    return round(float(rho), 4)


def top_k_overlap(
    heuristic_ranking: List[Tuple],
    ml_ranking: List[Tuple],
    k: int = 3,
) -> float:
    """
    What fraction of the heuristic's top-k tasks also appear in ML's top-k?

    Parameters
    ----------
    k : int
        How many top tasks to compare.

    Returns
    -------
    float in [0, 1]
    """
    if len(heuristic_ranking) < k or len(ml_ranking) < k:
        k = min(len(heuristic_ranking), len(ml_ranking))
        if k == 0:
            return float("nan")

    top_heuristic = {item[0].task_id for item in heuristic_ranking[:k]}
    top_ml = {item[0].task_id for item in ml_ranking[:k]}

    overlap = len(top_heuristic & top_ml)
    return round(overlap / k, 4)


def score_difference_stats(
    heuristic_ranking: List[Tuple],
    ml_ranking: List[Tuple],
) -> Dict[str, float]:
    """
    Compare raw scores assigned by each method to the same tasks.

    Returns mean absolute difference and max absolute difference,
    which tell us whether the two methods agree numerically.
    """
    heuristic_scores = {item[0].task_id: item[1] for item in heuristic_ranking}
    ml_scores = {item[0].task_id: item[1] for item in ml_ranking}

    common_ids = list(set(heuristic_scores.keys()) & set(ml_scores.keys()))
    if not common_ids:
        return {"mean_abs_diff": float("nan"), "max_abs_diff": float("nan")}

    diffs = np.abs([
        heuristic_scores[tid] - ml_scores[tid]
        for tid in common_ids
    ])

    return {
        "mean_abs_diff": round(float(diffs.mean()), 4),
        "max_abs_diff": round(float(diffs.max()), 4),
        "std_diff": round(float(diffs.std()), 4),
    }


def full_comparison_report(
    heuristic_ranking: List[Tuple],
    ml_ranking: List[Tuple],
    k: int = 3,
) -> Dict:
    """
    Run all comparison metrics and return them as a dict.
    """
    rho = spearman_rank_correlation(heuristic_ranking, ml_ranking)
    overlap = top_k_overlap(heuristic_ranking, ml_ranking, k=k)
    score_stats = score_difference_stats(heuristic_ranking, ml_ranking)

    return {
        "spearman_rho": rho,
        f"top_{k}_overlap": overlap,
        **score_stats,
    }
