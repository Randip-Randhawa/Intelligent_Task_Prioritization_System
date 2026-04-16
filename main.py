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
import json
from datetime import datetime

import pandas as pd

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
    metrics = ml_scorer.train(train_df, verbose=True)

    print("\n[Step 3] Ranking the same demo tasks with the ML model...")
    ml_ranked = ml_scorer.rank(tasks, reference_time=reference_time)
    print_ml_ranking(
        ml_ranked,
        reference_time=reference_time,
        title="ML Model Task Ranking (Random Forest Regressor)",
    )
    return ml_ranked, ml_scorer, train_df, metrics


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


def create_output_dir(base_dir="results"):
    """Create a timestamped output directory for run artifacts."""
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{run_stamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _ranking_to_df(ranking, reference_time, has_features=False):
    rows = []
    for idx, item in enumerate(ranking, start=1):
        task = item[0]
        score = item[1]
        row = {
            "rank": idx,
            "task_id": task.task_id,
            "name": task.name,
            "subject": task.subject,
            "score": score,
            "difficulty": task.difficulty,
            "priority": task.priority,
            "estimated_hours": task.estimated_hours,
            "available_hours": task.available_hours,
            "hours_until_deadline": round(task.hours_until_deadline(reference_time), 2),
        }
        if has_features and len(item) > 2 and isinstance(item[2], dict):
            feats = item[2]
            row.update({
                "urgency": feats.get("urgency"),
                "difficulty_norm": feats.get("difficulty"),
                "priority_norm": feats.get("priority"),
                "time_pressure": feats.get("time_pressure"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _save_plots_if_available(output_dir, heuristic_df, ml_df, merged_df):
    """Save PNG charts if matplotlib is installed; otherwise write a note file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        with open(
            os.path.join(output_dir, "plot_generation_note.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "Plots were not generated because matplotlib is not installed.\n"
                "Install it with: pip install matplotlib\n"
            )
        return False

    # Horizontal bar chart comparing heuristic and ML scores by task.
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(merged_df))
    width = 0.4
    ax.bar(
        [i - width / 2 for i in x],
        merged_df["heuristic_score"],
        width=width,
        label="Heuristic",
    )
    ax.bar(
        [i + width / 2 for i in x],
        merged_df["ml_score"],
        width=width,
        label="ML",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged_df["task_id"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Heuristic vs ML Scores by Task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_comparison_by_task.png"), dpi=150)
    plt.close(fig)

    # Scatter plot of score agreement.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged_df["heuristic_score"], merged_df["ml_score"], alpha=0.8)
    lo = float(min(merged_df["heuristic_score"].min(), merged_df["ml_score"].min()))
    hi = float(max(merged_df["heuristic_score"].max(), merged_df["ml_score"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Heuristic Score")
    ax.set_ylabel("ML Score")
    ax.set_title("Score Agreement (Heuristic vs ML)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_agreement_scatter.png"), dpi=150)
    plt.close(fig)

    # Top-10 ranking bars for heuristic.
    fig, ax = plt.subplots(figsize=(12, 6))
    h_sorted = heuristic_df.sort_values("rank").head(10)
    ax.barh(h_sorted["task_id"], h_sorted["score"])
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title("Top Tasks by Heuristic Ranking")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heuristic_top_tasks.png"), dpi=150)
    plt.close(fig)

    # Top-10 ranking bars for ML.
    fig, ax = plt.subplots(figsize=(12, 6))
    m_sorted = ml_df.sort_values("rank").head(10)
    ax.barh(m_sorted["task_id"], m_sorted["score"])
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title("Top Tasks by ML Ranking")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ml_top_tasks.png"), dpi=150)
    plt.close(fig)

    return True


def export_run_artifacts(
    tasks,
    reference_time,
    heuristic_ranked,
    ml_ranked,
    comparison,
    train_df,
    ml_metrics,
):
    """Persist run artifacts (tables, metrics, and optional plots) to disk."""
    output_dir = create_output_dir()

    heuristic_df = _ranking_to_df(heuristic_ranked, reference_time, has_features=True)
    ml_df = _ranking_to_df(ml_ranked, reference_time, has_features=False)

    heuristic_df.to_csv(
        os.path.join(output_dir, "heuristic_ranking.csv"),
        index=False,
    )
    ml_df.to_csv(
        os.path.join(output_dir, "ml_ranking.csv"), index=False
    )

    merged_df = (
        heuristic_df[["task_id", "name", "rank", "score"]]
        .rename(columns={"rank": "heuristic_rank", "score": "heuristic_score"})
        .merge(
            ml_df[["task_id", "rank", "score"]].rename(
                columns={"rank": "ml_rank", "score": "ml_score"}
            ),
            on="task_id",
            how="inner",
        )
    )
    merged_df["rank_diff"] = merged_df["heuristic_rank"] - merged_df["ml_rank"]
    merged_df["score_diff"] = merged_df["heuristic_score"] - merged_df["ml_score"]
    merged_df.to_csv(os.path.join(output_dir, "ranking_comparison.csv"), index=False)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_time": reference_time.isoformat(timespec="seconds"),
        "n_demo_tasks": len(tasks),
        "n_training_samples": int(len(train_df)),
        "ml_metrics": ml_metrics,
        "comparison": comparison,
    }
    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(os.path.join(output_dir, "analysis_report.md"), "w", encoding="utf-8") as f:
        f.write("# Run Analysis Report\n\n")
        f.write(f"- Generated at: {summary_payload['generated_at']}\n")
        f.write(f"- Reference time: {summary_payload['reference_time']}\n")
        f.write(f"- Demo tasks: {summary_payload['n_demo_tasks']}\n")
        f.write(f"- Training samples: {summary_payload['n_training_samples']}\n\n")

        f.write("## Model Metrics\n\n")
        for key, value in ml_metrics.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## Comparison Metrics\n\n")
        for key, value in comparison.items():
            f.write(f"- {key}: {value}\n")

        f.write("\n## Saved Files\n\n")
        f.write("- heuristic_ranking.csv\n")
        f.write("- ml_ranking.csv\n")
        f.write("- ranking_comparison.csv\n")
        f.write("- summary_metrics.json\n")

    plots_generated = _save_plots_if_available(output_dir, heuristic_df, ml_df, merged_df)

    print("\n[Artifacts] Results saved to:", output_dir)
    if plots_generated:
        print("[Artifacts] PNG visualizations were generated.")
    else:
        print("[Artifacts] PNG visualizations were skipped (see note file).")


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
    ml_ranked, ml_scorer, train_df, ml_metrics = run_phase2(tasks, reference_time)

    # Comparison
    comparison = run_comparison(heuristic_ranked, ml_ranked, k=3)

    # Persist run artifacts
    export_run_artifacts(
        tasks=tasks,
        reference_time=reference_time,
        heuristic_ranked=heuristic_ranked,
        ml_ranked=ml_ranked,
        comparison=comparison,
        train_df=train_df,
        ml_metrics=ml_metrics,
    )

    print("\n" + "=" * 70)
    print("  Run complete. See README.md for interpretation of results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
