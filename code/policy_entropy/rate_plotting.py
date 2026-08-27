"""Artifacts and compact rate-performance plots for evaluation-only sweeps."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    value = _jsonify(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def save_rate_sweep_outputs(
    summaries: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Save the common CSV table plus JSON summaries and run metadata."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "rate_sweep_summary.csv")
    json_path = os.path.join(output_dir, "rate_sweep_summary.json")
    metadata_path = os.path.join(output_dir, "rate_sweep_metadata.json")

    fieldnames = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _csv_value(summary.get(key)) for key in fieldnames})

    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(_jsonify(summaries), output_file, indent=2, sort_keys=True)
    with open(metadata_path, "w", encoding="utf-8") as output_file:
        json.dump(_jsonify(metadata), output_file, indent=2, sort_keys=True)

    return {
        "summary_csv": csv_path,
        "summary_json": json_path,
        "metadata_json": metadata_path,
    }


def _finite_xy(
    summaries: Iterable[Dict[str, Any]],
    x_key: str,
    y_key: str,
    constrained_only: bool = False,
) -> tuple:
    points = []
    for summary in summaries:
        if constrained_only and summary.get("mode") == "tf_policy":
            continue
        x_value = summary.get(x_key)
        y_value = summary.get(y_key)
        if x_value is None or y_value is None:
            continue
        if np.isfinite(float(x_value)) and np.isfinite(float(y_value)):
            points.append((float(x_value), float(y_value), summary.get("rate_label", "")))
    points.sort(key=lambda item: (item[0], item[2]))
    if not points:
        return np.array([]), np.array([]), []
    return (
        np.asarray([point[0] for point in points]),
        np.asarray([point[1] for point in points]),
        [point[2] for point in points],
    )


def _plot_curve(
    summaries: List[Dict[str, Any]],
    output_path: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    constrained_only: bool = False,
    unrestricted_reference: bool = False,
) -> Optional[str]:
    x_values, y_values, _ = _finite_xy(
        summaries, x_key, y_key, constrained_only=constrained_only
    )
    if x_values.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x_values, y_values, marker="o", label="Greedy / Top-K")
    if unrestricted_reference:
        unrestricted = next(
            (
                summary.get(y_key)
                for summary in summaries
                if summary.get("mode") == "tf_policy" and summary.get(y_key) is not None
            ),
            None,
        )
        if unrestricted is not None and np.isfinite(float(unrestricted)):
            ax.axhline(
                float(unrestricted), linestyle="--", linewidth=1.2, label="Unrestricted"
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if unrestricted_reference:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_rate_sweep_plots(
    summaries: List[Dict[str, Any]],
    output_dir: str,
    title_prefix: str,
) -> Dict[str, Optional[str]]:
    """Generate the required budget/performance and coding-baseline plots."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, Optional[str]] = {}
    paths["hits1_vs_budget"] = _plot_curve(
        summaries,
        os.path.join(output_dir, "hits1_vs_budget.png"),
        "nominal_budget_bits",
        "hits_at_1",
        "Nominal hard budget (bits/hop)",
        "Hits@1",
        f"{title_prefix}: Hits@1 vs hard action budget",
        constrained_only=True,
        unrestricted_reference=True,
    )
    paths["ped_vs_budget"] = _plot_curve(
        summaries,
        os.path.join(output_dir, "ped_vs_budget.png"),
        "nominal_budget_bits",
        "ped",
        "Nominal hard budget (bits/hop)",
        "Path edit distance",
        f"{title_prefix}: PED vs hard action budget",
        constrained_only=True,
        unrestricted_reference=True,
    )
    paths["hits1_vs_empirical_rate"] = _plot_curve(
        summaries,
        os.path.join(output_dir, "hits1_vs_empirical_rate.png"),
        "mean_fixed_budget_bits",
        "hits_at_1",
        "Empirical mean fixed-rank bits/hop",
        "Hits@1",
        f"{title_prefix}: Hits@1 vs empirical fixed-rank rate",
    )

    constrained = [summary for summary in summaries if summary.get("mode") != "tf_policy"]
    x_values, task_values, _ = _finite_xy(
        constrained,
        "nominal_budget_bits",
        "mean_step_shannon_code_bits",
    )
    x_agnostic, agnostic_values, _ = _finite_xy(
        constrained,
        "nominal_budget_bits",
        "mean_task_agnostic_shannon_bits",
    )
    prior_path = os.path.join(output_dir, "task_conditioned_vs_agnostic.png")
    if x_values.size and x_agnostic.size and np.array_equal(x_values, x_agnostic):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(x_values, task_values, marker="o", label="Task-conditioned execution policy")
        ax.plot(x_agnostic, agnostic_values, marker="s", label="Graph-structural prior")
        ax.set_xlabel("Nominal hard budget (bits/hop)")
        ax.set_ylabel("Mean integer Shannon length (bits/action)")
        ax.set_title(f"{title_prefix}: task-conditioned vs task-agnostic coding")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(prior_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["task_conditioned_vs_agnostic"] = prior_path
    else:
        paths["task_conditioned_vs_agnostic"] = None

    return paths
