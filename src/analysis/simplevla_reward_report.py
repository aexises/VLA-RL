from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) in sys.path:
        sys.path.remove(str(script_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_cache_root = Path(tempfile.gettempdir()) / "simplevla_reward_report"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_run_metrics(metrics_path: Path) -> pd.DataFrame:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            row = {"step": payload["step"], **payload["metrics"]}
            rows.append(row)
    return pd.DataFrame(rows)


def load_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for summary_path in root.glob("**/summary.json"):
        if "SimpleVLA-RL" not in str(summary_path):
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["run_dir"] = str(summary_path.parent)
        rows.append(payload)
    return pd.DataFrame(rows)


def build_report(results_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    root = Path(results_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(root)
    if summaries.empty:
        report = {"runs": 0, "message": "No summary.json files found."}
        (output / "reward_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    summaries.to_csv(output / "reward_summary_table.csv", index=False)
    grouped = summaries.groupby(["task_suite_name", "reward_type"], as_index=False).agg(
        best_validation_success_mean=("best_validation_success", "mean"),
        final_validation_success_mean=("final_validation_success", "mean"),
        mean_cumulative_reward_mean=("mean_cumulative_reward", "mean"),
        mean_advantage_std_mean=("mean_advantage_std", "mean"),
        negative_reward_ratio_mean=("negative_reward_ratio", "mean"),
    )
    grouped.to_csv(output / "reward_comparison_table.csv", index=False)

    tau_only = summaries[summaries["reward_type"] == "clipped_dense"].copy()
    if not tau_only.empty:
        tau_only = tau_only.sort_values(["task_suite_name", "tau_clip"])
        tau_only.to_csv(output / "tau_sweep_table.csv", index=False)

    for task_suite_name, task_frame in grouped.groupby("task_suite_name"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].bar(task_frame["reward_type"], task_frame["final_validation_success_mean"], color=["tab:blue", "tab:green", "tab:orange"][: len(task_frame)])
        axes[0].set_title(f"{task_suite_name} final validation")
        axes[0].set_ylim(0.0, 1.05)
        axes[1].bar(task_frame["reward_type"], task_frame["mean_advantage_std_mean"], color="tab:red")
        axes[1].set_title(f"{task_suite_name} mean advantage std")
        fig.savefig(output / f"{task_suite_name}_comparison.png", dpi=160)
        plt.close(fig)

    for summary in summaries.to_dict(orient="records"):
        metrics_path = Path(summary["run_dir"]) / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        metrics = load_run_metrics(metrics_path)
        if metrics.empty or "val/test_score/all" not in metrics.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(metrics["step"], metrics["val/test_score/all"], label="val/test_score/all")
        if "critic/rewards/mean" in metrics.columns:
            ax.plot(metrics["step"], metrics["critic/rewards/mean"], label="critic/rewards/mean")
        ax.set_title(f'{summary["task_suite_name"]} | {summary["reward_type"]}')
        ax.legend()
        fig.savefig(output / f'{summary["task_suite_name"]}_{summary["reward_type"]}_curve.png', dpi=160)
        plt.close(fig)

    report = {
        "runs": int(len(summaries)),
        "tasks": sorted(summaries["task_suite_name"].dropna().unique().tolist()),
        "reward_types": sorted(summaries["reward_type"].dropna().unique().tolist()),
    }
    (output / "reward_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate SimpleVLA-RL reward comparison outputs.")
    parser.add_argument("--results-root", default=".", help="Directory to search for summary.json artifacts.")
    parser.add_argument("--output-dir", default="results/simplevla_reward_report", help="Directory for report outputs.")
    args = parser.parse_args()
    build_report(args.results_root, args.output_dir)
