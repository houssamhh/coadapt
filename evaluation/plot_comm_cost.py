"""
evaluation/plot_comm_cost.py
-------------------------
Bar chart comparing average communication cost (bytes/frame) between
the baseline (all in-range CAVs) and the LLM-selected approach.
One subplot per fusion method to avoid scale issues.

Accepts one or more eval/baseline CSV pairs so train and test splits
can be merged into a single plot. Scenarios are sorted chronologically
and replaced by short IDs (S01, S02, ...).

Usage
-----
# Single split:
python evaluation/plot_comm_cost.py \
    --eval_csv     results/acsos2026/gemma4/opv2v_test/eval.csv \
    --baseline_csv results/acsos2026/gemma4/opv2v_test/eval_baseline.csv \
    --output       results/artifact_evaluation/comm_cost.png

# Train + test merged:
python evaluation/plot_comm_cost.py \
    --eval_csv     results/acsos2026/gemma4/opv2v_train/eval.csv \
                   results/acsos2026/gemma4/opv2v_test/eval.csv \
    --baseline_csv results/acsos2026/gemma4/opv2v_train/eval_baseline.csv \
                   results/acsos2026/gemma4/opv2v_test/eval_baseline.csv \
    --output       results/artifact_evaluation/comm_cost.png
"""

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _format_bytes(val):
    """Human-readable byte string."""
    if val >= 1e6:
        return f"{int(round(val/1e6))}M"
    if val >= 1e3:
        return f"{int(round(val/1e3))}K"
    return f"{int(round(val))}"


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scenario_datetime(name: str) -> datetime:
    """Parse YYYY_MM_DD_HH_MM_SS scenario name to datetime for sorting."""
    try:
        return datetime.strptime(name, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return datetime.min


def main():
    parser = argparse.ArgumentParser(
        description="Plot communication cost: baseline vs LLM-selected")
    parser.add_argument("--eval_csv", nargs="+", required=True,
                        help="One or more paths to eval.csv (LLM pipeline)")
    parser.add_argument("--baseline_csv", nargs="+", required=True,
                        help="One or more paths to eval_baseline.csv")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: show interactively)")
    args = parser.parse_args()

    if len(args.eval_csv) != len(args.baseline_csv):
        parser.error("--eval_csv and --baseline_csv must have the same number of entries")

    # -- Load and merge all rows -----------------------------------------------
    pipeline_rows = []
    for path in args.eval_csv:
        pipeline_rows.extend(_load_csv(path))

    baseline_rows = []
    for path in args.baseline_csv:
        baseline_rows.extend(_load_csv(path))

    if not pipeline_rows or not baseline_rows:
        print("One or both CSVs are empty.")
        return

    # -- Build baseline lookup -------------------------------------------------
    baseline_map = {}
    baseline_ap_map = {}
    for r in baseline_rows:
        key = (r["scenario"], r["selection_frame"])
        baseline_map[key] = float(r.get("avg_comm_bytes_per_frame", 0))
        baseline_ap_map[key] = float(r.get("avg_ap_07", 0))

    # -- Sort pipeline rows chronologically ------------------------------------
    pipeline_rows.sort(key=lambda r: (
        _scenario_datetime(r["scenario"]),
        int(r.get("selection_frame", 0)),
    ))

    # -- Assign short scenario IDs in chronological order ---------------------
    scenario_id: dict = {}
    for r in pipeline_rows:
        sc = r["scenario"]
        if sc not in scenario_id:
            scenario_id[sc] = f"S{len(scenario_id) + 1:02d}"

    # -- Group by fusion method ------------------------------------------------
    groups = defaultdict(lambda: {"labels": [], "pipeline": [], "baseline": [],
                                  "pipeline_ap": [], "baseline_ap": []})
    window_counter: dict = {}

    for r in pipeline_rows:
        key = (r["scenario"], r["selection_frame"])
        if key not in baseline_map:
            continue

        sc  = r["scenario"]
        sid = scenario_id[sc]
        window_counter[sc] = window_counter.get(sc, 0) + 1
        w_num = window_counter[sc]

        fm = r.get("fusion_method", "unknown")
        groups[fm]["labels"].append(f"{sid}/W{w_num}")
        groups[fm]["pipeline"].append(float(r.get("avg_comm_bytes_per_frame", 0)))
        groups[fm]["baseline"].append(baseline_map[key])
        groups[fm]["pipeline_ap"].append(float(r.get("avg_ap_07", 0)))
        groups[fm]["baseline_ap"].append(baseline_ap_map[key])

    if not groups:
        print("No matching windows found between the two CSVs.")
        return

    # Order: early, intermediate, late
    fm_order = [fm for fm in ("early", "intermediate", "late") if fm in groups]
    fm_order += [fm for fm in groups if fm not in fm_order]

    # -- Plot: one subplot per fusion method -----------------------------------
    n_panels = len(fm_order)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(7.16, 2.5 * n_panels),  # IEEE double-column
                             squeeze=False)

    for ax_idx, fm in enumerate(fm_order):
        ax = axes[ax_idx, 0]
        data = groups[fm]
        labels = data["labels"]
        baseline_costs = data["baseline"]
        pipeline_costs = data["pipeline"]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, baseline_costs, width,
               color="#4C9BE8", label="Default", hatch="o")
        ax.bar(x + width / 2, pipeline_costs, width,
               color="#E8724C", label="CoAdapt", hatch="//")

        ax.set_ylabel("Avg. comm. cost / frame (in B)", fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: _format_bytes(v)))
        ax.set_title(f"{fm.capitalize()} fusion", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

        # Annotate values on top of bars
        # y_max = max(max(baseline_costs), max(pipeline_costs)) if baseline_costs else 1
        # offset = y_max * 0.02
        # for i, (b, p) in enumerate(zip(baseline_costs, pipeline_costs)):
        #     ax.text(x[i] - width / 2, b + offset,
        #             _format_bytes(b), ha="center", va="bottom", fontsize=6)
        #     ax.text(x[i] + width / 2, p + offset,
        #             _format_bytes(p), ha="center", va="bottom", fontsize=6)

        # Secondary y-axis: AP@0.7
        ax2 = ax.twinx()
        ax2.plot(x, data["baseline_ap"], marker="o", markersize=3, linewidth=1,
                 color="#1a5276", linestyle="-", alpha=0.95,
                 label="Default AP@0.7", zorder=5)
        ax2.plot(x, data["pipeline_ap"], marker="x", markersize=3, linewidth=1,
                 color="#b03a2e", linestyle="-", alpha=0.95,
                 label="CoAdapt AP@0.7", zorder=5)
        ax2.set_ylabel("AP@0.7", fontsize=10)
        ax2.tick_params(axis="y", labelsize=10)
        ax2.set_ylim(0, 1.05)

        # Combine legends — only on the first panel (early fusion)
        y_max_bar = max(max(baseline_costs), max(pipeline_costs)) if baseline_costs else 1
        ax.set_ylim(bottom=0, top=y_max_bar * 1.3)
        # if ax_idx == 0:
        #     lines1, labels1 = ax.get_legend_handles_labels()
        #     lines2, labels2 = ax2.get_legend_handles_labels()
        #     ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
        #               loc="upper left", framealpha=0.9, edgecolor="gray")

    # fig.suptitle("Communication Cost: Baseline vs. LLM-selected",
                #  fontsize=9, fontweight="bold", y=0.998)
    fig.tight_layout(pad=0.4)

    # Print summary per fusion method
    for fm in fm_order:
        data = groups[fm]
        b_mean = np.mean(data["baseline"])
        p_mean = np.mean(data["pipeline"])
        saving = (b_mean - p_mean) / b_mean * 100 if b_mean > 0 else 0
        print(f"  {fm:14s}  baseline={_format_bytes(b_mean)}  "
              f"pipeline={_format_bytes(p_mean)}  saving={saving:.1f}%")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
