"""
evaluation/plot_cav_selection.py
-----------------------------
Plot in-range CAVs vs LLM-selected CAVs with available bandwidth.
Accepts one or more selection.csv files (e.g. train + test); scenarios are
merged and sorted chronologically. Scenario names are replaced by short IDs
(S01, S02, ...).

Usage
-----
# Single split:
python evaluation/plot_cav_selection.py \
    --selection_csv results/acsos2026/gemma4/opv2v_test/selection.csv \
    --output        results/artifact_evaluation/cav_selection.png

# Train + test merged and sorted chronologically:
python evaluation/plot_cav_selection.py \
    --selection_csv results/acsos2026/gemma4/opv2v_train/selection.csv \
                    results/acsos2026/gemma4/opv2v_test/selection.csv \
    --output        results/artifact_evaluation/cav_selection.png
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scenario_datetime(name: str) -> datetime:
    """Parse YYYY_MM_DD_HH_MM_SS scenario name to datetime for sorting."""
    try:
        return datetime.strptime(name, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return datetime.min


# Bandwidth tier thresholds (must match llm_robot_and_strategy_selector.py)
_BW_HIGH   = 50.0
_BW_MEDIUM = 20.0
_BW_LOW    = 5.0
_THRESH_HIGH_MED = (_BW_HIGH + _BW_MEDIUM) / 2    # 35 Mbps
_THRESH_MED_LOW  = (_BW_MEDIUM + _BW_LOW)  / 2    # 12.5 Mbps


def _row_bandwidth(r: dict) -> float:
    raw_bw = r.get("bandwidth_mbps", "").strip()
    if raw_bw:
        return float(raw_bw)
    _fallback = {"low": _BW_LOW, "medium": _BW_MEDIUM, "high": _BW_HIGH}
    return _fallback.get(r.get("network_congestion", "low").strip().lower(), _BW_LOW)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot in-range vs LLM-selected CAVs with bandwidth")
    parser.add_argument("--selection_csv", nargs="+", required=True,
                        help="One or more paths to selection.csv files")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: show interactively)")
    args = parser.parse_args()

    # -- Load and merge all rows ----------------------------------------------
    all_rows = []
    for path in args.selection_csv:
        all_rows.extend(_load_csv(path))

    if not all_rows:
        print("No data found in selection CSV(s).")
        return

    # -- Sort rows chronologically by scenario datetime, then frame_idx ------
    all_rows.sort(key=lambda r: (
        _scenario_datetime(r["scenario"]),
        int(r.get("frame_idx", 0)),
    ))

    # -- Assign short scenario IDs in chronological order --------------------
    seen_scenarios: list = []
    scenario_id: dict = {}
    for r in all_rows:
        sc = r["scenario"]
        if sc not in scenario_id:
            scenario_id[sc] = f"S{len(scenario_id) + 1:02d}"
            seen_scenarios.append(sc)

    # -- Build series ---------------------------------------------------------
    window_counter: dict = {}
    labels = []
    in_range_counts = []
    selected_counts = []
    bandwidth_values = []

    for r in all_rows:
        sc  = r["scenario"]
        sid = scenario_id[sc]
        window_counter[sc] = window_counter.get(sc, 0) + 1
        w_num = window_counter[sc]

        labels.append(f"{sid}/W{w_num}")
        in_range_counts.append(len([c for c in r["in_range_cavs"].split("|") if c]))
        n_sel = len([c for c in r["selected_cavs"].split("|") if c])
        selected_counts.append(n_sel)
        bandwidth_values.append(_row_bandwidth(r))

    # Clip selected to at most in-range
    selected_counts = [min(s, ir) for s, ir in zip(selected_counts, in_range_counts)]

    # -- Stats: percentage reduction ------------------------------------------
    total_in_range = sum(in_range_counts)
    total_selected = sum(selected_counts)
    reductions = [
        (ir - s) / ir * 100 if ir > 0 else 0.0
        for ir, s in zip(in_range_counts, selected_counts)
    ]
    print(f"\n=== Participant reduction stats ({len(labels)} windows) ===")
    print(f"  Total in-range (default):  {total_in_range}")
    print(f"  Total selected (LLM):      {total_selected}")
    print(f"  Overall reduction:         {(total_in_range - total_selected) / total_in_range * 100:.1f}%")
    print(f"  Mean per-window reduction: {np.mean(reductions):.1f}%  (std {np.std(reductions):.1f}%)")
    print(f"  Median per-window:         {np.median(reductions):.1f}%")
    print()

    # -- Plot -----------------------------------------------------------------
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.16, 2.5))  # IEEE double-column width

    ax.plot(x, in_range_counts, marker="o", markersize=3, linewidth=1.0,
            color="#4C9BE8", label="Default")
    ax.plot(x, selected_counts, marker="x", markersize=3, linewidth=1.0,
            color="#E8724C", label="CoAdapt")

    ax.set_xlabel("Scenario / Window", fontsize=10)
    ax.set_ylabel("Number of CAVs", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=10)
    y_max = max(max(in_range_counts), max(selected_counts))
    ax.set_ylim(bottom=0, top=y_max + 2)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)

    # -- Secondary y-axis: bandwidth (Mbps) -----------------------------------
    ax2 = ax.twinx()
    ax2.plot(x, bandwidth_values, marker="D", markersize=2, linewidth=0.9,
             color="#6BBF59", alpha=0.8, linestyle="--", label="Bandwidth (Mbps)")
    bw_max = max(bandwidth_values) * 1.15
    ax2.set_ylabel("Bandwidth (Mbps)", fontsize=10)
    ax2.set_ylim(bottom=0, top=bw_max)
    ax2.tick_params(axis="y", labelsize=10)

    # -- Legend: top-right ----------------------------------------------------
    # lines1, labs1 = ax.get_legend_handles_labels()
    # lines2, labs2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
    #           loc="upper right", framealpha=0.9, edgecolor="gray")

    fig.tight_layout(pad=0.4)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
