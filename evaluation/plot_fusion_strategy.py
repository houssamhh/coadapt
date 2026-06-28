"""
evaluation/plot_fusion_strategy.py
--------------------------------
Plot fusion strategy (y-axis) vs. scenario/window (x-axis) with bandwidth
evolution on a secondary y-axis.

Accepts one or more selection.csv files (e.g. train + test); scenarios are
merged and sorted chronologically. Scenario names are replaced by short IDs
(S01, S02, ...).

Usage
-----
# Single split:
python evaluation/plot_fusion_strategy.py \
    --selection_csv results/acsos2026/gemma4/opv2v_test/selection.csv \
    --output        results/artifact_evaluation/fusion_strategy.png

# Train + test merged and sorted chronologically:
python evaluation/plot_fusion_strategy.py \
    --selection_csv results/acsos2026/gemma4/opv2v_train/selection.csv \
                    results/acsos2026/gemma4/opv2v_test/selection.csv \
    --output        results/artifact_evaluation/fusion_strategy.png
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scenario_datetime(name: str) -> datetime:
    try:
        return datetime.strptime(name, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return datetime.min


# Fusion method → numeric level (higher = more data shared)
_FM_LEVEL = {"late": 0, "intermediate": 1, "early": 2}
_FM_COLOR = {"late": "#E8724C", "intermediate": "#F5C242", "early": "#4C9BE8"}
_FM_LABELS = {0: "Late", 1: "Intermediate", 2: "Early"}

_BW_HIGH   = 50.0
_BW_MEDIUM = 20.0
_BW_LOW    = 5.0


def _row_bandwidth(r: dict) -> float:
    raw_bw = r.get("bandwidth_mbps", "").strip()
    if raw_bw:
        return float(raw_bw)
    fallback = {"low": _BW_LOW, "medium": _BW_MEDIUM, "high": _BW_HIGH}
    return fallback.get(r.get("network_congestion", "low").strip().lower(), _BW_LOW)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot fusion strategy and bandwidth evolution per window")
    parser.add_argument("--selection_csv", nargs="+", required=True,
                        help="One or more paths to selection.csv files")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: show interactively)")
    args = parser.parse_args()

    # -- Load and merge -------------------------------------------------------
    all_rows = []
    for path in args.selection_csv:
        all_rows.extend(_load_csv(path))

    if not all_rows:
        print("No data found in selection CSV(s).")
        return

    # -- Sort chronologically -------------------------------------------------
    all_rows.sort(key=lambda r: (
        _scenario_datetime(r["scenario"]),
        int(r.get("frame_idx", 0)),
    ))

    # -- Assign short scenario IDs --------------------------------------------
    scenario_id: dict = {}
    for r in all_rows:
        sc = r["scenario"]
        if sc not in scenario_id:
            scenario_id[sc] = f"S{len(scenario_id) + 1:02d}"

    # -- Build series ---------------------------------------------------------
    window_counter: dict = {}
    labels = []
    fm_levels = []
    fm_names = []
    bandwidth_values = []

    for r in all_rows:
        sc  = r["scenario"]
        sid = scenario_id[sc]
        window_counter[sc] = window_counter.get(sc, 0) + 1
        w_num = window_counter[sc]

        fm = r.get("fusion_method", "unknown").strip().lower()
        labels.append(f"{sid}/W{w_num}")
        fm_levels.append(_FM_LEVEL.get(fm, -1))
        fm_names.append(fm)
        bandwidth_values.append(_row_bandwidth(r))

    x = np.arange(len(labels))

    # -- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.16, 2.5))  # IEEE double-column width

    # Scatter: one dot per window, coloured by fusion method
    colors = [_FM_COLOR.get(fm, "#aaaaaa") for fm in fm_names]
    ax.scatter(x, fm_levels, c=colors, s=10, zorder=3)

    # Step line connecting the dots
    ax.step(x, fm_levels, where="mid", linewidth=0.8, color="#555555",
            alpha=0.5, zorder=2)

    ax.set_xlabel("Scenario / Window", fontsize=10)
    ax.set_ylabel("Fusion strategy", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([_FM_LABELS[i] for i in [0, 1, 2]], fontsize=10)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(axis="y", alpha=0.3)

    # -- Secondary y-axis: bandwidth ------------------------------------------
    ax2 = ax.twinx()
    ax2.plot(x, bandwidth_values, marker="D", markersize=2, linewidth=0.9,
             color="#6BBF59", alpha=0.8, linestyle="--", label="Bandwidth (Mbps)")
    bw_max = max(bandwidth_values) * 1.15
    ax2.set_ylabel("Bandwidth (Mbps)", fontsize=10)
    ax2.set_ylim(bottom=0, top=bw_max)
    ax2.tick_params(axis="y", labelsize=10)

    # -- Legend ---------------------------------------------------------------
    patches = [mpatches.Patch(color=_FM_COLOR[fm], label=fm.capitalize())
               for fm in ("early", "intermediate", "late")]
    bw_line = plt.Line2D([0], [0], color="#6BBF59", linewidth=0.9,
                         linestyle="--", marker="D", markersize=3,
                         label="Bandwidth (Mbps)")
    # ax.legend(handles=patches + [bw_line], fontsize=8,
            #   loc="upper right", framealpha=0.9, edgecolor="gray")

    fig.tight_layout(pad=0.4)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
