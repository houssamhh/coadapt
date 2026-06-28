"""
analyze/plot_bev_positions.py
------------------------------
Bird's-eye view (BEV) of CAV positions for a given scenario and frame.
Shows ego, in-range, selected, and out-of-range vehicles with communication
range circle. Suitable for inclusion in a research paper.

Usage
-----
python evaluation/plot_bev_positions.py \
    --dataset_root /path/to/opv2v/test \
    --selection_csv results/acsos2026/gemma4/opv2v_test/selection.csv \
    --scenario 2021_08_22_07_52_02 \
    --frame 000071 \
    --output results/artifact_evaluation/bev_example.pdf

If --scenario / --frame are omitted, the first row in selection_csv is used.
"""

import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_pose(yaml_path):
    """Load [x, y, z, roll, yaw, pitch] from a CAV YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.UnsafeLoader)
    pose = data.get("lidar_pose") or data.get("true_ego_pos") or data.get("true_ego_pose")
    if pose is None:
        return None
    return list(pose)


def _yaw_to_rad(yaw_deg):
    """Convert yaw in degrees to radians for arrow direction."""
    return math.radians(yaw_deg)


def main():
    parser = argparse.ArgumentParser(
        description="BEV plot of CAV positions for a scenario/frame")
    parser.add_argument("--dataset_root", required=True,
                        help="Root of the dataset (e.g., opv2v/test)")
    parser.add_argument("--selection_csv", required=True,
                        help="Path to selection.csv")
    parser.add_argument("--scenario", default=None,
                        help="Scenario folder name (default: first in CSV)")
    parser.add_argument("--frame", default=None,
                        help="Frame name, e.g. 000071 (default: first in CSV)")
    parser.add_argument("--all", action="store_true",
                        help="Generate a figure for every row in the CSV")
    parser.add_argument("--output", default=None,
                        help="Output path (.pdf/.png), or output directory when --all is used")
    args = parser.parse_args()

    # -- Read selection CSV to get metadata ------------------------------------
    with open(args.selection_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Empty selection CSV.")
        return

    # -- Batch mode: generate for all rows -------------------------------------
    if args.all:
        out_dir = args.output or "analyze/bev_figures"
        os.makedirs(out_dir, exist_ok=True)
        for i, row in enumerate(rows):
            sc = row["scenario"]
            fr = row["frame_name"]
            out_path = os.path.join(out_dir, f"{sc}_{fr}.pdf")
            print(f"[{i+1}/{len(rows)}] {sc} / {fr} -> {out_path}")
            _plot_one(row, args.dataset_root, out_path)
        print(f"\nDone. {len(rows)} figures saved to {out_dir}/")
        return

    # -- Single mode -----------------------------------------------------------
    target = None
    for r in rows:
        if args.scenario and r["scenario"] != args.scenario:
            continue
        if args.frame and r["frame_name"] != args.frame:
            continue
        target = r
        break

    if target is None:
        print(f"No matching row for scenario={args.scenario}, frame={args.frame}")
        return

    _plot_one(target, args.dataset_root, args.output)


def _plot_one(row, dataset_root, output_path):
    """Generate a single BEV figure for one selection row."""
    scenario = row["scenario"]
    frame_name = row["frame_name"]
    ego_id = row["ego_cav"]
    com_range = float(row["com_range_m"])
    all_cavs = [c for c in row["all_cavs"].split("|") if c]
    in_range = [c for c in row["in_range_cavs"].split("|") if c]
    selected = [c for c in row["selected_cavs"].split("|") if c]

    # -- Load positions from YAML files ----------------------------------------
    scenario_dir = os.path.join(dataset_root, scenario)
    if not os.path.isdir(scenario_dir):
        for sub in os.listdir(dataset_root):
            candidate = os.path.join(dataset_root, sub, scenario)
            if os.path.isdir(candidate):
                scenario_dir = candidate
                break
        else:
            print(f"  [skip] Scenario directory not found: {scenario_dir}")
            return

    positions = {}  # cav_id -> (x, y, yaw_deg)

    for cav_id in all_cavs:
        yaml_path = os.path.join(scenario_dir, cav_id, f"{frame_name}.yaml")
        if not os.path.exists(yaml_path):
            continue
        pose = _load_pose(yaml_path)
        if pose is None:
            continue
        positions[cav_id] = (pose[0], pose[1], pose[4])

    if ego_id not in positions:
        print(f"  [skip] Ego {ego_id} not found — missing YAML at "
              f"{os.path.join(scenario_dir, ego_id, frame_name + '.yaml')}")
        return

    # -- Plot ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # IEEE single-column width

    ego_x, ego_y, _ = positions[ego_id]

    # Communication range circle
    circle = plt.Circle((ego_x, ego_y), com_range,
                        fill=False, linestyle="--", linewidth=0.8,
                        color="#888888")
    ax.add_patch(circle)

    # Plot each CAV
    arrow_len = com_range * 0.12
    for cav_id, (cx, cy, yaw) in positions.items():
        if cav_id == ego_id:
            color = "#2E86AB"
            marker = "^"
            size = 80
            zorder = 10
        elif cav_id in selected:
            color = "#E84855"
            marker = "s"
            size = 55
            zorder = 8
        elif cav_id in in_range:
            color = "#F6AE2D"
            marker = "o"
            size = 45
            zorder = 6
        else:
            color = "#AAAAAA"
            marker = "x"
            size = 35
            zorder = 4

        ax.scatter(cx, cy, c=color, marker=marker, s=size, zorder=zorder,
                   edgecolors="black", linewidths=0.4)

        # Heading arrow
        yaw_rad = _yaw_to_rad(yaw)
        dx = arrow_len * math.cos(yaw_rad)
        dy = arrow_len * math.sin(yaw_rad)
        ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->, head_width=0.3, head_length=0.3", color=color, lw=1.2),
                    zorder=zorder - 1)

    # Axis settings
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Legend by category
    legend_handles = [
        plt.Line2D([0], [0], linestyle="none", marker="^", color="#2E86AB",
                   markeredgecolor="black", markeredgewidth=0.4, markersize=8,
                   label="Ego CAV"),
        plt.Line2D([0], [0], linestyle="none", marker="s", color="#E84855",
                   markeredgecolor="black", markeredgewidth=0.4, markersize=7,
                   label="CoAdapt-selected CAVs"),
        plt.Line2D([0], [0], linestyle="none", marker="o", color="#F6AE2D",
                   markeredgecolor="black", markeredgewidth=0.4, markersize=6,
                   label="Not selected"),
        # plt.Line2D([0], [0], linestyle="none", marker="x", color="#AAAAAA",
        #            markeredgecolor="black", markeredgewidth=0.4, markersize=6,
        #            label="Out of range"),
        plt.Line2D([0], [0], linestyle="--", color="#888888",
                   label=f"Com. range ({com_range:.0f}m)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
              framealpha=0.9)

    # Auto-zoom with padding
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    pad = max(com_range * 0.05, 5)
    ax.set_xlim(min(min(all_x), ego_x - com_range) - pad,
                max(max(all_x), ego_x + com_range) + pad)
    ax.set_ylim(min(min(all_y), ego_y - com_range) - pad,
                max(max(all_y), ego_y + com_range) + pad)

    fig.tight_layout(pad=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
