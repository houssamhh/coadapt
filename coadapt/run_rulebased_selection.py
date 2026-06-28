"""
pipeline/run_rulebased_selection.py
-------------------------------------
Naive rule-based CAV selection (Phase 1 replacement).

Rules
-----
  - Bandwidth LOW  (< 12.5 Mbps) → late fusion
  - Bandwidth MEDIUM or HIGH (≥ 12.5 Mbps) → intermediate fusion
  - Always select the ego vehicle + the K nearest in-range CAVs
    (K is configurable; if fewer than K are in range, all are selected)

Produces a selection.csv with the same schema as run_pipeline.py so that
Phase 2 inference (run_pipeline.py --skip_selection) can be used directly.

Usage
-----
python coadapt/run_rulebased_selection.py \
    --dataset_root   opv2v_data_dumping/train \
    --output_dir     results/pipeline/rulebased/opv2v_train \
    --k              3 \
    --reselect_every 50 \
    --min_in_range   3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from typing import List

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from coadapt.scene_abstraction_module import iter_frames

# ════════════════════════════════════════════════════════════════════════════
# Bandwidth helpers (mirrors llm_robot_and_strategy_selector.py)
# ════════════════════════════════════════════════════════════════════════════

_BASE_BW = {"high": 50.0, "medium": 20.0, "low": 5.0}
_THRESH_HIGH_MED = (_BASE_BW["high"] + _BASE_BW["medium"]) / 2   # 35.0 Mbps
_THRESH_MED_LOW  = (_BASE_BW["medium"] + _BASE_BW["low"])  / 2   # 12.5 Mbps


def generate_bandwidth(scenario_name: str, frame_id: int) -> float:
    parts = scenario_name.split("_")
    try:
        hour = int(parts[3])
    except (IndexError, ValueError):
        hour = 12
    if 6 <= hour <= 11:
        base = _BASE_BW["high"]
    elif 12 <= hour <= 17:
        base = _BASE_BW["medium"]
    else:
        base = _BASE_BW["low"]
    sinusoidal = 0.15 * base * math.sin(2 * math.pi * frame_id / 50)
    digest = hashlib.md5(f"{scenario_name}_{frame_id}".encode()).hexdigest()
    noise = (int(digest[:8], 16) / 0xFFFFFFFF) * 2 - 1
    return max(0.1, base + sinusoidal + 0.05 * base * noise)


def bw_to_tier(bw: float) -> str:
    if bw >= _THRESH_HIGH_MED:
        return "high"
    if bw >= _THRESH_MED_LOW:
        return "medium"
    return "low"


# ════════════════════════════════════════════════════════════════════════════
# Rule-based selector
# ════════════════════════════════════════════════════════════════════════════

def rule_based_select(
    scenario_data: dict,
    bandwidth_mbps: float,
    k: int,
) -> dict:
    """Apply the naive rule to a single frame.

    Parameters
    ----------
    scenario_data : dict
        Output of scenario_describer.load_frame().
    bandwidth_mbps : float
        Available bandwidth for this frame.
    k : int
        Maximum number of non-ego CAVs to include (nearest-first).

    Returns
    -------
    dict with keys: selected_cavs, fusion_method, reason
    """
    tier = bw_to_tier(bandwidth_mbps)
    fusion_method = "late" if tier == "low" else "intermediate"

    ego  = scenario_data["ego_cav"]
    cr   = scenario_data["com_range"]
    cavs = scenario_data["cavs"]

    # In-range non-ego CAVs sorted by distance (nearest first)
    non_ego_in_range: List[str] = sorted(
        (cid for cid, info in cavs.items()
         if cid != ego and info["dist_to_ego"] <= cr),
        key=lambda cid: cavs[cid]["dist_to_ego"],
    )

    selected_non_ego = non_ego_in_range[:k]
    selected_cavs    = [ego] + selected_non_ego

    reason = (
        f"Rule-based: bandwidth={bandwidth_mbps:.1f} Mbps (tier={tier}), "
        f"fusion={fusion_method}, selected {len(selected_non_ego)} nearest "
        f"non-ego CAV(s) out of {len(non_ego_in_range)} in range."
    )

    return {
        "selected_cavs": selected_cavs,
        "fusion_method": fusion_method,
        "reason":        reason,
    }


# ════════════════════════════════════════════════════════════════════════════
# CSV schema (must match run_pipeline.py SELECTION_FIELDS)
# ════════════════════════════════════════════════════════════════════════════

SELECTION_FIELDS = [
    "scenario", "frame_idx", "frame_name",
    "ego_cav", "com_range_m", "bandwidth_mbps",
    "all_cavs", "in_range_cavs",
    "selected_cavs", "n_selected",
    "fusion_method", "reason", "llm_response_time_s",
]


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Naive rule-based CAV selection (Phase 1 replacement)")
    p.add_argument("--dataset_root",   required=True,
                   help="Root folder of the dataset split")
    p.add_argument("--output_dir",     default="results/pipeline/rulebased",
                   help="Directory where selection.csv is written")
    p.add_argument("--k",              type=int, default=3,
                   help="Max non-ego CAVs to select (nearest K). Default: 3")
    p.add_argument("--reselect_every", type=int, default=None, metavar="N",
                   help="Re-apply rules every N frames. Default: first frame only")
    p.add_argument("--min_in_range",   type=int, default=1, metavar="N",
                   help="Skip frames with fewer than N CAVs in range (default: 1)")
    p.add_argument("--com_range",      type=float, default=None,
                   help="Override communication range (m)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    selection_csv_path = os.path.join(args.output_dir, "selection.csv")

    scenario_paths = sorted(
        p for p in (
            os.path.join(args.dataset_root, d)
            for d in os.listdir(args.dataset_root)
        )
        if os.path.isdir(p)
    )

    if not scenario_paths:
        print(f"[rulebased] No scenarios found in {args.dataset_root}")
        sys.exit(1)

    print(f"[rulebased] dataset_root  = {args.dataset_root}")
    print(f"[rulebased] output_dir    = {args.output_dir}")
    print(f"[rulebased] k             = {args.k}")
    print(f"[rulebased] reselect_every= {args.reselect_every}")
    print(f"[rulebased] scenarios     = {len(scenario_paths)}")

    with open(selection_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SELECTION_FIELDS)
        writer.writeheader()

        for sp in scenario_paths:
            scenario_name = os.path.basename(sp)
            print(f"\n[rulebased] Scenario: {scenario_name}")

            for frame_name, frame_idx, scenario_data in iter_frames(
                sp,
                reselect_every=args.reselect_every,
                min_in_range=args.min_in_range,
                com_range_override=args.com_range,
            ):
                bandwidth = generate_bandwidth(scenario_name, int(frame_name))
                result    = rule_based_select(scenario_data, bandwidth, args.k)

                cavs     = scenario_data["cavs"]
                cr       = scenario_data["com_range"]
                in_range = [cid for cid, info in cavs.items()
                            if info["dist_to_ego"] <= cr]
                selected = result["selected_cavs"]

                print(f"  frame={frame_name}  bw={bandwidth:.1f} Mbps  "
                      f"tier={bw_to_tier(bandwidth)}  "
                      f"fusion={result['fusion_method']}  "
                      f"selected={selected}")

                writer.writerow({
                    "scenario":            scenario_name,
                    "frame_idx":           frame_idx,
                    "frame_name":          frame_name,
                    "ego_cav":             scenario_data["ego_cav"],
                    "com_range_m":         cr,
                    "bandwidth_mbps":      round(bandwidth, 2),
                    "all_cavs":            "|".join(sorted(cavs.keys())),
                    "in_range_cavs":       "|".join(sorted(in_range)),
                    "selected_cavs":       "|".join(str(c) for c in selected),
                    "n_selected":          len(selected),
                    "fusion_method":       result["fusion_method"],
                    "reason":              result["reason"],
                    "llm_response_time_s": 0.0,   # no LLM query
                })
                csvfile.flush()

    print(f"\n[rulebased] Selection saved to {selection_csv_path}")


if __name__ == "__main__":
    main()
