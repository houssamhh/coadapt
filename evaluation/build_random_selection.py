"""
analyze/build_random_selection.py
----------------------------------
Generate a random-selection baseline that mirrors LLama's selection.csv:
  - Same fusion strategy per window
  - Same number of CAVs selected
  - CAVs are chosen randomly from in-range vehicles (ego always included,
    remaining slots filled with randomly shuffled non-ego in-range CAVs)

Selection is deterministic: the RNG is seeded with
  hash(f"{scenario}_{frame_name}") so results are reproducible across runs
  but differ from LLama's choices.

Output: results/continuous_bandwidth/random/opv2v_{split}/selection.csv

Usage
-----
python analyze/build_random_selection.py \
    --llama_root  results/continuous_bandwidth/llama-3.3-70b \
    --output_root results/continuous_bandwidth/random \
    --splits      opv2v_train opv2v_test
"""

import argparse
import csv
import hashlib
import os
import random

SELECTION_FIELDS = [
    "scenario", "frame_idx", "frame_name", "ego_cav",
    "com_range_m", "bandwidth_mbps",
    "all_cavs", "in_range_cavs", "selected_cavs", "n_selected",
    "fusion_method", "reason", "llm_response_time_s",
]


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _deterministic_rng(scenario, frame_name):
    """Seed an RNG deterministically from (scenario, frame_name)."""
    seed_str = f"{scenario}_{frame_name}"
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    return rng


def build_random_selection(llama_root, output_root, split):
    sel_path = os.path.join(llama_root, split, "selection.csv")
    rows = _load_csv(sel_path)
    if not rows:
        print(f"  [skip] No selection.csv at {sel_path}")
        return

    out_rows = []
    n_changed = 0
    n_same_forced = 0

    for row in rows:
        scenario   = row["scenario"]
        frame_name = row["frame_name"]
        ego_cav    = row["ego_cav"]
        n_selected = int(row["n_selected"])
        fusion_method = row["fusion_method"]

        in_range = [c for c in row["in_range_cavs"].split("|") if c]
        llama_selected = set(c for c in row["selected_cavs"].split("|") if c)

        # Separate ego and non-ego in-range CAVs
        non_ego_in_range = [c for c in in_range if c != ego_cav]

        # Number of non-ego slots to fill
        n_non_ego = max(0, n_selected - 1)  # -1 for ego

        rng = _deterministic_rng(scenario, frame_name)

        if n_non_ego >= len(non_ego_in_range):
            # Must take all non-ego in-range CAVs — no randomness possible
            chosen_non_ego = non_ego_in_range[:]
            n_same_forced += 1
        else:
            # Shuffle and pick n_non_ego; retry if identical to llama's set
            shuffled = non_ego_in_range[:]
            rng.shuffle(shuffled)
            chosen_non_ego = shuffled[:n_non_ego]

            # If by chance the same as llama, rotate by one to differ
            if set(chosen_non_ego) == (llama_selected - {ego_cav}):
                chosen_non_ego = shuffled[1:n_non_ego + 1] or shuffled[:n_non_ego]

            n_changed += 1

        selected = [ego_cav] + chosen_non_ego
        actual_n = len(selected)

        out_rows.append({
            "scenario":         scenario,
            "frame_idx":        row["frame_idx"],
            "frame_name":       frame_name,
            "ego_cav":          ego_cav,
            "com_range_m":      row["com_range_m"],
            "bandwidth_mbps":   row["bandwidth_mbps"],
            "all_cavs":         row["all_cavs"],
            "in_range_cavs":    row["in_range_cavs"],
            "selected_cavs":    "|".join(selected),
            "n_selected":       actual_n,
            "fusion_method":    fusion_method,
            "reason":           f"Random: {actual_n} CAV(s) selected randomly from {len(in_range)} in-range.",
            "llm_response_time_s": 0.0,
        })

    out_path = os.path.join(output_root, split, "selection.csv")
    _write_csv(out_path, out_rows, SELECTION_FIELDS)
    print(f"  [{split}] {len(out_rows)} rows written to {out_path}")
    print(f"           {n_changed} randomly shuffled, {n_same_forced} forced (all non-ego taken)")


def main():
    parser = argparse.ArgumentParser(
        description="Build random CAV selection mirroring LLaMA strategy/count")
    parser.add_argument("--llama_root",
                        default="results/continuous_bandwidth/llama-3.3-70b",
                        help="Root of LLaMA results")
    parser.add_argument("--output_root",
                        default="results/continuous_bandwidth/random",
                        help="Output root for random results")
    parser.add_argument("--splits", nargs="+",
                        default=["opv2v_train", "opv2v_test"])
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n=== {split} ===")
        build_random_selection(args.llama_root, args.output_root, split)


if __name__ == "__main__":
    main()
