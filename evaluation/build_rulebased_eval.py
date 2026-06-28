"""
analyze/build_rulebased_eval.py
--------------------------------
Build eval.csv for the rule-based approach by looking up pre-computed
baseline eval results for each (scenario, frame, fusion_method) window.

The rule-based selector picks a fusion method per window (based on
bandwidth tier) but always uses all in-range CAVs — identical to the
baseline evaluation. So performance metrics are looked up directly from
the corresponding baseline eval_baseline.csv.

Output format matches the LLM eval.csv:
  scenario, selection_frame, n_frames_inferred, fusion_method,
  all_cavs, in_range_cavs, selected_cavs, n_selected,
  avg_ap_03, avg_ap_05, avg_ap_07,
  total_comm_bytes, avg_comm_bytes_per_frame

Usage
-----
python analyze/build_rulebased_eval.py \
    --rulebased_root  results/continuous_bandwidth/rulebased \
    --baseline_root   results/continuous_bandwidth/baseline \
    --splits          opv2v_train opv2v_test
"""

import argparse
import csv
import os

FUSION_METHODS = ("early", "intermediate", "late")

EVAL_FIELDS = [
    "scenario", "selection_frame", "n_frames_inferred", "fusion_method",
    "all_cavs", "in_range_cavs", "selected_cavs", "n_selected",
    "avg_ap_03", "avg_ap_05", "avg_ap_07",
    "total_comm_bytes", "avg_comm_bytes_per_frame",
]


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_eval(rulebased_root, baseline_root, split):
    selection_path = os.path.join(rulebased_root, split, "selection.csv")
    selection_rows = _load_csv(selection_path)
    if not selection_rows:
        print(f"  [skip] No selection.csv found at {selection_path}")
        return

    # Load baseline eval for each fusion method → keyed by (scenario, frame)
    baseline_maps = {}
    for fm in FUSION_METHODS:
        bpath = os.path.join(baseline_root, split, fm, "eval_baseline.csv")
        rows = _load_csv(bpath)
        if not rows:
            print(f"  [warn] Missing baseline eval: {bpath}")
        baseline_maps[fm] = {
            (r["scenario"], r["selection_frame"]): r for r in rows
        }

    out_rows = []
    n_matched = 0
    n_missing = 0

    for sel in selection_rows:
        fm = sel.get("fusion_method", "").strip().lower()
        if fm not in FUSION_METHODS:
            print(f"  [skip] Unknown fusion_method '{fm}' for "
                  f"{sel['scenario']}/{sel['frame_name']}")
            n_missing += 1
            continue

        key = (sel["scenario"], sel["frame_name"])
        brow = baseline_maps[fm].get(key)
        if brow is None:
            print(f"  [miss] No baseline match for {key} / {fm}")
            n_missing += 1
            continue

        # Use selected_cavs and n_selected from rulebased, metrics from baseline
        out_row = {
            "scenario":               brow["scenario"],
            "selection_frame":        brow["selection_frame"],
            "n_frames_inferred":      brow["n_frames_inferred"],
            "fusion_method":          fm,
            "all_cavs":               brow["all_cavs"],
            "in_range_cavs":          brow["in_range_cavs"],
            "selected_cavs":          sel["selected_cavs"],
            "n_selected":             sel["n_selected"],
            "avg_ap_03":              brow["avg_ap_03"],
            "avg_ap_05":              brow["avg_ap_05"],
            "avg_ap_07":              brow["avg_ap_07"],
            "total_comm_bytes":       brow["total_comm_bytes"],
            "avg_comm_bytes_per_frame": brow["avg_comm_bytes_per_frame"],
        }
        out_rows.append(out_row)
        n_matched += 1

    out_path = os.path.join(rulebased_root, split, "eval.csv")
    _write_csv(out_path, out_rows, EVAL_FIELDS)
    print(f"  [{split}] {n_matched} rows written to {out_path}  "
          f"({n_missing} missing)")


def main():
    parser = argparse.ArgumentParser(
        description="Build rulebased eval.csv from baseline eval results")
    parser.add_argument("--rulebased_root",
                        default="results/continuous_bandwidth/rulebased",
                        help="Root of rule-based results")
    parser.add_argument("--baseline_root",
                        default="results/continuous_bandwidth/baseline",
                        help="Root of baseline results")
    parser.add_argument("--splits", nargs="+",
                        default=["opv2v_train", "opv2v_test"],
                        help="Dataset splits to process")
    args = parser.parse_args()

    for split in args.splits:
        print(f"\n=== {split} ===")
        build_eval(args.rulebased_root, args.baseline_root, split)


if __name__ == "__main__":
    main()
