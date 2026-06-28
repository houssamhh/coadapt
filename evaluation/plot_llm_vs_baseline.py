"""
analyze/plot_llm_vs_baseline.py
--------------------------------
For each LLM under results/continuous_bandwidth/, create eval_baseline.csv
(same windows as the LLM eval, but with all in-range CAVs — looked up from
results/continuous_bandwidth/baseline/) and generate a comm-cost vs precision
plot per fusion method, similar to plot_comm_cost.py.

Usage
-----
# All LLMs, all splits found automatically:
python analyze/plot_llm_vs_baseline.py

# Filter to a specific split or LLM:
python analyze/plot_llm_vs_baseline.py --split opv2v_train
python analyze/plot_llm_vs_baseline.py --llm gemma4 --split opv2v_test

# Custom roots:
python analyze/plot_llm_vs_baseline.py --results_root results/continuous_bandwidth
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "continuous_bandwidth")
FIGURES_DIR  = os.path.join(SCRIPT_DIR, "figures")

LLM_DISPLAY = {
    "gemma4":       "Gemma 4 31B",
    "gpt20b":       "GPT-OSS 20B",
    "gpt120b":      "GPT-OSS 120B",
    "llama3.3-70b": "LLaMA 3.3 70B",
}
FUSION_METHODS = ("early", "intermediate", "late")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_bytes(val):
    if val >= 1e9:
        return f"{val/1e9:.1f}G"
    if val >= 1e6:
        return f"{val/1e6:.1f}M"
    if val >= 1e3:
        return f"{val/1e3:.1f}K"
    return f"{val:.0f}B"


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Step 1 — build eval_baseline.csv for a single LLM + split
# ---------------------------------------------------------------------------
def build_eval_baseline(llm_dir: str, split: str, baseline_root: str) -> str | None:
    """
    Match each row in {llm_dir}/{split}/eval.csv against the corresponding
    baseline row in baseline/{split}/{fusion_method}/eval_baseline.csv, then
    write {llm_dir}/{split}/eval_baseline.csv.

    Returns the output path, or None if eval.csv is missing.
    """
    eval_path = os.path.join(llm_dir, split, "eval.csv")
    if not os.path.exists(eval_path):
        return None

    llm_rows = _load_csv(eval_path)
    if not llm_rows:
        return None

    # Load all fusion baselines once
    baseline_maps: dict[str, dict] = {}
    for fm in FUSION_METHODS:
        bpath = os.path.join(baseline_root, split, fm, "eval_baseline.csv")
        brows = _load_csv(bpath)
        baseline_maps[fm] = {
            (r["scenario"], r["selection_frame"]): r for r in brows
        }

    out_rows = []
    fieldnames = None
    for row in llm_rows:
        fm  = row.get("fusion_method", "")
        key = (row.get("scenario", ""), row.get("selection_frame", ""))
        bmap = baseline_maps.get(fm, {})
        if key in bmap:
            brow = bmap[key]
            if fieldnames is None:
                fieldnames = list(brow.keys())
            out_rows.append(brow)

    if not out_rows:
        print(f"  [skip] no matching baseline rows for {os.path.basename(llm_dir)}/{split}")
        return None

    out_path = os.path.join(llm_dir, split, "eval_baseline.csv")
    _write_csv(out_path, out_rows, fieldnames)
    print(f"  [wrote] {os.path.relpath(out_path, REPO_ROOT)}  ({len(out_rows)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Step 2 — plot comm cost + AP for one LLM + split
# ---------------------------------------------------------------------------
def plot_llm_vs_baseline(llm_name: str, llm_dir: str, split: str,
                         out_dir: str) -> None:
    eval_path     = os.path.join(llm_dir, split, "eval.csv")
    baseline_path = os.path.join(llm_dir, split, "eval_baseline.csv")

    if not os.path.exists(eval_path) or not os.path.exists(baseline_path):
        return

    llm_rows  = _load_csv(eval_path)
    base_rows = _load_csv(baseline_path)

    # Index baseline by (scenario, selection_frame)
    baseline_map    = {(r["scenario"], r["selection_frame"]): float(r.get("avg_comm_bytes_per_frame", 0))
                       for r in base_rows}
    baseline_ap_map = {(r["scenario"], r["selection_frame"]): float(r.get("avg_ap_07", 0))
                       for r in base_rows}

    # Group LLM rows by fusion method
    groups = defaultdict(lambda: {
        "labels": [], "llm_cost": [], "base_cost": [],
        "llm_ap": [],  "base_ap": [],
    })
    sc_counter: dict = {}

    for row in llm_rows:
        key = (row.get("scenario", ""), row.get("selection_frame", ""))
        if key not in baseline_map:
            continue
        sc = row["scenario"]
        sc_counter[sc] = sc_counter.get(sc, 0) + 1

        fm = row.get("fusion_method", "unknown")
        g  = groups[fm]
        g["labels"].append(f"{sc[-8:]}/W{sc_counter[sc]}")
        g["llm_cost"].append(float(row.get("avg_comm_bytes_per_frame", 0)))
        g["base_cost"].append(baseline_map[key])
        g["llm_ap"].append(float(row.get("avg_ap_07", 0)))
        g["base_ap"].append(baseline_ap_map[key])

    if not groups:
        print(f"  [skip] no matched windows for {llm_name}/{split}")
        return

    fm_order = [fm for fm in FUSION_METHODS if fm in groups]
    fm_order += [fm for fm in groups if fm not in fm_order]
    n_panels  = len(fm_order)

    max_labels = max(len(groups[fm]["labels"]) for fm in fm_order)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(max(10, max_labels * 0.8), 4 * n_panels),
        squeeze=False,
    )

    display_name = LLM_DISPLAY.get(llm_name, llm_name)
    fig.suptitle(
        f"Communication Cost vs. Precision — {display_name} ({split})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax_idx, fm in enumerate(fm_order):
        ax = axes[ax_idx, 0]
        g  = groups[fm]

        labels     = g["labels"]
        base_costs = g["base_cost"]
        llm_costs  = g["llm_cost"]
        x     = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, base_costs, width, color="#4C9BE8",
               label="Baseline (all CAVs)")
        ax.bar(x + width / 2, llm_costs,  width, color="#E8724C",
               label=f"{display_name}")

        ax.set_ylabel("Comm. cost / frame")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: _format_bytes(v)))
        ax.set_title(f"{fm.capitalize()} fusion — {len(labels)} window(s)",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bar tops
        y_max  = max(max(base_costs), max(llm_costs)) if base_costs else 1
        offset = y_max * 0.02
        ax.set_ylim(bottom=0, top=y_max * 1.3)
        for i, (b, p) in enumerate(zip(base_costs, llm_costs)):
            ax.text(x[i] - width / 2, b + offset,
                    _format_bytes(b), ha="center", va="bottom", fontsize=7)
            ax.text(x[i] + width / 2, p + offset,
                    _format_bytes(p), ha="center", va="bottom", fontsize=7)

        # Right axis: AP@0.7
        ax2 = ax.twinx()
        ax2.plot(x, g["base_ap"], marker="o", markersize=6, linewidth=2.2,
                 color="#1a5276", label="Baseline AP@0.7", zorder=5)
        ax2.plot(x, g["llm_ap"],  marker="s", markersize=6, linewidth=2.2,
                 color="#b03a2e", label=f"{display_name} AP@0.7", zorder=5)
        ax2.set_ylabel("AP@0.7")
        ax2.set_ylim(0, 1.05)

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
                  loc="upper right", bbox_to_anchor=(1.0, 1.0),
                  framealpha=0.9, edgecolor="gray")

        # Summary stats
        b_mean  = np.mean(base_costs) if base_costs else 0
        p_mean  = np.mean(llm_costs)  if llm_costs  else 0
        saving  = (b_mean - p_mean) / b_mean * 100 if b_mean > 0 else 0
        b_ap    = np.mean(g["base_ap"])
        p_ap    = np.mean(g["llm_ap"])
        print(f"    {fm:14s}  baseline={_format_bytes(b_mean):>8s}  "
              f"llm={_format_bytes(p_mean):>8s}  saving={saving:+.1f}%  "
              f"AP baseline={b_ap:.3f}  AP llm={p_ap:.3f}")

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"llm_vs_baseline_{llm_name}_{split}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot]  {os.path.relpath(out_path, REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create eval_baseline.csv per LLM+split and plot comm cost vs precision")
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    parser.add_argument("--llm",   default=None, help="Filter to a single LLM dir name")
    parser.add_argument("--split", default=None, help="Filter to a single split (e.g. opv2v_train)")
    parser.add_argument("--out_dir", default=FIGURES_DIR)
    args = parser.parse_args()

    results_root  = os.path.abspath(args.results_root)
    baseline_root = os.path.join(results_root, "baseline")

    if not os.path.isdir(baseline_root):
        print(f"Baseline directory not found: {baseline_root}")
        return

    # Discover LLM directories
    llm_dirs = {}
    for name in sorted(os.listdir(results_root)):
        if name == "baseline":
            continue
        d = os.path.join(results_root, name)
        if not os.path.isdir(d):
            continue
        if args.llm and name != args.llm:
            continue
        llm_dirs[name] = d

    if not llm_dirs:
        print("No LLM directories found.")
        return

    # Discover splits
    all_splits = set()
    for d in llm_dirs.values():
        for entry in os.listdir(d):
            if os.path.isdir(os.path.join(d, entry)) and entry.startswith("opv2v_"):
                all_splits.add(entry)
    if args.split:
        all_splits = {args.split} if args.split in all_splits else set()

    if not all_splits:
        print("No matching splits found.")
        return

    for llm_name, llm_dir in llm_dirs.items():
        for split in sorted(all_splits):
            eval_path = os.path.join(llm_dir, split, "eval.csv")
            if not os.path.exists(eval_path):
                continue
            print(f"\n{llm_name} / {split}")
            build_eval_baseline(llm_dir, split, baseline_root)
            plot_llm_vs_baseline(llm_name, llm_dir, split, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
