"""
analyze/plot_approach_comparison.py
------------------------------------
Bar chart comparing average communication cost (bytes/frame) and average
AP@0.7 across LLM approaches, rule-based, and the default baseline —
considering only intermediate fusion windows.

Auto-discovers approach directories under results_root (skips "baseline").
Train and test splits are merged.

Usage
-----
python evaluation/plot_approach_comparison.py \
    --results_root results/acsos2026 \
    --splits opv2v_train opv2v_test \
    --output results/artifact_evaluation/approach_comparison.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Display names for known directory names
_DISPLAY_NAMES = {
    "gemma4":        "Gemma 4 31B",
    "gpt20b":        "GPT-OSS 20B",
    "gpt120b":       "GPT-OSS 120B",
    "llama-3.3-70b": "LLama 3.3 70B",
    "llama3.3-70b":  "LLama 3.3 70B",
    "rulebased":     "Rule-based",
    "baseline":      "Default",
}


def _format_bytes(val):
    if val >= 1e6:
        return f"{int(round(val/1e6))}M"
    if val >= 1e3:
        return f"{int(round(val/1e3))}K"
    return f"{int(round(val))}"


def _load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stats(rows, fusion_method=None):
    """Return (avg_comm_bytes, avg_ap07), optionally filtered to a fusion method."""
    if fusion_method:
        rows = [r for r in rows if r.get("fusion_method", "").strip() == fusion_method]
    if not rows:
        return None, None
    comm = np.mean([float(r["avg_comm_bytes_per_frame"]) for r in rows])
    ap   = np.mean([float(r["avg_ap_07"]) for r in rows])
    return comm, ap


def main():
    parser = argparse.ArgumentParser(
        description="Compare approaches: comm cost and AP@0.7 for intermediate fusion")
    parser.add_argument("--results_root", default="results/acsos2026")
    parser.add_argument("--splits", nargs="+", default=["opv2v_train", "opv2v_test"])
    parser.add_argument("--output", default="results/artifact_evaluation/approach_comparison.png")
    args = parser.parse_args()

    root = args.results_root

    # -- Discover LLM/rulebased approaches ------------------------------------
    approach_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d != "baseline"
    ])

    # -- Collect stats per approach -------------------------------------------
    entries = []  # list of (display_name, comm_bytes, ap07)

    for approach in approach_dirs:
        rows = []
        for split in args.splits:
            eval_path = os.path.join(root, approach, split, "eval.csv")
            rows.extend(_load_csv(eval_path))

        comm, ap = _stats(rows)
        if comm is None:
            print(f"  [skip] {approach}: no rows found")
            continue

        name = _DISPLAY_NAMES.get(approach, approach)
        entries.append((name, comm, ap))
        print(f"  {name:20s}  comm={_format_bytes(comm):>8s}  AP@0.7={ap:.4f}")

    # -- Default baseline: intermediate fusion with all CAVs ------------------
    baseline_rows = []
    for split in args.splits:
        bpath = os.path.join(root, "baseline", split, "intermediate", "eval_baseline.csv")
        baseline_rows.extend(_load_csv(bpath))

    b_comm, b_ap = _stats(baseline_rows)
    if b_comm is not None:
        entries.append(("Default", b_comm, b_ap))
        print(f"  {'Default':20s}  comm={_format_bytes(b_comm):>8s}  AP@0.7={b_ap:.4f}")

    if not entries:
        print("No data found.")
        return

    # -- Deduplicate display names (keep first occurrence) --------------------
    seen = set()
    unique_entries = []
    for e in entries:
        if e[0] not in seen:
            seen.add(e[0])
            unique_entries.append(e)
    entries = unique_entries

    names   = [e[0] for e in entries]
    comms   = [e[1] for e in entries]
    aps     = [e[2] for e in entries]

    x = np.arange(len(names))

    # -- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4)) 

    # Color: highlight "Default" differently
    colors = ["#4C9BE8" if n == "Default" else "#E8724C" for n in names]
    bars = ax.bar(x, comms, color=colors, width=0.5, zorder=2)

    ax.set_ylabel("Avg. comm. cost / frame (in B)", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _format_bytes(v)))
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(bottom=0, top=max(comms) * 1.3)
    ax.grid(axis="y", alpha=0.3, zorder=1)

    # Annotate bars
    for bar, val in zip(bars, comms):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(comms) * 0.01,
                _format_bytes(val), ha="center", va="bottom", fontsize=10)

    # Secondary y-axis: AP@0.7
    ax2 = ax.twinx()
    ax2.plot(x, aps, marker="o", markersize=5, linewidth=1.2,
             color="#1a5276", linestyle="-", zorder=5, label="AP@0.7")
    for xi, ap in zip(x, aps):
        ax2.text(xi + 0.05, ap + 0.01, f"{ap:.3f}", fontsize=10,
                 color="#1a5276", va="bottom")
    ax2.set_ylabel("AP@0.7", fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(0, 1.05)

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color="#E8724C", label="CoAdapt / Rule-based"),
        mpatches.Patch(color="#4C9BE8", label="Default"),
        plt.Line2D([0], [0], color="#1a5276", marker="o", markersize=4,
                   linewidth=1.2, label="AP@0.7"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.9, edgecolor="gray")

    fig.tight_layout(pad=0.4)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
