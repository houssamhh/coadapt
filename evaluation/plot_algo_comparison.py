"""
analyze/plot_algo_comparison.py
-------------------------------
Grouped bar chart + line overlay comparing intermediate fusion algorithms.
  X-axis:  scenario / window label
  Left  Y: average communication cost per frame (bars, one per algorithm)
  Right Y: AP@0.7 precision (line curves, one per algorithm)

Auto-discovers all eval_<algo>_<compression>.csv files in --results_dir.

Usage
-----
python analyze/plot_algo_comparison.py \
    --results_dir results/pipeline/gemma4/opv2v_train \
    --output      analyze/figures/algo_comparison.pdf
"""

import argparse
import csv
import glob
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Styling ──────────────────────────────────────────────────────────────────
_COLOURS = [
    "#1f77b4", "#e8724c", "#2ca02c", "#9467bd",
    "#8c564b", "#d62728", "#17becf", "#bcbd22",
]
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def _format_bytes(val, _pos=None):
    """Human-readable byte string for axis labels."""
    if val >= 1e9:
        return f"{val / 1e9:.1f} GB"
    if val >= 1e6:
        return f"{val / 1e6:.1f} MB"
    if val >= 1e3:
        return f"{val / 1e3:.1f} KB"
    return f"{val:.0f} B"


def _pretty_label(algo: str, compression: str) -> str:
    """Human-readable legend label from file naming convention."""
    # Capitalise known acronyms, title-case the rest
    known = {"coalign": "CoAlign", "cobevt": "CoBEVT", "v2vnet": "V2VNet",
             "v2x_vit": "V2X-ViT", "where2comm": "Where2comm",
             "disconet": "DiscoNet", "fcooper": "F-Cooper"}
    name = known.get(algo.lower(), algo.replace("_", " ").title())
    comp = "w/ compr." if compression == "compression" else "w/o compr."
    return f"{name} ({comp})"


def main():
    parser = argparse.ArgumentParser(
        description="Compare intermediate fusion algorithms: comm cost vs precision")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing eval_<algo>_<compression>.csv files")
    parser.add_argument("--ap_key", default="avg_ap_07",
                        help="AP column to use (default: avg_ap_07)")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: show interactively)")
    args = parser.parse_args()

    # ── Discover eval files ──────────────────────────────────────────────────
    pattern = os.path.join(args.results_dir, "eval_*.csv")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "baseline" not in os.path.basename(f).lower()]

    if not files:
        print(f"No eval_*.csv files found in {args.results_dir}")
        return

    # Parse algo and compression from filename:  eval_<algo>_<compression>.csv
    regex = re.compile(r"eval_(.+)_(compression|no-compression)\.csv$")

    # series_name -> list of row dicts (preserving CSV order)
    series: OrderedDict = OrderedDict()

    for fpath in files:
        m = regex.search(os.path.basename(fpath))
        if not m:
            print(f"  [skip] {os.path.basename(fpath)}")
            continue

        algo, compression = m.group(1), m.group(2)
        label = _pretty_label(algo, compression)

        with open(fpath, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print(f"  [skip] {label}: empty CSV")
            continue

        series[label] = rows
        print(f"  Found: {label} ({len(rows)} windows)")

    if not series:
        print("No data to plot.")
        return

    # ── Build a unified x-axis from all files ────────────────────────────────
    # Use the first series to define window labels (all files share the same
    # rows from selection.csv).  Fall back to union if they differ.
    first_rows = list(series.values())[0]
    window_counter: dict = {}
    x_labels = []
    x_keys = []  # (scenario, selection_frame) for matching across series

    for r in first_rows:
        sc = r["scenario"]
        window_counter[sc] = window_counter.get(sc, 0) + 1
        w_num = window_counter[sc]
        x_labels.append(f"{sc[-5:]}/W{w_num}")  # last 5 chars of scenario for brevity
        x_keys.append((r["scenario"], r["selection_frame"]))

    n_windows = len(x_labels)
    n_series = len(series)

    # ── Extract per-series cost and AP arrays aligned to x_keys ──────────────
    aligned = OrderedDict()  # label -> {"costs": [...], "aps": [...]}
    for label, rows in series.items():
        row_map = {}
        for r in rows:
            key = (r["scenario"], r["selection_frame"])
            row_map[key] = r

        costs = []
        aps = []
        for key in x_keys:
            r = row_map.get(key)
            if r:
                costs.append(float(r.get("avg_comm_bytes_per_frame", 0)))
                aps.append(float(r.get(args.ap_key, 0)))
            else:
                costs.append(0.0)
                aps.append(0.0)

        aligned[label] = {"costs": np.array(costs), "aps": np.array(aps)}
        print(f"  {label}: mean cost={_format_bytes(np.mean(costs))}, "
              f"mean AP={np.mean(aps):.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig_width = max(12, n_windows * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(n_windows)
    total_bar_width = 0.75
    bar_width = total_bar_width / n_series

    # Bars: communication cost (left y-axis)
    for idx, (label, data) in enumerate(aligned.items()):
        colour = _COLOURS[idx % len(_COLOURS)]
        offset = -total_bar_width / 2 + bar_width * (idx + 0.5)
        ax.bar(x + offset, data["costs"], bar_width,
               color=colour, alpha=0.7, label=f"{label} (cost)")

    ax.set_ylabel("Avg. comm. cost / frame", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_bytes))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    # Lines: precision (right y-axis)
    ax2 = ax.twinx()
    for idx, (label, data) in enumerate(aligned.items()):
        colour = _COLOURS[idx % len(_COLOURS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        ax2.plot(x, data["aps"], color=colour, marker=marker, markersize=6,
                 linewidth=2.2, alpha=0.95, zorder=5,
                 label=f"{label} (AP)")

    ap_label = args.ap_key.replace("avg_", "").upper().replace("_", "@")
    ax2.set_ylabel(ap_label, fontsize=11)
    ax2.set_ylim(0, 1.05)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.6, n_windows - 0.4)

    # Combined legend — placed below the plot to avoid overlap
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=min(n_series, 3), framealpha=0.9, edgecolor="gray")

    ax.set_title("Intermediate fusion algorithms: communication cost & precision",
                 fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'Algorithm':<35s} {'Mean Cost':>12s} {'Mean AP':>10s}")
    print("-" * 60)
    for label, data in aligned.items():
        print(f"  {label:<33s} {_format_bytes(np.mean(data['costs'])):>12s} "
              f"{np.mean(data['aps']):>10.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
