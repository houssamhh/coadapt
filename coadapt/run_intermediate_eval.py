"""
Run intermediate-fusion-only evaluation for a specific algorithm, reusing
the window plan from an existing selection.csv.  This is used to compare
multiple intermediate fusion models (with or without compression) on exactly
the same set of frames and CAV selections.

Output is saved in the same directory as --selection_csv with the name:
    eval_{algorithm}_{compression|no-compression}.csv

Usage
-----
# LLM-selected CAVs (default):
python coadapt/run_intermediate_eval.py \
    --dataset_root  /path/to/dataset/test \
    --selection_csv results/pipeline/gemma3/opv2v_test/selection.csv \
    --model_dir     trained-models/Models/pointpillar_attentive_fusion \
    --algorithm     attentive_fusion \
    --compression

# Baseline (all in-range CAVs):
python coadapt/run_intermediate_eval.py \
    --dataset_root  /path/to/dataset/test \
    --selection_csv results/pipeline/gemma3/opv2v_test/selection.csv \
    --model_dir     trained-models/Models/pointpillar_cobevt/pointpillar_CoBEVT_nocompression \
    --algorithm     cobevt \
    --no-compression \
    --baseline
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import pickle
import sys
import types
from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np

# ── make repo root importable ────────────────────────────────────────────────
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ════════════════════════════════════════════════════════════════════════════
# CSV field definitions
# ════════════════════════════════════════════════════════════════════════════

EVAL_FIELDS = [
    "scenario", "selection_frame", "n_frames_inferred", "fusion_method",
    "all_cavs", "in_range_cavs", "selected_cavs", "n_selected",
    "avg_ap_03", "avg_ap_05", "avg_ap_07",
    "total_comm_bytes", "avg_comm_bytes_per_frame",
]


# ════════════════════════════════════════════════════════════════════════════
# Communication cost — actual transmitted tensor measurement
# ════════════════════════════════════════════════════════════════════════════

class _CommMeter:
    """Hooks into the model to capture the actual transmitted tensor size.

    For *compression* models the transmitted payload is the output of the
    compressor's **encoder** (before the decoder reconstructs it on the
    receiver side).

    For *no-compression* models the transmitted payload is the feature map
    fed into the fusion module (backbone output, possibly after shrink_conv).

    CoAlign is handled correctly: only layer-0 features are transmitted,
    and the hook is placed on ``fusion_net[0]`` (or ``naive_compressor.encoder``
    when compression is active).
    """

    def __init__(self, model):
        import torch.nn as nn

        self.bytes_this_forward: int = 0
        self._handles = []

        if hasattr(model, "naive_compressor"):
            # NaiveCompressor: transmitted payload = encoder output.
            h = model.naive_compressor.encoder.register_forward_hook(
                self._capture_tensor)
            self._handles.append(h)
            print("[comm] Hooked naive_compressor.encoder (compressed tensor)")
        elif getattr(model, "compression", False) and hasattr(model, "compression_layer"):
            # AutoEncoder: transmitted payload = last encoder layer output
            # (bottleneck tensor before decoder reconstructs).
            enc = model.compression_layer.encoder
            h = enc[-1].register_forward_hook(self._capture_tensor)
            self._handles.append(h)
            print("[comm] Hooked compression_layer.encoder[-1] (autoencoder compressed tensor)")
        elif hasattr(model, "fusion_net"):
            # No compression → measure features entering the fusion module.
            if isinstance(model.fusion_net, nn.ModuleList):
                # CoAlign: only layer-0 features are transmitted.
                h = model.fusion_net[0].register_forward_pre_hook(
                    self._capture_pre_hook)
                self._handles.append(h)
                print("[comm] Hooked fusion_net[0] pre-hook (CoAlign layer-0)")
            else:
                h = model.fusion_net.register_forward_pre_hook(
                    self._capture_pre_hook)
                self._handles.append(h)
                print("[comm] Hooked fusion_net pre-hook (uncompressed features)")
        else:
            print("[comm] WARNING: could not find a hook point — comm bytes will be 0")

    # -- hook callbacks --------------------------------------------------------

    def _capture_tensor(self, _module, _input, output):
        """Post-hook: captures the encoder output tensor."""
        # output: [N_total_cavs, C_compressed, H, W]
        self._measure(output)

    def _capture_pre_hook(self, _module, inputs):
        """Pre-hook: captures the first positional argument (feature tensor)."""
        # inputs[0]: [N_total_cavs, C, H, W]
        self._measure(inputs[0])

    def _measure(self, tensor):
        """Compute bytes for non-ego CAVs from a stacked [N, C, H, W] tensor."""
        n_total = tensor.shape[0]          # all CAVs in the batch element
        n_non_ego = max(n_total - 1, 0)    # ego doesn't transmit
        per_cav_bytes = (tensor.shape[1] * tensor.shape[2]
                         * tensor.shape[3] * tensor.element_size())
        self.bytes_this_forward = n_non_ego * per_cav_bytes

    # -- lifecycle -------------------------------------------------------------

    def reset(self):
        self.bytes_this_forward = 0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ════════════════════════════════════════════════════════════════════════════

def run_eval(args) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils

    tag = "compression" if args.compression else "no-compression"
    out_name = f"eval_{args.algorithm}_{tag}.csv"
    out_dir  = os.path.dirname(os.path.abspath(args.selection_csv))
    eval_csv = os.path.join(out_dir, out_name)

    print(f"[eval] Intermediate fusion evaluation")
    print(f"[eval] Algorithm: {args.algorithm}  |  Compression: {tag}")
    print(f"[eval] Model dir: {args.model_dir}")
    print(f"[eval] Mode: {'baseline (all in-range CAVs)' if args.baseline else 'LLM-selected CAVs'}")
    print(f"[eval] Output: {eval_csv}")

    # -- Dataset classes -------------------------------------------------------

    def _filter_base_data(base_data: OrderedDict, selected_ids: list) -> OrderedDict:
        keep = set(str(c) for c in selected_ids)
        return OrderedDict(
            (k, v) for k, v in base_data.items()
            if (isinstance(v, dict) and v.get("ego", False)) or str(k) in keep
        )

    class _CavFilterMixin:
        _active_filter: list = None
        def retrieve_base_data(self, idx, **kwargs):
            base = super().retrieve_base_data(idx, **kwargs)
            if self._active_filter is not None:
                base = _filter_base_data(base, self._active_filter)
            return base

    class FilteredIntermediate(_CavFilterMixin, IntermediateFusionDataset): pass

    # -- Load hypes and model --------------------------------------------------

    opt = types.SimpleNamespace(model_dir=args.model_dir, fusion_method="intermediate")
    hypes = yaml_utils.load_yaml(None, opt)
    hypes["validate_dir"] = args.dataset_root
    hypes["num_agents"] = None

    # -- Read selection CSV ----------------------------------------------------

    with open(args.selection_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Only use rows where the LLM selected intermediate fusion
    valid_rows = [
        r for r in all_rows
        if r.get("fusion_method", "").strip() == "intermediate"
        and r.get("selected_cavs", "").strip() not in ("", "0")
        and r.get("in_range_cavs", "").strip() != ""
    ]
    print(f"[eval] {len(valid_rows)}/{len(all_rows)} usable rows from selection.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- AP helpers ------------------------------------------------------------

    def _empty_stat():
        return {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}

    def _compute_ap(stat) -> dict:
        stat_copy = copy.deepcopy(stat)
        aps = {}
        for iou in (0.3, 0.5, 0.7):
            if stat_copy[iou]["gt"] == 0:
                aps[iou] = float("nan")
            else:
                ap, _, _ = eval_utils.calculate_ap(stat_copy, iou, global_sort_detections=False)
                aps[iou] = round(float(ap), 4)
        return aps

    # -- Scenario index --------------------------------------------------------

    def _build_scenario_index(dataset) -> dict:
        root = dataset.params.get("validate_dir") or dataset.params.get("root_dir")
        folders = sorted(x for x in os.listdir(root)
                         if os.path.isdir(os.path.join(root, x)))
        index = {}
        for i, folder in enumerate(folders):
            if i not in dataset.scenario_database:
                continue
            sc_db = dataset.scenario_database[i]
            first_cav = next(iter(sc_db))
            timestamps = sorted(ts for ts in sc_db[first_cav] if ts != "ego")
            index[folder] = (i, {ts: j for j, ts in enumerate(timestamps)})
        return index

    def _find_flat_idx(scenario_index, dataset, scenario_name, frame_name):
        if scenario_name not in scenario_index:
            return None
        sc_idx, ts_map = scenario_index[scenario_name]
        if frame_name not in ts_map:
            return None
        base = dataset.len_record[sc_idx - 1] if sc_idx > 0 else 0
        return base + ts_map[frame_name]

    # -- Cache scenario index --------------------------------------------------

    _cache_key  = hashlib.md5(os.path.abspath(args.dataset_root).encode()).hexdigest()[:12]
    _cache_path = os.path.join(out_dir, f".scenario_index_{_cache_key}.pkl")

    if os.path.exists(_cache_path):
        print(f"[eval] Loading cached scenario index from {_cache_path} ...")
        with open(_cache_path, "rb") as _f:
            scenario_index = pickle.load(_f)
        print(f"[eval] Scenario index ready ({len(scenario_index)} scenarios) [from cache].")
    else:
        print(f"[eval] Building scenario index (scanning {args.dataset_root}) ...")
        base_ds_tmp = FilteredIntermediate(hypes, visualize=False, train=False)
        scenario_index = _build_scenario_index(base_ds_tmp)
        del base_ds_tmp
        os.makedirs(out_dir, exist_ok=True)
        with open(_cache_path, "wb") as _f:
            pickle.dump(scenario_index, _f)
        print(f"[eval] Scenario index ready ({len(scenario_index)} scenarios) — cached.")

    # -- Build windows ---------------------------------------------------------

    def _build_windows(valid_rows, scenario_index):
        by_scenario: Dict[str, list] = defaultdict(list)
        for r in valid_rows:
            by_scenario[r["scenario"]].append(r)
        for rows in by_scenario.values():
            rows.sort(key=lambda r: int(r.get("frame_idx", 0)))

        windows: list = []

        for scenario_name, rows in by_scenario.items():
            if scenario_name not in scenario_index:
                continue
            _, ts_map = scenario_index[scenario_name]
            all_frames = sorted(ts_map.keys(), key=lambda f: ts_map[f])

            for i, row in enumerate(rows):
                sel_frame = row["frame_name"]
                if sel_frame not in ts_map:
                    continue
                start_idx = ts_map[sel_frame]

                if i + 1 < len(rows):
                    next_frame = rows[i + 1]["frame_name"]
                    end_idx = ts_map.get(next_frame, len(all_frames))
                else:
                    end_idx = len(all_frames)

                window_frames = [f for f in all_frames
                                 if start_idx <= ts_map[f] < end_idx]

                # Baseline: all in-range CAVs; otherwise: LLM-selected CAVs
                if args.baseline:
                    cav_ids = [c for c in row["in_range_cavs"].split("|") if c]
                else:
                    cav_ids = [c for c in row["selected_cavs"].split("|") if c]

                windows.append({
                    "meta":    row,
                    "cav_ids": cav_ids,
                    "frames":  window_frames,
                })

        return windows

    windows = _build_windows(valid_rows, scenario_index)

    if not windows:
        print("[eval] No windows to evaluate — nothing to do")
        return

    total_frames = sum(len(w["frames"]) for w in windows)
    print(f"[eval] {len(windows)} selection windows  |  {total_frames} frames total")

    # -- Inference -------------------------------------------------------------

    base_ds = FilteredIntermediate(hypes, visualize=False, train=False)
    base_ds.max_cav = 256

    class PlanDataset(Dataset):
        def __init__(self, base_ds, windows, scenario_index):
            self.ds = base_ds
            self.entries = []
            for g_idx, window in enumerate(windows):
                for frame_name in window["frames"]:
                    flat_idx = _find_flat_idx(
                        scenario_index, base_ds,
                        window["meta"]["scenario"], frame_name,
                    )
                    if flat_idx is None:
                        continue
                    self.entries.append({
                        "flat_idx":      flat_idx,
                        "cav_ids":       window["cav_ids"],
                        "group_idx":     g_idx,
                    })

        def __len__(self): return len(self.entries)

        def __getitem__(self, i):
            entry = self.entries[i]
            if not args.baseline:
                self.ds._active_filter = entry["cav_ids"]
            try:
                return self.ds[entry["flat_idx"]]
            except (AssertionError, KeyError, TypeError):
                return None
            finally:
                self.ds._active_filter = None

        def collate_batch_test(self, batch):
            return self.ds.collate_batch_test(batch)

    plan_ds = PlanDataset(base_ds, windows, scenario_index)
    if len(plan_ds) == 0:
        print("[eval] No frames matched — nothing to do")
        return

    def _collate_skip_none(batch):
        batch = [x for x in batch if x is not None]
        if not batch:
            return None
        return plan_ds.collate_batch_test(batch)

    loader = DataLoader(plan_ds, batch_size=1, num_workers=4,
                        collate_fn=_collate_skip_none,
                        shuffle=False, drop_last=False,
                        persistent_workers=True, prefetch_factor=2)

    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    # Set up communication cost meter (hooks into the model)
    comm_meter = _CommMeter(model)
    window_comm_bytes = [0] * len(windows)   # accumulated per window

    window_stats = [_empty_stat() for _ in windows]

    for i, batch in enumerate(tqdm(loader, desc=f"  intermediate ({args.algorithm})")):
        if batch is None:
            continue
        entry     = plan_ds.entries[i]
        g_idx     = entry["group_idx"]
        batch_dev = train_utils.to_device(batch, device)

        comm_meter.reset()
        with torch.no_grad():
            pred_box, pred_score, gt_box = \
                inference_utils.inference_intermediate_fusion(batch_dev, model, base_ds)

        # Accumulate actual transmitted bytes for this frame
        window_comm_bytes[g_idx] += comm_meter.bytes_this_forward

        for iou in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box,
                                       window_stats[g_idx], iou)

    comm_meter.remove()

    # -- Collect results -------------------------------------------------------

    all_results: List[dict] = []
    for g_idx, window in enumerate(windows):
        ap = _compute_ap(window_stats[g_idx])
        meta = window["meta"]
        n_frames = max(len(window["frames"]), 1)
        total_bytes = window_comm_bytes[g_idx]
        all_results.append({
            "scenario":         meta["scenario"],
            "selection_frame":  meta["frame_name"],
            "n_frames_inferred": len(window["frames"]),
            "fusion_method":    "intermediate",
            "all_cavs":         meta.get("all_cavs", ""),
            "in_range_cavs":    meta.get("in_range_cavs", ""),
            "selected_cavs":    "|".join(window["cav_ids"]),
            "n_selected":       len(window["cav_ids"]),
            "avg_ap_03":        ap[0.3],
            "avg_ap_05":        ap[0.5],
            "avg_ap_07":        ap[0.7],
            "total_comm_bytes":         total_bytes,
            "avg_comm_bytes_per_frame": round(total_bytes / n_frames, 1),
        })

    if not all_results:
        print("[eval] No results to save")
        return

    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # -- Summary ---------------------------------------------------------------

    def _mean_ap(key):
        vals = [float(r[key]) for r in all_results
                if str(r[key]) not in ("", "nan")]
        return np.mean(vals) if vals else float("nan")

    n_windows = len(all_results)
    n_frames  = sum(r["n_frames_inferred"] for r in all_results)
    print(f"\n[eval] Results saved to {eval_csv}")
    print(f"[eval] {args.algorithm} ({tag})  |  {n_windows} windows  |  {n_frames} frames")
    print(f"  AP@0.3={_mean_ap('avg_ap_03'):.4f}  "
          f"AP@0.5={_mean_ap('avg_ap_05'):.4f}  "
          f"AP@0.7={_mean_ap('avg_ap_07'):.4f}")

    comm_vals = [r["avg_comm_bytes_per_frame"] for r in all_results
                 if r["avg_comm_bytes_per_frame"] > 0]
    if comm_vals:
        print(f"  Avg comm cost/frame: {np.mean(comm_vals):,.0f} bytes  "
              f"({np.mean(comm_vals)/1024:,.1f} KB)")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a specific intermediate fusion algorithm "
                    "using the window plan from selection.csv")

    p.add_argument("--dataset_root",  required=True,
                   help="Root folder of the dataset split to evaluate")
    p.add_argument("--selection_csv", required=True,
                   help="Path to selection.csv from the LLM pipeline")
    p.add_argument("--model_dir",     required=True,
                   help="Path to the trained model directory (must contain config.yaml)")
    p.add_argument("--algorithm",     required=True,
                   help="Algorithm name for the output filename "
                        "(e.g., attentive_fusion, cobevt, fcooper, v2vnet)")

    comp = p.add_mutually_exclusive_group(required=True)
    comp.add_argument("--compression", action="store_true", dest="compression",
                      help="Model uses compression")
    comp.add_argument("--no-compression", action="store_false", dest="compression",
                      help="Model does not use compression")

    p.add_argument("--baseline", action="store_true",
                   help="Use all in-range CAVs instead of LLM-selected (baseline mode)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
