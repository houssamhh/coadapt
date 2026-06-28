"""
Baseline comparison: uses the same fusion method chosen by the LLM (from
selection.csv) but includes ALL in-range CAVs instead of the LLM-selected
subset. 

Usage
-----
python pipeline/run_baseline.py \
    --dataset_root       /path/to/dataset/test \
    --selection_csv      results/model_name/selection.csv \
    --model_early        trained-models/Models/pixor_early_fusion \
    --model_intermediate trained-models/Models/pointpillar_attentive_fusion \
    --model_late         trained-models/Models/pointpillar_late_fusion \
    --output_dir         results/model_name
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


EVAL_FIELDS = [
    "scenario", "selection_frame", "n_frames_inferred", "fusion_method",
    "all_cavs", "in_range_cavs", "selected_cavs", "n_selected",
    "avg_ap_03", "avg_ap_05", "avg_ap_07",
    "total_comm_bytes", "avg_comm_bytes_per_frame",
]


def _pcd_point_count(pcd_path: str) -> int:
    """Read POINTS count from a PCD file header"""
    try:
        with open(pcd_path, "rb") as f:
            for raw_line in f:
                line = raw_line.decode("ascii", errors="ignore").strip()
                if line.startswith("POINTS"):
                    return int(line.split()[1])
                if line.startswith("DATA"):
                    break
    except Exception:
        pass
    return 0


def _load_pcd_numpy(pcd_path: str):
    """Load a PCD file and return (N, 4) float32 array [x, y, z, intensity]."""
    from opencood.utils.pcd_utils import pcd_to_numpy_open3d_equivalent
    try:
        return pcd_to_numpy_open3d_equivalent(pcd_path)
    except Exception:
        return np.empty((0, 4), dtype=np.float32)


def _count_yaml_objects(yaml_path: str) -> int:
    """Count the number of vehicle annotations in a CAV's YAML file."""
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return len(data.get("vehicles", {}))
    except Exception:
        return 0


def _actual_comm_bytes_early(pcd_path: str) -> int:
    """Early fusion: raw point cloud (N, 4) float32 = N × 16 bytes."""
    return _pcd_point_count(pcd_path) * 4 * 4


def _actual_comm_bytes_intermediate(pcd_path: str, lidar_range, voxel_size,
                                     max_points_per_voxel, max_voxels) -> int:
    """Intermediate fusion: actual voxelised data size.

    Loads the point cloud, filters to lidar range, computes unique occupied
    voxels, and returns the exact byte count for:
      voxel_features (M, T, 4) float32 + voxel_coords (M, 3) int32
      + voxel_num_points (M,) int32
    where M = actual occupied voxels, T = max_points_per_voxel.
    """
    pcd_np = _load_pcd_numpy(pcd_path)
    if pcd_np.shape[0] == 0:
        return 0

    from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points
    pcd_np = mask_points_by_range(pcd_np, lidar_range)
    pcd_np = mask_ego_points(pcd_np)
    if pcd_np.shape[0] == 0:
        return 0

    origin = np.array(lidar_range[:3], dtype=np.float32)
    vs = np.array(voxel_size, dtype=np.float32)
    voxel_idx = np.floor((pcd_np[:, :3] - origin) / vs).astype(np.int32)
    n_occupied = min(len(np.unique(voxel_idx, axis=0)), max_voxels)

    T = max_points_per_voxel
    return n_occupied * (T * 4 * 4 + 3 * 4 + 4)


def _actual_comm_bytes_late(yaml_path: str) -> int:
    """Late fusion: bounding-box detections.  n_objects × 7 × 4 bytes."""
    return _count_yaml_objects(yaml_path) * 7 * 4


def _compute_window_comm(windows_by_fm, dataset_root):
    """Compute total_comm_bytes / avg_comm_bytes_per_frame for early and late windows.

    Intermediate fusion comm cost is measured via _CommMeter hooks during inference.
    """
    for fm, windows in windows_by_fm.items():
        if fm == "intermediate":
            # Will be filled in by _CommMeter during inference; skip here.
            continue
        for window in windows:
            scenario = window["meta"]["scenario"]
            ego_cav  = window["meta"].get("ego_cav", "")
            cav_ids  = window.get("selected_cav_ids",
                                  window.get("in_range_cav_ids", []))
            non_ego  = [c for c in cav_ids if c != ego_cav]

            total = 0
            for frame_name in window["frames"]:
                for cav_id in non_ego:
                    pcd    = os.path.join(dataset_root, scenario,
                                          cav_id, f"{frame_name}.pcd")
                    yaml_p = os.path.join(dataset_root, scenario,
                                          cav_id, f"{frame_name}.yaml")
                    if fm == "early":
                        total += _actual_comm_bytes_early(pcd)
                    elif fm == "late":
                        total += _actual_comm_bytes_late(yaml_p)

            n_frames = max(len(window["frames"]), 1)
            window["total_comm_bytes"]         = total
            window["avg_comm_bytes_per_frame"] = total / n_frames


class _CommMeter:
    """Hooks into an intermediate-fusion model to measure transmitted bytes.

    Auto-detects the compression module:
    - ``naive_compressor`` (NaiveCompressor) — used by PointPillar models.
    - ``compression_layer`` (AutoEncoder) — used by VoxelNet models.
    - No compression: hook on ``fusion_net`` pre-hook (full feature map).
    """
    def __init__(self, model):
        import torch.nn as nn
        self.bytes_this_forward: int = 0
        self._handles = []

        if hasattr(model, "naive_compressor"):
            h = model.naive_compressor.encoder.register_forward_hook(
                self._capture_post)
            self._handles.append(h)
            print("[comm] Hooked naive_compressor.encoder (compressed tensor)")
        elif getattr(model, "compression", False) and hasattr(model, "compression_layer"):
            enc = model.compression_layer.encoder
            h = enc[-1].register_forward_hook(self._capture_post)
            self._handles.append(h)
            print("[comm] Hooked compression_layer.encoder[-1] (autoencoder compressed tensor)")
        elif hasattr(model, "fusion_net"):
            if isinstance(model.fusion_net, nn.ModuleList):
                h = model.fusion_net[0].register_forward_pre_hook(
                    self._capture_pre)
                self._handles.append(h)
                print("[comm] Hooked fusion_net[0] pre-hook (CoAlign layer-0)")
            else:
                h = model.fusion_net.register_forward_pre_hook(
                    self._capture_pre)
                self._handles.append(h)
                print("[comm] Hooked fusion_net pre-hook (uncompressed features)")
        else:
            print("[comm] WARNING: no hook point found — comm bytes will be 0")

    def _capture_post(self, _m, _i, output):
        self._measure(output)

    def _capture_pre(self, _m, inputs):
        self._measure(inputs[0])

    def _measure(self, tensor):
        n_non_ego = max(tensor.shape[0] - 1, 0)
        per_cav = (tensor.shape[1] * tensor.shape[2]
                   * tensor.shape[3] * tensor.element_size())
        self.bytes_this_forward = n_non_ego * per_cav

    def reset(self):
        self.bytes_this_forward = 0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ════════════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════════════

def run_baseline(args) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
    from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils

    print("[baseline] All-CAV baseline (fusion method from LLM, all in-range CAVs)")

    # -- No CAV filtering -- use all in-range CAVs ----------------------------
    # We still use a mixin but it does NOT filter — every CAV is kept.
    # This is equivalent to the standard dataset behavior.

    class _NoFilterMixin:
        _active_filter: list = None  # unused, kept for PlanDataset compat
        def retrieve_base_data(self, idx, **kwargs):
            return super().retrieve_base_data(idx, **kwargs)

    class BaselineLate(_NoFilterMixin, LateFusionDataset): pass
    class BaselineEarly(_NoFilterMixin, EarlyFusionDataset): pass
    class BaselineIntermediate(_NoFilterMixin, IntermediateFusionDataset): pass

    _BASELINE = {
        "late":         BaselineLate,
        "early":        BaselineEarly,
        "intermediate": BaselineIntermediate,
    }

    model_map = {
        "early":        args.model_early,
        "intermediate": args.model_intermediate,
        "late":         args.model_late,
    }

    # -- Read selection CSV ---------------------------------------------------

    with open(args.selection_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    valid_rows = [
        r for r in all_rows
        if r.get("fusion_method", "").strip() not in ("", "unknown")
        and r.get("in_range_cavs", "").strip() != ""
    ]
    print(f"[baseline] {len(valid_rows)}/{len(all_rows)} rows with valid fusion method")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- AP helpers -----------------------------------------------------------

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

    # -- Scenario index helpers -----------------------------------------------

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

    # -- Build window plans (same logic, but using in_range_cavs) -------------

    def _build_windows(valid_rows, scenario_index, model_map):
        by_scenario: Dict[str, list] = defaultdict(list)
        for r in valid_rows:
            if not model_map.get(r["fusion_method"]):
                continue
            by_scenario[r["scenario"]].append(r)
        for rows in by_scenario.values():
            rows.sort(key=lambda r: int(r.get("frame_idx", 0)))

        windows_by_fm: Dict[str, list] = defaultdict(list)

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

                fm = row["fusion_method"]
                # KEY DIFFERENCE: use in_range_cavs, not selected_cavs
                windows_by_fm[fm].append({
                    "meta":             row,
                    "in_range_cav_ids": [c for c in row["in_range_cavs"].split("|") if c],
                    "frames":           window_frames,
                })

        return windows_by_fm

    # -- Plan dataset (no filtering) ------------------------------------------

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
                        "flat_idx":  flat_idx,
                        "group_idx": g_idx,
                    })

        def __len__(self): return len(self.entries)

        def __getitem__(self, i):
            entry = self.entries[i]
            try:
                return self.ds[entry["flat_idx"]]
            except (AssertionError, KeyError, TypeError):
                return None

        def collate_batch_test(self, batch):
            return self.ds.collate_batch_test(batch)

    # -- Run inference --------------------------------------------------------

    all_results: List[dict] = []

    active_fms = [fm for fm in ("late", "intermediate", "early")
                  if model_map.get(fm)
                  and any(r["fusion_method"] == fm for r in valid_rows)]

    if not active_fms:
        print("[baseline] No valid fusion methods with model paths — nothing to do")
        return

    # Cache scenario index for faster loading on subsequent runs with the same dataset.
    _cache_key  = hashlib.md5(os.path.abspath(args.dataset_root).encode()).hexdigest()[:12]
    _cache_path = os.path.join(args.output_dir, f".scenario_index_{_cache_key}.pkl")

    if os.path.exists(_cache_path):
        print(f"[baseline] Loading cached scenario index from {_cache_path} ...")
        with open(_cache_path, "rb") as _f:
            scenario_index = pickle.load(_f)
        print(f"[baseline] Scenario index ready ({len(scenario_index)} scenarios) [from cache].")
    else:
        print(f"[baseline] Building scenario index (scanning {args.dataset_root}) ...")
        _index_fm    = active_fms[0]
        _index_opt   = types.SimpleNamespace(model_dir=model_map[_index_fm],
                                             fusion_method=_index_fm)
        _index_hypes = yaml_utils.load_yaml(None, _index_opt)
        _index_hypes["validate_dir"] = args.dataset_root
        _index_hypes["num_agents"] = None
        _index_ds      = _BASELINE[_index_fm](_index_hypes, visualize=False, train=False)
        scenario_index = _build_scenario_index(_index_ds)
        del _index_ds
        os.makedirs(args.output_dir, exist_ok=True)
        with open(_cache_path, "wb") as _f:
            pickle.dump(scenario_index, _f)
        print(f"[baseline] Scenario index ready ({len(scenario_index)} scenarios) — cached.")

    windows_by_fm = _build_windows(valid_rows, scenario_index, model_map)

    # Pre-load hypes for each active fusion method (needed for comm cost)
    hypes_map = {}
    for fm in active_fms:
        opt = types.SimpleNamespace(model_dir=model_map[fm], fusion_method=fm)
        hypes_map[fm] = yaml_utils.load_yaml(None, opt)

    print("[baseline] Computing communication costs (early/late) ...")
    _compute_window_comm(windows_by_fm, args.dataset_root)

    for fm in active_fms:
        print(f"\n[baseline] Inference: method={fm}  (ALL in-range CAVs)")

        hypes = hypes_map[fm]
        hypes["validate_dir"] = args.dataset_root
        hypes["num_agents"] = None

        base_ds = _BASELINE[fm](hypes, visualize=False, train=False)
        base_ds.max_cav = 256

        windows = windows_by_fm.get(fm, [])
        if not windows:
            print("  [warn] no windows for this fusion method — skipping")
            continue

        total_frames = sum(len(w["frames"]) for w in windows)
        print(f"  {len(windows)} selection windows  |  {total_frames} frames total")

        plan_ds = PlanDataset(base_ds, windows, scenario_index)
        if len(plan_ds) == 0:
            print("  [warn] no frames matched — skipping")
            continue

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
        _, model = train_utils.load_saved_model(model_map[fm], model)
        model.eval()

        # For intermediate fusion: measure actual transmitted tensor bytes via hooks.
        # For early/late: comm cost was already computed by _compute_window_comm.
        comm_meter = _CommMeter(model) if fm == "intermediate" else None
        window_comm_bytes = [0] * len(windows)  # hook-accumulated (intermediate only)

        window_stats = [_empty_stat() for _ in windows]

        for i, batch in enumerate(tqdm(loader, desc=f"  {fm}")):
            if batch is None:
                continue
            entry     = plan_ds.entries[i]
            g_idx     = entry["group_idx"]
            batch_dev = train_utils.to_device(batch, device)

            if comm_meter is not None:
                comm_meter.reset()

            with torch.no_grad():
                if fm == "late":
                    pred_box, pred_score, gt_box, _ = \
                        inference_utils.inference_late_fusion(
                            batch_dev, model, base_ds, return_output_dict=True)
                elif fm == "early":
                    pred_box, pred_score, gt_box = \
                        inference_utils.inference_early_fusion(batch_dev, model, base_ds)
                elif fm == "intermediate":
                    pred_box, pred_score, gt_box = \
                        inference_utils.inference_intermediate_fusion(batch_dev, model, base_ds)

            if comm_meter is not None:
                window_comm_bytes[g_idx] += comm_meter.bytes_this_forward

            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box,
                                           window_stats[g_idx], iou)

        if comm_meter is not None:
            comm_meter.remove()

        for g_idx, window in enumerate(windows):
            ap = _compute_ap(window_stats[g_idx])
            meta     = window["meta"]
            in_range = window["in_range_cav_ids"]
            n_frames = max(len(window["frames"]), 1)

            if fm == "intermediate":
                total_bytes = window_comm_bytes[g_idx]
            else:
                total_bytes = window.get("total_comm_bytes", 0)

            all_results.append({
                "scenario":         meta["scenario"],
                "selection_frame":  meta["frame_name"],
                "n_frames_inferred": len(window["frames"]),
                "fusion_method":    fm,
                "all_cavs":         meta.get("all_cavs", ""),
                "in_range_cavs":    meta.get("in_range_cavs", ""),
                "selected_cavs":    "|".join(in_range),  # baseline = all in range
                "n_selected":       len(in_range),
                "avg_ap_03":        ap[0.3],
                "avg_ap_05":        ap[0.5],
                "avg_ap_07":        ap[0.7],
                "total_comm_bytes":         total_bytes,
                "avg_comm_bytes_per_frame": round(total_bytes / n_frames, 1),
            })

        fm_results = [r for r in all_results if r["fusion_method"] == fm]
        def _mean_ap(key):
            vals = [float(r[key]) for r in fm_results
                    if str(r[key]) not in ("", "nan")]
            return np.mean(vals) if vals else float("nan")
        print(f"  Overall (mean of {len(fm_results)} windows) "
              f"AP@0.3={_mean_ap('avg_ap_03'):.4f}  "
              f"AP@0.5={_mean_ap('avg_ap_05'):.4f}  "
              f"AP@0.7={_mean_ap('avg_ap_07'):.4f}")

    if not all_results:
        print("[baseline] No results to save")
        return

    eval_csv = os.path.join(args.output_dir, "eval_baseline.csv")
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[baseline] Results saved to {eval_csv}")

    print("\n-- Baseline: Average AP by fusion method --")
    for fm in sorted(set(r["fusion_method"] for r in all_results)):
        fm_rows = [r for r in all_results if r["fusion_method"] == fm]
        n_windows = len(fm_rows)
        n_frames  = sum(r["n_frames_inferred"] for r in fm_rows)
        for key, label in [("avg_ap_03", "0.3"), ("avg_ap_05", "0.5"), ("avg_ap_07", "0.7")]:
            vals = [float(r[key]) for r in fm_rows if str(r[key]) not in ("", "nan")]
            if vals:
                print(f"  {fm:14s} AP@{label}: mean={np.mean(vals):.4f}  "
                      f"windows={n_windows}  frames={n_frames}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline: same fusion method as LLM, but ALL in-range CAVs")

    p.add_argument("--dataset_root",       required=True)
    p.add_argument("--selection_csv",      required=True,
                   help="Path to selection.csv from the LLM pipeline")
    p.add_argument("--model_early",        default=None)
    p.add_argument("--model_intermediate", default=None)
    p.add_argument("--model_late",         default=None)
    p.add_argument("--output_dir",         default="results/baseline",
                   help="Directory where eval_baseline.csv is saved")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_baseline(args)
