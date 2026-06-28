"""
    End-to-end pipeline: LLM robot/strategy selection followed by cooperative-
perception inference with exactly the selected CAVs and fusion method.

Phase 1 - LLM selection
  Iterates over every scenario in dataset_root, calls the LLM every
  --reselect_every frames, and saves results to <output_dir>/selection.csv.

Phase 2 - Inference (optional)
  Reads selection.csv, runs inference for each planned frame using the
  selected CAVs and fusion method, and saves per-frame AP results to
  <output_dir>/eval.csv.
  Requires --model_early / --model_intermediate / --model_late.
  Skipped entirely if no model paths are provided.

Usage
-----
# Full pipeline (selection + inference):
python coadapt/run_pipeline.py \
    --dataset_root       /path/to/dataset/test \
    --llm_model          google/gemma-3-27b-it \
    --model_intermediate trained-models/Models/pointpillar_attentive_fusion \
    --model_early        trained-models/Models/pixor_early_fusion \
    --model_late         trained-models/Models/pointpillar_late_fusion \
    --output_dir         results/pipeline_run \
    --reselect_every     5 \
    --min_in_range       3

# Selection only (no inference):
python coadapt/run_pipeline.py \
    --dataset_root  /path/to/dataset/test \
    --llm_model     google/gemma-3-27b-it \
    --output_dir    results/pipeline_run \
    --reselect_every 5

# Inference only (re-use an existing selection.csv):
python coadapt/run_pipeline.py \
    --dataset_root       /path/to/dataset/test \
    --skip_selection \
    --selection_csv      results/pipeline_run/selection.csv \
    --model_intermediate trained-models/Models/pointpillar_attentive_fusion \
    --model_early        trained-models/Models/pixor_early_fusion \
    --model_late         trained-models/Models/pointpillar_late_fusion \
    --output_dir         results/pipeline_run
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import types
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import numpy as np

# ── make repo root importable ────────────────────────────────────────────────
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from coadapt.scene_abstraction_module import iter_frames
from coadapt.llm_robot_and_strategy_selector import (
    RobotAndStrategySelector,
    generate_bandwidth,
)


# ════════════════════════════════════════════════════════════════════════════
# CSV field definitions
# ════════════════════════════════════════════════════════════════════════════

SELECTION_FIELDS = [
    "scenario", "frame_idx", "frame_name",
    "ego_cav", "com_range_m", "bandwidth_mbps",
    "all_cavs", "in_range_cavs",
    "selected_cavs", "n_selected",
    "fusion_method", "reason", "llm_response_time_s",
]

EVAL_FIELDS = [
    "scenario", "selection_frame", "n_frames_inferred", "fusion_method",
    "all_cavs", "in_range_cavs", "selected_cavs", "n_selected",
    "avg_ap_03", "avg_ap_05", "avg_ap_07",
    "total_comm_bytes", "avg_comm_bytes_per_frame",
]


# ════════════════════════════════════════════════════════════════════════════
# Communication cost — actual measurement
# ════════════════════════════════════════════════════════════════════════════

def _pcd_point_count(pcd_path: str) -> int:
    """Read POINTS count from a PCD file header (only reads a few lines)."""
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


def _actual_comm_bytes_late(yaml_path: str) -> int:
    """Late fusion: bounding-box detections.  n_objects × 7 × 4 bytes."""
    return _count_yaml_objects(yaml_path) * 7 * 4


def _compute_window_comm(windows_by_fm, dataset_root):
    """Compute total_comm_bytes / avg_comm_bytes_per_frame for early and late windows.

    Intermediate fusion is intentionally skipped here — its comm cost is
    measured via _CommMeter hooks during inference.
    """
    for fm, windows in windows_by_fm.items():
        for window in windows:
            scenario = window["meta"]["scenario"]
            ego_cav  = window["meta"].get("ego_cav", "")
            cav_ids  = window.get("selected_cav_ids",
                                  window.get("in_range_cav_ids", []))
            non_ego  = [c for c in cav_ids if c != ego_cav]

            total = 0
            for frame_name in window["frames"]:
                for cav_id in non_ego:
                    pcd  = os.path.join(dataset_root, scenario,
                                        cav_id, f"{frame_name}.pcd")
                    yaml_p = os.path.join(dataset_root, scenario,
                                          cav_id, f"{frame_name}.yaml")

                    if fm == "early":
                        total += _actual_comm_bytes_early(pcd)
                    elif fm == "late":
                        total += _actual_comm_bytes_late(yaml_p)
                    # intermediate: measured via _CommMeter hooks during inference

            n_frames = max(len(window["frames"]), 1)
            window["total_comm_bytes"]         = total
            window["avg_comm_bytes_per_frame"] = total / n_frames


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — LLM selection
# ════════════════════════════════════════════════════════════════════════════

def run_selection(args, selection_csv_path: str) -> None:
    """Run the LLM over every scenario and save selections to CSV."""

    abs_root = os.path.abspath(args.dataset_root)
    print(f"[pipeline] Phase 1 — LLM selection")
    print(f"[pipeline] dataset_root = {args.dataset_root}")
    print(f"[pipeline] resolved to  = {abs_root}")

    scenario_paths = sorted(
        d for d in glob.glob(os.path.join(args.dataset_root, "*"))
        if os.path.isdir(d)
    )
    if not scenario_paths:
        print(f"[pipeline] No scenarios found in {abs_root}")
        return

    print(f"[pipeline] Found {len(scenario_paths)} scenarios")

    # Load previous AP results for feedback to the LLM (optional)
    prev_ap_lookup: dict = {}
    if args.prev_eval_csv and os.path.exists(args.prev_eval_csv):
        with open(args.prev_eval_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                prev_ap_lookup[(row["scenario"], row["frame_name"])] = {
                    "ap_03": row.get("ap_03", ""),
                    "ap_05": row.get("ap_05", ""),
                    "ap_07": row.get("ap_07", ""),
                }
        print(f"[pipeline] Loaded {len(prev_ap_lookup)} AP entries from {args.prev_eval_csv}")

    selector = RobotAndStrategySelector(
        args.llm_model,
        hf_token=args.hf_token,
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
    )

    with open(selection_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SELECTION_FIELDS)
        writer.writeheader()

        for sp in scenario_paths:
            scenario_name = os.path.basename(sp)
            print(f"\n[pipeline] Scenario: {scenario_name}")

            prev_result = None
            prev_frame  = None

            for frame_name, frame_idx, scenario_data in iter_frames(
                sp,
                reselect_every=args.reselect_every,
                min_in_range=args.min_in_range,
                com_range_override=args.com_range,
            ):
                bandwidth = generate_bandwidth(scenario_name, int(frame_name))
                prev_ap    = prev_ap_lookup.get((scenario_name, prev_frame)) \
                             if prev_frame else None

                print(f"  frame={frame_name}  bandwidth={bandwidth:.1f} Mbps  "
                      f"in-range CAVs: {[c for c, i in scenario_data['cavs'].items() if i['dist_to_ego'] <= scenario_data['com_range']]}")

                import time as _time
                _t0 = _time.monotonic()
                parsed, raw_response = selector.select(
                    scenario_data,
                    bandwidth_mbps=bandwidth,
                    prev_result=prev_result,
                    prev_ap=prev_ap,
                )
                llm_response_time = round(_time.monotonic() - _t0, 2)
                print(f"  Raw response ({llm_response_time}s):\n{raw_response}\n")

                selected      = parsed.get("selected_cavs", [])
                fusion_method = parsed.get("fusion_method", "unknown")
                reason        = parsed.get("reason", "")
                cavs          = scenario_data["cavs"]
                cr            = scenario_data["com_range"]
                in_range      = [cid for cid, info in cavs.items()
                                 if info["dist_to_ego"] <= cr]

                print(f"  Selected: {selected}  |  Fusion: {fusion_method}")

                writer.writerow({
                    "scenario":           scenario_name,
                    "frame_idx":          frame_idx,
                    "frame_name":         frame_name,
                    "ego_cav":            scenario_data["ego_cav"],
                    "com_range_m":        cr,
                    "bandwidth_mbps":     round(bandwidth, 2),
                    "all_cavs":           "|".join(sorted(cavs.keys())),
                    "in_range_cavs":      "|".join(sorted(in_range)),
                    "selected_cavs":      "|".join(str(c) for c in selected),
                    "n_selected":         len(selected),
                    "fusion_method":      fusion_method,
                    "reason":             reason,
                    "llm_response_time_s": llm_response_time,
                })
                csvfile.flush()

                prev_result = parsed
                prev_frame  = frame_name

    print(f"\n[pipeline] Selection saved to {selection_csv_path}")


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Inference
# ════════════════════════════════════════════════════════════════════════════

def _run_inference(args, selection_csv_path: str, eval_csv_path: str) -> None:
    """Run inference for each planned frame and save AP results to CSV."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
    from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
    from opencood.tools import train_utils, inference_utils
    from opencood.utils import eval_utils

    print(f"\n[pipeline] Phase 2 — Inference")

    # -- CAV filtering helpers ------------------------------------------------

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

    class FilteredLate(_CavFilterMixin, LateFusionDataset): pass
    class FilteredEarly(_CavFilterMixin, EarlyFusionDataset): pass
    class FilteredIntermediate(_CavFilterMixin, IntermediateFusionDataset): pass

    _FILTERED = {
        "late":         FilteredLate,
        "early":        FilteredEarly,
        "intermediate": FilteredIntermediate,
    }

    model_map = {
        "early":        args.model_early,
        "intermediate": args.model_intermediate,
        "late":         args.model_late,
    }

    # -- Read selection CSV ---------------------------------------------------

    with open(selection_csv_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    valid_rows = [
        r for r in all_rows
        if r.get("fusion_method", "").strip() not in ("", "unknown")
        and r.get("selected_cavs", "").strip() not in ("", "0")
    ]
    print(f"[pipeline] {len(valid_rows)}/{len(all_rows)} rows with valid selection")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- AP helpers -----------------------------------------------------------

    def _empty_stat():
        return {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}

    def _compute_ap(stat) -> dict:
        """Compute AP without mutating stat (calculate_ap modifies lists in-place)."""
        import copy
        stat_copy = copy.deepcopy(stat)
        aps = {}
        for iou in (0.3, 0.5, 0.7):
            if stat_copy[iou]["gt"] == 0:
                aps[iou] = float("nan")
            else:
                ap, _, _ = eval_utils.calculate_ap(stat_copy, iou, global_sort_detections=False)
                aps[iou] = round(float(ap), 4)
        return aps

    # -- Comm meter (intermediate fusion only) --------------------------------

    class _CommMeter:
        """Hooks into an intermediate-fusion model to measure transmitted bytes.

        Auto-detects the compression module:
        - ``naive_compressor`` (NaiveCompressor) — used by PointPillar models.
          Hook on ``naive_compressor.encoder`` (post-hook).
        - ``compression_layer`` (AutoEncoder) — used by VoxelNet models.
          Hook on ``compression_layer.encoder[-1]`` (post-hook) to capture the
          bottleneck tensor before the decoder reconstructs it.
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

    # -- Build window plans ---------------------------------------------------
    # Each selection row governs all frames from its frame_idx up to (but not
    # including) the next selection frame in the same scenario.  We expand
    # each selection into a "window" of consecutive frames and tag every frame
    # with its parent selection index so we can aggregate AP per window.

    def _build_windows(valid_rows, scenario_index, model_map):
        """
        Returns a dict  {fusion_method: [window, ...]}
        where each window is:
          {
            "meta":            original selection CSV row dict,
            "selected_cav_ids": list[str],
            "frames":          list of frame_name strings to infer,
            "group_idx":       int  (position in the windows list for this fm),
          }
        """
        # group selections by scenario, sort by frame_idx
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

                # end = start of next selection, or end of scenario
                if i + 1 < len(rows):
                    next_frame = rows[i + 1]["frame_name"]
                    end_idx = ts_map.get(next_frame, len(all_frames))
                else:
                    end_idx = len(all_frames)

                window_frames = [f for f in all_frames
                                 if start_idx <= ts_map[f] < end_idx]

                fm = row["fusion_method"]
                windows_by_fm[fm].append({
                    "meta":             row,
                    "selected_cav_ids": [c for c in row["selected_cavs"].split("|") if c],
                    "frames":           window_frames,
                })

        return windows_by_fm

    # -- Plan dataset wrapper -------------------------------------------------
    # Each entry carries its group_idx so we can accumulate AP per window.

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
                        "selected_cavs": window["selected_cav_ids"],
                        "group_idx":     g_idx,
                    })

        def __len__(self): return len(self.entries)

        def __getitem__(self, i):
            entry = self.entries[i]
            self.ds._active_filter = entry["selected_cavs"]
            try:
                return self.ds[entry["flat_idx"]]
            except (AssertionError, KeyError, TypeError):
                # Some frames have missing/corrupt ego data — skip them.
                return None
            finally:
                self.ds._active_filter = None

        def collate_batch_test(self, batch):
            return self.ds.collate_batch_test(batch)

    # -- Run inference per fusion method --------------------------------------

    all_results: List[dict] = []

    # Build scenario_index ONCE using the first available fusion method.
    # All fusion-method datasets share the same validate_dir so the index is
    # identical — no need to re-scan the dataset for every fm (~10 min each).
    active_fms = [fm for fm in ("late", "intermediate", "early")
                  if model_map.get(fm)
                  and any(r["fusion_method"] == fm for r in valid_rows)]

    if not active_fms:
        print("[pipeline] No valid fusion methods with model paths — skipping inference")
        return

    # Cache scenario_index to a pickle so repeated runs skip the ~10 min scan.
    # Cache is keyed by dataset_root; delete the file to force a rescan.
    import pickle, hashlib
    _cache_key  = hashlib.md5(os.path.abspath(args.dataset_root).encode()).hexdigest()[:12]
    _cache_path = os.path.join(args.output_dir, f".scenario_index_{_cache_key}.pkl")

    if os.path.exists(_cache_path):
        print(f"\n[pipeline] Loading cached scenario index from {_cache_path} ...")
        with open(_cache_path, "rb") as _f:
            scenario_index = pickle.load(_f)
        print(f"[pipeline] Scenario index ready ({len(scenario_index)} scenarios) [from cache].")
    else:
        print(f"\n[pipeline] Building scenario index (scanning {args.dataset_root}) ...")
        _index_fm    = active_fms[0]
        _index_opt   = types.SimpleNamespace(model_dir=model_map[_index_fm],
                                             fusion_method=_index_fm)
        _index_hypes = yaml_utils.load_yaml(None, _index_opt)
        _index_hypes["validate_dir"] = args.dataset_root
        _index_hypes["num_agents"] = None
        _index_ds      = _FILTERED[_index_fm](_index_hypes, visualize=False, train=False)
        scenario_index = _build_scenario_index(_index_ds)
        del _index_ds
        os.makedirs(args.output_dir, exist_ok=True)
        with open(_cache_path, "wb") as _f:
            pickle.dump(scenario_index, _f)
        print(f"[pipeline] Scenario index ready ({len(scenario_index)} scenarios) — cached to {_cache_path}.")

    windows_by_fm = _build_windows(valid_rows, scenario_index, model_map)

    # Pre-load hypes for each active fusion method (needed for comm cost + inference)
    hypes_map = {}
    for fm in active_fms:
        opt = types.SimpleNamespace(model_dir=model_map[fm], fusion_method=fm)
        hypes_map[fm] = yaml_utils.load_yaml(None, opt)
        hypes_map[fm]["validate_dir"] = args.dataset_root
        hypes_map[fm]["num_agents"] = None

    print("[pipeline] Computing communication costs ...")
    _compute_window_comm(windows_by_fm, args.dataset_root)

    for fm in active_fms:
        print(f"\n[pipeline] Inference: method={fm}")

        hypes = hypes_map[fm]
        base_ds = _FILTERED[fm](hypes, visualize=False, train=False)
        # Raise max_cav AFTER __init__ so the scenario_database scan uses the
        # fast default (7), but retrieve_base_data can return all CAVs.
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

        # one stat accumulator per window (group_idx)
        window_stats = [_empty_stat() for _ in windows]

        # For intermediate fusion: measure actual transmitted tensor bytes via hooks.
        # For early/late: comm cost was already computed by _compute_window_comm.
        comm_meter = _CommMeter(model) if fm == "intermediate" else None
        window_comm_bytes = [0] * len(windows)   # hook-accumulated (intermediate only)

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

        # one output row per window
        for g_idx, window in enumerate(windows):
            ap = _compute_ap(window_stats[g_idx])
            meta = window["meta"]

            if fm == "intermediate":
                total_bytes = window_comm_bytes[g_idx]
            else:
                total_bytes = window.get("total_comm_bytes", 0)
            n_frames = max(len(window["frames"]), 1)

            all_results.append({
                "scenario":         meta["scenario"],
                "selection_frame":  meta["frame_name"],
                "n_frames_inferred": len(window["frames"]),
                "fusion_method":    fm,
                "all_cavs":         meta.get("all_cavs", ""),
                "in_range_cavs":    meta.get("in_range_cavs", ""),
                "selected_cavs":    "|".join(window["selected_cav_ids"]),
                "n_selected":       meta.get("n_selected", ""),
                "avg_ap_03":        ap[0.3],
                "avg_ap_05":        ap[0.5],
                "avg_ap_07":        ap[0.7],
                "total_comm_bytes":         total_bytes,
                "avg_comm_bytes_per_frame": round(total_bytes / n_frames, 1),
            })

        # Compute overall as mean of per-window APs (avoids double cumsum
        # that happens when re-aggregating tp/fp after calculate_ap has
        # already converted them to cumulative values in-place).
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
        print("[pipeline] No inference results to save")
        return

    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n[pipeline] Eval results saved to {eval_csv_path}")

    print("\n-- Average AP by fusion method --")
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
    p = argparse.ArgumentParser(description="LLM cooperative-perception pipeline")

    # Dataset
    p.add_argument("--dataset_root",   required=True,
                   help="Root folder of the dataset split to evaluate")
    p.add_argument("--com_range",      type=float, default=None,
                   help="Override communication range (m). "
                        "Defaults to value in data_protocol.yaml")
    p.add_argument("--reselect_every", type=int, default=None, metavar="N",
                   help="Re-query the LLM every N frames. Default: first frame only")
    p.add_argument("--min_in_range",   type=int, default=1, metavar="N",
                   help="Skip frames with fewer than N CAVs in range (default: 1)")

    # LLM
    p.add_argument("--llm_model",      default=None,
                   help="HuggingFace model ID for the selector LLM")
    p.add_argument("--hf_token",       default=None,
                   help="HuggingFace token for gated models")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--quantization",  type=int, default=None,
                   choices=[4, 8, 16, 32],
                   help="Quantization level for the LLM (4/8=bitsandbytes, "
                        "16=fp16, 32=fp32). Ignored for GPT-OSS models.")
    p.add_argument("--prev_eval_csv",  default=None,
                   help="Previous eval.csv to feed AP feedback to the LLM")

    # Selection control
    p.add_argument("--skip_selection", action="store_true",
                   help="Skip Phase 1 and use an existing selection CSV")
    p.add_argument("--selection_csv",  default=None,
                   help="Path to an existing selection CSV (required with --skip_selection)")

    # Inference models
    p.add_argument("--model_early",        default=None)
    p.add_argument("--model_intermediate", default=None)
    p.add_argument("--model_late",         default=None)

    # Output
    p.add_argument("--output_dir", default="results/pipeline",
                   help="Directory where selection.csv and eval.csv are saved")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    selection_csv = (args.selection_csv if args.skip_selection
                     else os.path.join(args.output_dir, "selection.csv"))
    eval_csv      = os.path.join(args.output_dir, "eval.csv")

    # Phase 1
    if args.skip_selection:
        if not selection_csv or not os.path.exists(selection_csv):
            print(f"[pipeline] --skip_selection requires a valid --selection_csv")
            sys.exit(1)
        print(f"[pipeline] Skipping Phase 1 — using {selection_csv}")
    else:
        if not args.llm_model:
            print("[pipeline] --llm_model is required for Phase 1 (or use --skip_selection)")
            sys.exit(1)
        run_selection(args, selection_csv)

    # Phase 2
    has_any_model = any([args.model_early, args.model_intermediate, args.model_late])
    if has_any_model:
        _run_inference(args, selection_csv, eval_csv)
    else:
        print("\n[pipeline] No model paths provided — skipping inference (Phase 2)")
        print(f"[pipeline] To run inference later, use --skip_selection "
              f"--selection_csv {selection_csv} with model paths")


if __name__ == "__main__":
    main()
