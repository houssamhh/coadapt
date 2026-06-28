# -*- coding: utf-8 -*-
# Leader runtime for distributed OpenCOOD inference (early / intermediate / late)
#
# Receives agent payloads over ZeroMQ, assembles them per frame, runs fusion on leader.
#
# Notes:
# - Early: agents send projected point clouds (Nx4 float32), leader stacks + preprocess + inference.
# - Intermediate: agents should send per-agent feature tensors right before fusion (model-specific).
# - Late: agents send predicted boxes/scores; leader projects to ego + NMS (no model forward).

import argparse
import csv
from email.mime import base
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import zmq
import msgpack

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.pcd_utils import mask_points_by_range
import opencood.utils.box_utils as box_utils


# ------------------------- CLI -------------------------

def test_parser():
    p = argparse.ArgumentParser("opencood distributed leader")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--fusion_method", type=str, required=True, choices=["early", "intermediate", "late"])
    p.add_argument("--comm_file", type=str, required=True)

    # ZMQ
    p.add_argument("--bind", type=str, default="tcp://0.0.0.0:5555",
                   help="ZMQ PULL bind address for agent payloads.")
    p.add_argument("--expected_agents", type=int, default=2,
                   help="How many agent payloads to wait for per frame (including ego if it sends).")
    p.add_argument("--frame_timeout_s", type=float, default=0.5,
                   help="Fuse with whatever arrived after this timeout (avoid deadlock under loss).")
    p.add_argument("--max_inflight_frames", type=int, default=50,
                   help="Buffer size for out-of-order frames.")
    p.add_argument("--max_frames", type=int, default=-1,
                   help="Stop after N fused frames. -1 means run forever.")

    # Eval
    p.add_argument("--do_eval", action="store_true",
                   help="If set, leader loads GT from dataset to compute AP.")
    p.add_argument("--global_sort_detections", action="store_true")

    return p.parse_args()


# ------------------------- Helpers -------------------------

def late_fuse_nms(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    nms_thresh: float = 0.2,
    topk: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Concatenate ego-frame detections from all agents and apply NMS.
    Expects each boxes array to be (N,7): x,y,z,dx,dy,dz,yaw and scores (N,).
    Returns torch tensors on CPU.
    """
    if not boxes_list:
        return None, None

    boxes = np.concatenate([b for b in boxes_list if b is not None and b.size > 0], axis=0) if boxes_list else np.zeros((0,7), np.float32)
    scores = np.concatenate([s for s in scores_list if s is not None and s.size > 0], axis=0) if scores_list else np.zeros((0,), np.float32)

    if boxes.shape[0] == 0:
        return None, None
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise RuntimeError(f"late_fuse_nms expects boxes (N,7), got {boxes.shape}")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise RuntimeError(f"late_fuse_nms expects scores (N,), got {scores.shape} vs boxes {boxes.shape}")

    # sort by score desc
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]

    if topk is not None and boxes.shape[0] > topk:
        boxes = boxes[:topk]
        scores = scores[:topk]

    # --- NMS: try common OpenCOOD function names (varies by fork) ---
    keep = None

    # Many forks use rotated BEV NMS
    if hasattr(box_utils, "nms_rotated"):
        keep = box_utils.nms_rotated(boxes, scores, nms_thresh)

    # Some use torch-based rotated NMS
    elif hasattr(box_utils, "nms_rotated_torch"):
        b = torch.from_numpy(boxes).float()
        s = torch.from_numpy(scores).float()
        keep = box_utils.nms_rotated_torch(b, s, nms_thresh)
        keep = keep.detach().cpu().numpy() if torch.is_tensor(keep) else np.asarray(keep)

    # Fallback: no NMS available -> keep everything (NOT ideal)
    if keep is None:
        # Brutal honesty: if you hit this path, your late fusion results won't be comparable.
        keep = np.arange(boxes.shape[0], dtype=np.int64)

    keep = np.asarray(keep, dtype=np.int64)

    boxes_f = torch.from_numpy(boxes[keep]).float()
    scores_f = torch.from_numpy(scores[keep]).float()
    return boxes_f, scores_f


def get_gt_for_frame(opencood_dataset, frame_id: int) -> Optional[torch.Tensor]:
    """
    Get GT boxes for a frame using the dataset sample, in the same format used by eval_utils.
    """
    sample = opencood_dataset[frame_id]
    sample = opencood_dataset.collate_batch_test([sample])

    # Most OpenCOOD datasets expose these
    if "ego" not in sample:
        return None
    ego = sample["ego"]
    if "object_bbx_center" not in ego or "object_bbx_mask" not in ego:
        return None

    # post_processor usually can generate GT from these fields
    gt_box_tensor = opencood_dataset.post_processor.generate_gt_bbx(sample)
    return gt_box_tensor

def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.element_size() * t.nelement()) if t is not None else 0


def recv_msg_pull(sock: zmq.Socket) -> Tuple[Dict[str, Any], List[bytes]]:
    """
    Receive a multipart message:
      [header_msgpack, payload0, payload1, ...]
    """
    parts = sock.recv_multipart()
    if not parts:
        raise RuntimeError("Empty ZMQ message")
    header = msgpack.loads(parts[0], raw=False)
    return header, parts[1:]


def now_s() -> float:
    return time.time()


# ------------------------- Assembly per fusion mode -------------------------

def assemble_early_batch_from_payloads(
    opencood_dataset,
    projected_list: List[np.ndarray],
) -> Dict[str, Any]:
    """
    Build a minimal OpenCOOD batch_data for early fusion from already-ego-projected point clouds.
    """
    assert len(projected_list) > 0
    stacked = np.vstack(projected_list).astype(np.float32, copy=False)

    # Preprocess to voxel/BEV/etc (same component used in dataset)
    stacked = mask_points_by_range(stacked, opencood_dataset.params["preprocess"]["cav_lidar_range"])
    lidar_dict = opencood_dataset.pre_processor.preprocess(stacked)

    # Collate into torch tensors shaped as OpenCOOD expects
    processed_lidar_torch_dict = opencood_dataset.pre_processor.collate_batch([lidar_dict])

    # Minimal batch for early fusion inference
    batch_data = {
        "ego": {
            "processed_lidar": processed_lidar_torch_dict,
            # Some pipelines expect this key to exist; safest to include identity.
            "transformation_matrix": torch.eye(4, dtype=torch.float32),
        }
    }
    return batch_data


def assemble_intermediate_batch_placeholder(
    feature_payloads: List[Tuple[str, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Placeholder: intermediate fusion requires model-specific keys.
    You must align this with your OpenCOOD model's forward() expected inputs.
    """
    raise NotImplementedError(
        "Intermediate fusion needs model-specific batch_data assembly. "
        "Decide what tensor agents send (e.g., per-agent BEV feature map right before fusion), "
        "then reconstruct the keys expected by inference_utils.inference_intermediate_fusion()."
    )


def fuse_late_on_leader(
    opencood_dataset,
    ego_pose: List[float],
    det_payloads: List[Dict[str, Any]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Late fusion on leader: project received boxes to ego and apply NMS/WBF.
    This avoids running the detector on the leader.
    """
    # Expected payload format per agent (example):
    # {
    #   "cav_id": "...",
    #   "boxes": np.ndarray (M,7) in agent frame OR ego frame (you decide),
    #   "scores": np.ndarray (M,),
    #   "pose": [x,y,z,roll,pitch,yaw] if boxes are in agent frame
    # }
    #
    # For minimal pain: have agents send boxes already in ego frame.
    # Then leader just concatenates and runs NMS via post_processor if possible.

    all_boxes = []
    all_scores = []

    for p in det_payloads:
        boxes = p.get("boxes", None)
        scores = p.get("scores", None)
        if boxes is None or scores is None:
            continue
        boxes_t = torch.from_numpy(np.asarray(boxes)).float()
        scores_t = torch.from_numpy(np.asarray(scores)).float()
        if boxes_t.numel() == 0:
            continue
        all_boxes.append(boxes_t)
        all_scores.append(scores_t)

    if not all_boxes:
        return None, None

    boxes_cat = torch.cat(all_boxes, dim=0)
    scores_cat = torch.cat(all_scores, dim=0)

    # Use post_processor NMS if available; otherwise fallback to a simple top-k.
    # Many OpenCOOD postprocessors expose a nms function internally; not standardized across all.
    # Here we do a conservative fallback: sort by score and keep all (you should replace with real NMS).
    # Brutal honesty: you should wire actual NMS from your post_processor for correctness.
    order = torch.argsort(scores_cat, descending=True)
    boxes_cat = boxes_cat[order]
    scores_cat = scores_cat[order]

    return boxes_cat, scores_cat


# ------------------------- Main -------------------------

def main():
    opt = test_parser()

    hypes = yaml_utils.load_yaml(None, opt)

    print("Dataset Building (leader)")
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)

    # Create model only if needed (early/intermediate). Late fusion leader can run without model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if opt.fusion_method in ("early", "intermediate"):
        print("Creating Model")
        model = train_utils.create_model(hypes)
        if torch.cuda.is_available():
            model.cuda()
        print("Loading Model from checkpoint")
        _, model = train_utils.load_saved_model(opt.model_dir, model)
        model.eval()

    # Comm log
    os.makedirs(os.path.dirname(opt.comm_file) or ".", exist_ok=True)
    comm_log_f = open(opt.comm_file, mode="w", newline="")
    comm_writer = csv.writer(comm_log_f)

    if opt.fusion_method == "early":
        comm_writer.writerow(["frame_id", "num_agents_received", "early_tx_bytes_total"])
    elif opt.fusion_method == "intermediate":
        comm_writer.writerow(["frame_id", "num_agents_received", "bytes_non_ego", "payload_shapes", "payload_dtype"])
    else:
        comm_writer.writerow(["frame_id", "num_agents_received", "late_tx_bytes_total", "num_boxes_total"])

    # Eval stats (optional)
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    # ZMQ setup
    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    pull.bind(opt.bind)
    pull.RCVTIMEO = 100  # ms; lets us implement timeout logic without blocking forever

    # Buffer per frame_id
    # key: (scenario_id, frame_id) -> dict(cav_id -> payload record)
    inflight: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    first_seen: Dict[Tuple[str, int], float] = {}
    order_q = deque()  # keep insertion order for eviction

    fused_count = 0

    def maybe_evict():
        while len(inflight) > opt.max_inflight_frames:
            k = order_q.popleft()
            inflight.pop(k, None)
            first_seen.pop(k, None)

    print(f"Leader listening on {opt.bind} | fusion={opt.fusion_method} | expected_agents={opt.expected_agents}")

    while True:
        if opt.max_frames > 0 and fused_count >= opt.max_frames:
            break

        # receive (non-blocking-ish)
        try:
            header, payload_parts = recv_msg_pull(pull)
        except zmq.error.Again:
            header = None

        if header is not None:
            scenario_id = str(header.get("scenario_id", "default"))
            frame_id = int(header["frame_id"])
            cav_id = str(header["cav_id"])
            payload_type = header.get("payload_type", opt.fusion_method)

            key = (scenario_id, frame_id)
            if key not in inflight:
                inflight[key] = {}
                first_seen[key] = now_s()
                order_q.append(key)
                maybe_evict()

            rec: Dict[str, Any] = {"header": header}

            # Decode depending on payload_type
            if payload_type == "early":
                # payload_parts[0] contains raw float32 bytes of Nx4 projected lidar
                shape = tuple(header["shape"])
                arr = np.frombuffer(payload_parts[0], dtype=np.float32).reshape(shape)
                rec["projected_lidar"] = arr
                rec["tx_bytes"] = int(header.get("nbytes", arr.nbytes))

            elif payload_type == "intermediate":
                # You decide what agents send. Common: one tensor per agent.
                # For now, just store raw bytes and metadata.
                rec["raw_parts"] = payload_parts
                rec["tx_bytes"] = sum(len(p) for p in payload_parts)
                rec["meta_shape"] = header.get("shape", None)
                rec["meta_dtype"] = header.get("dtype", None)

            elif payload_type == "late":
                # Expect two parts: boxes bytes + scores bytes, with shapes in header
                # Example header fields: boxes_shape, scores_shape
                boxes_shape = tuple(header["boxes_shape"])
                scores_shape = tuple(header["scores_shape"])
                boxes = np.frombuffer(payload_parts[0], dtype=np.float32).reshape(boxes_shape)
                scores = np.frombuffer(payload_parts[1], dtype=np.float32).reshape(scores_shape)
                rec["boxes"] = boxes
                rec["scores"] = scores
                rec["tx_bytes"] = int(header.get("nbytes", boxes.nbytes + scores.nbytes))
            else:
                # Unknown payload; drop
                continue

            inflight[key][cav_id] = rec

        # Decide if any frame is ready to fuse
        # Strategy: fuse when expected_agents received OR timeout expired.
        ready_keys = []
        t = now_s()
        for key, cav_map in inflight.items():
            if len(cav_map) >= opt.expected_agents:
                ready_keys.append(key)
            # else:
            #     if (t - first_seen.get(key, t)) >= opt.frame_timeout_s and len(cav_map) > 0:
            #         ready_keys.append(key)

        # Fuse ready frames (in order of frame_id)
        ready_keys.sort(key=lambda k: (k[0], k[1]))

        for key in ready_keys:
            scenario_id, frame_id = key
            cav_map = inflight.pop(key, {})
            first_seen.pop(key, None)
            # also remove from order_q if present (not critical if left; eviction handles missing)
            # (optional) clean order_q lazily

            num_agents = len(cav_map)

            if opt.fusion_method == "early":
                projected_list = [rec["projected_lidar"] for rec in cav_map.values() if "projected_lidar" in rec]
                if not projected_list:
                    continue

                batch_data = assemble_early_batch_from_payloads(opencood_dataset, projected_list)
                # Pull GT fields from the dataset sample for this frame
                sample = opencood_dataset[frame_id]          # calls __getitem__
                sample = opencood_dataset.collate_batch_test([sample])  # make it shaped like inference expects

                batch_data["ego"]["object_bbx_center"] = sample["ego"]["object_bbx_center"]
                batch_data["ego"]["object_bbx_mask"]   = sample["ego"]["object_bbx_mask"]
                batch_data["ego"]["object_ids"]        = sample["ego"]["object_ids"]
                # batch_data["ego"]["anchor_box"]        = sample["ego"]["anchor_box"]

                # comm logging
                early_bytes_total = sum(int(rec.get("tx_bytes", 0)) for rec in cav_map.values())
                comm_writer.writerow([frame_id, num_agents, early_bytes_total])

                # inference
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(
                        batch_data, model, opencood_dataset
                    )
                    n_pred = 0 if pred_box_tensor is None else int(pred_box_tensor.shape[0])
                    print(f"[leader] fused frame={frame_id} agents={num_agents} preds={n_pred}")

            elif opt.fusion_method == "intermediate":
                # Placeholder until you define the payload and mapping to model inputs
                bytes_total = sum(int(rec.get("tx_bytes", 0)) for rec in cav_map.values())
                shapes = [rec.get("meta_shape", None) for rec in cav_map.values()]
                dtypes = [rec.get("meta_dtype", None) for rec in cav_map.values()]
                comm_writer.writerow([frame_id, num_agents, bytes_total, shapes, dtypes])
                raise NotImplementedError("Intermediate leader assembly not implemented yet (model-specific).")

            else:  # late
                boxes_list = []
                scores_list = []
                late_bytes_total = 0
                num_boxes_total = 0

                for rec in cav_map.values():
                    if "boxes" not in rec or "scores" not in rec:
                        continue
                    b = rec["boxes"]
                    s = rec["scores"]

                    # sanity
                    if b.ndim != 2 or b.shape[1] != 7:
                        raise RuntimeError(f"[leader late] expected boxes (N,7), got {b.shape}")
                    if s.ndim != 1 or s.shape[0] != b.shape[0]:
                        raise RuntimeError(f"[leader late] scores {s.shape} vs boxes {b.shape}")

                    boxes_list.append(b.astype(np.float32, copy=False))
                    scores_list.append(s.astype(np.float32, copy=False))

                    late_bytes_total += int(rec.get("tx_bytes", b.nbytes + s.nbytes))
                    num_boxes_total += int(b.shape[0])

                comm_writer.writerow([frame_id, num_agents, late_bytes_total, num_boxes_total])

                # --- REAL late fusion: concat + NMS (boxes are assumed already in ego frame) ---
                pred_box_tensor, pred_score = late_fuse_nms(
                    boxes_list, scores_list,
                    nms_thresh=0.2,     # tune as needed
                    topk=None
                )

                # --- GT for eval (optional) ---
                gt_box_tensor = None
                if opt.do_eval:
                    gt_box_tensor = get_gt_for_frame(opencood_dataset, frame_id)

                n_pred = 0 if pred_box_tensor is None else int(pred_box_tensor.shape[0])
                print(f"[leader] late fused frame={frame_id} agents={num_agents} in_boxes={num_boxes_total} out_boxes={n_pred}")

            # Eval (only if gt is available and enabled)
            if opt.do_eval and (gt_box_tensor is not None) and (pred_box_tensor is not None) and (pred_score is not None):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            fused_count += 1

    # Finalize
    comm_log_f.close()
    print(f"Saved communication log to: {opt.comm_file}")

    if opt.do_eval:
        eval_utils.eval_final_results(result_stat, opt.model_dir, opt.global_sort_detections)


if __name__ == "__main__":
    main()