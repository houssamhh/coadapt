import argparse
import csv
import os
import socket
import time
import math
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.data_utils.datasets as datasets_mod
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils, box_utils
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.utils.transformation_utils import x1_to_x2

from opencood.tools.distributed.common import make_server_socket, np_nbytes, recv_message, send_message


def parse_args():
    parser = argparse.ArgumentParser("OpenCOOD distributed leader")
    parser.add_argument("--model_dir", type=str, required=True, help="Path containing config.yaml and checkpoints")
    parser.add_argument("--fusion_method", type=str, required=True, choices=["early", "intermediate", "late"])
    parser.add_argument("--bind_host", type=str, default="0.0.0.0")
    parser.add_argument("--bind_port", type=int, default=28080)
    parser.add_argument("--num_agents", type=int, default=-1,
                        help="Number of agent processes. -1 (default) means all agents (dataset.max_cav).")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Number of frames to evaluate. -1 means all frames in the dataset.")
    parser.add_argument("--frame_timeout_s", type=float, default=10.0)
    parser.add_argument("--comm_file", type=str, required=True, help="CSV file for communication logs")
    parser.add_argument("--global_sort_detections", action="store_true")
    return parser.parse_args()


def _payload_bytes(fusion_method, payload):
    if fusion_method == "early":
        return np_nbytes(payload["lidar_np"])
    if fusion_method == "intermediate":
        total = 0
        for _, value in payload["processed_features"].items():
            if isinstance(value, list):
                for x in value:
                    total += np_nbytes(x)
            else:
                total += np_nbytes(value)
        return total
    processed = payload["processed"]
    total = 0
    if "processed_lidar" in processed:
        for _, value in processed["processed_lidar"].items():
            if isinstance(value, list):
                for x in value:
                    total += np_nbytes(x)
            else:
                total += np_nbytes(value)
    return total


def _collect_connections(server_sock, num_agents):
    by_slot = {}
    while len(by_slot) < num_agents:
        conn, addr = server_sock.accept()
        try:
            hello = recv_message(conn)
            if hello.get("msg_type") != "hello":
                conn.close()
                continue
            slot = int(hello["agent_slot"])
            by_slot[slot] = conn
            print(f"agent connected slot={slot} from {addr[0]}:{addr[1]}")
        except Exception:
            conn.close()
            raise
    return by_slot


def _split_payloads(messages):
    payloads = []
    for msg in messages:
        if msg.get("status") != "ok":
            continue
        payload = msg.get("payload")
        if payload:
            payloads.append(payload)
    return payloads


def _find_ego_payload(payloads):
    for p in payloads:
        if bool(p.get("ego", False)):
            return p
    return payloads[0] if payloads else None


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _build_batch_early(dataset, payloads):
    ego_payload = _find_ego_payload(payloads)
    if ego_payload is None:
        return None
    ego_pose = ego_payload["params"]["lidar_pose"]

    projected_lidar_stack = []
    object_stack = []
    object_id_stack = []
    selected_count = 0
    early_tx_non_ego = 0

    for p in payloads:
        lidar_pose = p["params"]["lidar_pose"]
        distance = math.sqrt((lidar_pose[0] - ego_pose[0]) ** 2 + (lidar_pose[1] - ego_pose[1]) ** 2)
        if distance > datasets_mod.COM_RANGE:
            continue
        selected_count += 1
        selected_cav_base = {"params": p["params"], "lidar_np": p["lidar_np"]}
        selected_cav_processed = dataset.get_item_single_car(selected_cav_base, ego_pose)
        projected_lidar_stack.append(selected_cav_processed["projected_lidar"])
        object_stack.append(selected_cav_processed["object_bbx_center"])
        object_id_stack += selected_cav_processed["object_ids"]
        if not bool(p.get("ego", False)):
            early_tx_non_ego += np_nbytes(p["lidar_np"])

    if not projected_lidar_stack:
        return None

    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)] if object_id_stack else []
    if object_stack:
        object_all = np.vstack(object_stack)
        object_all = object_all[unique_indices] if unique_indices else np.zeros((0, 7))
    else:
        object_all = np.zeros((0, 7))

    max_num = dataset.params["postprocess"]["max_num"]
    object_bbx_center = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    count = min(object_all.shape[0], max_num)
    if count > 0:
        object_bbx_center[:count, :] = object_all[:count, :]
        mask[:count] = 1

    projected = np.vstack(projected_lidar_stack)
    projected, object_bbx_center, mask = dataset.augment(projected, object_bbx_center, mask)
    projected = mask_points_by_range(projected, dataset.params["preprocess"]["cav_lidar_range"])

    object_valid = object_bbx_center[mask == 1]
    object_valid, range_mask = box_utils.mask_boxes_outside_range_numpy(
        object_valid,
        dataset.params["preprocess"]["cav_lidar_range"],
        dataset.params["postprocess"]["order"],
        return_mask=True,
    )
    mask[:] = 0
    object_bbx_center[:] = 0
    valid_count = min(object_valid.shape[0], max_num)
    if valid_count > 0:
        object_bbx_center[:valid_count] = object_valid[:valid_count]
        mask[:valid_count] = 1
    if unique_indices:
        unique_indices = list(np.array(unique_indices)[range_mask])

    lidar_dict = dataset.pre_processor.preprocess(projected)
    anchor_box = dataset.post_processor.generate_anchor_box()
    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask
    )

    ego_dict = {
        "object_bbx_center": object_bbx_center,
        "object_bbx_mask": mask,
        "object_ids": [object_id_stack[i] for i in unique_indices] if unique_indices else [],
        "anchor_box": anchor_box,
        "processed_lidar": lidar_dict,
        "label_dict": label_dict,
        "comm_meta": {
            "num_agents_selected": int(selected_count),
            "early_tx_bytes_total": int(early_tx_non_ego),
        },
    }
    return dataset.collate_batch_test([{"ego": ego_dict}])


def _build_batch_intermediate(dataset, payloads):
    ego_payload = _find_ego_payload(payloads)
    if ego_payload is None:
        return None

    ego_pose = ego_payload["params"]["lidar_pose"]
    processed_features = []
    object_stack = []
    object_id_stack = []
    velocity = []
    time_delay = []
    infra = []
    spatial_correction_matrix = []
    base_data_dict = OrderedDict()

    ordered = sorted(payloads, key=lambda x: (not bool(x.get("ego", False)), str(x.get("cav_id"))))
    for p in ordered:
        lidar_pose = p["params"]["lidar_pose"]
        distance = math.sqrt((lidar_pose[0] - ego_pose[0]) ** 2 + (lidar_pose[1] - ego_pose[1]) ** 2)
        if distance > datasets_mod.COM_RANGE:
            continue

        cav_id = p["cav_id"]
        base_data_dict[cav_id] = {"params": p["params"]}
        processed_features.append(p["processed_features"])
        object_stack.append(p["object_bbx_center"])
        object_id_stack += p["object_ids"]
        velocity.append(p["velocity"])
        time_delay.append(float(p.get("time_delay", 0.0)))
        cav_int = _safe_int(cav_id, default=1)
        infra.append(1 if cav_int < 0 else 0)
        spatial_correction_matrix.append(p["params"]["spatial_correction_matrix"])

    if not processed_features:
        return None

    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)] if object_id_stack else []
    if object_stack:
        object_all = np.vstack(object_stack)
        object_all = object_all[unique_indices] if unique_indices else np.zeros((0, 7))
    else:
        object_all = np.zeros((0, 7))

    max_num = dataset.params["postprocess"]["max_num"]
    object_bbx_center = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    count = min(object_all.shape[0], max_num)
    if count > 0:
        object_bbx_center[:count] = object_all[:count]
        mask[:count] = 1

    cav_num = len(processed_features)
    merged_feature_dict = dataset.merge_features_to_dict(processed_features)
    anchor_box = dataset.post_processor.generate_anchor_box()
    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center, anchors=anchor_box, mask=mask
    )

    velocity = velocity + (dataset.max_cav - len(velocity)) * [0.0]
    time_delay = time_delay + (dataset.max_cav - len(time_delay)) * [0.0]
    infra = infra + (dataset.max_cav - len(infra)) * [0.0]
    spatial_correction_matrix = np.stack(spatial_correction_matrix)
    if dataset.max_cav > len(spatial_correction_matrix):
        padding_eye = np.tile(np.eye(4)[None], (dataset.max_cav - len(spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)
    pairwise_t_matrix = dataset.get_pairwise_transformation(base_data_dict, dataset.max_cav)

    ego_dict = {
        "object_bbx_center": object_bbx_center,
        "object_bbx_mask": mask,
        "object_ids": [object_id_stack[i] for i in unique_indices] if unique_indices else [],
        "anchor_box": anchor_box,
        "processed_lidar": merged_feature_dict,
        "label_dict": label_dict,
        "cav_num": cav_num,
        "velocity": velocity,
        "time_delay": time_delay,
        "infra": infra,
        "spatial_correction_matrix": spatial_correction_matrix,
        "pairwise_t_matrix": pairwise_t_matrix,
        "comm_meta": {"num_agents_selected": int(cav_num)},
    }
    return dataset.collate_batch_test([{"ego": ego_dict}])


def _build_batch_late(dataset, payloads):
    ego_payload = _find_ego_payload(payloads)
    if ego_payload is None:
        return None
    ego_cav_id = ego_payload["cav_id"]
    ego_pose = ego_payload["params"]["lidar_pose"]

    frame_dict = OrderedDict()
    ordered = sorted(payloads, key=lambda x: (not bool(x.get("ego", False)), str(x.get("cav_id"))))
    for p in ordered:
        lidar_pose = p["params"]["lidar_pose"]
        distance = math.sqrt((lidar_pose[0] - ego_pose[0]) ** 2 + (lidar_pose[1] - ego_pose[1]) ** 2)
        if distance > datasets_mod.COM_RANGE:
            continue
        cav_content = dict(p["processed"])
        cav_content["transformation_matrix"] = x1_to_x2(lidar_pose, ego_pose)
        update_id = "ego" if p["cav_id"] == ego_cav_id else p["cav_id"]
        frame_dict[update_id] = cav_content

    if not frame_dict:
        return None
    return dataset.collate_batch_test([frame_dict])


def _recv_for_frame(conn: socket.socket, frame_id: int, frame_timeout_s: float):
    conn.settimeout(frame_timeout_s)
    msg = recv_message(conn)
    if msg.get("msg_type") != "frame_data":
        raise RuntimeError(f"unexpected message type: {msg.get('msg_type')}")
    if int(msg.get("frame_id", -1)) != frame_id:
        raise RuntimeError(f"frame mismatch: expected {frame_id}, got {msg.get('frame_id')}")
    return msg


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.comm_file) or ".", exist_ok=True)
    opt = SimpleNamespace(model_dir=args.model_dir)
    hypes = yaml_utils.load_yaml(None, opt)

    dataset = build_dataset(hypes, visualize=False, train=False)
    max_frames = len(dataset) if args.max_frames < 0 else min(args.max_frames, len(dataset))
    num_agents = dataset.max_cav if args.num_agents < 0 else args.num_agents
    print(f"frames to evaluate: {max_frames} (dataset size: {len(dataset)})")
    print(f"agents expected:    {num_agents} (dataset.max_cav: {dataset.max_cav})")
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    evaluated = 0

    server_sock = make_server_socket(args.bind_host, args.bind_port)
    print(f"leader listening on {args.bind_host}:{args.bind_port}")
    connections = _collect_connections(server_sock, num_agents)

    eval_save_dir = os.path.dirname(os.path.abspath(args.comm_file))

    with open(args.comm_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Columns:
        #   input_bytes      – bytes sent from non-ego agents to leader
        #                      early:        raw point cloud numpy bytes
        #                      intermediate: preprocessed feature numpy bytes
        #                      late:         preprocessed lidar numpy bytes
        #   output_bytes     – (late only) raw model-output tensor bytes per non-ego CAV
        #   total_latency_ms – leader tick_sent → leader recv (full round-trip per agent)
        #   processing_ms    – leader tick_sent → agent send  (leader→agent delay + agent processing)
        #   network_rx_ms    – agent send → leader recv        (agent→leader network delay)
        #   NOTE: processing_ms + network_rx_ms ≈ total_latency_ms (exact in Mininet; approx on real nets)
        writer.writerow(["frame_id", "agent_slot", "status", "payload_type",
                         "input_bytes", "output_bytes",
                         "total_latency_ms", "processing_ms", "network_rx_ms"])

        for frame_id in range(max_frames):
            # Send ticks to all agents, recording the timestamp per slot
            tick_sent_ts = {}
            for slot in sorted(connections.keys()):
                tick_sent_ts[slot] = time.time()
                send_message(
                    connections[slot],
                    {"msg_type": "frame_tick", "frame_id": frame_id, "fusion_method": args.fusion_method},
                )

            finished = 0
            frame_msgs = []
            frame_input_bytes = 0
            for slot in sorted(connections.keys()):
                msg = _recv_for_frame(connections[slot], frame_id, args.frame_timeout_s)
                recv_ts = time.time()
                frame_msgs.append(msg)
                status = msg.get("status", "unknown")
                payload = msg.get("payload")
                payload_type = payload.get("payload_type") if payload else ""
                # only count non-ego agents toward comm cost
                is_ego = bool(payload.get("ego", False)) if payload else True
                input_bytes = (_payload_bytes(args.fusion_method, payload) if (payload and not is_ego) else 0)
                frame_input_bytes += input_bytes
                total_latency_ms = int((recv_ts - tick_sent_ts[slot]) * 1000)
                agent_send_ts = msg.get("send_ts")
                if agent_send_ts is not None:
                    network_rx_ms  = int((recv_ts - agent_send_ts) * 1000)
                    processing_ms  = int((agent_send_ts - tick_sent_ts[slot]) * 1000)
                else:
                    network_rx_ms = processing_ms = -1  # agent didn't include send_ts
                writer.writerow([frame_id, slot, status, payload_type,
                                 input_bytes, 0,
                                 total_latency_ms, processing_ms, network_rx_ms])
                if status == "finished":
                    finished += 1

            f.flush()
            if finished == num_agents:
                print(f"all agents finished at frame {frame_id}")
                break

            payloads = _split_payloads(frame_msgs)
            if not payloads:
                continue

            if args.fusion_method == "early":
                batch_data = _build_batch_early(dataset, payloads)
            elif args.fusion_method == "intermediate":
                batch_data = _build_batch_intermediate(dataset, payloads)
            else:
                batch_data = _build_batch_late(dataset, payloads)
            if batch_data is None:
                continue

            frame_output_bytes = 0
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                if args.fusion_method == "early":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(
                        batch_data, model, dataset
                    )
                elif args.fusion_method == "intermediate":
                    pred_box_tensor, pred_score, gt_box_tensor = (
                        inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
                    )
                else:
                    pred_box_tensor, pred_score, gt_box_tensor, output_dict = inference_utils.inference_late_fusion(
                        batch_data, model, dataset, return_output_dict=True
                    )
                    # output_bytes: raw model-output tensor bytes for each non-ego CAV
                    # (proxy for what a real late-fusion agent would transmit after local detection)
                    for cav_id, cav_out in output_dict.items():
                        if cav_id == "ego":
                            continue
                        for v in cav_out.values():
                            if isinstance(v, torch.Tensor):
                                frame_output_bytes += v.numel() * v.element_size()

                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)
                evaluated += 1

            # per-frame summary row (agent_slot=-1); latency = total wall time for the frame
            frame_wall_ms = int((time.time() - min(tick_sent_ts.values())) * 1000)
            writer.writerow([frame_id, -1, "summary", args.fusion_method,
                             frame_input_bytes, frame_output_bytes,
                             frame_wall_ms, -1, -1])
            f.flush()

    for slot, conn in connections.items():
        try:
            send_message(conn, {"msg_type": "shutdown"})
        except Exception:
            pass
        conn.close()

    server_sock.close()
    if evaluated > 0:
        eval_utils.eval_final_results(result_stat, eval_save_dir, args.global_sort_detections)
    print(f"communication log saved to {args.comm_file}")
    print(f"eval results saved to {eval_save_dir}")


if __name__ == "__main__":
    main()
