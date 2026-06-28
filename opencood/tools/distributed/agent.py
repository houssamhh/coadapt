import argparse
import time
from types import SimpleNamespace

from opencood.data_utils.datasets import build_dataset
import opencood.hypes_yaml.yaml_utils as yaml_utils

from opencood.tools.distributed.common import (
    connect_with_retry,
    recv_message,
    select_cav_for_slot,
    send_message,
)


def parse_args():
    parser = argparse.ArgumentParser("OpenCOOD distributed agent")
    parser.add_argument("--model_dir", type=str, required=True, help="Path containing config.yaml")
    parser.add_argument("--fusion_method", type=str, required=True, choices=["early", "intermediate", "late"])
    parser.add_argument("--agent_slot", type=int, required=True, help="0 = ego, 1..N = non-ego agents")
    parser.add_argument("--leader_host", type=str, default="leader")
    parser.add_argument("--leader_port", type=int, default=28080)
    parser.add_argument("--connect_timeout_s", type=float, default=120.0)
    parser.add_argument("--max_frames", type=int, default=-1)
    return parser.parse_args()


def _build_payload(dataset, fusion_method, base_data_dict, cav_id, cav_base):
    if fusion_method == "early":
        return {
            "payload_type": "early_raw_points",
            "cav_id": cav_id,
            "ego": bool(cav_base.get("ego", False)),
            "params": cav_base["params"],
            "lidar_np": cav_base["lidar_np"],
        }

    if fusion_method == "intermediate":
        ego_pose = None
        for _id, content in base_data_dict.items():
            if bool(content.get("ego", False)):
                ego_pose = content["params"]["lidar_pose"]
                break
        if ego_pose is None:
            raise RuntimeError("ego pose not found in frame")
        processed = dataset.get_item_single_car(cav_base, ego_pose)
        return {
            "payload_type": "intermediate_features",
            "cav_id": cav_id,
            "ego": bool(cav_base.get("ego", False)),
            "params": cav_base["params"],
            "time_delay": float(cav_base.get("time_delay", 0.0)),
            "processed_features": processed["processed_features"],
            "velocity": processed["velocity"],
            "object_bbx_center": processed["object_bbx_center"],
            "object_ids": processed["object_ids"],
        }

    processed = dataset.get_item_single_car(cav_base)
    return {
        "payload_type": "late_processed_lidar",
        "cav_id": cav_id,
        "ego": bool(cav_base.get("ego", False)),
        "params": cav_base["params"],
        "processed": processed,
    }


def main():
    args = parse_args()
    opt = SimpleNamespace(model_dir=args.model_dir)
    hypes = yaml_utils.load_yaml(None, opt)
    dataset = build_dataset(hypes, visualize=False, train=False)
    frame_count = len(dataset)

    sock = connect_with_retry(args.leader_host, args.leader_port, timeout_s=args.connect_timeout_s)
    send_message(
        sock,
        {
            "msg_type": "hello",
            "agent_slot": args.agent_slot,
            "fusion_method": args.fusion_method,
            "frame_count": frame_count,
            "ts": time.time(),
        },
    )

    max_frames = frame_count if args.max_frames < 0 else min(frame_count, args.max_frames)
    while True:
        msg = recv_message(sock)
        msg_type = msg.get("msg_type")
        if msg_type == "shutdown":
            break
        if msg_type != "frame_tick":
            continue

        frame_id = int(msg["frame_id"])
        if frame_id >= max_frames:
            send_message(
                sock,
                {
                    "msg_type": "frame_data",
                    "agent_slot": args.agent_slot,
                    "frame_id": frame_id,
                    "status": "finished",
                    "fusion_method": args.fusion_method,
                },
            )
            continue

        cur_ego_pose_flag = getattr(dataset, 'cur_ego_pose_flag', True)
        base_data_dict = dataset.retrieve_base_data(frame_id, cur_ego_pose_flag=cur_ego_pose_flag)
        cav_id, cav_base = select_cav_for_slot(base_data_dict, args.agent_slot)
        if cav_id is None:
            send_message(
                sock,
                {
                    "msg_type": "frame_data",
                    "agent_slot": args.agent_slot,
                    "frame_id": frame_id,
                    "status": "absent",
                    "fusion_method": args.fusion_method,
                },
            )
            continue

        payload = _build_payload(dataset, args.fusion_method, base_data_dict, cav_id, cav_base)
        send_message(
            sock,
            {
                "msg_type": "frame_data",
                "agent_slot": args.agent_slot,
                "frame_id": frame_id,
                "status": "ok",
                "fusion_method": args.fusion_method,
                "payload": payload,
                "send_ts": time.time(),  # timestamp just before sending; used by leader to measure network_rx delay
            },
        )

    sock.close()


if __name__ == "__main__":
    main()
