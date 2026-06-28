# -*- coding: utf-8 -*-
# Author: Houssam Hajj Hassan


import math
import torch

# import opencood.data_utils.datasets  # for COM_RANGE
COM_RANGE = 70 
from opencood.tools import train_utils
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2

import argparse

import zmq
import msgpack

import numpy as np
import time
from typing import Optional, Dict, Any

import math
# import opencood.data_utils.datasets  # for COM_RANGE

# import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--comm_file', type=str, required=True,
                        help='Path to communication metrics')
    parser.add_argument('--cav_id', type=str, required=True,
                        help='ID of CAV')
    parser.add_argument('--connect', type=str, required=True,
                        help='ZMQ endpoint of leader, e.g. tcp://127.0.0.1:5555')
    parser.add_argument('--scenario_id', type=str, default='default',
                        help='Scenario identifier')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Stop after sending N frames (-1 = all)')
    parser.add_argument('--sleep_s', type=float, default=0.0,
                        help='Optional sleep between frames')
    opt = parser.parse_args()
    return opt



def send_late_dets(sock: zmq.Socket,
                   scenario_id: str,
                   frame_id: int,
                   cav_id: str,
                   boxes: np.ndarray,
                   scores: np.ndarray):
    boxes = np.ascontiguousarray(boxes.astype(np.float32, copy=False))
    scores = np.ascontiguousarray(scores.astype(np.float32, copy=False))

    header: Dict[str, Any] = {
        "version": 1,
        "scenario_id": scenario_id,
        "frame_id": int(frame_id),
        "cav_id": str(cav_id),
        "payload_type": "late",
        "dtype": "f4",
        "boxes_shape": list(boxes.shape),     # [M,7]
        "scores_shape": list(scores.shape),   # [M]
        "nbytes": int(boxes.nbytes + scores.nbytes),
        "ts": time.time(),
        "boxes_in_ego": True,  # we will send ego-frame boxes below
    }

    sock.send_multipart([
        msgpack.dumps(header, use_bin_type=True),
        boxes.tobytes(),
        scores.tobytes(),
    ])
    
def send_projected_lidar(sock: zmq.Socket,
                         scenario_id: str,
                         frame_id: int,
                         cav_id: str,
                         projected_lidar: np.ndarray):
    # enforce float32 contiguous
    projected_lidar = np.ascontiguousarray(projected_lidar.astype(np.float32, copy=False))
    header: Dict[str, Any] = {
        "version": 1,
        "scenario_id": scenario_id,
        "frame_id": int(frame_id),
        "cav_id": str(cav_id),
        "payload_type": "early",
        "dtype": "f4",
        "shape": list(projected_lidar.shape),   # [N,4]
        "nbytes": int(projected_lidar.nbytes),
        "ts": time.time(),
    }
    sock.send_multipart([
        msgpack.dumps(header, use_bin_type=True),
        projected_lidar.tobytes(),
    ])
    
def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']

    hypes = yaml_utils.load_yaml(None, opt)
    
     # ZMQ PUSH
    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.connect(opt.connect)
    
    cav_id_target = str(opt.cav_id)

  

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    
    if opt.fusion_method == "early":
        sent = 0
        for i in range(len(opencood_dataset)):
            base_data = opencood_dataset.retrieve_base_data(i)

            ego_pose = None
            for cav_id, cav_content in base_data.items():
                if cav_content["ego"]:
                    ego_pose = cav_content["params"]["lidar_pose"]
            assert ego_pose is not None, f"No ego found for frame {i}"

            for cav_id, cav_content in base_data.items():
                if cav_id == opt.cav_id:
                    print("CAV ID: ", cav_id)
                    lidar = cav_content["lidar_np"]

                    print("Processing frames: ", i)
                    # this is exactly what agents should do
                    processed = opencood_dataset.get_item_single_car(
                        cav_content,
                        ego_pose
                    )

                    print("Projecting lidar")
                    projected_lidar = processed["projected_lidar"]

                    # ensure float32 contiguous (important for ZMQ serialization later)
                    projected_lidar = projected_lidar.astype("float32", copy=False)
                    # before sending
                    cav_pose = cav_content["params"]["lidar_pose"]
                    dist = math.sqrt((cav_pose[0] - ego_pose[0])**2 + (cav_pose[1] - ego_pose[1])**2)
                    if dist > COM_RANGE:
                        continue  # don't send this frame for this agent (native would skip)
                    send_projected_lidar(
                        push,
                        scenario_id=opt.scenario_id,
                        frame_id=i,
                        cav_id=cav_id_target,
                        projected_lidar=projected_lidar,
                    )
                    sent += 1
                    print(f"sent frame={i} cav={cav_id_target} points={projected_lidar.shape[0]} bytes={projected_lidar.nbytes}")
                    
    elif opt.fusion_method == "late":
        # ---- Create model on agent (late fusion requires local detection) ----
        print("Creating Model (agent late)")
        model = train_utils.create_model(hypes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("Loading Model from checkpoint (agent late)")
        _, model = train_utils.load_saved_model(opt.model_dir, model)
        model.eval()

        # pcd utils (same as native pipeline)
        from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, shuffle_points

        sent = 0
        for frame_id in range(len(opencood_dataset)):
            base_data = opencood_dataset.retrieve_base_data(frame_id)

            # ---- get ego pose (for COM_RANGE + projection) ----
            ego_pose = None
            for _, cav_content in base_data.items():
                if cav_content.get("ego", False):
                    ego_pose = cav_content["params"]["lidar_pose"]
                    break
            if ego_pose is None:
                raise RuntimeError(f"No ego found for frame {frame_id}")

            # ---- get this agent content ----
            cav_content = None
            cav_pose = None
            for cav_id, c in base_data.items():
                if str(cav_id) == cav_id_target:
                    cav_content = c
                    cav_pose = c["params"]["lidar_pose"]
                    break
            if cav_content is None:
                continue  # agent absent this frame

            # ---- COM_RANGE filter ----
            dx = float(cav_pose[0] - ego_pose[0])
            dy = float(cav_pose[1] - ego_pose[1])
            if math.sqrt(dx * dx + dy * dy) > COM_RANGE:
                continue

            # ---- preprocess lidar in *agent* frame (late fusion local detection) ----
            lidar_np = cav_content["lidar_np"].astype(np.float32, copy=False)
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_ego_points(lidar_np)
            lidar_np = mask_points_by_range(
                lidar_np, opencood_dataset.params["preprocess"]["cav_lidar_range"]
            )

            lidar_dict = opencood_dataset.pre_processor.preprocess(lidar_np)
            processed_lidar = opencood_dataset.pre_processor.collate_batch([lidar_dict])

            # anchor box (torch float32)
            anchor_box = opencood_dataset.post_processor.generate_anchor_box()
            anchor_box_torch = torch.as_tensor(anchor_box, dtype=torch.float32)

            # ---- build minimal single-agent batch in the expected OpenCOOD format ----
            single_batch = {
                "ego": {
                    "processed_lidar": processed_lidar,
                    "anchor_box": anchor_box_torch,
                    "transformation_matrix": torch.eye(4, dtype=torch.float32),
                }
            }

            # ---- forward + postprocess ----
            with torch.no_grad():
                single_batch = train_utils.to_device(single_batch, device)

                # IMPORTANT: pass the whole batch to the model
                # output = model(single_batch)
                output = model(single_batch["ego"])   # NOT model(single_batch)

                # normalize to {"ego": ...} for post_process
                wrapped_output = output if (isinstance(output, dict) and "ego" in output) else {"ego": output}

                pred_box_tensor, pred_score = opencood_dataset.post_processor.post_process(
                    single_batch, wrapped_output
                )

            # ---- to numpy + enforce shapes ----
            if pred_box_tensor is None or pred_score is None:
                boxes = np.zeros((0, 7), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
            else:
                    boxes = pred_box_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                    scores = pred_score.detach().cpu().numpy().astype(np.float32, copy=False)

                    # scores should be (N,)
                    if scores.ndim > 1:
                        scores = scores.reshape(-1)

                    # Case A: boxes already (N,7)
                    if boxes.ndim == 2 and boxes.shape[1] == 7:
                        pass

                    # Case B: boxes are corners (N,8,3) -> convert to (N,7)
                    elif boxes.ndim == 3 and boxes.shape[1] == 8 and boxes.shape[2] == 3:
                        # Convert corners to (x,y,z,dx,dy,dz,yaw)
                        # NOTE: function name/order can vary across OpenCOOD forks.
                        # These are the two most common:
                        if hasattr(box_utils, "corner_to_center"):
                            boxes = box_utils.corner_to_center(boxes, order="lwh")  # -> (N,7) in many versions
                        elif hasattr(box_utils, "corner_to_center_torch"):
                            boxes_t = torch.from_numpy(boxes)
                            boxes = box_utils.corner_to_center_torch(boxes_t, order="lwh").numpy()
                        else:
                            raise RuntimeError("OpenCOOD box_utils has no corner_to_center* function; need your version's helper name.")

                        boxes = boxes.astype(np.float32, copy=False)

                    else:
                        raise RuntimeError(f"Unexpected pred_box format: {boxes.shape}")

            # ---- project boxes agent -> ego (centers + yaw) ----
            if boxes.shape[0] > 0:
                T = x1_to_x2(cav_pose, ego_pose)  # agent -> ego
                T = torch.as_tensor(T, dtype=torch.float32)

                centers = torch.as_tensor(boxes[:, 0:3], dtype=torch.float32).reshape(-1, 3)
                centers_ego = box_utils.project_points_by_matrix_torch(centers, T).cpu().numpy()
                boxes[:, 0:3] = centers_ego

                # yaw correction (IMPORTANT for late fusion)
                # pose convention assumed: yaw at index 4, box yaw at index 6 (OpenCOOD default)
                boxes[:, 6] = boxes[:, 6] + float(ego_pose[4] - cav_pose[4])

            # ---- send to leader ----
            send_late_dets(
                push,
                scenario_id=opt.scenario_id,
                frame_id=frame_id,
                cav_id=cav_id_target,
                boxes=boxes,
                scores=scores,
            )

            sent += 1
            print(
                f"sent late frame={frame_id} cav={cav_id_target} "
                f"num_boxes={boxes.shape[0]} bytes={boxes.nbytes + scores.nbytes}"
            )
            

        if opt.sleep_s > 0:
            time.sleep(opt.sleep_s)
    # if opt.show_sequence:
    #     vis.destroy_window()

if __name__ == '__main__':
    main()
