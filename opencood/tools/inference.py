# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import csv
import argparse
import os
import time
from tqdm import tqdm

import torch
# import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--comm_file', type=str, required=True,
                        help='Path to communication metrics')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    
    # ------------------ COMM LOGGING (INTERMEDIATE via hook) ------------------
    fusion_hook_handle = None
    comm_cache = {}

    comm_log_f = None
    comm_writer = None
    comm_log_path = None

    def tensor_nbytes(t: torch.Tensor) -> int:
        return int(t.element_size() * t.nelement()) if t is not None else 0

    if opt.fusion_method == "intermediate":
        # Hook cache
        comm_cache = {"_current_record_len": None, "k": None, "bytes_non_ego": None, "shape": None, "dtype": None}

        def fusion_net_hook(module, inputs, output):
            if len(inputs) < 1:
                return
            regroup_feature = inputs[0]
            if not torch.is_tensor(regroup_feature) or regroup_feature.dim() != 5:
                comm_cache["k"] = None
                comm_cache["bytes_non_ego"] = None
                comm_cache["shape"] = tuple(regroup_feature.shape) if torch.is_tensor(regroup_feature) else None
                comm_cache["dtype"] = str(regroup_feature.dtype) if torch.is_tensor(regroup_feature) else None
                return

            k = comm_cache.get("_current_record_len", None)
            comm_cache["k"] = k
            comm_cache["shape"] = tuple(regroup_feature.shape)
            comm_cache["dtype"] = str(regroup_feature.dtype)

            if k is None or k <= 1:
                comm_cache["bytes_non_ego"] = 0
                return

            non_ego = regroup_feature[0, 1:k]  # exclude ego index 0
            comm_cache["bytes_non_ego"] = tensor_nbytes(non_ego)

        fusion_hook_handle = model.fusion_net.register_forward_hook(fusion_net_hook)

        comm_log_path = os.path.join(opt.comm_file)
        comm_log_f = open(comm_log_path, mode="w", newline="")
        comm_writer = csv.writer(comm_log_f)
        comm_writer.writerow(["frame_id", "k_record_len", "bytes_non_ego", "regroup_shape", "dtype"])

    elif opt.fusion_method == "early":
        comm_log_path = os.path.join(opt.comm_file)
        comm_log_f = open(comm_log_path, mode="w", newline="")
        comm_writer = csv.writer(comm_log_f)
        comm_writer.writerow(["frame_id", "k_selected", "early_tx_bytes_total"])

    else:
        # late: skip for now or add later
        comm_log_path = os.path.join(opt.comm_file)
        comm_log_f = open(comm_log_path, mode="w", newline="")
        comm_writer = csv.writer(comm_log_f)
        comm_writer.writerow(["frame_id", "k_record_len", "num_boxes_non_ego", "late_tx_bytes_non_ego"])
# -------------------------------------------------------------------------

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    # Added: ---------------- COMM LOGGING (EARLY FUSION) ----------------
    # comm_log_path = os.path.join(opt.model_dir, f"comm_{opt.fusion_method}.csv")
    # comm_log_f = open(comm_log_path, mode="w", newline="")
    # comm_writer = csv.writer(comm_log_f)

    # # Header
    # comm_writer.writerow([
    #     "frame_id",
    #     "fusion_method",
    #     "num_agents",
    #     "early_tx_bytes_total"
    # ])
    # -------------------------------------------------------------

    # if opt.show_sequence:
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()

    #     vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    #     vis.get_render_option().point_size = 1.0
    #     vis.get_render_option().show_coordinate_frame = True

    #     # used to visualize lidar points
    #     vis_pcd = o3d.geometry.PointCloud()
    #     # used to visualize object bounding box, maximum 50
    #     vis_aabbs_gt = []
    #     vis_aabbs_pred = []
    #     for _ in range(50):
    #         vis_aabbs_gt.append(o3d.geometry.LineSet())
    #         vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        
        #Added for intermediate --------
        if opt.fusion_method == "intermediate":
    # batch_size=1 assumption
            try:
                k = int(batch_data["ego"]["record_len"][0].item())
            except Exception:
                k = None
            comm_cache["_current_record_len"] = k
            comm_cache["k"] = None
            comm_cache["bytes_non_ego"] = None
            comm_cache["shape"] = None
            comm_cache["dtype"] = None

        elif opt.fusion_method == "early":
            comm_meta = batch_data.get("ego", {}).get("comm_meta", {})
            k = comm_meta.get("num_agents_selected", None)  # YOU ADDED THIS IN DATASET
            early_bytes = comm_meta.get("early_tx_bytes_total", None)
            comm_writer.writerow([i, k, early_bytes])
        # ----------------------------------------------------------

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, late_output_dict = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset,
                                                          return_output_dict=True)
                if comm_writer is not None:
                    k = len(batch_data)
                    num_boxes_non_ego = 0
                    late_tx_bytes_non_ego = 0

                    for cav_id in batch_data.keys():
                        if cav_id == 'ego':
                            continue

                        cav_data = {cav_id: batch_data[cav_id]}
                        cav_output = {cav_id: late_output_dict[cav_id]}
                        cav_pred_box_tensor, cav_pred_score, _ = \
                            opencood_dataset.post_process(cav_data, cav_output)

                        if cav_pred_box_tensor is None or cav_pred_score is None:
                            continue

                        num_boxes_non_ego += int(cav_pred_box_tensor.shape[0])
                        late_tx_bytes_non_ego += tensor_nbytes(cav_pred_box_tensor)
                        late_tx_bytes_non_ego += tensor_nbytes(cav_pred_score)

                    comm_writer.writerow([i, k, num_boxes_non_ego, late_tx_bytes_non_ego])
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
                    # Added: ----
                if opt.fusion_method == "intermediate" and comm_writer is not None:
                    comm_writer.writerow([
                        i,
                        comm_cache["k"],
                        comm_cache["bytes_non_ego"],
                        comm_cache["shape"],
                        comm_cache["dtype"],
                    ])
    # ------
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

    # Added: close file
    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    
    if comm_log_f is not None:
        comm_log_f.close()
        print(f"Saved communication log to: {comm_log_path}")

    if fusion_hook_handle is not None:
        fusion_hook_handle.remove()
    #Added: save communication cost:
    
    # if opt.show_sequence:
    #     vis.destroy_window()

if __name__ == '__main__':
    main()
