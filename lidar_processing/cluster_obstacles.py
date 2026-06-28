import open3d as o3d
import os
import numpy as np

from interpret_pointcloud import visualize_pcd_with_colored_obstacles, load_point_cloud, visualize_pcd


pcd_path = r".\opv2v_data_dumping\sample_opv2v_data\341\000060.pcd"


def crop_forward_roi(pcd, x_min=0.0, x_max=50.0,
                    y_min=-10.0, y_max=10.0,
                    z_min=-2.0, z_max=2.0):
        """
        Keeping only the forward-facing vue
        """
        pts = np.asarray(pcd.points)
        mask = (
                (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
                (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
        )
        idx = np.where(mask)[0]
        return pcd.select_by_index(idx)


def remove_ground_ransac(pcd,
                        distance_threshold=0.2,
                        ransac_n=3,
                        num_iterations=200):
    """
    Segement the dominant plane and treat it as ground
    """

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    return non_ground, ground, plane_model


def cluster_dbscan(pcd, eps=0.6, min_points=20):
    """
    Clustering objects
    """
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    return labels


def extract_obstacle_clusters(pcd, labels, min_cluster_size=30, max_cluster_size=200000):
    """Returns a list of (cluster_id, cluster_pcd)"""
    clusters = []
    valid_ids = [i for i in np.unique(labels) if i != -1]
    for cid in valid_ids:
        idx = np.where(labels == cid)[0]
        n = len(idx)
        if n < min_cluster_size or n > max_cluster_size:
            continue
        clusters.append((int(cid), pcd.select_by_index(idx)))

    return clusters

def save_clusters(clusters, out_dir="obstacles"):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for cid, cpcd in clusters:
        path = os.path.join(out_dir, f"obstacle_{cid:03d}.pcd")
        o3d.io.write_point_cloud(path, cpcd)
        paths.append(path)
    return paths
            


def pipeline(input_path,
            downsample=False,
            voxel_size=0.15,
            x_min=0.0, x_max=50.0,
            y_min=-10.0, y_max=10.0,
            z_min=-2.0, z_max=2.0,
            do_ground_removal=True,
            ground_dist=0.2,
            eps=0.6, min_points=20,
            min_cluster_size=30):
    
    
    # 1) Load PCD
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud {input_path} is empty.")
    
    # 2) Optional downsampling for speeding up processing
    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    
    # 3) Forward ROI cropping
    pcd_roi = crop_forward_roi(pcd,
                            x_min=x_min, x_max=x_max,
                            y_min=y_min, y_max=y_max,
                            z_min=z_min, z_max=z_max)    
    
    
    # 4) Optional ground removal
    if do_ground_removal and not pcd_roi.is_empty():
        pcd_ng, pcd_ground, plane = remove_ground_ransac(pcd_roi,
                                                        distance_threshold=ground_dist)    
        
    else:
        pcd_ng, pcd_ground, plane = pcd_roi, o3d.geometry.PointCloud(), None
    
    # 5) Clustering obstacles
    if pcd_ng.is_empty():
        return{
            "roi": pcd_roi,
            "non_ground": pcd_ng,
            "ground": pcd_ground,
            "clusters": [],
            "saved_paths": []
        }
        
    labels = cluster_dbscan(pcd_ng, eps=eps, min_points=min_points)
    clusters = extract_obstacle_clusters(pcd_ng, labels, min_cluster_size=min_cluster_size)
    
    # 6) Saving each obstacle as its own PCD file
    saved_paths = save_clusters(clusters, out_dir="obstacles")
    
    return {
        "roi": pcd_roi,
        "non_ground": pcd_ng,
        "ground": pcd_ground,
        "clusters": clusters,
        "saved_paths": saved_paths,
        "plane_model": plane
    }
    
    
    
    
if __name__ == '__main__':
    obstacle_path = r"path_to_obstacle"
    # visualize_pcd_with_colored_obstacles(load_point_cloud(pcd_path))
    pcd = o3d.io.read_point_cloud(obstacle_path)
    o3d.visualization.draw_geometries([pcd])
    
    # npy_path = r"path_to_npy"
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.load(npy_path)[:, :3])
    # o3d.visualization.draw_geometries([pcd])

# if __name__ == "__main__":
#     result = pipeline(
#         input_path=pcd_path,
#         voxel_size=0.15,
#         x_min=0.0, x_max=60.0,
#         y_min=-12.0, y_max=12.0,
#         z_min=-2.5, z_max=2.5,
#         do_ground_removal=True,
#         ground_dist=0.2,
#         eps=0.7,
#         min_points=25,
#         min_cluster_size=40
#     )

    # print(f"ROI points: {len(result['roi'].points)}")
    # print(f"Non-ground points: {len(result['non_ground'].points)}")
    # print(f"Extracted obstacles: {len(result['clusters'])}")
    # print("Saved obstacle clouds to:", result["saved_paths"][:5], "..." if len(result["saved_paths"]) > 5 else "")