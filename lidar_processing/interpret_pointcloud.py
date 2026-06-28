import math
import open3d as o3d
from open3d.t.geometry import TriangleMesh
import numpy as np
import random


pcd_path = r"path_to_pcd"         # V2X-R dataset


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)

    # Flipping along the y-axis to match the PNG -- not needed anymore since we're flipping along the z axis
    points = np.asarray(pcd.points)
    points[:, 1] = -points[:, 1]
    pcd.points = o3d.utility.Vector3dVector(points)

    # Flipping along the z-axis to match the PNG
    # points = np.asarray(pcd.points)
    # points[:, 2] = -points[:, 2]
    # pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

# TODO: Combine multiple point cloud layers

def segment_ground(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=200):
    if len(pcd.points) <= 4:
        print(f"Not enough points to run plane segmentation for file {pcd}. Only {pcd.points} points are found.")
        exit
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.2,
            ransac_n=3,
            num_iterations=200
        )    
        ground = pcd.select_by_index(inliers)
        objects = pcd.select_by_index(inliers, invert=True)
        return ground, objects
    except RuntimeError as e:
    # Check if it’s the specific RANSAC error
        if "There must be at least 'ransac_n' points" in str(e):
            print("Skipping: not enough points in this point cloud")
            plane_model, inliers = None, []
        else:
            # Re-raise if it’s a different runtime error
            raise




# def cluster_obstacles(objects, eps=0.5, min_points=10, ego_distance_threshold=2.0, ego_size_thresh=(5.0, 2.0)):
#     labels = np.array(objects.cluster_dbscan(eps=eps, min_points=10))
#     clusters = [objects.select_by_index(np.where(labels == i)[0]) for i in np.unique(labels) if i != -1]
#     return clusters


def cluster_obstacles(objects, eps=0.5, min_points=10, ego_distance_threshold=2.1, ego_size_threshold=(5.0, 2.0)):
    labels = np.array(objects.cluster_dbscan(eps=eps, min_points=10))
    clusters = []

    for i in np.unique(labels):
        if i == -1:
            continue # ignore noise

        cluster = objects.select_by_index(np.where(labels == i)[0])
        points = np.asarray(cluster.points)    
        centroid = points.mean(axis=0)
        size = points.max(axis=0) - points.min(axis=0)

        # Filtering out ego vehicle clusters
        if np.linalg.norm(centroid[:2]) > ego_distance_threshold:
            clusters.append(cluster)
        
    return clusters

def describe_obstacles(clusters):
    obstacle_info = []
    for i, cluster in enumerate(clusters):
        points = np.asarray(cluster.points)
        centroid = points.mean(axis=0)
        size = points.max(axis=0) - points.min(axis=0)
        obstacle_info.append({
            "id": i,
            "centroid_relative_to_robot": centroid.tolist(),
            "bounding_box": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
                "size": size.tolist()
            },
            "num_points": len(points),
            "density": len(points) / np.prod(size) if np.prod(size) != 0 else -1
            # "type_guess": "unknown"  # could be filled if we add ML-based classification
        })
    return obstacle_info

def obstacles_to_text(obstacle_info):
    text_descriptions = []
    for obs in obstacle_info:
        c = obs["centroid_relative_to_robot"]
        s = obs["bounding_box"]["size"]
        d = obs["density"]

        distance = np.linalg.norm(c)        # distance from ego vehicle
        distance = math.trunc(distance)

        text_descriptions.append(
            f"Obstacle {obs['id']}: at x={c[0]:.2f}m, y={c[1]:.2f}m, z={c[2]:.2f}m; "
            f"Distance from ego vehicle: {distance} m; "
            f"Approximate size x={s[0]:.2f}m, y={s[1]:.2f}m, z={s[2]:.2f}m; "
            f"Density={d:.2f}"
        )
    return "\n".join(text_descriptions)


# This function is used to just provide the distances of obstacles for the LLM prompt. Currently used for Gemma models.
def obstacle_distances_to_text(obstacle_info):
    text_descriptions = []
    for obs in obstacle_info:
        c = obs["centroid_relative_to_robot"]
        distance = np.linalg.norm(c)        # distance from ego vehicle
        distance = math.trunc(distance)      # truncate decimal part
        text_descriptions.append(
            f"Obstacle {obs['id']}: Distance = {distance} m; "
        )
    return "\n".join(text_descriptions)

def visualize_pcd(pcd):
    ground, objects = segment_ground(pcd)
    ground = ground.paint_uniform_color([0, 0, 1.0])
    objects = objects.paint_uniform_color([0, 1.0, 0])
    o3d.visualization.draw_geometries([ground, objects],
                                zoom=0.8,
                                front=[-0.4999, -0.1659, -0.8499],
                                lookat=[2.1813, 2.0619, 2.0999],
                                up=[0.1204, -0.9852, 0.1215])



def color_and_save_pcd(pcd, output_path):           # colors obstacles differently from the ground
    # Segment ground and objects
    ground, objects = segment_ground(pcd)           # PC objects
    ground.paint_uniform_color([0, 0, 1.0])  # blue ground
    objects.paint_uniform_color([0, 1.0, 0]) # green obstacles

    combined = ground + objects
    o3d.io.write_point_cloud(output_path, combined)
    print("Saved colored point cloud as PCD!")  

def visualize_pcd_with_colored_obstacles(pcd):
    # Segment ground and objects
    ground, objects = segment_ground(pcd)           # PC objects
    ground.paint_uniform_color([0, 0, 1.0])  # blue ground
    objects.paint_uniform_color([0, 1.0, 0]) # green obstacles

    # Cluster objects
    clusters = cluster_obstacles(objects)

    # Assign a unique color to each cluster based on its ID
    obstacle_meshes = []
    centroid_spheres = []
    for i, cluster in enumerate(clusters):
        np.random.seed(i)
        color = np.random.rand(3)

        # Compute centroid
        points = np.asarray(cluster.points)
        centroid = points.mean(axis=0)

        obstacle_mesh: TriangleMesh = o3d.t.geometry.TriangleMesh.create_text(f"{i}", depth=1).to_legacy()
        obstacle_mesh.paint_uniform_color((1, 0, 0))

        scale = 0.1

        location = (centroid[0], centroid[1], centroid[2])

        obstacle_mesh.transform([[scale, 0, 0, location[0]], [0, scale, 0, location[1]], [0, 0, scale, location[2]],
                            [0, 0, 0, 1]])
        
        obstacle_meshes.append(obstacle_mesh)

        # print(f"Obstacle {i} centroid at: {centroid}")  # console label

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    # Visualize everything
    o3d.visualization.draw_geometries([origin, ground, objects, *obstacle_meshes, *centroid_spheres],
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215],
                                    mesh_show_back_face=True)
    


def visualize_pcd_by_density(pcd, k_neighbors=50, save_path=None, width=3840, height=2160):
    points = np.asarray(pcd.points)
    n = len(points)

    # Estimate local density via mean distance to k nearest neighbors
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    mean_dists = np.zeros(n)
    for i in range(n):
        [_, _, dist_sq] = kdtree.search_knn_vector_3d(points[i], k_neighbors + 1)
        mean_dists[i] = np.mean(np.sqrt(dist_sq[1:]))  # exclude self (dist=0)
    density = 1.0 / (mean_dists + 1e-6)

    dist_from_origin = np.linalg.norm(points, axis=1)

    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    # Exponential proximity: drops off steeply from center → intense red at origin
    proximity = np.exp(-2.0 * normalize(dist_from_origin))

    # Score: high = dense + close → red; low = sparse + far → green
    score = 0.3 * normalize(density) + 0.7 * proximity

    # Vectorized color gradient: green → yellow → orange → red
    colors = np.zeros((n, 3))

    mask = score <= 1/3
    t = score[mask] / (1/3)
    colors[mask, 0] = t     # R: 0→1
    colors[mask, 1] = 1.0   # G: 1

    mask = (score > 1/3) & (score <= 2/3)
    t = (score[mask] - 1/3) / (1/3)
    colors[mask, 0] = 1.0
    colors[mask, 1] = 1.0 - 0.5 * t  # G: 1→0.5

    mask = score > 2/3
    t = (score[mask] - 2/3) / (1/3)
    colors[mask, 0] = 1.0
    colors[mask, 1] = 0.5 * (1.0 - t)  # G: 0.5→0

    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = pcd.points
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    vis.add_geometry(pcd_colored)
    vis.add_geometry(origin)

    vc = vis.get_view_control()
    vc.set_zoom(0.8)
    vc.set_front([-0.4999, -0.1659, -0.8499])
    vc.set_lookat([2.1813, 2.0619, 2.0999])
    vc.set_up([0.1204, -0.9852, 0.1215])

    if save_path:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path, do_render=True)
        print(f"Saved to {save_path}")

    vis.run()
    vis.destroy_window()


def get_obstacles_within_d_meters(obstacles_description, distance):
    obstacles_within_d_meters = []
    for obs in obstacles_description:
        c = obs["centroid_relative_to_robot"]
        s = obs["bounding_box"]["size"]
        d = obs["density"]

        distance_to_ego = np.linalg.norm(c)        # distance from ego vehicle
    
        if distance_to_ego <= distance: 
            obstacles_within_d_meters.append(obs["id"])
    return obstacles_within_d_meters




# pcd = load_point_cloud(pcd_path)
# ground, objects = segment_ground(pcd)
# clusters = cluster_obstacles(objects)
# obstacles_description = describe_obstacles(clusters)