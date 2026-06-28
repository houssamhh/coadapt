# # -*- coding: utf-8 -*-
# # Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# # License: TDG-Attribution-NonCommercial-NoDistrib


# """
# Utility functions related to point cloud
# """

# import open3d as o3d
import numpy as np


def pcd_to_numpy_with_rgb(path):
    fields = None
    points_expected = None
    data_type = None
    header_lines = 0

    # Read header only
    with open(path, "r") as f:
        for line in f:
            header_lines += 1
            line = line.strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("POINTS"):
                points_expected = int(line.split()[1])
            elif line.startswith("DATA"):
                data_type = line.split()[1].lower()
                break

    if data_type != "ascii":
        raise ValueError(f"Unsupported DATA type: {data_type}")

    if fields is None:
        raise ValueError("Missing FIELDS in PCD header")

    col = {name: i for i, name in enumerate(fields)}
    for req in ("x", "y", "z", "rgb"):
        if req not in col:
            raise ValueError(f"Missing field '{req}' in PCD")

    # Load all data fast
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)

    if points_expected is not None and data.shape[0] != points_expected:
        raise ValueError(f"POINTS mismatch: header={points_expected}, loaded={data.shape[0]}")

    xyz = data[:, [col["x"], col["y"], col["z"]]]

    # Packed float rgb -> uint32 bits -> unpack bytes
    rgb_u32 = data[:, col["rgb"]].view(np.uint32)
    r = (rgb_u32 >> 16) & 0xFF
    g = (rgb_u32 >> 8) & 0xFF
    b = rgb_u32 & 0xFF
    rgb = np.stack([r, g, b], axis=1).astype(np.uint8)

    return xyz, rgb

def pcd_to_numpy_open3d_equivalent(path):
    fields = None
    data_type = None
    header_bytes = 0

    with open(path, "rb") as f:
        for line_bytes in f:
            header_bytes += len(line_bytes)
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("DATA"):
                data_type = line.split()[1].lower()
                break

    col = {name: i for i, name in enumerate(fields)}
    n_fields = len(fields)

    if data_type == "binary":
        # All CARLA PCD fields are float32 (4 bytes) — read in one shot
        with open(path, "rb") as f:
            f.seek(header_bytes)
            data = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, n_fields)
    elif data_type == "ascii":
        # np.fromstring is significantly faster than np.loadtxt
        with open(path, "r") as f:
            f.seek(header_bytes)
            data = np.fromstring(f.read(), dtype=np.float32, sep=" ").reshape(-1, n_fields)
    else:
        raise ValueError(f"Unsupported PCD DATA type: {data_type!r}")

    xyz = data[:, [col["x"], col["y"], col["z"]]]
    rgb_u32 = data[:, col["rgb"]].view(np.uint32)
    intensity = (((rgb_u32 >> 16) & 0xFF).astype(np.float32) / 255.0).reshape(-1, 1)
    return np.hstack((xyz, intensity)).astype(np.float32)

def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    
    # ORIGINAL CODE
    # import open3d as o3d
    # pcd = o3d.io.read_point_cloud(pcd_file)

    # xyz = np.asarray(pcd.points)
    
    # # we save the intensity in the first channel
    # intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    
    # pcd_np = np.hstack((xyz, intensity))

    # return np.asarray(pcd_np, dtype=np.float32)
    
    # UPDATED CODE
    return pcd_to_numpy_open3d_equivalent(pcd_file)
    
    


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, 4)

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lida data, shape: (n, 4)
    """

    lidar_xyz = lidar_data[:, :3].T
    # (3, n) -> (4, n), homogeneous transformation
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    lidar_int = lidar_data[:, 3]

    # transform to ego vehicle space, (3, n)
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :]
    # (n, 3)
    project_lidar_xyz = project_lidar_xyz.T
    # concatenate the intensity with xyz, (n, 4)
    projected_lidar = np.hstack((project_lidar_xyz,
                                 np.expand_dims(lidar_int, -1)))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list
