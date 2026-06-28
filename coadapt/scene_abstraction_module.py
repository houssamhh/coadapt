"""
Reads a scenario directory and produces a plain-text description of each
queried frame, suitable for passing to an LLM.

----------
iter_frames(scenario_path, ...)
    Generator that yields (frame_name, scenario_data) for each frame that
    should be queried according to the reselect_every / min_in_range settings.

build_scene_description(scenario_data, ...)
    Returns a formatted string describing a single frame.
"""

from __future__ import annotations

import glob
import math
import os
import sys
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

# ── make opencood importable when this file is run directly ─────────────────
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from opencood.hypes_yaml.yaml_utils import load_yaml

# optional — open3d may not be available on all nodes
try:
    from lidar_processing.interpret_pointcloud import (
        load_point_cloud,
        segment_ground,
        cluster_obstacles,
        describe_obstacles,
        obstacles_to_text,
    )
    _LIDAR_AVAILABLE = True
except Exception:
    _LIDAR_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# LiDAR scene description helper
# ════════════════════════════════════════════════════════════════════════════

def _scene_abstraction_module(cav_dir: str, frame_name: str) -> Optional[str]:
    """Return obstacle text for a single CAV frame, or None if unavailable."""
    if not _LIDAR_AVAILABLE:
        return None
    pcd_path = os.path.join(cav_dir, f"{frame_name}.pcd")
    if not os.path.exists(pcd_path):
        return None
    try:
        pcd = load_point_cloud(pcd_path)
        _, objects = segment_ground(pcd)
        clusters = cluster_obstacles(objects)
        obstacle_info = describe_obstacles(clusters)
        return obstacles_to_text(obstacle_info) or None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# Direction helpers
# ════════════════════════════════════════════════════════════════════════════

_COMPASS_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def _bearing_label(dx: float, dy: float) -> str:
    """Return an 8-direction compass label for the vector (dx, dy).

    Assumes +x = East, +y = North (standard CARLA / right-hand map frame).
    """
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return _COMPASS_8[int((angle + 22.5) / 45) % 8]


# ════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ════════════════════════════════════════════════════════════════════════════

def _cav_dirs(scenario_path: str) -> List[str]:
    """Return sorted list of CAV sub-folder paths.

    Accepts both purely-numeric names (e.g. '302') and 'cav_<id>' names
    (e.g. 'cav_208').  RSU folders ('rsu_*') are always excluded.
    """
    def _is_cav(path: str) -> bool:
        name = os.path.basename(path)
        if not os.path.isdir(path):
            return False
        if name.startswith("rsu_"):
            return False
        return name.isdigit() or name.startswith("cav_")

    return sorted(d for d in glob.glob(os.path.join(scenario_path, "*")) if _is_cav(d))


def _load_frame_yaml(cav_dir: str, frame_name: Optional[str]) -> Optional[dict]:
    """Load the YAML for a given frame (or the first frame if frame_name is None)."""
    if frame_name is None:
        yamls = sorted(glob.glob(os.path.join(cav_dir, "*.yaml")))
        return load_yaml(yamls[0]) if yamls else None

    path = os.path.join(cav_dir, f"{frame_name}.yaml")
    return load_yaml(path) if os.path.exists(path) else None


def _all_frame_names(scenario_path: str) -> List[str]:
    """Return sorted list of frame stems present in any CAV folder."""
    frame_set: set = set()
    for cav_dir in _cav_dirs(scenario_path):
        for y in glob.glob(os.path.join(cav_dir, "*.yaml")):
            frame_set.add(os.path.splitext(os.path.basename(y))[0])
    return sorted(frame_set)


def _ego_spawn_position(protocol: dict) -> Optional[np.ndarray]:
    """Return (x, y) spawn position of the ego from data_protocol.yaml."""
    try:
        spawn = protocol["scenario"]["single_cav_list"][0]["spawn_position"]
        return np.array([spawn[0], spawn[1]], dtype=float)
    except (KeyError, IndexError):
        return None


def _protocol_com_range(protocol: dict, default: float = 70.0) -> float:
    """Extract communication range (metres) from data_protocol.yaml."""
    try:
        return float(
            protocol["scenario"]["single_cav_list"][0]["v2x"]["communication_range"]
        )
    except (KeyError, IndexError, TypeError):
        return default


# ════════════════════════════════════════════════════════════════════════════
# Scenario loading
# ════════════════════════════════════════════════════════════════════════════

def load_frame(scenario_path: str,
               frame_name: Optional[str] = None,
               com_range_override: Optional[float] = None,
               with_lidar: bool = True) -> dict:
    """Load CAV positions at a specific frame and return a scenario_data dict.

    Parameters
    ----------
    scenario_path : str
        Path to the scenario folder.
    frame_name : str or None
        Frame stem to load (e.g. '000068').  None loads the first frame.
    com_range_override : float or None
        If provided, overrides the communication range from data_protocol.yaml.

    Returns
    -------
    dict with keys:
        ego_cav    : str
        com_range  : float  (metres)
        cavs       : {cav_id: {"pos": [x, y, z], "dist_to_ego": float}}
        frame_name : str
    Returns an empty dict if no CAV data is found.
    """
    protocol_path = os.path.join(scenario_path, "data_protocol.yaml")
    protocol = load_yaml(protocol_path) if os.path.exists(protocol_path) else {}

    com_range  = com_range_override if com_range_override is not None \
                 else _protocol_com_range(protocol, default=70.0)
    ego_spawn  = _ego_spawn_position(protocol)

    actual_frame: Optional[str] = frame_name
    cavs: Dict[str, dict] = {}

    for cav_dir in _cav_dirs(scenario_path):
        cav_id     = os.path.basename(cav_dir)
        frame_data = _load_frame_yaml(cav_dir, frame_name)
        if frame_data is None:
            continue

        # record the actual frame name on the first successful load
        if actual_frame is None:
            yamls = sorted(glob.glob(os.path.join(cav_dir, "*.yaml")))
            if yamls:
                actual_frame = os.path.splitext(os.path.basename(yamls[0]))[0]

        pose = frame_data.get("lidar_pose") or frame_data.get("true_ego_pose")
        if pose is None:
            continue
        resolved_frame = actual_frame or frame_name or ""
        cavs[cav_id] = {
            "pos":        list(pose[:3]),
            "scene_text": _scene_abstraction_module(cav_dir, resolved_frame) if with_lidar else None,
        }

    if not cavs:
        return {}

    # determine ego: CAV closest to spawn position, or first alphabetically
    if ego_spawn is not None:
        ego_cav = min(
            cavs,
            key=lambda cid: np.linalg.norm(np.array(cavs[cid]["pos"][:2]) - ego_spawn),
        )
    else:
        ego_cav = sorted(cavs.keys())[0]

    ego_xy = np.array(cavs[ego_cav]["pos"][:2])
    for cav_id, info in cavs.items():
        info["dist_to_ego"] = float(
            np.linalg.norm(np.array(info["pos"][:2]) - ego_xy)
        )

    return {
        "ego_cav":    ego_cav,
        "com_range":  com_range,
        "cavs":       cavs,
        "frame_name": actual_frame or "unknown",
    }


# ════════════════════════════════════════════════════════════════════════════
# Frame iterator
# ════════════════════════════════════════════════════════════════════════════

def iter_frames(
    scenario_path: str,
    reselect_every: Optional[int] = None,
    min_in_range: int = 1,
    com_range_override: Optional[float] = None,
) -> Generator[Tuple[str, int, dict], None, None]:
    """Yield (frame_name, frame_idx, scenario_data) for each queried frame.

    Parameters
    ----------
    scenario_path : str
        Path to the scenario folder.
    reselect_every : int or None
        Query every N-th frame.  None means only the first frame.
    min_in_range : int
        Skip frames where fewer than this many CAVs are within range of ego.
    com_range_override : float or None
        Override the communication range from data_protocol.yaml.
    """
    if reselect_every is None:
        query_indices = [None]   # None means "first frame"
        all_frames: List[str] = []
    else:
        all_frames = _all_frame_names(scenario_path)
        if not all_frames:
            return
        query_indices = [
            (i, f) for i, f in enumerate(all_frames)
            if i % reselect_every == 0
        ]

    for entry in query_indices:
        if entry is None:
            frame_name_query = None
            frame_idx = 0
        else:
            frame_idx, frame_name_query = entry

        data = load_frame(scenario_path, frame_name_query, com_range_override)
        if not data:
            continue

        in_range_count = sum(
            1 for info in data["cavs"].values()
            if info["dist_to_ego"] <= data["com_range"]
        )
        if in_range_count < min_in_range:
            continue

        # resolve frame_idx when only first frame is queried
        if reselect_every is None:
            fn = data["frame_name"]
            all_frames_tmp = _all_frame_names(scenario_path)
            frame_idx = all_frames_tmp.index(fn) if fn in all_frames_tmp else 0

        yield data["frame_name"], frame_idx, data


# ════════════════════════════════════════════════════════════════════════════
# Description builder
# ════════════════════════════════════════════════════════════════════════════

def build_scene_description(
    scenario_data: dict,
    bandwidth_mbps: float = 50.0,
    prev_result: Optional[dict] = None,
    prev_ap: Optional[dict] = None,
) -> str:
    """Return a plain-text scene description for a single frame.

    Parameters
    ----------
    scenario_data : dict
        Output of load_frame().
    bandwidth_mbps : float
        Available bandwidth in Mbps (continuous value).
    prev_result : dict or None
        Previous LLM decision: {selected_cavs, fusion_method, reason}.
    prev_ap : dict or None
        AP metrics from the previous inference cycle: {ap_03, ap_05, ap_07}.
    """
    ego    = scenario_data["ego_cav"]
    cr     = scenario_data["com_range"]
    cavs   = scenario_data["cavs"]
    ego_xy = np.array(cavs[ego]["pos"][:2])

    sorted_cavs  = sorted(cavs.items(), key=lambda x: x[1]["dist_to_ego"])
    in_range_ids = [cid for cid, info in sorted_cavs if info["dist_to_ego"] <= cr]

    lines = [
        f"Ego CAV: {ego}",
        f"Comm range: {cr:.0f} m",
        f"Available bandwidth: {bandwidth_mbps:.1f} Mbps",
        "",
        "CAVs (sorted by distance to ego):",
    ]

    for cav_id, info in sorted_cavs:
        x, y, _ = info["pos"]
        d = info["dist_to_ego"]
        if cav_id == ego:
            direction = "-"
            status    = "EGO"
        else:
            direction = _bearing_label(x - ego_xy[0], y - ego_xy[1])
            status    = "in-range" if d <= cr else "out-of-range"
        lines.append(
            f"  {cav_id} [{status}]  x={x:.1f} y={y:.1f}  dist={d:.1f}m  dir={direction}"
        )
        scene_text = info.get("scene_text")
        if scene_text:
            for obs_line in scene_text.splitlines():
                lines.append(f"    {obs_line}")

    non_ego_in_range = [cid for cid in in_range_ids if cid != ego]
    if len(non_ego_in_range) >= 2:
        lines += ["", "Pairwise distances (in-range non-ego CAVs):"]
        for i, a in enumerate(non_ego_in_range):
            for b in non_ego_in_range[i + 1:]:
                pa = np.array(cavs[a]["pos"][:2])
                pb = np.array(cavs[b]["pos"][:2])
                lines.append(f"  {a} - {b}: {np.linalg.norm(pa - pb):.1f}m")

    if prev_result:
        lines += [
            "",
            "Previous decision:",
            f"  Selected: {', '.join(str(c) for c in prev_result.get('selected_cavs', []))}",
            f"  Fusion: {prev_result.get('fusion_method', 'unknown')}",
            f"  Reason: {prev_result.get('reason', '')}",
        ]
        if prev_ap:
            lines.append(
                f"  AP@0.3: {prev_ap.get('ap_03', 'n/a')}  "
                f"AP@0.5: {prev_ap.get('ap_05', 'n/a')}  "
                f"AP@0.7: {prev_ap.get('ap_07', 'n/a')}"
            )

    lines += ["", "Output the final JSON on the last line."]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or not sys.argv[1].endswith(".pcd"):
        print("Usage: python scene_abstraction_module.py <path/to/scan.pcd> [output.txt]")
        sys.exit(1)

    pcd_path    = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not _LIDAR_AVAILABLE:
        print("ERROR: lidar_processing dependencies (open3d) are not installed.")
        sys.exit(1)

    if not os.path.exists(pcd_path):
        print(f"ERROR: file not found: {pcd_path}")
        sys.exit(1)

    print(f"Processing: {pcd_path}\n")

    pcd      = load_point_cloud(pcd_path)
    _, objs  = segment_ground(pcd)
    clusters = cluster_obstacles(objs)
    info     = describe_obstacles(clusters)
    text     = obstacles_to_text(info) or "(no obstacles detected)"

    print(text)

    if output_path:
        with open(output_path, "w") as f:
            f.write(text + "\n")
        print(f"\nResults saved to: {output_path}")
