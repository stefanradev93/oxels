import os
import glob
import sys
import time
import json
import random
from datetime import datetime
import uuid

import numpy as np
import trimesh
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy as _cpy
from collections import OrderedDict
from typing import Tuple


# If you have oxels in a parent directory, adjust as needed
parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

from oxels.obj_3d import (
    render_faster,
    sample_random_on_mesh_fast_o3d,
    get_visible_matches_with_indices_o3d,
)

# === GLOBAL CONFIG ===
SHAPENET_DIR = "./ShapeNet_Split/train"
BLACKLIST = {"04090263"}

SCENENN_DIR = "./SceneNN_Split/train"
BAD_SCENES = {}

OUTPUT_DIR = f"contrastive_3d/train/{str(uuid.uuid4())}"
RESOLUTION = (640, 480)
F1 = 500
F2 = 500

RATIO_OF_POINTS_TO_RENDER = 1/8
NUM_CAMERAS = 2

# — Number of image‐pairs to generate total —
NUM_PAIRS = 1000  # ← change as desired

# =============================================================================
#  -- CACHES TO SPEED UP REPEATED LOADS --
# =============================================================================
# Cache for any SceneNN scene we load (keyed by scene_id)
MAX_SCENE_CACHE = 100
_SCENE_CACHE: OrderedDict[str, Tuple] = OrderedDict()

# Cache for each raw ShapeNet .obj we load (keyed by absolute model_path).
_MAX_SHAPENET_MESH_CACHE = 200
_SHAPENET_MESH_CACHE: OrderedDict[str, o3d.geometry.TriangleMesh] = OrderedDict()


# =============================================================================
# 1. Load a random SceneNN mesh and detect floor plane + up‐axis, with caching
# =============================================================================
def load_scene_mesh_and_detect_floor(scenenn_dir: str, bad_scenes: set):
    """
    1) Builds a list of all valid SceneNN-scene folders (excluding bad_scenes).
    2) Randomly picks one, loads its .ply into an Open3D TriangleMesh.
    3) Samples a large point cloud, runs RANSAC to detect the floor plane, and returns:
       - scene_mesh (Open3D TriangleMesh)
       - up_axis (int: 0=x,1=y,2=z)
       - plane_normal (np.ndarray length 3, pointing "up")
       - floor_height (float, slightly above actual floor)
       - ceiling_height (float)
       - min_bound, max_bound (axis‐aligned bounds of scene)
       - all sampled points (np.ndarray of shape (N,3))
    4) If the same scene_id has already been loaded once, returns the cached result.
    """
    # 1) list valid scene folders
    scene_folders = [
        f for f in glob.glob(f"{scenenn_dir}/*")
        if os.path.isdir(f) and os.path.basename(f) not in bad_scenes
    ]
    if not scene_folders:
        raise RuntimeError(f"No valid scenes found under {scenenn_dir}")

    # 2) pick one randomly
    selected_scene = random.choice(scene_folders)
    scene_id = os.path.basename(selected_scene)

    # 3) if already cached, return the tuple
    if scene_id in _SCENE_CACHE:
        return _SCENE_CACHE[scene_id]

    mesh_path = os.path.join(selected_scene, f"{scene_id}.ply")
    print(f"[load_scene] Loading scene {scene_id} from {mesh_path}")
    scene_mesh = o3d.io.read_triangle_mesh(mesh_path)
    scene_mesh.compute_vertex_normals()

    # Sample many points for RANSAC
    pcd = scene_mesh.sample_points_uniformly(number_of_points=200000)
    points = np.asarray(pcd.points)

    # Heuristic for up-axis: the axis with smallest extent
    bounds = scene_mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound
    extent = max_bound - min_bound
    up_axis = int(np.argmin(extent))  # 0=x,1=y,2=z
    up_vector = np.zeros(3)
    up_vector[up_axis] = 1.0

    # RANSAC to find dominant plane (floor)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c])
    # Ensure normal points "up"
    if plane_normal.dot(up_vector) < 0:
        plane_normal = -plane_normal
        d = -d

    # floor height = min of inlier points along up_axis + small epsilon
    floor_points = points[inliers]
    floor_height = float(np.min(floor_points[:, up_axis])) + 0.01
    # ceiling = near top of bounding box
    ceiling_height = float(max_bound[up_axis] - 0.1)

    # Cache it all in a tuple for reuse
    if len(_SCENE_CACHE) >= MAX_SCENE_CACHE:
        _SCENE_CACHE.popitem(last=False)
    _SCENE_CACHE[scene_id] = (
        scene_mesh,
        up_axis,
        plane_normal,
        floor_height,
        ceiling_height,
        min_bound,
        max_bound,
        points
    )
    return _SCENE_CACHE[scene_id]


# =============================================================================
# 2. Sample candidate points inside the bounding box & filter by clearance & interior‐check
# =============================================================================
def sample_interior_points(scene_mesh: o3d.geometry.TriangleMesh,
                           points_all: np.ndarray,
                           up_axis: int,
                           floor_height: float,
                           ceiling_height: float,
                           min_bound: np.ndarray,
                           max_bound: np.ndarray,
                           num_samples: int = 5000,
                           min_clearance: float = 0.05):
    """
    (Exactly as before.)
    """
    t0 = time.time()
    # 1) generate num_samples random points within the axis‐aligned box
    min_x, min_y, min_z = min_bound
    max_x, max_y, max_z = max_bound
    if up_axis == 0:
        x_vals = np.random.uniform(floor_height, ceiling_height, num_samples)
        y_vals = np.random.uniform(min_y + 0.1, max_y - 0.1, num_samples)
        z_vals = np.random.uniform(min_z + 0.1, max_z - 0.1, num_samples)
    elif up_axis == 1:
        x_vals = np.random.uniform(min_x + 0.1, max_x - 0.1, num_samples)
        y_vals = np.random.uniform(floor_height, ceiling_height, num_samples)
        z_vals = np.random.uniform(min_z + 0.1, max_z - 0.1, num_samples)
    else:  # up_axis == 2
        x_vals = np.random.uniform(min_x + 0.1, max_x - 0.1, num_samples)
        y_vals = np.random.uniform(min_y + 0.1, max_y - 0.1, num_samples)
        z_vals = np.random.uniform(floor_height, ceiling_height, num_samples)
    sampled_points = np.vstack([x_vals, y_vals, z_vals]).T

    # 2) build RaycastingScene for clearance test
    scene_rt = o3d.t.geometry.RaycastingScene()
    scene_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh)
    _ = scene_rt.add_triangles(scene_mesh_t)
    queries = o3d.core.Tensor(sampled_points, dtype=o3d.core.Dtype.Float32)
    distances = scene_rt.compute_distance(queries).numpy()
    clear_mask = distances > min_clearance
    candidates = sampled_points[clear_mask]
    distances = distances[clear_mask]

    # 3) horizontal ray‐cast check for interior‐ness
    axes = [0, 1, 2]
    axes.remove(up_axis)
    ray_dirs = []
    for ax in axes:
        vec = np.zeros(3); vec[ax] = 1.0
        ray_dirs.append(vec.copy())
        ray_dirs.append(-vec.copy())
    ray_dirs = np.array(ray_dirs, dtype=np.float32)  # shape (4,3)
    num_cand = candidates.shape[0]
    origins = np.repeat(candidates, repeats=len(ray_dirs), axis=0).astype(np.float32)
    dirs = np.tile(ray_dirs, (num_cand, 1)).astype(np.float32)
    dir_norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.divide(dirs, dir_norms, out=np.zeros_like(dirs), where=dir_norms != 0)
    rays = np.hstack((origins, dirs))
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ray_results = scene_rt.cast_rays(rays_tensor)
    t_hit = ray_results["t_hit"].numpy().reshape(num_cand, len(ray_dirs))
    hit_count = np.sum(np.isfinite(t_hit), axis=1)
    interior_mask = hit_count >= 3
    interior_points = candidates[interior_mask]
    interior_distances = distances[interior_mask]

    return interior_points, interior_distances


# =============================================================================
# 3. Place multiple objects within the interior points
# =============================================================================
def place_objects(interior_points: np.ndarray,
                  interior_distances: np.ndarray,
                  min_object_distance: float = 0.35,
                  max_objects: int = 20):
    """
    (Exactly as before.)
    """
    t0 = time.time()
    if interior_points.shape[0] == 0:
        raise RuntimeError("No interior points to place objects.")
    order = np.argsort(-interior_distances)
    pts_sorted = interior_points[order]
    dists_sorted = interior_distances[order]
    cluster_center = pts_sorted[0]
    num_objects = np.random.randint(1, max_objects + 1)
    object_positions = [cluster_center.copy()]

    remaining_points = pts_sorted[1:].copy()
    remaining_dist = dists_sorted[1:].copy()

    while len(object_positions) < num_objects and remaining_points.shape[0] > 0:
        valid_mask = np.ones(len(remaining_points), dtype=bool)
        for placed in object_positions:
            dists = np.linalg.norm(remaining_points - placed.reshape(1, 3), axis=1)
            valid_mask &= (dists >= min_object_distance)
        if not np.any(valid_mask):
            print(f"[place_objects] Warning: no valid position for object {len(object_positions)+1}")
            break
        valid_pts = remaining_points[valid_mask]
        valid_dist = remaining_dist[valid_mask]
        weights = valid_dist / np.sum(valid_dist)
        idx_choice = np.random.choice(len(valid_pts), p=weights)
        object_positions.append(valid_pts[idx_choice].copy())
        remaining_points = np.delete(remaining_points, np.where(valid_mask)[0][idx_choice], axis=0)
        remaining_dist = np.delete(remaining_dist, np.where(valid_mask)[0][idx_choice], axis=0)

    return np.vstack(object_positions), cluster_center


def compute_camera_poses(interior_points: np.ndarray,
                         cluster_center: np.ndarray,
                         plane_normal: np.ndarray,
                         floor_height: float,
                         ceiling_height: float,
                         up_axis: int,
                         scene_mesh: o3d.geometry.TriangleMesh,
                         target_offset_range: float = 0.5,
                         cam_height_min: float = 0.1,
                         cam_height_max: float = 4.0,
                         min_cluster_dist: float = 0.5,
                         max_cluster_dist: float = 4.0,
                         min_cam_separation: float = 0.5,
                         max_cam_separation: float = 4.0,
                         max_angle_diff_deg: float = 65.0,
                         max_relax_passes: int = 5):
    """
    Find two camera poses that both see `cluster_center`, subject to constraints.
    Uses randomized sampling + raycasting in Open3D. Includes a maximum number of
    relaxation passes to avoid infinite loops.

    Parameters:
        interior_points: (N×3) array of candidate points inside the scene
        cluster_center:  (3,) point to center both cameras on
        plane_normal:    (3,) normal of the floor plane (world-up direction)
        floor_height:    minimum camera height above floor
        ceiling_height:  maximum camera height below ceiling
        up_axis:         index (0/1/2) corresponding to the "up" coordinate in interior_points
        scene_mesh:      an Open3D TriangleMesh of the scene geometry
        target_offset_range: random jitter around cluster_center for "look-at" target
        cam_height_min:  minimum camera height above floor
        cam_height_max:  maximum camera height above floor
        min_cluster_dist: minimum allowed distance from camera to cluster_center
        max_cluster_dist: maximum allowed distance from camera to cluster_center
        min_cam_separation: minimum allowed distance between the two cameras
        max_cam_separation: maximum allowed distance between the two cameras
        max_angle_diff_deg: max allowed angle (in degrees) between camera viewing directions
        max_relax_passes: maximum times to relax constraints before giving up

    Returns:
        A dict with keys {"pose1", "pose2", "cam1_pos", "cam2_pos", "R_cam1", "R_cam2", "target1", "target2"}.

    Raises:
        RuntimeError if no valid camera(s) can be found after max_relax_passes.
    """
    t0 = time.time()
    scene_rt    = o3d.t.geometry.RaycastingScene()
    scene_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(scene_mesh)
    _ = scene_rt.add_triangles(scene_mesh_t)

    world_up = plane_normal.copy()
    # Save original parameters so we can restore before second camera
    base_params = {
        "cam_height_min":    cam_height_min,
        "cam_height_max":    cam_height_max,
        "min_cluster_dist":  min_cluster_dist,
        "max_cluster_dist":  max_cluster_dist,
        "min_cam_separation": min_cam_separation,
        "max_cam_separation": max_cam_separation,
        "max_angle_diff_deg": max_angle_diff_deg,
        "target_offset_range": target_offset_range
    }

    # --- FIND FIRST CAMERA ---
    first_relax_count = 0
    camera_position1 = None
    R_cam1 = None
    while True:
        # 1) Filter by height band
        cam_min_h = floor_height + cam_height_min
        cam_max_h = min(floor_height + cam_height_max, ceiling_height - 0.1)
        height_mask = (
            (interior_points[:, up_axis] >= cam_min_h) &
            (interior_points[:, up_axis] <= cam_max_h)
        )
        candidates = interior_points[height_mask]
        if candidates.shape[0] == 0:
            candidates = interior_points.copy()

        first_target = cluster_center + np.random.uniform(
            -target_offset_range, target_offset_range, size=3
        )
        first_target[up_axis] = cluster_center[up_axis]

        # Try up to 100 random interior points
        for attempt in range(100):
            idx = np.random.randint(len(candidates))
            cand = candidates[idx]
            dist_to_cluster = np.linalg.norm(cand - cluster_center)
            if not (min_cluster_dist <= dist_to_cluster <= max_cluster_dist):
                continue

            # Raycast from cand to first_target
            dir_vec = first_target - cand
            dist = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / (dist + 1e-8)
            ray = o3d.core.Tensor(
                np.hstack((cand.astype(np.float32),
                           dir_unit.astype(np.float32))).reshape((1, 6)),
                dtype=o3d.core.Dtype.Float32
            )
            res = scene_rt.cast_rays(ray)
            t_hit = res["t_hit"].numpy()[0]
            if np.isfinite(t_hit) and t_hit >= dist:
                # Construct camera coordinate frame: forward = dir_unit, right, up
                forward = dir_unit
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 1e-6:
                    right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
                right /= np.linalg.norm(right)
                cam_up = np.cross(right, forward)
                cam_up /= np.linalg.norm(cam_up)
                R_cam1 = np.stack([right, cam_up, forward], axis=1)
                camera_position1 = cand.copy()
                break  # found first camera

        # If we found a camera_position1, break out of relax loop
        if camera_position1 is not None:
            break

        # Otherwise: relax constraints
        first_relax_count += 1
        if first_relax_count >= max_relax_passes:
            raise RuntimeError("Failed to find a valid first camera pose "
                               f"after {max_relax_passes} relaxation passes.")
        # Relax by 20%
        min_cluster_dist *= 0.8
        max_cluster_dist *= 1.2
        cam_height_min = max(0.0, cam_height_min * 0.8)
        cam_height_max *= 1.2

    # --- FIND SECOND CAMERA ---
    # Restore original constraints
    cam_height_min       = base_params["cam_height_min"]
    cam_height_max       = base_params["cam_height_max"]
    min_cluster_dist     = base_params["min_cluster_dist"]
    max_cluster_dist     = base_params["max_cluster_dist"]
    min_cam_separation   = base_params["min_cam_separation"]
    max_cam_separation   = base_params["max_cam_separation"]
    max_angle_diff_deg   = base_params["max_angle_diff_deg"]
    target_offset_range  = base_params["target_offset_range"]

    second_relax_count = 0
    camera_position2 = None
    R_cam2 = None
    angle_deg = None

    # Compute first camera's viewing direction
    first_cam_dir = (first_target - camera_position1)
    first_cam_dir /= (np.linalg.norm(first_cam_dir) + 1e-8)

    while True:
        second_target = cluster_center + np.random.uniform(
            -target_offset_range, target_offset_range, size=3
        )
        second_target[up_axis] = cluster_center[up_axis]

        # Try up to 100 random interior points
        for attempt in range(100):
            idx = np.random.randint(len(candidates))
            cand = candidates[idx]

            sep = np.linalg.norm(cand - camera_position1)
            if not (min_cam_separation <= sep <= max_cam_separation):
                continue

            dist_cluster = np.linalg.norm(cand - cluster_center)
            if not (min_cluster_dist <= dist_cluster <= max_cluster_dist):
                continue

            dir_vec = second_target - cand
            dist = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / (dist + 1e-8)
            ray = o3d.core.Tensor(
                np.hstack((cand.astype(np.float32),
                           dir_unit.astype(np.float32))).reshape((1, 6)),
                dtype=o3d.core.Dtype.Float32
            )
            res = scene_rt.cast_rays(ray)
            t_hit = res["t_hit"].numpy()[0]
            if not (np.isfinite(t_hit) and t_hit >= dist):
                continue

            # Angle between first_cam_dir and new dir_unit
            angle = np.arccos(np.clip(np.dot(first_cam_dir, dir_unit), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            if angle_deg > max_angle_diff_deg:
                continue

            # Construct second camera's coordinate frame
            forward = dir_unit
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
            right /= np.linalg.norm(right)
            cam_up = np.cross(right, forward)
            cam_up /= np.linalg.norm(cam_up)
            R_cam2 = np.stack([right, cam_up, forward], axis=1)
            camera_position2 = cand.copy()
            break  # found second camera

        # If we found a valid second camera, break
        if camera_position2 is not None:
            break

        # Otherwise: relax constraints
        second_relax_count += 1
        if second_relax_count >= max_relax_passes:
            raise RuntimeError("Failed to find a valid second camera pose "
                               f"after {max_relax_passes} relaxation passes.")
        # Relax by 20% (and widen angle by +10°)
        min_cam_separation *= 0.8
        max_cam_separation *= 1.2
        min_cluster_dist *= 0.8
        max_cluster_dist *= 1.2
        max_angle_diff_deg += 10.0
        cam_height_min = max(0.0, cam_height_min * 0.8)
        cam_height_max *= 1.2

    # --- BUILD 4×4 "look-at" POSES for both cameras ---
    pose1 = np.eye(4, dtype=float)
    pose1[:3, 0] = R_cam1[:, 0]
    pose1[:3, 1] = R_cam1[:, 1]
    pose1[:3, 2] = -R_cam1[:, 2]
    pose1[:3, 3] = camera_position1

    pose2 = np.eye(4, dtype=float)
    pose2[:3, 0] = R_cam2[:, 0]
    pose2[:3, 1] = R_cam2[:, 1]
    pose2[:3, 2] = -R_cam2[:, 2]
    pose2[:3, 3] = camera_position2

    return {
        "pose1":      pose1,
        "pose2":      pose2,
        "cam1_pos":   camera_position1,
        "cam2_pos":   camera_position2,
        "R_cam1":     R_cam1,
        "R_cam2":     R_cam2,
        "target1":    first_target,
        "target2":    second_target
    }


# =============================================================================
# 5. Load random ShapeNet objects and transform them to object_positions, with caching
# =============================================================================
def _get_base_shapenet_mesh(model_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a raw ShapeNet .obj once and cache it (up to 1000 entries).
    On cache hit: return a fresh copy of the cached mesh.
    On miss: load, cache (evicting oldest if needed), and return the new mesh.
    """
    # ——— Cache hit: move to end (MRU) and return a copy ———
    if model_path in _SHAPENET_MESH_CACHE:
        # pop then re-insert to mark as most-recently used
        original = _SHAPENET_MESH_CACHE.pop(model_path)
        _SHAPENET_MESH_CACHE[model_path] = original
        try:
            return original.copy()
        except AttributeError:
            return _cpy.deepcopy(original)

    # ——— Cache miss: load from disk ———
    tm = trimesh.load(model_path, process=False)
    if isinstance(tm, trimesh.Scene):
        geoms = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("Scene has no Trimesh parts")
        tm = trimesh.util.concatenate(geoms)

    # extract or synthesize vertex colors
    vis = tm.visual
    if hasattr(vis, "vertex_colors") and len(vis.vertex_colors):
        vc = vis.vertex_colors
    elif hasattr(vis, "to_color"):
        vc = vis.to_color().vertex_colors
    else:
        vc = np.tile([200, 200, 200, 255], (len(tm.vertices), 1))
    vc = (vc[:, :3] / 255.0).astype(np.float64)

    # build Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
    mesh.compute_vertex_normals()

    # ——— Evict oldest if we're at capacity ———
    if len(_SHAPENET_MESH_CACHE) >= _MAX_SHAPENET_MESH_CACHE:
        _SHAPENET_MESH_CACHE.popitem(last=False)

    # ——— Cache a copy and return the original ———
    try:
        _SHAPENET_MESH_CACHE[model_path] = mesh.copy()
    except AttributeError:
        _SHAPENET_MESH_CACHE[model_path] = _cpy.deepcopy(mesh)

    return mesh


def load_and_transform_objects(object_positions: np.ndarray):
    """
    For each 3D position in object_positions:
      - Pick a random ShapeNet .obj (avoiding BLACKLIST), try up to 10 times.
      - Call _get_base_shapenet_mesh(model_path) to get a cached/raw Open3D mesh.
      - Clone via .copy() (or deepcopy) so we don't alter the cache.
      - Apply random scale, rotation, translation → append to object_meshes.
      - If loading fails entirely, fall back to a simple sphere at "pos".
    """
    t0 = time.time()
    object_meshes = []

    for pos in object_positions:
        base_mesh = None
        for attempt in range(10):
            category_dirs = [d for d in glob.glob(f"{SHAPENET_DIR}/*") if os.path.isdir(d)]
            category_ids = [os.path.basename(cat) for cat in category_dirs if os.path.basename(cat) not in BLACKLIST]
            if not category_ids:
                continue
            category = random.choice(category_ids)
            model_dirs = [
                d for d in glob.glob(f"{SHAPENET_DIR}/{category}/*/")
                if os.path.isdir(d) and not d.endswith(".zip")
            ]
            if not model_dirs:
                continue

            model_path = os.path.join(random.choice(model_dirs),
                                      "models", "model_normalized.obj")
            if not os.path.exists(model_path):
                continue

            try:
                base_mesh = _get_base_shapenet_mesh(model_path)
                break
            except Exception as e:
                continue

        if base_mesh is None:
            # fallback: simple blue sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.07)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])
            sphere.translate(pos, relative=False)
            object_meshes.append(sphere)
            continue

        # Make a writable copy of the cached mesh:
        try:
            obj = base_mesh.copy()
        except AttributeError:
            obj = _cpy.deepcopy(base_mesh)

        # Random scale + rotation + translation
        S = np.diag(np.random.uniform(0.3, 0.75, size=3).tolist() + [1.0])
        rot = Rotation.from_euler("xyz", np.random.uniform(0, 360, 3), degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot.dot(S[:3, :3])
        T[:3, 3] = pos
        obj.transform(T)

        object_meshes.append(obj)

    return object_meshes


# =============================================================================
# 6. Visualization of scene + objects + camera frames + lines (Open3D)
# =============================================================================
def visualize_scene(scene_mesh: o3d.geometry.TriangleMesh,
                    object_meshes: list,
                    camera_info: dict,
                    cluster_center: np.ndarray):
    """
    (Unchanged. Blocks until window is closed.)
    """
    cam1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    cam1_frame.rotate(camera_info["R_cam1"], center=(0, 0, 0))
    cam1_frame.translate(camera_info["cam1_pos"])
    cam2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    cam2_frame.rotate(camera_info["R_cam2"], center=(0, 0, 0))
    cam2_frame.translate(camera_info["cam2_pos"])

    pts1 = [camera_info["cam1_pos"], cluster_center]
    set1 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts1),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    set1.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    pts2 = [camera_info["cam2_pos"], cluster_center]
    set2 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts2),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    set2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in [scene_mesh] + object_meshes + [cam1_frame, cam2_frame, set1, set2]:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size = 5.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord)
    vis.run()
    vis.destroy_window()


# =============================================================================
# 9. Sample points and find matches between two rendered views
# =============================================================================
def sample_and_match(scene_mesh: o3d.geometry.TriangleMesh,
                     object_meshes: list,
                     pose1: np.ndarray,
                     pose2: np.ndarray,
                     resolution: tuple,
                     f1: float,
                     f2: float,
                     cluster_center,
                     interior_points,
                     up_axis,
                     interior_dists,
                     num_points: int = 2048):
    """
    Sample points from the scene and find matches between two rendered views.
    """
    t_start = time.time()
    rp1 = pose1.copy()
    rp2 = pose2.copy()
    rp1[:3, 2] = -rp1[:3, 2]
    rp2[:3, 2] = -rp2[:3, 2]

    # Render both views
    I1 = render_faster(
        scene_mesh,
        object_meshes,
        rp1,
        resolution,
        f1,
        interior_points=interior_points,
        interior_dists=interior_dists,
    )

    I2 = render_faster(
        scene_mesh,
        object_meshes,
        rp2,
        resolution,
        f2,
        interior_points=interior_points,
        interior_dists=interior_dists,
    )

    # Build raycasting scene
    combined = o3d.geometry.TriangleMesh()
    for m in [scene_mesh] + object_meshes:
        combined += m
    ray_scene = o3d.t.geometry.RaycastingScene()
    _ = ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(combined))

    # Sample points
    try:
        xyz_all, uv1_all = sample_random_on_mesh_fast_o3d(
            ray_scene=ray_scene,
            triangle_mesh=combined,
            N_max=num_points,
            camera_pose=pose1,
            f=f1,
            resolution=resolution
        )
    except Exception as e:
        print(f"[sample_and_match] Error sampling points: {e}")
        xyz_all = np.zeros((0, 3))
        uv1_all = np.zeros((0, 2))
    if len(xyz_all) == 0:
        raise RuntimeError("No visible points sampled from combined mesh.")

    # Find matches
    match_mask, uv2_all, indices = get_visible_matches_with_indices_o3d(
        ray_scene=ray_scene,
        xyz=xyz_all,
        camera_pose=pose2,
        f=f2,
        resolution=resolution
    )

    t_end = time.time()
    return I1, I2, xyz_all, uv1_all, match_mask, uv2_all, indices


# =============================================================================
# 10. Visualize matched points in image space
# =============================================================================
def visualize_matches(I1: np.ndarray,
                      I2: np.ndarray,
                      uv1_all: np.ndarray,
                      uv2_all: np.ndarray,
                      match_mask: np.ndarray,
                      resolution: tuple):
    """
    (Unchanged.)
    """
    combined_image = np.concatenate([I1, I2], axis=1)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(combined_image, origin="lower")
    uv1_vis = uv1_all[match_mask]
    uv2_vis = uv2_all
    num_matches = len(uv1_vis)
    colors = np.random.rand(num_matches, 3)
    for i in range(num_matches):
        x1, y1 = uv1_vis[i]
        x2, y2 = uv2_vis[i]
        x1 = resolution[0] - x1
        x2 = resolution[0] - x2
        clr = colors[i]
        ax.scatter(x1, y1, color=clr, s=20, marker="o")
        ax.scatter(x2 + resolution[0], y2, color=clr, s=20, marker="o")
    ax.set_title("Matched Points (Left: View 1, Right: View 2)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 11. Export everything into OUTPUT_DIR/scene_timestamp
# =============================================================================
def apply_augmentations(image: np.ndarray):
    """Gaussian blur + brightness/contrast adjustments."""
    ksize = random.choice([1, 3, 5])
    if ksize > 1:
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image, ksize, alpha, beta

def export_dataset(images: list,
                   poses: list,
                   uv_sets: list,
                   xyz_set: list,
                   match_masks: list,
                   indices_set: list,
                   output_dir: str,
                   resolution: tuple,
                   f1: float):
    """
    Exports each pair of images and their matches, but instead of listing UV coordinates,
    creates two binary masks (mask1, mask2) of length W*H, where a 1 indicates a matched
    pixel, and a permutation array perm of length W*H such that perm[p1] = p2 if pixel
    p1 in image1 matches pixel p2 in image2 (else -1).

    Parameters:
      images:      [I1, I2] floating‐point RGB arrays in [0,1], each shape (H, W, 3)
      poses:       [pose1, pose2] each a 4×4 camera‐to‐world matrix
      uv_sets:     [uv1_all, uv2_all], each an (M,2) array of (u,v) pixel locations for sampled points
                   with (0,0) at top‐left. M is the number of sampled 3D points visible in view1/view2.
      xyz_set:     [xyz1_all, xyz2_all] each an (M,3) array of the corresponding 3D points (same length as uv_sets)
      match_masks: 2×2 list‐of‐lists of boolean arrays; match_masks[i][j] is a boolean mask of length M
                   indicating which sampled points in view i have a valid match to view j.
      indices_set: [indices1, indices2], each a length‐M array of the original indices of those sampled points.
      output_dir:  Path to the folder in which to create a new "scene_<timestamp>/" subfolder.
      resolution:  (W, H) tuple giving image width and height in pixels.
      f1:          Focal length in pixels (stored in JSON as metadata).
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Create a new unique subdirectory for this export using UUID
    random_name = str(uuid.uuid4())
    scene_dir = os.path.join(output_dir, random_name)
    os.makedirs(scene_dir, exist_ok=True)

    W, H = resolution

    meta = {
        "scene_id": random_name,
        "resolution": [int(W), int(H)],
        "focal_length": float(f1),
        "views": [],
        "matches": []
    }

    # --------------------------
    # 1) Save each image + pose
    # --------------------------
    for i, (img, pose) in enumerate(zip(images, poses)):
    # 1) drop alpha if present
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # 2) only scale floats in [0,1]
        if img.dtype == np.uint8:
            img_uint8 = img
        else:
            img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

        img_path = os.path.join(scene_dir, f"view{i}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

        meta["views"].append({
            "image": f"view{i}.png",
            "pose": pose.tolist(),
            "augmentations": None
        })

    # --------------------------------------------
    # 2) For each pair (i,j), create masks & perm
    # --------------------------------------------
    N = len(images)
    for i in range(N):
        for j in range(i + 1, N):
            mask_ij = match_masks[i][j]  # boolean array, length = M
            if mask_ij is None or np.sum(mask_ij) == 0:
                continue

            # uv1_all, uv2_all: shape (M,2), with (u,v) in top-left origin
            uv1_all = np.array(uv_sets[i], dtype=int)  # (M,2)
            uv2_all = np.array(uv_sets[j], dtype=int)  # (M,2)

            # Masked UV coordinates for matched points only
            uv1_matched = uv1_all[mask_ij]  # shape (K,2)
            uv2_matched = uv2_all

            # Build mask1 and mask2 as length W*H arrays of 0/1
            # Initialize to zeros
            mask1 = np.zeros((W * H,), dtype=int)
            mask2 = np.zeros((W * H,), dtype=int)
            # Build permutation array, default -1
            perm = np.full((W * H,), -1, dtype=int)

            # For each matched pair, compute flat indices and set
            for (u1, v1), (u2, v2) in zip(uv1_matched, uv2_matched):
                if not (0 <= u1 < W and 0 <= v1 < H and 0 <= u2 < W and 0 <= v2 < H):
                    # Skip any that fall outside due to rounding errors
                    continue
                idx1 = v1 * W + u1
                idx2 = v2 * W + u2
                mask1[idx1] = 1
                mask2[idx2] = 1
                perm[idx1] = int(idx2)

            # Convert masks and perm to Python lists for JSON
            mask1_list = mask1.tolist()
            mask2_list = mask2.tolist()
            perm_list  = perm.tolist()

            match_entry = {
                "view1": i,
                "view2": j,
                "mask1": mask1_list,
                "mask2": mask2_list,
                "perm": perm_list
            }
            meta["matches"].append(match_entry)

    # --------------------------
    # 3) Write meta.json
    # --------------------------
    with open(os.path.join(scene_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[export_dataset] Exported to {scene_dir} (took {time.time() - t0:.3f} s)")


# =============================================================================
# 12. MAIN PIPELINE
#      Now loops over NUM_PAIRS, choosing a random SceneNN each iteration
# =============================================================================
def main():
    pair_idx = 0
    while pair_idx < NUM_PAIRS:
        t_start = time.time()
        print(f"\n==== Generating pair {pair_idx + 1}/{NUM_PAIRS} ====")
        try:
            # 1) Load a random scene + detect floor (cached if re‐used)
            (scene_mesh, up_axis, plane_normal,
             floor_height, ceiling_height,
             min_bound, max_bound, points_all) = load_scene_mesh_and_detect_floor(
                SCENENN_DIR, BAD_SCENES
            )

            # 2) Sample interior points (once for this scene)
            interior_pts, interior_dists = sample_interior_points(
                scene_mesh=scene_mesh,
                points_all=points_all,
                up_axis=up_axis,
                floor_height=floor_height,
                ceiling_height=ceiling_height,
                min_bound=min_bound,
                max_bound=max_bound,
                num_samples=5000,
                min_clearance=0.05
            )

            # 3) Place objects (once for this scene)
            object_positions, cluster_center = place_objects(
                interior_points=interior_pts,
                interior_distances=interior_dists,
                min_object_distance=0.35,
                max_objects=15
            )

            # 4) Compute camera poses (new random poses for this scene)
            cam_info = compute_camera_poses(
                interior_points=interior_pts,
                cluster_center=cluster_center,
                plane_normal=plane_normal,
                floor_height=floor_height,
                ceiling_height=ceiling_height,
                up_axis=up_axis,
                scene_mesh=scene_mesh,
                target_offset_range=0.5,
                cam_height_min=0.5,
                cam_height_max=3.0,
                min_cluster_dist=1.0,
                max_cluster_dist=3.0,
                min_cam_separation=0.5,
                max_cam_separation=3.0,
                max_angle_diff_deg=65.0,
            )

            pose1 = cam_info["pose1"]
            pose2 = cam_info["pose2"]

            # 5) Load & transform objects into that scene (with caching on raw meshes)
            object_meshes = load_and_transform_objects(object_positions)

            # 6) Visualize the scene, objects, camera frames, etc.
            #visualize_scene(scene_mesh, object_meshes, cam_info, cluster_center)

            # 7) Sample points & find matches between the two views
            I1, I2, xyz_all, uv1_all, match_mask, uv2_all, indices = sample_and_match(
                scene_mesh=scene_mesh,
                object_meshes=object_meshes,
                pose1=pose1,
                pose2=pose2,
                resolution=RESOLUTION,
                f1=F1,
                f2=F2,
                interior_points=interior_pts,
                cluster_center=cluster_center,
                up_axis=up_axis,
                interior_dists=interior_dists,
                num_points=int(RESOLUTION[0]*RESOLUTION[1]*RATIO_OF_POINTS_TO_RENDER)
            )

            #8) Visualize matched points
            # visualize_matches(
            #     I1=I1,
            #     I2=I2,
            #     uv1_all=uv1_all,
            #     uv2_all=uv2_all,
            #     match_mask=match_mask,
            #     resolution=RESOLUTION
            # )

            # 9) Export dataset (timestamps guarantee unique subfolders per pair)
            images = [I1, I2]
            poses = [pose1, pose2]
            uv_sets = [uv1_all, uv2_all]
            xyz_set = [xyz_all, xyz_all]
            match_masks = [[None] * NUM_CAMERAS for _ in range(NUM_CAMERAS)]
            match_masks[0][1] = match_mask
            match_masks[1][0] = match_mask
            indices_set = [indices, indices]

            export_dataset(
                images=images,
                poses=poses,
                uv_sets=uv_sets,
                xyz_set=xyz_set,
                match_masks=match_masks,
                indices_set=indices_set,
                output_dir=OUTPUT_DIR,
                resolution=RESOLUTION,
                f1=F1
            )

            # Pause 1 second so the next timestamp folder differs
            print(f"[export_dataset] Exported (took {time.time() - t_start:.3f}s)")

            # If all operations succeed, increment the counter to proceed to the next pair
            pair_idx += 1

        except RuntimeError as e:
            print(f"\nCaught a RuntimeError: {e}")
            print(f"Retrying generation for pair {pair_idx + 1}.\n")
            continue


if __name__ == "__main__":
    main()
