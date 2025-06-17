import numpy as np
import trimesh
from typing import Tuple
import open3d as o3d
import time
from trimesh.ray.ray_pyembree import RayMeshIntersector


def sample_random_on_mesh(
    mesh: trimesh.Trimesh, N_max: int, camera_pose: np.ndarray, f: float, resolution: Tuple[int, int]
):
    """
    Sample random points on the mesh that visible from a given camera view

    Parameters
    ----------
    mesh        : trimesh.Trimesh
        The scene
    N_max       : int
        Maximum number of visible pixel to be selected
    camera_pose : np.ndarray of floats of shape (4,4)
        The camera pose matrix (3x3 camera orientation matrix + 1x3 camera position vector + [0,0,0,1] row)
    f           : float
        Camera's focal length
    resolution  : Tuple[int,int]
        (Width,Height) in pixels

    Returns
    -------
    np.ndarray of float of shape (M,3)
        3D points on the mesh (M <= N_max)
    np.ndarray int of of shape (M,2)
        corresponding pixels as seen by the camera
    """

    # Samples camera pixels
    u = np.random.uniform(0, resolution[0], size=N_max)
    v = np.random.uniform(0, resolution[1], size=N_max)

    # x,y,z in camera plane (focus is at [0,0,0])
    x_cam = (u - resolution[0] // 2) / f
    y_cam = (v - resolution[1] // 2) / f
    z_cam = -np.ones_like(x_cam)

    # pixels directions in camera coordiates
    dirs_cam = np.vstack([x_cam, y_cam, z_cam]).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    # dircetions and origins (in mesh coordinates) of rays from camera focus through camera pixels
    dirs = (camera_pose[:3, :3] @ dirs_cam.T).T
    origins = np.tile(camera_pose[:3, 3], (N_max, 1))

    loc, ray_id, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=dirs,
        multiple_hits=False,  # only the first hit per ray
    )

    return loc, np.stack([u[ray_id], v[ray_id]], axis=1)

def sample_random_on_mesh_fast_o3d(
    ray_scene: o3d.t.geometry.RaycastingScene,
    triangle_mesh: o3d.geometry.TriangleMesh,
    N_max: int,
    camera_pose: np.ndarray,
    f: float,
    resolution: tuple[int, int],
    tol: float = 1e-6
):
    t0 = time.perf_counter()
    point_sample_count: int = N_max*5
    # Step 1: Sample surface points
    sampled_points = triangle_mesh.sample_points_uniformly(number_of_points=point_sample_count)
    sampled_points_np = np.asarray(sampled_points.points)
    t1 = time.perf_counter()

    # Step 2: Transform to camera coordinates
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    points_cam = (sampled_points_np - t) @ R
    z = -points_cam[:, 2]
    valid = z > 0
    points_cam = points_cam[valid]
    sampled_points_np = sampled_points_np[valid]
    z = z[valid]
    t2 = time.perf_counter()

    # Step 3: Project into 2D
    u = (points_cam[:, 0] * f / z) + resolution[0] / 2
    v = (points_cam[:, 1] * f / z) + resolution[1] / 2
    in_frame = (u >= 0) & (u < resolution[0]) & (v >= 0) & (v < resolution[1])
    u = u[in_frame]
    v = v[in_frame]
    visible_points = sampled_points_np[in_frame]
    t3 = time.perf_counter()

    # Step 4: Raycasting using Open3D
    ray_origins = np.tile(camera_pose[:3, 3], (len(visible_points), 1))
    ray_dirs = visible_points - ray_origins
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)

    rays = np.hstack((ray_origins, ray_dirs)).astype(np.float32)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = ray_scene.cast_rays(rays_tensor)

    # Step 5: Filter hits
    hit_mask = hits['t_hit'].isfinite().numpy()
    hit_pts = ray_origins + hits['t_hit'].numpy()[..., None] * ray_dirs
    dists = np.linalg.norm(visible_points - hit_pts, axis=1)
    ok_hits = (dists < tol) & hit_mask

    final_visible_points = visible_points[ok_hits]
    final_uv = np.stack([u[ok_hits], v[ok_hits]], axis=1)
    t4 = time.perf_counter()

    # Step 6: Subsample if needed
    if len(final_visible_points) > N_max:
        idx = np.random.choice(len(final_visible_points), N_max, replace=False)
        final_visible_points = final_visible_points[idx]
        final_uv = final_uv[idx]

    # === Timing Output ===
    # print("Visibility-aware sampling (Open3D):")
    # print(f"- Surface sampling        : {t1 - t0:.4f}s")
    # print(f"- Transform & filtering   : {t2 - t1:.4f}s")
    # print(f"- Projection              : {t3 - t2:.4f}s")
    # print(f"- Raycasting visibility   : {t4 - t3:.4f}s")
    # print(f"==> Total                 : {t4 - t0:.4f}s")

    return final_visible_points, final_uv.astype(int)


def get_visible_matches(
    mesh: trimesh.Trimesh,
    xyz: np.ndarray,
    camera_pose: np.ndarray,
    f: float,
    resolution: Tuple[int, int],
    tol: float = 1e-6,
):
    """
    Finds all the 3D points on the mesh that are visible from a given camera view

    Parameters
    ----------
    mesh        : trimesh.Trimesh
        The scene
    xyz         : np.ndarray of shape (M,3)
        Point on the mesh
    camera_pose : np.ndarray of floats of shape (4,4)
        The camera pose matrix (3x3 camera orientation matrix + 1x3 camera position vector + [0,0,0,1] row)
    f           : float
        Camera's focal length
    resolution  : Tuple[int,int]
        (Width,Height) in pixels
    tol         : float
        Tollerance on 3D matches to be considered a match

    Returns
    -------
    np.ndarray of bool shape (M,)
        Matches mask
    np.ndarray of int of shape (L,2)
        Matches as seens by the camera (L <= M)
    """
    W, H = resolution

    # Find the lines/directions connecting all the points to the camera
    origins = np.tile(camera_pose[:3, 3], (len(xyz), 1))  # (M,3)
    dirs = xyz - origins  # (M,3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # Rotate the direction into camera coordinates to get pixel values
    dirs_cam = (np.linalg.inv(camera_pose)[:3, :3] @ dirs.T).T
    u = np.round(-(dirs_cam[:, 0] / dirs_cam[:, 2]) * f + W // 2).astype(int)
    v = np.round(-(dirs_cam[:, 1] / dirs_cam[:, 2]) * f + H // 2).astype(int)

    # Find visible pixels
    match_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # Find intersections with the mesh, this time from the camera's viewpoint
    xyz2, ray_id, _ = mesh.ray.intersects_location(
        ray_origins=origins[match_mask], ray_directions=dirs[match_mask], multiple_hits=False
    )

    # It's not a match if the new interesction is far from the original points (e.g., because of obstructions)
    diff = np.linalg.norm(xyz[match_mask][ray_id] - xyz2, axis=1)
    match_mask[ray_id[diff > tol]] = False

    return match_mask, np.stack([u, v], axis=1)[match_mask]


def get_visible_matches_with_indices_o3d(
    ray_scene: o3d.t.geometry.RaycastingScene,
    xyz: np.ndarray,
    camera_pose: np.ndarray,
    f: float,
    resolution: Tuple[int, int],
    tol: float = 1e-6
):
    print("version (Open3D raycasting)")
    t0 = time.perf_counter()

    W, H = resolution

    # === Step 1: Compute ray directions and projection ===
    origins = np.tile(camera_pose[:3, 3], (len(xyz), 1))  # (M, 3)
    dirs = xyz - origins
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    # Project to 2D
    R_inv = np.linalg.inv(camera_pose[:3, :3])
    dirs_cam = (R_inv @ dirs.T).T
    u = np.round(-(dirs_cam[:, 0] / dirs_cam[:, 2]) * f + W // 2).astype(int)
    v = np.round(-(dirs_cam[:, 1] / dirs_cam[:, 2]) * f + H // 2).astype(int)

    match_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    t1 = time.perf_counter()

    # === Step 2: Ray intersection with Open3D ===
    xyz_visible = xyz[match_mask]
    origins_visible = origins[match_mask]
    dirs_visible = dirs[match_mask]

    rays = np.hstack((origins_visible, dirs_visible)).astype(np.float32)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = ray_scene.cast_rays(rays_tensor)

    hit_mask = hits['t_hit'].isfinite().numpy()
    ray_ids = np.where(hit_mask)[0]
    hit_pts = origins_visible[ray_ids] + hits['t_hit'].numpy()[ray_ids][:, None] * dirs_visible[ray_ids]

    t2 = time.perf_counter()

    # === Step 3: Occlusion filtering using distance ===
    ray_xyz_indices = np.where(match_mask)[0][ray_ids]
    original_points = xyz[ray_xyz_indices]
    dist_diff = np.linalg.norm(original_points - hit_pts, axis=1)
    valid_hits = dist_diff <= tol

    final_mask = np.zeros_like(match_mask)
    final_mask[ray_xyz_indices[valid_hits]] = True

    pixel_coords = np.stack([u[final_mask], v[final_mask]], axis=1)
    visible_xyz_indices = np.where(final_mask)[0]

    t3 = time.perf_counter()

    # === Debug profiling ===
    # print("Timing breakdown (Open3D):")
    # print(f"- Direction + projection : {t1 - t0:.4f}s")
    # print(f"- Ray intersection        : {t2 - t1:.4f}s")
    # print(f"- Occlusion filtering     : {t3 - t2:.4f}s")
    # print(f"==> Total                 : {t3 - t0:.4f}s")

    return final_mask, pixel_coords, visible_xyz_indices