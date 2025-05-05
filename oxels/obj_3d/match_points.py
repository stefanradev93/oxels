import numpy as np
import trimesh
from typing import Tuple

def sample_random_on_mesh(mesh: trimesh.Trimesh, N_max: int, camera_pose: np.ndarray, f: float, resolution: Tuple[int,int]):
    u = np.random.uniform(0, resolution[0], size=N_max)
    v = np.random.uniform(0, resolution[1], size=N_max)

    # build ray origins & directions for cam1
    #    camera coords: x_cam = (u-cx)/fx, y_cam=(v-cy)/fy, z_cam=-1
    x_cam = (u - resolution[0]//2)/f
    y_cam = (v - resolution[1]//2)/f
    dirs_cam = np.vstack([x_cam, y_cam, -np.ones_like(x_cam)]).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    dirs = (camera_pose[:3,:3] @ dirs_cam.T).T
    origins = np.tile(camera_pose[:3,3], (N_max,1))

    loc, ray_id, _ = mesh.ray.intersects_location(
        ray_origins   = origins,
        ray_directions= dirs,
        multiple_hits = False  # only the first hit per ray
    )

    return loc, np.stack([u[ray_id], v[ray_id]], axis=1)


def get_visible_matches(mesh: trimesh.Trimesh, xyz: np.ndarray, camera_pose: np.ndarray, f: float, resolution: Tuple[int,int], tol: float = 1e-6):
    W,H = resolution
    origins = np.tile(camera_pose[:3, 3], (len(xyz),1))       # (M,3)
    dirs = xyz - origins                       # (M,3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    dirs_cam = (np.linalg.inv(camera_pose)[:3,:3] @ dirs.T).T
    u = -(dirs_cam[:,0] / dirs_cam[:,2]) * f + W//2
    v = -(dirs_cam[:,1] / dirs_cam[:,2]) * f + H//2

    match_mask = (u>=0)&(u<W)&(v>=0)&(v<H)

    xyz2, ray_id, _ = mesh.ray.intersects_location(
        ray_origins    = origins[match_mask],
        ray_directions = dirs[match_mask],
        multiple_hits  = False
    )

    diff = np.linalg.norm(xyz[match_mask][ray_id] - xyz2, axis=1)
    match_mask[ray_id[diff > tol]] = False

    return match_mask, np.stack([u,v], axis=1)[match_mask]