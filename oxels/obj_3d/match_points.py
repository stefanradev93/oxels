import numpy as np
import trimesh
from typing import Tuple

def sample_random_on_mesh(mesh: trimesh.Trimesh, N_max: int, camera_pose: np.ndarray, f: float, resolution: Tuple[int,int]):
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

    #Samples camera pixels
    u = np.random.uniform(0, resolution[0], size=N_max)
    v = np.random.uniform(0, resolution[1], size=N_max)

    #x,y,z in camera plane (focus is at [0,0,0])
    x_cam = (u - resolution[0]//2)/f
    y_cam = (v - resolution[1]//2)/f
    z_cam = -np.ones_like(x_cam)

    #pixels directions in camera coordiates
    dirs_cam = np.vstack([x_cam, y_cam, z_cam]).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    #dircetions and origins (in mesh coordinates) of rays from camera focus through camera pixels
    dirs = (camera_pose[:3,:3] @ dirs_cam.T).T
    origins = np.tile(camera_pose[:3,3], (N_max,1))

    loc, ray_id, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=dirs,
        multiple_hits=False,  # only the first hit per ray
    )

    return loc, np.stack([u[ray_id], v[ray_id]], axis=1)


def get_visible_matches(mesh: trimesh.Trimesh, xyz: np.ndarray, camera_pose: np.ndarray, f: float, resolution: Tuple[int,int], tol: float = 1e-6):
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
    W,H = resolution

    #Find the lines/directions connecting all the points to the camera
    origins = np.tile(camera_pose[:3, 3], (len(xyz),1))       # (M,3)
    dirs = xyz - origins                       # (M,3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    #Rotate the direction into camera coordinates to get pixel values
    dirs_cam = (np.linalg.inv(camera_pose)[:3,:3] @ dirs.T).T
    u = np.round(-(dirs_cam[:,0] / dirs_cam[:,2]) * f + W//2).astype(int)
    v = np.round(-(dirs_cam[:,1] / dirs_cam[:,2]) * f + H//2).astype(int)

    #Find visible pixels
    match_mask = (u>=0)&(u<W)&(v>=0)&(v<H)

    #Find intersections with the mesh, this time from the camera's viewpoint
    xyz2, ray_id, _ = mesh.ray.intersects_location(
        ray_origins=origins[match_mask], ray_directions=dirs[match_mask], multiple_hits=False
    )

    #It's not a match if the new interesction is far from the original points (e.g., because of obstructions)
    diff = np.linalg.norm(xyz[match_mask][ray_id] - xyz2, axis=1)
    match_mask[ray_id[diff > tol]] = False

    return match_mask, np.stack([u, v], axis=1)[match_mask]
