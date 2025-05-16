import numpy as np
import cv2
import trimesh
from typing import Tuple


def render(scene: trimesh.Scene, pose: np.ndarray, resolution: Tuple[int, int], f: float, visible: bool = False):
    # set the camera
    scene.camera_transform = pose
    scene.camera.resolution = resolution
    scene.camera.focal = (f, f)

    # render to PNG bytes
    png_bytes = scene.save_image(resolution=resolution, visible=visible)

    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255


def get_pose(M: np.ndarray, d: np.ndarray):
    pose = np.zeros((4, 4))
    pose[3, 3] = 1
    pose[:3, :3] = M
    pose[:3, 3] = d
    return pose
