import numpy as np
import trimesh
from typing import Tuple, Union
import open3d as o3d
import time

# Old (slower) rendering version
def render(scene, pose, resolution, f, visible=False):
    t0 = time.perf_counter()
    scene.camera_transform = pose
    scene.camera.resolution = resolution
    scene.camera.focal = (f, f)
    png_bytes = scene.save_image(resolution=resolution, visible=visible)
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255

def render_faster(scene_mesh, object_meshes, pose: np.ndarray, resolution: tuple[int,int], f: float):
    """
    Render Open3D meshes so that the camera is *exactly* at `pose`
    (world‐to‐camera = inv(pose)), with focal length f and image size.
    """

    w, h = resolution
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)

    # 2) Add all geometry
    for mesh in [scene_mesh] + object_meshes:
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.6, 0.6, 0.6])
        vis.add_geometry(mesh)

    vis.poll_events()
    vis.update_renderer()

    # 3) Build pinhole intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    intrinsic.set_intrinsics(width=w, height=h, fx=f, fy=f, cx=w/2, cy=h/2)

    # 4) Compute world‐to‐camera extrinsic
    extrinsic = np.linalg.inv(pose)

    # 5) Package into PinholeCameraParameters
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrinsic
    cam_params.extrinsic = extrinsic

    # 6) Apply them directly to the view control
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    # 7) Capture and return
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return img  # float image in [0,1]



def get_pose(M: np.ndarray, d: np.ndarray):
    pose = np.zeros((4, 4))
    pose[3, 3] = 1
    pose[:3, :3] = M
    pose[:3, 3] = d
    return pose
