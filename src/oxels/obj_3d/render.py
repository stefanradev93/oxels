import numpy as np
import trimesh
from typing import Tuple, Union
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
import time
import random

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

_app_initialized = False
_renderer_initialized= False
_render_count = 0  # Keep track of which sample we're on


# Working Render, No intensity change
def render_faster(scene_mesh, object_meshes, pose, resolution, f,
                  interior_points, interior_dists, debug=False):

    global _renderer_initialized, _offscreen_renderer, _render_count
    t_start = time.time()

    width, height = resolution

    # 1) Initialize OffscreenRenderer once
    if not _renderer_initialized:
        _offscreen_renderer = rendering.OffscreenRenderer(width, height)
        _renderer_initialized = True

    # 2) Use OffscreenRenderer's scene
    scene = _offscreen_renderer.scene

    # 3) Geometry setup
    scene_mesh.compute_vertex_normals()
    for obj in object_meshes:
        obj.compute_vertex_normals()
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = (1.0, 1.0, 1.0, 1.0)
    material.base_reflectance = 0.5

    scene.clear_geometry()
    scene.add_geometry("scene", scene_mesh, material)
    for i, obj in enumerate(object_meshes):
        scene.add_geometry(f"object_{i}", obj, material)

    # 4) lighting variation
    min_open = 0.1
    mask = (interior_dists > min_open)
    if not np.any(mask):
        mask = np.ones(len(interior_points), dtype=bool)
    pts = interior_points[mask]
    dists = interior_dists[mask]
    cam_pos = pose[:3, 3]
    cam_forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    cam_forward /= (np.linalg.norm(cam_forward) + 1e-8)

    # Multiple light position selection strategies with increased randomness
    selection_strategy = np.random.choice(['weighted', 'uniform', 'far_bias', 'close_bias', 'fully_random'])
    
    if selection_strategy == 'weighted':
        # Original weighted selection with added randomness
        vecs = pts - cam_pos
        dist_cam = np.linalg.norm(vecs, axis=1) + 1e-8
        w_dist = 1.0 / dist_cam
        dirs = vecs / dist_cam[:, None]
        w_align = np.clip(dirs.dot(cam_forward), 0.0, 1.0)
        w_open = np.clip(dists, 0.0, None)
        weights = w_dist * w_align * w_open
        # Add random noise to weights to increase variability
        weights += np.random.uniform(0, 0.3 * np.max(weights), len(weights))
        if np.all(weights <= 0):
            weights = w_open + 1e-6
        probs = weights / np.sum(weights)
        idx = np.random.choice(len(pts), p=probs)
    elif selection_strategy == 'uniform':
        # Uniform random selection
        idx = np.random.choice(len(pts))
    elif selection_strategy == 'far_bias':
        # Bias toward farther points
        dist_cam = np.linalg.norm(pts - cam_pos, axis=1) + 1e-8
        weights = dist_cam / np.sum(dist_cam)
        idx = np.random.choice(len(pts), p=weights)
    elif selection_strategy == 'close_bias':
        # Bias toward closer points
        dist_cam = np.linalg.norm(pts - cam_pos, axis=1) + 1e-8
        weights = (1.0 / dist_cam)
        weights = weights / np.sum(weights)
        idx = np.random.choice(len(pts), p=weights)
    else:  # fully_random
        # Completely ignore interior points and use random positions around camera
        random_radius = np.random.uniform(2.0, 10.0)
        random_angles = np.random.uniform(0, 2*np.pi, 2)  # spherical coordinates
        random_elevation = np.random.uniform(-np.pi/3, np.pi/3)  # limit elevation
        x = random_radius * np.cos(random_elevation) * np.cos(random_angles[0])
        y = random_radius * np.sin(random_elevation)
        z = random_radius * np.cos(random_elevation) * np.sin(random_angles[0])
        light_pos = cam_pos + np.array([x, y, z])
        idx = None  # Skip the normal selection
    
    if idx is not None:
        light_pos = pts[idx]
    
    # Directional variation with more randomness
    direction_mode = np.random.choice(['to_camera', 'from_camera', 'side_lighting', 'random', 'hemisphere', 'scattered'])
    
    if direction_mode == 'to_camera':
        base_dir = (cam_pos - light_pos)
    elif direction_mode == 'from_camera':
        base_dir = (light_pos - cam_pos)
    elif direction_mode == 'side_lighting':
        # Perpendicular to camera-light vector
        cam_to_light = light_pos - cam_pos
        up = np.array([0.0, 1.0, 0.0])
        side = np.cross(cam_to_light, up)
        side = side / (np.linalg.norm(side) + 1e-8)
        base_dir = side
    elif direction_mode == 'hemisphere':
        # Random direction in upper hemisphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi/2)  # upper hemisphere only
        base_dir = np.array([
            np.sin(phi) * np.cos(theta),
            np.cos(phi),
            np.sin(phi) * np.sin(theta)
        ])
    elif direction_mode == 'scattered':
        # Scattered around a general downward direction
        general_down = np.array([0.0, -1.0, 0.0])
        scatter_angle = np.random.uniform(0, np.pi/2)  # up to 90 degrees scatter
        random_axis = np.random.randn(3)
        random_axis = random_axis / (np.linalg.norm(random_axis) + 1e-8)
        # Rotate general_down around random_axis by scatter_angle
        cos_a = np.cos(scatter_angle)
        sin_a = np.sin(scatter_angle)
        base_dir = (cos_a * general_down + 
                   sin_a * np.cross(random_axis, general_down) +
                   (1 - cos_a) * np.dot(random_axis, general_down) * random_axis)
    else:  # random
        # Completely random direction
        base_dir = np.random.randn(3)
    
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-8)
    
    # Perturbation amount for more randomness
    perturbation_strength = np.random.uniform(15.0, 45.0)  # 15° to 45° instead of 5° to 25°
    
    up_ref = np.array([0.0, 1.0, 0.0])
    if abs(base_dir.dot(up_ref)) > 0.9:
        up_ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(base_dir, up_ref); u /= np.linalg.norm(u) + 1e-8
    v = np.cross(base_dir, u);       v /= np.linalg.norm(v) + 1e-8
    
    yaw = np.deg2rad(np.random.uniform(-perturbation_strength, perturbation_strength))
    pitch = np.deg2rad(np.random.uniform(-perturbation_strength, perturbation_strength))
    perturbed_sun_dir = base_dir + np.tan(yaw)*u + np.tan(pitch)*v
    perturbed_sun_dir = perturbed_sun_dir / (np.linalg.norm(perturbed_sun_dir) + 1e-8)

    # Modified lighting profiles with intermediate shadow option
    shadow_mode = np.random.choice(['medium', 'soft', 'medium_hard_blend', 'no_shadows'], 
                                  p=[0.35, 0.25, 0.35, 0.05])
    
    intrinsic_mat = np.array([[f, 0, width/2.0],
                              [0, f, height/2.0],
                              [0, 0, 1.0]], dtype=np.float64)
    eye = pose[:3, 3]
    forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    center = eye + forward
    up = pose[:3, :3] @ np.array([0.0, 1.0, 0.0])
    scene.camera.set_projection(intrinsic_mat, 0.1, 1000.0, width, height)
    scene.camera.look_at(center, eye, up)

    if shadow_mode == 'medium_hard_blend':
        chosen_profile = scene.LightingProfile.HARD_SHADOWS

        # — render once with medium shadows —
        scene.set_lighting(scene.LightingProfile.MED_SHADOWS,
                           perturbed_sun_dir.astype(np.float32))
        med_o3d = _offscreen_renderer.render_to_image()
        med_img = np.asarray(med_o3d)

        # — render again with hard shadows —
        scene.set_lighting(scene.LightingProfile.HARD_SHADOWS,
                           perturbed_sun_dir.astype(np.float32))
        hard_o3d = _offscreen_renderer.render_to_image()
        hard_img = np.asarray(hard_o3d)

        # — blend 50/50 and rotate into final image —
        blended = ((med_img.astype(np.float32) + hard_img.astype(np.float32)) / 2.0
                  ).clip(0, 255).astype(np.uint8)
        img = np.rot90(np.rot90(blended, k=1), k=1)
    else:
        if shadow_mode == 'medium':
            chosen_profile = scene.LightingProfile.MED_SHADOWS
        elif shadow_mode == 'soft':
            chosen_profile = scene.LightingProfile.SOFT_SHADOWS
        else:  # no_shadows
            chosen_profile = scene.LightingProfile.NO_SHADOWS
    
        scene.set_lighting(chosen_profile, perturbed_sun_dir.astype(np.float32))

        # 6) Render
        img_o3d = _offscreen_renderer.render_to_image()
        img = np.asarray(img_o3d)
        img = np.rot90(img, k=1)
        img = np.rot90(img, k=1)

    # debug info
    if debug:
        print(f"[DEBUG] Rendered image shape: {img.shape}")
        print(f"[DEBUG] Selection strategy: {selection_strategy}")
        print(f"[DEBUG] Direction mode: {direction_mode}")
        print(f"[DEBUG] Perturbation: ±{perturbation_strength:.1f}°")
        print(f"[DEBUG] Final light direction: {perturbed_sun_dir}")
        print(f"[DEBUG] Shadow mode: {shadow_mode}")
        print(f"[DEBUG] Lighting profile: {chosen_profile}")

    _render_count += 1
    t_end = time.time()
    print(f"[render_faster] Sample {_render_count} rendered in {t_end - t_start:.3f}s")

    return img

def get_pose(M: np.ndarray, d: np.ndarray):
    pose = np.zeros((4, 4))
    pose[3, 3] = 1
    pose[:3, :3] = M
    pose[:3, 3] = d
    return pose
