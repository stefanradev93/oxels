import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
from torchvision import transforms
from oxels.models import Contrastive_Model
import hnswlib
import cv2

def crop_image_patches(image: Image.Image, grid, patch_size=256):
    """
    Given an image and a list of (x, y) top-left positions, return
    a list of 256x256 cropped image patches.
    """
    patches = []
    for x, y in grid:
        # Crop box: (left, upper, right, lower)
        box = (x, y, x + patch_size, y + patch_size)
        patch = image.crop(box)
        patches.append(patch)
    return patches

def get_patch_grid(image_width, image_height, patch_size=256):
    """
    Compute a grid of top-left (x, y) coordinates for 256x256 patches
    that fully cover the image with uniform overlap.
    """

    def compute_positions(image_dim):
        # Step 1: Compute number of patches needed
        num_patches = image_dim // patch_size
        remainder = image_dim % patch_size
        if remainder != 0:
            num_patches += 1

        if num_patches == 1:
            return [0]

        # Step 2: Compute stride so patches are evenly spaced to cover the image
        stride = (image_dim - patch_size) / (num_patches - 1)

        # Step 3: Generate start positions
        positions = [int(round(i * stride)) for i in range(num_patches)]
        return positions

    x_positions = compute_positions(image_width)
    y_positions = compute_positions(image_height)

    grid = [(x, y) for y in y_positions for x in x_positions]
    return grid

def visualize_oxels_patch(oxels_tensor, n_oxels_to_show=None):
    """
    Visualize a single oxels tensor (num_oxels, H, W):
    - Show all channels as grayscale images (no RGB composite)
    Returns a list of images (as numpy arrays) for grid plotting.
    """
    oxels = oxels_tensor.detach().cpu()
    C, H, W = oxels.shape
    if n_oxels_to_show is None:
        n_oxels_to_show = C
    imgs = []
    for i in range(min(n_oxels_to_show, C)):
        g = oxels[i]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)  # normalize to [0,1]
        g = g.numpy()
        imgs.append(np.stack([g, g, g], axis=-1))  # (H, W, 3)
    return imgs

def process_patches_with_model(patches, model_ckpt_path, model_config):
    """
    Given a list of PIL.Image patches, load the model from checkpoint and process each patch.
    Returns a list of output tensors (num_oxels, 256, 256) for each patch.
    """
    model = Contrastive_Model.load_from_checkpoint(model_ckpt_path)
    model.freeze()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    outputs = []
    for patch in patches:
        tensor = transform(patch).unsqueeze(0).to(device)  # (1, 3, 256, 256)
        with torch.no_grad():
            out = model(tensor)
        outputs.append(out.squeeze(0).cpu())  # (num_oxels, 256, 256)
    return outputs

def recombine_oxels_patches(patches, grid, image_size, patch_size=256):
    """
    Recombine oxel patches into a full image-sized embedding with smooth linear blending in overlaps.
    
    Args:
        patches: list of torch.Tensor, each (num_oxels, patch_size, patch_size)
        grid: list of (x, y) top-left positions for each patch
        image_size: (H, W) tuple for the full image
        patch_size: int, size of each patch (default 256)
        
    Returns:
        full_embedding: torch.Tensor of shape (num_oxels, H, W)
    """
    H, W = image_size
    num_oxels = patches[0].shape[0]

    device = patches[0].device
    embedding = torch.zeros((num_oxels, H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    # Linear weights from 0→1→0 for smooth horizontal/vertical blending
    weight_x = torch.linspace(0, 1, steps=patch_size // 2)
    weight_x = torch.cat([weight_x, 1 - weight_x], dim=0).view(1, 1, patch_size)  # (1, 1, W)
    
    weight_y = torch.linspace(0, 1, steps=patch_size // 2)
    weight_y = torch.cat([weight_y, 1 - weight_y], dim=0).view(1, patch_size, 1)  # (1, H, 1)
    
    weight_map = weight_x * weight_y  # (1, H, W)

    for patch, (x, y) in zip(patches, grid):
        y_end = min(H, y + patch_size)
        x_end = min(W, x + patch_size)
        patch_h = y_end - y
        patch_w = x_end - x

        # Crop if patch exceeds image bounds
        patch_cropped = patch[:, :patch_h, :patch_w]
        weight_cropped = weight_map[:, :patch_h, :patch_w]

        embedding[:, y:y_end, x:x_end] += patch_cropped * weight_cropped
        weight_sum[:, y:y_end, x:x_end] += weight_cropped

    # Normalize weighted sum
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    full_embedding = embedding / weight_sum

    return full_embedding.cpu()
    

def load_all_image_pairs():
    train_dir = 'Downstream_Image_Matching/train'

    for scene in os.listdir(train_dir):
        scene_path = os.path.join(train_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        pair_csv = os.path.join(scene_path, 'pair_covisibility.csv')
        images_dir = os.path.join(scene_path, 'images')
        if not os.path.exists(pair_csv) or not os.path.isdir(images_dir):
            continue

        with open(pair_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or '-' not in row[0]:
                    continue

                img1_name, img2_name = row[0].split('-')
                img1_path = os.path.join(images_dir, img1_name + '.jpg')
                img2_path = os.path.join(images_dir, img2_name + '.jpg')
                try:
                    with Image.open(img1_path) as img1:
                        img1.load()
                        img1 = img1.convert('RGB')
                    with Image.open(img2_path) as img2:
                        img2.load()
                        img2 = img2.convert('RGB')
                except Exception as e:
                    print(f"Error loading {img1_path} or {img2_path}: {e}")
                    continue

                grid1 = get_patch_grid(img1.width, img1.height)
                grid2 = get_patch_grid(img2.width, img2.height)

                patches1 = crop_image_patches(img1, grid1)
                patches2 = crop_image_patches(img2, grid2)

                # Model config for loading (update as needed)
                model_ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoints/last.ckpt'))
                model_config = dict(
                    data_root="./contrastive_3d_local",  # or appropriate path
                    stage_channels=[16, 16, 32, 32],
                    num_res_blocks=[1, 1, 1, 1],
                    num_oxels=32,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    dropout_stages=[-2, -1],
                    dropout=0.1,
                    attention_stages=[],
                    lr_div_factor=25.0,
                    lr_final_div_factor=1e4,
                    lr_pct_start=0.05,
                    train_batch_size=2,
                    val_batch_size=2,
                    image_size=256,
                    contrastive_loss_weight=0.5,
                )

                processed_patches1 = process_patches_with_model(patches1, model_ckpt_path, model_config)
                # processed_patches2 = process_patches_with_model(patches2, model_ckpt_path, model_config)

                # Recombine patches for image 1
                full_embedding1 = recombine_oxels_patches(processed_patches1, grid1, (img1.height, img1.width), patch_size=256)

                # Visualize the recombined embedding for image 1
                n_oxels_to_show = full_embedding1.shape[0]
                imgs = visualize_oxels_patch(full_embedding1, n_oxels_to_show=n_oxels_to_show)
                n_imgs = len(imgs)
                n_cols = int(np.ceil(np.sqrt(n_imgs)))
                n_rows = int(np.ceil(n_imgs / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
                axes = np.array(axes).reshape(n_rows, n_cols)
                for idx, img in enumerate(imgs):
                    row = idx // n_cols
                    col = idx % n_cols
                    ax = axes[row, col]
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f'Channel {idx}')
                # Hide unused axes
                for idx in range(n_imgs, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].axis('off')
                plt.suptitle(f'Recombined oxel embedding for {img1_name}')
                plt.tight_layout()
                plt.show()

                # --- Efficient KNN matching using hnswlib ---
                # Recombine patches for image 2
                processed_patches2 = process_patches_with_model(patches2, model_ckpt_path, model_config)
                full_embedding2 = recombine_oxels_patches(processed_patches2, grid2, (img2.height, img2.width), patch_size=256)

                # Flatten embeddings: (num_oxels, H, W) -> (H*W, num_oxels)
                H1, W1 = img1.height, img1.width
                H2, W2 = img2.height, img2.width
                oxel1_flat = full_embedding1.permute(1, 2, 0).reshape(-1, full_embedding1.shape[0]).numpy()  # (H1*W1, num_oxels)
                oxel2_flat = full_embedding2.permute(1, 2, 0).reshape(-1, full_embedding2.shape[0]).numpy()  # (H2*W2, num_oxels)

                # Use hnswlib for fast KNN
                ids2 = np.arange(W2*H2)
                dim = oxel2_flat.shape[1]
                tree_2 = hnswlib.Index(space='l2', dim=dim)
                tree_2.init_index(max_elements=2*W2*H2, ef_construction=200, M=16)
                tree_2.add_items(oxel2_flat, ids2)
                tree_2.set_ef(50)

                # Query for 2 nearest neighbors for ratio test
                labels, distances = tree_2.knn_query(oxel1_flat, k=2)
                labels0 = labels[:, 0]
                labels1 = labels[:, 1]
                d0 = distances[:, 0]
                d1 = distances[:, 1]

                # Ratio test (Lowe's): keep matches where d0/d1 < 0.5
                ratio_thresh = 0.3
                good_matches = d0 < ratio_thresh * d1
                matched_train_idx = np.where(good_matches)[0]  # indices in image1
                matched_query_idx = labels0[good_matches]      # indices in image2

                # Estimate fundamental matrix
                #F, inliers = estimate_fundamental_matrix_ransac(x1, y1, x2, y2)

                # Convert flat indices to (y, x) coordinates
                y1, x1 = np.divmod(matched_train_idx, W1)
                y2, x2 = np.divmod(matched_query_idx, W2)

                # Show matches on the original images
                # Generate random colors (N, 3) in RGB
                num_matches = len(x1)
                colors = np.random.rand(num_matches, 3)

                # Show matches on the original images
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                axes[0].imshow(img1)
                axes[1].imshow(img2)

                # Plot each match with its unique color
                for i in range(num_matches):
                    c = colors[i]
                    axes[0].scatter(x1[i], y1[i], c=[c], s=4)
                    axes[1].scatter(x2[i], y2[i], c=[c], s=4)

                axes[0].set_title(f'Matched keypoints in {img1_name}')
                axes[1].set_title(f'Matched keypoints in {img2_name}')
                for ax in axes:
                    ax.axis('off')

                plt.suptitle('Oxel-based HNSWlib matches with color-correspondence')
                plt.tight_layout()
                plt.show()

                print(f"Loaded pair, shapes: {img1.size}, {img2.size}")
                print(f"# of patches: {len(grid1)}")

                break
        break

def estimate_fundamental_matrix_ransac(x1, y1, x2, y2):
    """
    Estimate the fundamental matrix using RANSAC from matched keypoints.
    Args:
        x1, y1: Arrays of x and y coordinates of matched points in image 1.
        x2, y2: Arrays of x and y coordinates of matched points in image 2.
    Returns:
        F: Fundamental matrix (3x3)
        inliers: Boolean mask of inliers selected by RANSAC
    """
    pts1 = np.stack([x1, y1], axis=-1)
    pts2 = np.stack([x2, y2], axis=-1)
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
    inliers = mask.ravel().astype(bool) if mask is not None else None
    return F, inliers

if __name__ == "__main__":
    load_all_image_pairs()
    print("Finished loading image pairs.")