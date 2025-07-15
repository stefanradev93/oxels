import json
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


@lru_cache(maxsize=128)
def _load_meta(meta_path_str: str) -> dict:
    """
    Cached loader for scene metadata JSON.
    """
    return json.loads(Path(meta_path_str).read_text())


class Contrastive3D(Dataset):
    """
    A dataset of (view1, view2, perm, flags, mask1, mask2)
    loaded lazily from exported folders. Caches the file index for fast startup.
    This version assumes images are already 256x256 and doesn't do random cropping.
    """
    def __init__(
        self,
        root: str = "./contrastive_3d_local",
        split: str = "train",
    ):
        super().__init__()
        
        # Color augmentations only
        self.pil_augs = T.Compose([
            T.ColorJitter(brightness=0.3, hue=(-0.1, 0.1), saturation=0.3),
            T.RandomAutocontrast(p=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(1.0, 3.0))], p=0.1),
            T.RandomPosterize(bits=5, p=0.1),
            T.RandomEqualize(p=0.1),
            T.RandomGrayscale(p=0.05),
        ])
        self.to_tensor = T.ToTensor()

        base = Path(root) / split
        if not base.exists():
            raise FileNotFoundError(f"Split folder not found: {base}")

        # Use the cached index of samples if it exists
        cache_path = base / f"{split}_samples.pt"
        if cache_path.exists():
            print(f"Loading cached samples from {cache_path}...")
            self.samples = torch.load(cache_path)
            print(f"Loaded {len(self.samples)} samples from cache.")
            return

        print(f"Cache not found at {cache_path}. Indexing from scratch...")
        self.samples = []
        total_scenes = 0
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir(): continue
            for scene_dir in sorted(run_dir.iterdir()):
                if not scene_dir.is_dir(): continue
                meta_path = scene_dir / "meta.json"
                if not meta_path.exists(): continue
                try:
                    meta = _load_meta(str(meta_path))
                except json.JSONDecodeError:
                    continue # Skip corrupted files
                views = meta.get("views", [])
                view_files = [v["image"] for v in views]
                matches = meta.get("matches", [])
                for mi in range(len(matches)):
                    self.samples.append((
                        str(scene_dir / view_files[matches[mi]["view1"]]),
                        str(scene_dir / view_files[matches[mi]["view2"]]),
                        str(meta_path),
                        mi,
                    ))
                total_scenes += 1
        
        if not self.samples: raise RuntimeError(f"No samples found under {base}")
        print(f"Saving indexed samples to cache at {cache_path}...")
        torch.save(self.samples, cache_path)
        print(f"[Contrastive3D_Fixed] Indexed {total_scenes} scenes -> {len(self.samples)} samples total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        v1_str, v2_str, meta_path_str, match_idx = self.samples[idx]

        # 1. Load original data
        img1 = Image.open(v1_str).convert("RGB")
        img2 = Image.open(v2_str).convert("RGB")
        orig_w, orig_h = img1.width, img1.height
        
        # Ensure images are 256x256
        if img1.size != (256, 256) or img2.size != (256, 256):
            img1 = img1.resize((256, 256))
            img2 = img2.resize((256, 256))

        # 2. Apply color augmentations
        img1 = self.pil_augs(img1)
        img2 = self.pil_augs(img2)

        # 3. Convert to tensors and flip horizontally
        img1_tensor = self.to_tensor(img1)
        img2_tensor = self.to_tensor(img2)
        view1 = TF.hflip(img1_tensor)
        view2 = TF.hflip(img2_tensor)

        # 4. Load metadata and get point coordinates
        meta = _load_meta(meta_path_str)
        m = meta["matches"][match_idx]
        
        # Create tensors for permutation and masks
        N = 256 * 256
        perm = torch.full((N,), -1, dtype=torch.int64)
        flags = torch.zeros(N, dtype=torch.bool)
        mask1 = torch.zeros(N, dtype=torch.bool)
        mask2 = torch.zeros(N, dtype=torch.bool)

        # Convert original indices to 256x256 space
        orig_perm = torch.from_numpy(np.asarray(m["perm"], dtype=np.int64))
        orig_mask1 = torch.from_numpy(np.asarray(m["mask1"], dtype=bool))
        
        # Get valid points and their coordinates in original space
        valid_p1_indices = torch.where(orig_mask1)[0]
        valid_p2_indices = orig_perm[valid_p1_indices]
        
        # Convert to 2D coordinates in original space
        p1_y_orig = valid_p1_indices // orig_w
        p1_x_orig = valid_p1_indices % orig_w
        p2_y_orig = valid_p2_indices // orig_w
        p2_x_orig = valid_p2_indices % orig_w
        
        # Scale coordinates to 256x256 space
        p1_y = (p1_y_orig.float() * 256 / orig_h).long()
        p1_x = (p1_x_orig.float() * 256 / orig_w).long()
        p2_y = (p2_y_orig.float() * 256 / orig_h).long()
        p2_x = (p2_x_orig.float() * 256 / orig_w).long()
        
        # Convert back to 1D indices in 256x256 space
        p1_indices = p1_y * 256 + p1_x
        p2_indices = p2_y * 256 + p2_x
        
        # Clip indices to valid range and filter out invalid ones
        valid_mask = (p1_indices >= 0) & (p1_indices < N) & (p2_indices >= 0) & (p2_indices < N)
        p1_indices = p1_indices[valid_mask]
        p2_indices = p2_indices[valid_mask]
        
        # Set masks and permutation
        mask1[p1_indices] = True
        mask2[p2_indices] = True
        perm[p1_indices] = p2_indices
        flags[p1_indices] = True

        # CRITICAL FIX: Fill non-shared pixels with random permutations
        # This matches the ImageNet PerspectiveTransform behavior
        non_shared_indices = torch.where(perm == -1)[0]
        if len(non_shared_indices) > 0:
            # Generate random indices for non-shared pixels
            random_indices = torch.randint(0, N, size=(len(non_shared_indices),), dtype=torch.int64)
            perm[non_shared_indices] = random_indices
            # Set flags to False for non-shared pixels (no geometric correspondence)
            flags[non_shared_indices] = False
            # Set mask1 to True for all pixels (all pixels in view1 are valid)
            mask1[non_shared_indices] = True
            # Set mask2 to True for all pixels (all pixels in view2 are valid)
            mask2[random_indices] = True

        # Ensure all indices are within valid range
        perm = torch.clamp(perm, 0, N - 1)

        return view1, view2, perm, flags, mask1, mask2 