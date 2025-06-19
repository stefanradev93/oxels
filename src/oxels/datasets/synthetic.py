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
    loaded lazily from exported folders of the form:
      root/
        train/ (or val, test)
          <run_uuid>/
            <scene_uuid>/
              view0.png
              view1.png
              meta.json

    Applies random 256x256 crop with augmentation. Matches outside the crop are removed
    and all output arrays are sized to the crop (N = crop_size^2), so downstream loss sees
    consistent flatten sizes without needing any change there.
    """
    pil_augs = T.Compose([
        T.ColorJitter(brightness=0.3, hue=(-0.1, 0.1), saturation=0.3),
        T.RandomAutocontrast(p=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(1.0, 3.0))], p=0.1),
        T.RandomPosterize(bits=5, p=0.1),
        T.RandomEqualize(p=0.1),
        T.RandomGrayscale(p=0.05),
    ])
    to_tensor = T.ToTensor()

    def __init__(
        self,
        root: str = "contrastive_3d",
        split: str = "train",
        crop_size: int = 256,
        log_every: int = 100,
    ):
        super().__init__()
        self.crop_size = crop_size
        base = Path(root) / split
        if not base.exists():
            raise FileNotFoundError(f"Split folder not found: {base}")

        # collect sample pointers (no large arrays)
        self.samples = []  # (v1_path, v2_path, meta_path_str, match_idx)
        total_scenes = 0
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_dirs = []
            for scene_dir in sorted(run_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                meta_path = scene_dir / "meta.json"
                if not meta_path.exists():
                    continue
                meta = _load_meta(str(meta_path))
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
        if not self.samples:
            raise RuntimeError(f"No samples found under {base}")
        print(f"[Contrastive3D] Indexed {total_scenes} scenes → {len(self.samples)} samples total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        v1_str, v2_str, meta_path_str, match_idx = self.samples[idx]

        # 1) load + augment PIL images
        img1 = self.pil_augs(Image.open(v1_str).convert("RGB"))
        img2 = self.pil_augs(Image.open(v2_str).convert("RGB"))

        # 2) random 256×256 crop
        orig_w, orig_h = img1.width, img1.height
        i, j, h_crop, w_crop = T.RandomCrop.get_params(img1, (self.crop_size, self.crop_size))
        img1_crop = TF.crop(img1, i, j, h_crop, w_crop)
        img2_crop = TF.crop(img2, i, j, h_crop, w_crop)

        # 3) to tensor [C,H,W]
        view1 = self.to_tensor(img1_crop)
        view2 = self.to_tensor(img2_crop)

        # 4) load metadata
        meta         = _load_meta(meta_path_str)
        m            = meta["matches"][match_idx]
        orig_perm    = np.asarray(m["perm"],  dtype=np.int64)
        orig_mask1   = np.asarray(m["mask1"], dtype=bool)
        orig_mask2   = np.asarray(m["mask2"], dtype=bool)

        # flatten coords for full image
        idxs = np.arange(orig_w * orig_h)
        r1   = idxs // orig_w
        c1   = idxs %  orig_w
        r2   = orig_perm // orig_w
        c2   = orig_perm %  orig_w

        # which fall inside this crop
        in1   = (r1 >= i) & (r1 < i+h_crop) & (c1 >= j) & (c1 < j+w_crop)
        in2   = (r2 >= i) & (r2 < i+h_crop) & (c2 >= j) & (c2 < j+w_crop)
        valid = in1 & in2

        # allocate per-crop arrays
        N_crop      = h_crop * w_crop
        perm_crop   = np.full((N_crop,), -1,       dtype=np.int64)
        mask1_crop  = np.zeros((N_crop,), dtype=bool)
        mask2_crop  = np.zeros((N_crop,), dtype=bool)
        flags_crop  = np.zeros((N_crop,), dtype=bool)

        # remap each valid pixel into crop-space
        valid_idxs = np.nonzero(valid)[0]
        r1v   = r1[valid_idxs] - i
        c1v   = c1[valid_idxs] - j
        pos1  = r1v * w_crop + c1v

        r2v   = r2[valid_idxs] - i
        c2v   = c2[valid_idxs] - j
        pos2  = r2v * w_crop + c2v

        perm_crop[pos1]   = pos2
        flags_crop[pos1]  = True
        mask1_crop[pos1]  = orig_mask1[valid_idxs]
        mask2_crop[pos1]  = orig_mask2[orig_perm[valid_idxs]]

        # ─── NEW CLAMP TO [0, N_crop-1] ─────────────────────────────────────────
        # avoids any -1 or >N_crop indices at gather time
        perm_crop = np.clip(perm_crop, 0, N_crop - 1)
        # ─────────────────────────────────────────────────────────────────────────

        # to tensors
        perm_tensor   = torch.from_numpy(perm_crop)                     # LongTensor [N_crop]
        flags_tensor  = torch.from_numpy(flags_crop.astype(np.uint8)).bool()
        mask1_tensor  = torch.from_numpy(mask1_crop.astype(np.uint8)).bool()
        mask2_tensor  = torch.from_numpy(mask2_crop.astype(np.uint8)).bool()

        return view1, view2, perm_tensor, flags_tensor, mask1_tensor, mask2_tensor
