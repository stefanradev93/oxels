import os
from collections.abc import Sequence
from typing import Literal
from pathlib import Path
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import lightning as L
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import WandbLogger

import wandb

from oxels.networks import SimpleUNet
from oxels.models.base_model import BaseModel
from oxels.datasets import DGPerspectiveDataset
from oxels.datasets.dg import MultiDomainDataset
from oxels.datasets.dg.mnist import transforms as dtransforms
from oxels.models.metrics_mixin import MetricsMixin
from oxels.callbacks import ShowOxels



class ColoredMNISTModel(BaseModel):
    def __init__(
        self,
        *,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        num_res_blocks: Sequence[int] = (2, 2, 2, 4),
        num_oxels: int = 64,
        num_norm_groups: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.004,
        dropout_stages: Sequence[int],
        dropout: float = 0.1,
        attention_stages: Sequence[int],
        lr_div_factor: float = 25.0,
        lr_final_div_factor: float = 1e4,
        lr_pct_start: float = 0.05,
        train_batch_size: int,
        val_batch_size: int,
        image_size: int = 256,
        contrastive_loss_weight: float = 0.5,
        frac_keep: float = 0.125,
    ):
        num_stages = len(stage_channels)
        has_attention = [False] * num_stages
        for stage in attention_stages:
            has_attention[stage] = True

        residual_dropout = [0.0] * num_stages
        for stage in dropout_stages:
            residual_dropout[stage] = dropout

        backbone = SimpleUNet(
            height=image_size,
            width=image_size,
            in_channels=3,
            out_channels=num_oxels,
            channels_of_stage=stage_channels,
            has_attention=has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=num_norm_groups,
            residual_dropout=residual_dropout,
        )
        super().__init__(
            backbone=backbone,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_div_factor=lr_div_factor,
            lr_final_div_factor=lr_final_div_factor,
            lr_pct_start=lr_pct_start,
            contrastive_loss_weight=contrastive_loss_weight,
        )
        self.frac_keep = frac_keep
        self.save_hyperparameters()

    def train_dataloader(self):
        dataset = DGPerspectiveDataset(
            path="datasets",
            dataset="ColoredMNIST",
            h=self.hparams.image_size,
            w=self.hparams.image_size,
            split="train",
            domain_split="id",
            seed=0,
            augmentations=transforms.Compose([
                transforms.Resize((32, 32))
            ]),
            frac_keep=self.frac_keep,
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,#len(os.sched_getaffinity(0)),
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = DGPerspectiveDataset(
            path="datasets",
            dataset="ColoredMNIST",
            h=self.hparams.image_size,
            w=self.hparams.image_size,
            split="val",
            domain_split="id",
            seed=0,
            augmentations=transforms.Compose([
                transforms.Resize((32, 32))
            ]),
            frac_keep=self.frac_keep,
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,#len(os.sched_getaffinity(0)),
            drop_last=False,
        )

    def val_dataloader_ood(self):
        dataset = DGPerspectiveDataset(
            path="datasets",
            dataset="ColoredMNIST",
            h=self.hparams.image_size,
            w=self.hparams.image_size,
            split="val",
            domain_split="ood",
            seed=0,
            augmentations=transforms.Compose([
                transforms.Resize((32, 32))
            ]),
            frac_keep=self.frac_keep,
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,#len(os.sched_getaffinity(0)),
            drop_last=False,
        )


class AttentionClassifier(nn.Module):
    def __init__(self, num_classes: int, num_heads: int, channel_dim: int):
        super().__init__()
        assert channel_dim % num_heads == 0, "channel_dim must be divisible by num_heads"
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.c_dim = channel_dim
        self.head_dim = channel_dim // num_heads

        # 1) Normalize input x (no bias shift)
        self.input_norm = nn.GroupNorm(num_groups=1, num_channels=channel_dim, eps=1e-6, affine=True)
        with torch.no_grad():
            self.input_norm.bias.fill_(0.0)
        self.input_norm.bias.requires_grad = False

        # 2) Linear projections for Q, K, V to C total dims
        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)

        # Initialize Q/K/V with Xavier uniform
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            nn.init.normal_(proj.bias, mean=0.0, std=1e-6)

        # 4) Normalize Q and K (layer norm over head dim)
        self.q_norm = nn.GroupNorm(num_groups=1, num_channels=self.head_dim, eps=1e-6, affine=True)
        self.k_norm = nn.GroupNorm(num_groups=1, num_channels=self.head_dim, eps=1e-6, affine=True)

        # 8) Output projection back to C, zero initialized
        #self.out_proj = nn.Linear(channel_dim, num_classes)
        # nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-6)  # this needs to be zero for uvit and residualuvit
        # nn.init.normal_(self.out_proj.bias, mean=0.0, std=1e-6)

        self.out_proj = nn.Sequential(
            nn.Linear(channel_dim, 150),
            nn.GELU(),
            nn.Linear(150, 100),
            nn.GELU(),
            nn.Linear(100, 50),
            nn.GELU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        HW = H * W

        # 1) Flatten spatial dims and apply input norm
        x_norm = self.input_norm(x)
        x_flat = x_norm.reshape(B, C, HW)  # (B, C, HW)
        x_flat = x_flat.permute(0, 2, 1).contiguous()  # (B, HW, C)

        # 2) Project to Q, K, V
        q = self.q_proj(x_flat)  # (B, HW, C)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # 3) Reshape for heads
        q = q.reshape(B, HW, self.num_heads, self.head_dim)
        k = k.reshape(B, HW, self.num_heads, self.head_dim)
        v = v.reshape(B, HW, self.num_heads, self.head_dim)

        # 4) Normalize Q, K
        q_perm = q.permute(0, 3, 1, 2).contiguous()  # (B, head_dim, HW, num_heads)
        k_perm = k.permute(0, 3, 1, 2).contiguous()  # (B, head_dim, HW, num_heads)
        q = self.q_norm(q_perm)  # norm over head_dim
        k = self.k_norm(k_perm)  # norm over head_dim
        q = q.permute(0, 2, 3, 1).contiguous()  # (B, HW, num_heads, head_dim)
        k = k.permute(0, 2, 3, 1).contiguous()  # (B, HW, num_heads, head_dim)

        # 5) Scale Q
        q = q * (self.head_dim**-0.5)  # (B, HW, num_heads, head_dim)

        # 6) Attention weights
        weights = torch.einsum("bqhd,bkhd->bhqk", q, k)  # (B, HW, heads, heads)
        weights = F.softmax(weights, dim=-1) # (B, HW, heads, heads)

        # 7) Weighted sum
        attn_vals = torch.einsum("bhqk,bkhd->bqhd", weights, v)  # (bB, hHW, gheads, kheads) @ (bB, kHW, hheads, dhead_dim) -> (bB, gHW, hheads, dhead_dim)

        # 8) Merge heads and final projection
        attn_vals = attn_vals.reshape(B, HW, self.c_dim)  # (B, HW, C)
        out = self.out_proj(attn_vals)  # (B, HW, C)
        #out = out.mean(dim=1)  # (B, C) (for image classifier)
        out = out.permute(0, 2, 1).reshape(B, self.num_classes, H, W)  # (B, C, HW)  (for stefans pixel classifier)

        # # 9) Reshape back to spatial
        # out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        #
        # # 2) importance score per pixel: sum over classes → [B, H, W]
        # #    (you could also use margin = |logit1 − logit0| for binary)
        # importance = out.sum(dim=1)
        #
        # # 3) flatten spatial → [B, H*W], pick the index of the max
        # imp_flat = importance.view(B, -1)
        # idx     = imp_flat.argmax(dim=1)      # [B]
        #
        # # 4) recover (i,j) coords
        # i = idx // W
        # j = idx %  W
        #
        # # 5) gather that pixel’s embedding: [B, C]
        # batch_idx = torch.arange(B, device=x.device)
        # sel_emb   = x[batch_idx, :, i, j]     # [B, C]
        #
        # # 6) final linear classifier → [B, K]
        # out = self.linear_classification_head(sel_emb)  # (B, 2, H, W)
        return out




class DGDataset(Dataset):
    splits = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
    }

    def __init__(
        self,
        *,
        path: str,
        dataset: str = "camelyon17",
        split: Literal["train", "val", "test"] = "train",
        domain_split: Literal["id", "ood"] = "id",
        seed: int = 0,
        augmentations: callable = None,
        domain_transforms: callable = None,
    ):
        self.path = Path(path)
        self.split = split
        self.domain_split = domain_split
        self.augmentations = augmentations
        self.domain_transforms = domain_transforms
        self.dataset = self._get_dataset(dataset, seed=seed)

    def __getitem__(self, item):
        with torch.device("cpu"):
            rgb, label, _ = self.dataset.__getitem__(item)
        rgb = rgb.numpy(force=True)
        return rgb, label[0]

    def __len__(self):
        return len(self.dataset)

    def _get_raw_dataset(self, dataset: str):
        match dataset.lower():
            case "camelyon17":
                from oxels.datasets.dg import Camelyon17

                return Camelyon17(self.path, download=True)
            case "coloredmnist":
                from oxels.datasets.dg import ColoredMNIST

                return ColoredMNIST(self.path, download=True, transform=self.augmentations, domain_transforms=self.domain_transforms)
            case "fmowregion":
                from oxels.datasets.dg import FMoWRegion

                return FMoWRegion(self.path, download=True)
            case "fmowyear":
                from oxels.datasets.dg import FMoWYear

                return FMoWYear(self.path, download=True)
            case "officehome":
                from oxels.datasets.dg import OfficeHome

                return OfficeHome(self.path, download=True)
            case "pacs":
                from oxels.datasets.dg import PACS

                return PACS(self.path, download=True)
            case "rotatedmnist":
                from oxels.datasets.dg import RotatedMNIST

                return RotatedMNIST(self.path, download=True)
            case "terraincognita":
                from oxels.datasets.dg import TerraIncognita

                return TerraIncognita(self.path, download=True)
            case "vlcs":
                from oxels.datasets.dg import VLCS

                return VLCS(self.path, download=True)
            case "povertymapurbanicity":
                from oxels.datasets.dg import PovertyMapUrbanicity

                return PovertyMapUrbanicity(self.path, download=True)
            case "povertymapcountry":
                from oxels.datasets.dg import PovertyMapCountry

                return PovertyMapCountry(self.path, download=True)
            case other:
                raise NotImplementedError(f"Unrecognized dataset: {other}")

    def _get_splits(self, dataset, seed):
        if self.domain_split == "id":
            domains = dataset.all_domains[:-1]
        elif self.domain_split == "ood":
            domains = dataset.all_domains[-1:]
        else:
            raise ValueError(f"Expected domain_split to be 'id' or 'ood', got {self.domain_split}.")

        # get the full domains
        domains = [dataset.domain(d) for d in domains]

        if self.domain_split != "ood":
            # get the seeded random splits for the given domain
            split = list(self.splits.keys()).index(self.split)
            generator = torch.Generator().manual_seed(seed)
            domains = [random_split(d, list(self.splits.values()), generator=generator)[split] for d in domains]

        return domains

    def _get_dataset(self, dataset, seed=0):
        raw_dataset = self._get_raw_dataset(dataset)
        splits = self._get_splits(raw_dataset, seed)

        return MultiDomainDataset(
            *splits, in_distribution=self.domain_split == "id", n_domains=len(raw_dataset.all_domains)
        )



class DGLinearClassifier(MetricsMixin, L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        dataset_name: str,
        train_batch_size: int,
        val_batch_size: int,
        num_oxels: int = 64,
        image_size: int = 256,
        weight_decay: float = 0.004,
        learning_rate: float = 1e-3,
        lr_pct_start: float = 0.05,
        lr_div_factor: float = 25.0,
        lr_final_div_factor: float = 1e4,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["backbone", "head", "dataset_name"],
        )
        self.backbone = backbone
        self.backbone.freeze()
        self.head = head
        self.dataset_name = dataset_name
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.train_acc = Accuracy(task="binary", num_classes=head.num_classes)
        self.val_acc = Accuracy(task="binary", num_classes=head.num_classes)
        self.test_acc = Accuracy(task="binary", num_classes=head.num_classes)

    def forward(self, x):
        oxels = self.backbone(x)
        prediction = self.head(oxels)
        return prediction

    def compute_loss(self, batch):
        images, labels = batch
        logits = self(images) # some scalar
        B, _, H, W = logits.shape
        # turn labels into single value
        labels = torch.argmax(labels, dim=1)
        labels = labels.to(dtype=logits.dtype)
        labels = labels.view(B, 1, 1, 1).repeat(1, 1, H, W)  # Now (B, 1, H, W)
        loss = self.loss_fn(logits, labels)
        return loss

    def compute_metrics(self, batch):
        images, labels = batch
        logits = self(images)
        B, _, H, W = logits.shape
        #logits = logits.view(logits.shape[0], -1)  # Flatten the spatial dimensions
        labels = torch.argmax(labels, dim=1)
        labels = labels.to(dtype=logits.dtype)
        labels_pix = labels.view(B, 1, 1, 1).repeat(1, 1, H, W)  # Now (B, 1, H, W)
        loss = self.loss_fn(logits, labels_pix)
        # choose the right Accuracy object based on stage
        # Lightning will set self.training, self.validating, self.testing flags
        if self.trainer.training:
            logits = torch.sigmoid(logits)  # Apply sigmoid to logits for binary classification
            logits = torch.mean(logits, dim=[1, 2, 3])
            acc = self.train_acc(logits, labels)
        elif self.trainer.validating:
            logits = torch.sigmoid(logits)  # Apply sigmoid to logits for binary classification
            logits = torch.mean(logits, dim=[1, 2, 3])
            acc = self.val_acc(logits, labels)
        else:  # testing
            logits = torch.sigmoid(logits)  # Apply sigmoid to logits for binary classification
            logits = torch.mean(logits, dim=[1, 2, 3])
            acc = self.test_acc(logits, labels)

        return {
            "loss": loss,
            "accuracy": acc,
        }

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay

        # only use parameters that requires grad
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.99))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=self.hparams.lr_div_factor,
            final_div_factor=self.hparams.lr_final_div_factor,
            pct_start=self.hparams.lr_pct_start,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.compute_metrics(batch)
        prefix = "validation/" if dataloader_idx == 0 else "validation/ood/"
        for key, value in metrics.items():
            key = f"{prefix}{key}"
            self.log(key,
                     value,
                     on_step=False,
                     on_epoch=True,
                     sync_dist=True,
                     prog_bar=(dataloader_idx == 0),  # maybe only show ID in prog bar
                     logger=True)

        return metrics["loss"]

    def train_dataloader(self):
        dataset = DGDataset(
            path="datasets",
            dataset=self.dataset_name,
            split="train",
            domain_split="id",
            seed=0,
            augmentations=transforms.Compose([
                transforms.Resize((self.hparams.image_size, self.hparams.image_size))
            ]),
            #domain_transforms= [
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #]
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,#len(os.sched_getaffinity(0)),
            drop_last=True,
        )

    def val_dataloader(self):
        val_id = DGDataset(
            path="datasets",
            dataset=self.dataset_name,
            split="val",
            domain_split="id",
            seed=0,
            augmentations=transforms.Compose([
                transforms.Resize((self.hparams.image_size, self.hparams.image_size))
            ]),
            #domain_transforms= [
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #]
        )

        val_ood = DGDataset(
            path="datasets",
            dataset=self.dataset_name,
            split="val",
            domain_split="ood",
            seed=0,
            augmentations=transforms.Compose([
                transforms.Resize((self.hparams.image_size, self.hparams.image_size))
            ]),
            #domain_transforms= [
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #    dtransforms.Compose(
            #        [dtransforms.BinarizeLabel(5), dtransforms.FlipLabel(0.5), dtransforms.ZeroChannel(0.0)]
            #    ),
            #]
        )

        return [DataLoader(val_id, batch_size=self.hparams.val_batch_size, shuffle=False,  num_workers=0, drop_last=False),
                DataLoader(val_ood, batch_size=self.hparams.val_batch_size, shuffle=False, num_workers=0, drop_last=False)]




backbone_run_names = [
    #"chocolate-terrain-296",
    #"twilight-firefly-295",
    #"feasible-pyramid-294",
    #"noble-river-293",
    #"dutiful-butterfly-292",
    #"daily-salad-291",
    #"honest-planet-290",
    #"clean-violet-289",
    #"fragrant-sponge-288",
    #"unique-energy-287",
    #"giddy-dew-286",
    #"denim-night-285",
    #"giddy-fire-284",
    #"sleek-microwave-283",
    #"deft-microwave-282",
    #"eager-elevator-281",
    #"fragrant-capybara-280",
    #"colorful-planet-279",
    #"treasured-valley-278",
    #"rosy-night-277",
    #"wise-plasma-276",
    #"generous-sun-275",
    #"driven-bush-274",
    #"amber-paper-273",
    #"dutiful-universe-272",
    #"legendary-plasma-271",
    #"whole-sea-270",
    #"avid-sound-269",
    #"rare-bush-268",
    #"eager-fire-267",
    #"rosy-capybara-266",
    #"sleek-wind-265",
    #"good-aardvark-264",
    #"honest-voice-263",
    #"ruby-star-262",
    #"mild-sea-261",
    #"azure-planet-260",
    #"lucky-tree-259",
    #"easy-cosmos-258",
    #"spring-mountain-257",
    #"stilted-waterfall-256",
    #"grateful-butterfly-255",
    #"desert-pyramid-253",
    #"clear-cloud-252",
    #"visionary-pyramid-251",
    #"electric-disco-250",
    #"dainty-firefly-249",
    #"swept-rain-248",
    #"summer-oath-247",
    #"leafy-firefly-245",
    #"clear-plant-244",
    "astral-forest-243",
    "snowy-bird-242",
    "vocal-pine-241",
    "comic-tree-240",
    "wild-star-239",
    "electric-shadow-236",
    "solar-cosmos-235",
    "radiant-durian-234",
    "super-shadow-233",
    "solar-tree-232",
    "proud-armadillo-231",
    "lucky-valley-230",
    "lilac-disco-227",
    "earnest-cherry-226",
    "earthy-deluge-225",
    "serene-resonance-224",
    "woven-lake-223",
    "twilight-oath-222",
    "smart-dawn-221"
]

for backbone_run_name in backbone_run_names:
    # ---- load backbone ----
    ckpt_dir = os.path.join("lightning_logs", str(backbone_run_name))
    with open(os.path.join(ckpt_dir, "wandb_run_id.txt"), "r") as f:
        backbone_run_id = f.read().strip()
    backbone = ColoredMNISTModel.load_from_checkpoint(os.path.join(ckpt_dir, "best_model.ckpt"))
    print(f"Loaded model from {os.path.join(ckpt_dir, 'best_model.ckpt')} with run id {backbone_run_id} and name {backbone_run_name}")
    num_oxels = backbone.hparams.num_oxels

    # ---- define classifier ----
    classifier = AttentionClassifier(num_classes=1, num_heads=1, channel_dim=num_oxels)
    print(f"Number of parameters in classifier: {sum(p.numel() for p in classifier.parameters())}")

    # ---- training setup ----


    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    print("Device Count:", torch.cuda.device_count())
    project = "ColoredMNIST-testing"
    total_steps = 12_000
    image_size = 32
    num_nodes = 1
    num_devices = 1
    train_batch_size = 128
    val_batch_size = 256
    learning_rate = 2e-3
    lr_pct_start = 0.05
    weight_decay = 1e-6

    model_config = dict(
        num_oxels=num_oxels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_pct_start=lr_pct_start,
        lr_div_factor=25.0,
        lr_final_div_factor=1e3,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        image_size=image_size,
    )

    trainer_config = dict(
        gradient_clip_val=3.0,
        gradient_clip_algorithm="value",
        max_steps=total_steps,
        accelerator="gpu",
        strategy="auto",
        devices=num_devices,
        precision="16-mixed",
        num_nodes=num_nodes,
    )

    model = DGLinearClassifier(
        backbone=backbone,
        head=classifier,
        dataset_name="ColoredMNIST",
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_oxels=num_oxels,
        image_size=image_size,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_pct_start=lr_pct_start,
        lr_div_factor=25.0,
        lr_final_div_factor=1e4)

    with torch.device("cpu"):
        train_images = [model.train_dataloader().dataset.dataset[i][0] for i in range(4)]
        train_images = torch.stack(train_images)
        validation_images = [model.val_dataloader()[0].dataset.dataset[i][0] for i in range(4)]
        validation_images = torch.stack(validation_images)
        validation_images_ood = [model.val_dataloader()[1].dataset.dataset[i][0] for i in range(4)]
        validation_images_ood = torch.stack(validation_images_ood)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: ", num_parameters)
    config = model_config | trainer_config

    # ---- logger ----
    run = wandb.init(
        entity="kl_divergence-rensselaer-polytechnic-institute",
        project=project,
        config=config,
        dir="wandb_results",
    )
    logger = WandbLogger(experiment=run)
    wandb.summary["num_parameters"] = num_parameters
    wandb.summary["backbone_run_id"] = backbone_run_id
    wandb.summary["backbone_run_name"] = backbone_run_name
    wandb.summary["model_type"] = "AttentionMLPPixelClassifier"
    run_name = wandb.run.name
    run_id = wandb.run.id
    print(f"name: {run_name} \t run id:{run_id}")

    # ---- trainer ----
    ckpt_dir = os.path.join("lightning_logs", str(run_name))
    # make sure the directory exists
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run_id)
    trainer = L.Trainer(
        **trainer_config,
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval="step"),
            callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="validation/loss/dataloader_idx_0",
                mode="min",
                save_top_k=1,
                filename="best_model",
                save_last=True,
            ),
            # ShowOxels(images=train_images, every_n_epochs=10, caption="Train Oxels"),
            # ShowOxels(images=validation_images, every_n_epochs=10, caption="Validation Oxels"),
            # ShowOxels(images=validation_images_ood, every_n_epochs=10, caption="Validation OOD Oxels"),
        ],
        logger=logger,
    )

    try:
        trainer.fit(model)
        result = trainer.callback_metrics["validation/loss/dataloader_idx_0"]
        wandb.summary["result"] = result
        del model, classifier, trainer, logger, run, backbone
        torch.cuda.empty_cache()  # clear GPU memory after each run
    finally:
        # clean up
        wandb.finish()