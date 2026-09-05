"""
PoreSR: Micro-CT Dataset

Dataset class supporting both 2D (single-slice) and 2.5D (multi-slice)
loading of paired HR/LR micro-CT training data.

Authors:
    Sonu Sudhikumar Seena (1), Anirban Chakraborty (2), Jingyue Hao (1), Lin Ma (1)

Implementation:
    Sonu Sudhikumar Seena

Affiliations:
    1. Department of Chemical Engineering, The University of Manchester,
       Oxford Road, Manchester M13 9PL, UK
    2. Department of Computational and Data Sciences (CDS),
       Indian Institute of Science Bangalore, Bangalore, Karnataka 560012, India

Paper:
    "Calibrated Degradation for Super-Resolution of Rock Micro-CT:
     Decoupling Image Fidelity from Petrophysical Accuracy"
    Computers & Geosciences, 2026

License: MIT
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MicroCTDataset(Dataset):
    """
    Dataset for paired HR/LR micro-CT slice loading.

    Supports both 2D (k=1, single-slice input) and 2.5D (k=5, five-slice
    stack input) modes. In 2.5D mode, K adjacent LR slices are stacked as
    independent input channels, with the centre HR slice as the target.

    During training, random crops are extracted. During validation, centre
    crops are used for deterministic evaluation.

    The LR crop origin is derived from the HR crop origin by integer division
    by the upscale factor, so a random HR origin that is not a multiple of
    four leaves the LR patch lagging its HR target by up to three HR pixels
    (0.75 LR pixels). This sub-LR-pixel jitter is applied identically to every
    architecture, so it affects all cells of the factorial equally. Validation
    crops are centred and exactly aligned.

    In 2.5D mode the first and last two slices of a split have no complete
    five-slice neighbourhood and are excluded as centre slices, so a 2.5D
    model sees four fewer centre slices per split than a 2D model. Training is
    step-based rather than epoch-based, so the number of optimisation steps and
    the number of patches seen are unaffected.

    Parameters
    ----------
    slice_indices : list of int
        Global slice indices for this split (train/val/test).
    data_root_hr : str
        Directory containing HR TIFF slices.
    data_root_lr : str
        Directory containing calibrated synthetic LR TIFF slices.
    k : int
        Number of adjacent slices. 1 for 2D, 5 for 2.5D.
    patch_size_hr : int
        HR patch spatial dimension. Default: 256.
    patches_per_image : int
        Number of random patches per image per epoch (training only).
    phase : str
        One of 'train' or 'val'. Controls cropping strategy.
    """

    def __init__(self, slice_indices, data_root_hr, data_root_lr,
                 k=1, patch_size_hr=256, patches_per_image=4, phase="train"):
        self.data_root_hr = data_root_hr
        self.data_root_lr = data_root_lr
        self.k = k
        self.patch_size_hr = patch_size_hr
        self.patch_size_lr = patch_size_hr // 4
        self.patches_per_image = patches_per_image
        self.phase = phase
        self.global_indices = slice_indices

        # Valid centre indices (margin for multi-slice context)
        half_k = k // 2
        self.valid_indices = list(range(half_k, len(slice_indices) - half_k))

    def __len__(self):
        if self.phase == "train":
            return len(self.valid_indices) * self.patches_per_image
        return len(self.valid_indices)

    def _load_slice(self, root, global_idx):
        """Load a single TIFF slice as a float32 tensor in [0, 1]."""
        path = os.path.join(root, f"slice_{global_idx:04d}.tif")
        # convert("L") returns an 8-bit greyscale image, so 16-bit TIFF input
        # is reduced to 8 bits here. This matches the loading used to produce
        # the published results and must not be changed without retraining.
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)

        # Normalise to [0, 1]. The 65535 branch is unreachable after
        # convert("L") and is retained only for inputs already in [0, 1].
        if arr.max() > 1.0:
            if arr.max() > 255:
                arr = arr / 65535.0
            else:
                arr = arr / 255.0

        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        if self.phase == "train":
            centre_local = self.valid_indices[idx // self.patches_per_image]
        else:
            centre_local = self.valid_indices[idx]

        half_k = self.k // 2

        # Load K adjacent LR slices
        lr_slices = []
        for offset in range(-half_k, half_k + 1):
            local_idx = centre_local + offset
            global_idx = self.global_indices[local_idx]
            lr_slices.append(self._load_slice(self.data_root_lr, global_idx))
        lr_stack = torch.stack(lr_slices, dim=0)  # [K, H_lr, W_lr]

        # Load K adjacent HR slices. Only the centre slice becomes the
        # training target; the neighbours are loaded so that the HR and LR
        # stacks are cropped identically.
        hr_slices = []
        for offset in range(-half_k, half_k + 1):
            local_idx = centre_local + offset
            global_idx = self.global_indices[local_idx]
            hr_slices.append(self._load_slice(self.data_root_hr, global_idx))
        hr_stack = torch.stack(hr_slices, dim=0)  # [K, H_hr, W_hr]

        # Patch extraction
        h_hr, w_hr = hr_stack.shape[1], hr_stack.shape[2]
        h_lr, w_lr = lr_stack.shape[1], lr_stack.shape[2]

        if self.phase == "train":
            top_hr = random.randint(0, h_hr - self.patch_size_hr)
            left_hr = random.randint(0, w_hr - self.patch_size_hr)
        else:
            top_hr = (h_hr - self.patch_size_hr) // 2
            left_hr = (w_hr - self.patch_size_hr) // 2

        # Integer division: see the class docstring on crop alignment.
        top_lr = top_hr // 4
        left_lr = left_hr // 4

        hr_patches = hr_stack[
            :,
            top_hr : top_hr + self.patch_size_hr,
            left_hr : left_hr + self.patch_size_hr,
        ]
        lr_patches = lr_stack[
            :,
            top_lr : top_lr + self.patch_size_lr,
            left_lr : left_lr + self.patch_size_lr,
        ]

        # Target is the centre HR slice
        hr_target = hr_patches[half_k].unsqueeze(0)  # [1, H, W]

        global_slice_idx = self.global_indices[centre_local]

        return {
            "lr": lr_patches,        # [K, 64, 64]
            "hr": hr_target,         # [1, 256, 256]
            "slice_idx": global_slice_idx,
        }
