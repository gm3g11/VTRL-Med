#!/usr/bin/env python3
"""
Stage 3 Dataset - Returns Images + Persistence Features
"""

import torch
import numpy as np
from torch.utils.data import Dataset

from topocl.stage2.stage2_unified_parallel import TopologyAwareAugmentation, compute_multiscale_persistence_full_image


class Stage3TopoImageDataset(Dataset):
    """Dataset for Stage 3 that returns both images and persistence features"""

    def __init__(self, images, labels, rois, augmenter, n_h0=96, n_h1=192):
        self.images = images
        self.labels = labels
        self.rois = rois
        self.augmenter = augmenter
        self.n_h0 = n_h0
        self.n_h1 = n_h1

    def __len__(self):
        return len(self.images)

    def _compute_persistence(self, img):
        """Compute multi-scale persistence features"""
        pd = compute_multiscale_persistence_full_image(
            img,
            n_h0=self.n_h0,
            n_h1=self.n_h1
        )
        return (
            torch.from_numpy(pd['h0']).float(),
            torch.from_numpy(pd['h1']).float()
        )

    def _img_to_tensor(self, img):
        """Convert numpy image (H, W, 3) to tensor (3, H, W)"""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_tensor

    def __getitem__(self, idx):
        img = self.images[idx]
        roi = self.rois[idx]  # Already a boolean numpy array
        label = self.labels[idx]

        # Generate two augmented views (roi is the mask)
        if np.random.rand() < 0.5:
            aug1 = self.augmenter.weak_augment(img, roi)
        else:
            aug1 = self.augmenter.strong_augment(img, roi)

        if np.random.rand() < 0.5:
            aug2 = self.augmenter.weak_augment(img, roi)
        else:
            aug2 = self.augmenter.strong_augment(img, roi)

        # Convert images to tensors
        img1_tensor = self._img_to_tensor(aug1)
        img2_tensor = self._img_to_tensor(aug2)

        # Compute persistence features
        h0_1, h1_1 = self._compute_persistence(aug1)
        h0_2, h1_2 = self._compute_persistence(aug2)

        return img1_tensor, img2_tensor, h0_1, h1_1, h0_2, h1_2, label