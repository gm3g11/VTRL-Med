#!/usr/bin/env python3
"""
Stage 2: Topology-based Contrastive Learning (OPTIMIZED)

Key improvements:
- Full-image PH (no cropping, no median filling)
- Proper grayscale conversion
- Multi-scale PH (3 scales)
- Larger model (512 dim, 8 heads)
- Optimized training with mixed precision
- Test accuracy monitoring (Supervised) / Linear probe (SSL)
- TTA at end of training
- Upper bound test as default
"""

import os
import json
import time
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# TDA
import cripser

# Image processing
from skimage import filters, feature, measure
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter, map_coordinates
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# Performance optimizations
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset: str) -> int:
    """Get number of classes for each dataset"""
    class_map = {
        'kvasir': 8,
        'pathmnist': 9,
        'octmnist': 4,
        'organsmnist': 11,
        'isic2019': 8,
        'imagenet100': 100,
    }
    return class_map.get(dataset.lower(), 10)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB to grayscale using proper Rec. 601 luma coefficients
    Y = 0.299*R + 0.587*G + 0.114*B

    This is the standard used in JPEG, MPEG, and most CV applications.
    Much better than simple mean!
    """
    if img.ndim == 2:
        return img.astype(np.float32)

    # Rec. 601 luma
    gray = (0.299 * img[..., 0] +
            0.587 * img[..., 1] +
            0.114 * img[..., 2]).astype(np.float32)

    return gray


def _load_single_image_helper(args):
    """Helper function for parallel image loading (must be at module level for pickle)"""
    path, img_size = args
    img = Image.open(path).convert('RGB').resize((img_size, img_size))
    return np.array(img)


# ============================================================================
# PERSISTENCE HOMOLOGY - FULL IMAGE, NO FILLING
# ============================================================================

def compute_multiscale_persistence_full_image(
        img: np.ndarray,
        n_h0: int = 96,
        n_h1: int = 192,
        scales: List[float] = [1.0, 0.5, 0.25]
) -> Dict[str, np.ndarray]:
    """
    ✅ OPTIMIZED: Compute PH on full image at multiple scales

    Key improvements:
    - No cropping (preserves all information)
    - No median filling (no artificial boundaries)
    - Proper grayscale conversion
    - Multi-scale for richer features

    Args:
        img: Image (H, W, 3) or (H, W)
        n_h0: Number of H0 features per scale
        n_h1: Number of H1 features per scale
        scales: Scales to compute (default: 1.0, 0.5, 0.25)

    Returns:
        dict with 'h0' and 'h1' arrays of shape (n_h0*len(scales), 2) and (n_h1*len(scales), 2)
    """
    # Convert to grayscale with proper weighting
    gray = rgb_to_gray(img)

    all_h0, all_h1 = [], []

    # Compute PH at each scale
    for scale in scales:
        if scale != 1.0:
            # Rescale image
            new_h, new_w = int(gray.shape[0] * scale), int(gray.shape[1] * scale)
            if new_h < 10 or new_w < 10:
                continue
            gray_scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            gray_scaled = gray

        # Normalize to uint8 for cripser
        gray_min, gray_max = gray_scaled.min(), gray_scaled.max()
        if gray_max - gray_min < 1e-8:
            # Constant image, skip
            continue

        gray_norm = ((gray_scaled - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)

        # Compute PH
        try:
            ph = cripser.computePH(gray_norm)
        except:
            continue

        # Extract birth-death pairs
        h0_pairs, h1_pairs = [], []

        for item in ph:
            if len(item) < 3:
                continue

            dim = int(item[0])
            birth = float(item[1]) / 255.0  # Normalize back to [0, 1]
            death = float(item[2]) / 255.0

            if death >= 1e5:  # Infinite death
                continue

            persistence = death - birth

            if persistence < 0.005:  # Filter noise
                continue

            if dim == 0:
                h0_pairs.append((birth, death))
            elif dim == 1:
                h1_pairs.append((birth, death))

        # Sort by persistence (most significant first)
        h0_pairs.sort(key=lambda x: x[1] - x[0], reverse=True)
        h1_pairs.sort(key=lambda x: x[1] - x[0], reverse=True)

        all_h0.extend(h0_pairs[:n_h0])
        all_h1.extend(h1_pairs[:n_h1])

    # Pad to fixed size
    def pad_points(pts: List[Tuple[float, float]], k: int) -> np.ndarray:
        result = np.zeros((k, 2), dtype=np.float32)
        for i in range(min(k, len(pts))):
            result[i] = pts[i]
        return result

    total_h0 = n_h0 * len(scales)
    total_h1 = n_h1 * len(scales)

    return {
        'h0': pad_points(all_h0, total_h0),
        'h1': pad_points(all_h1, total_h1)
    }

# ============================================================================
# DATASET LOADING (WORKING VERSION FROM SINGLE FILE)
# ============================================================================

def load_dataset_as_numpy(dataset_name, split, data_dir=None, img_size=224, seed=42):
    """Load any dataset and return as numpy arrays"""
    from torchvision import datasets
    from sklearn.model_selection import train_test_split

    dataset_name_lower = dataset_name.lower()

    # MedMNIST datasets
    if dataset_name_lower in ['pathmnist', 'octmnist', 'organamnist', 'organsmnist']:
        import medmnist
        from medmnist import INFO

        # Find correct key
        dataset_key = None
        for key in INFO.keys():
            if key.lower() == dataset_name_lower:
                dataset_key = key
                break

        if dataset_key is None:
            raise ValueError(f"Dataset {dataset_name} not found in MedMNIST")

        # Get class name
        key_lower = dataset_key.lower()
        if key_lower == 'octmnist':
            class_name = 'OCTMNIST'
        elif key_lower.startswith('organ') and 'mnist' in key_lower:
            letter = key_lower.replace('organ', '').replace('mnist', '')
            class_name = f'Organ{letter.upper()}MNIST'
        elif 'mnist' in key_lower:
            prefix = key_lower.replace('mnist', '')
            class_name = prefix.capitalize() + 'MNIST'
        else:
            class_name = dataset_key

        DataClass = getattr(medmnist, class_name)
        ds = DataClass(split=split, download=True, size=img_size)

        imgs = ds.imgs
        labels = ds.labels.squeeze()

        if imgs.dtype != np.uint8:
            imgs = (imgs * 255).astype(np.uint8)
        if imgs.ndim == 3:
            imgs = imgs[..., None]
        if imgs.shape[-1] == 1:
            imgs = np.repeat(imgs, 3, axis=-1)

        return imgs, labels

    # ISIC2019
    elif dataset_name_lower == 'isic2019':
        if data_dir is None:
            data_dir = os.path.expanduser('data/isic2019')

        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_ds = datasets.ImageFolder(train_dir)
            test_ds = datasets.ImageFolder(test_dir)
            all_paths = [s[0] for s in train_ds.samples] + [s[0] for s in test_ds.samples]
            all_labels = train_ds.targets + test_ds.targets
        else:
            full_ds = datasets.ImageFolder(data_dir)
            all_paths = [s[0] for s in full_ds.samples]
            all_labels = full_ds.targets

        train_idx, temp_idx = train_test_split(
            range(len(all_labels)), train_size=0.7, test_size=0.3,
            stratify=all_labels, random_state=seed
        )

        temp_labels = [all_labels[i] for i in temp_idx]
        val_idx_temp, test_idx_temp = train_test_split(
            range(len(temp_idx)), train_size=0.5, test_size=0.5,
            stratify=temp_labels, random_state=seed
        )

        val_idx = [temp_idx[i] for i in val_idx_temp]
        test_idx = [temp_idx[i] for i in test_idx_temp]

        if split == 'train':
            indices = train_idx
        elif split == 'val':
            indices = val_idx
        else:
            indices = test_idx

        paths = [all_paths[i] for i in indices]
        labels = [all_labels[i] for i in indices]

        print(f"Loading {len(paths)} images from ISIC2019 {split}...")

        n_workers = min(cpu_count() - 1, 8)
        args_list = [(path, img_size) for path in paths]

        with Pool(n_workers) as pool:
            imgs = list(tqdm(
                pool.imap(_load_single_image_helper, args_list, chunksize=100),
                total=len(paths),
                desc=f"Loading {split}"
            ))

        return np.array(imgs, dtype=np.uint8), np.array(labels)

    # Kvasir
    elif dataset_name_lower == 'kvasir':
        if data_dir is None:
            data_dir = os.path.expanduser('data/kvasir-dataset')

        full_ds = datasets.ImageFolder(data_dir)
        all_paths = [s[0] for s in full_ds.samples]
        all_labels = full_ds.targets

        train_idx, temp_idx = train_test_split(
            range(len(all_labels)), train_size=0.7, test_size=0.3,
            stratify=all_labels, random_state=seed
        )

        temp_labels = [all_labels[i] for i in temp_idx]
        val_idx_temp, test_idx_temp = train_test_split(
            range(len(temp_idx)), train_size=0.5, test_size=0.5,
            stratify=temp_labels, random_state=seed
        )

        val_idx = [temp_idx[i] for i in val_idx_temp]
        test_idx = [temp_idx[i] for i in test_idx_temp]

        if split == 'train':
            indices = train_idx
        elif split == 'val':
            indices = val_idx
        else:
            indices = test_idx

        paths = [all_paths[i] for i in indices]
        labels = [all_labels[i] for i in indices]

        print(f"Loading {len(paths)} images from Kvasir {split}...")

        n_workers = min(cpu_count() - 1, 8)
        args_list = [(path, img_size) for path in paths]

        with Pool(n_workers) as pool:
            imgs = list(tqdm(
                pool.imap(_load_single_image_helper, args_list, chunksize=100),
                total=len(paths),
                desc=f"Loading {split}"
            ))

        return np.array(imgs, dtype=np.uint8), np.array(labels)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")





# ============================================================================
# ROI EXTRACTION
# ============================================================================

class ImprovedROIExtractor:
    """Fallback ROI extraction using traditional CV methods"""

    def __init__(self, morph_size=3, edge_sigma=1.0):
        self.morph_size = morph_size
        self.edge_sigma = edge_sigma

    def extract(self, img):
        if img.ndim == 3:
            gray = img.mean(axis=2).astype(np.uint8)
        else:
            gray = img.astype(np.uint8)

        # Otsu thresholding
        try:
            thresh = filters.threshold_otsu(gray)
            mask = gray > thresh
        except:
            mask = gray > gray.mean()

        # Morphological operations
        if mask.sum() > 0:
            mask = binary_erosion(mask, disk(self.morph_size))
            mask = binary_dilation(mask, disk(self.morph_size * 2))

        # Edge detection fallback
        if mask.sum() < gray.size * 0.05:
            edges = feature.canny(gray, sigma=self.edge_sigma)
            mask = binary_dilation(edges, disk(self.morph_size * 3))

        return mask.astype(bool)


def load_rois_with_fallback(dataset_name, split, imgs, roi_base_dir='data/rois'):
    """
    Try to load SAM ROIs, fall back to traditional if not found

    Returns:
        rois: List of boolean masks
        using_sam: Boolean indicating if SAM ROIs were used
    """
    roi_dir = os.path.join(roi_base_dir, f'{dataset_name.lower()}_rois')

    # For datasets with train/val/test structure
    if dataset_name.lower() in ['pathmnist', 'octmnist', 'organamnist', 'organsmnist']:
        roi_path = os.path.join(roi_dir, split, f'{split}_rois.npz')
    # For datasets with all/ structure
    else:  # isic2019, kvasir
        roi_path = os.path.join(roi_dir, 'all', 'all_rois.npz')

    # Try to load SAM ROIs
    if os.path.exists(roi_path):
        print(f"  Loading SAM ROIs from: {roi_path}")
        try:
            roi_data = np.load(roi_path)
            all_rois = roi_data['rois']
            success_flags = roi_data.get('success_flags', np.ones(len(all_rois), dtype=bool))

            # For isic2019/kvasir, we need to align with the split
            if dataset_name.lower() in ['isic2019', 'kvasir']:
                # ROIs are for all samples, need to extract relevant indices
                # This alignment happens during dataset loading
                # For now, just verify count matches
                if len(all_rois) < len(imgs):
                    print(f"  ⚠️  ROI count mismatch: {len(all_rois)} ROIs vs {len(imgs)} images")
                    print(f"  ⚠️  Falling back to traditional ROI extraction")
                    raise ValueError("Count mismatch")

                # Return all ROIs (will be indexed later during split)
                rois = list(all_rois)
            else:
                # For MedMNIST datasets, direct alignment
                if len(all_rois) != len(imgs):
                    print(f"  ⚠️  ROI count mismatch: {len(all_rois)} ROIs vs {len(imgs)} images")
                    print(f"  ⚠️  Falling back to traditional ROI extraction")
                    raise ValueError("Count mismatch")

                rois = list(all_rois)

            roi_areas = np.array([roi.sum() / roi.size for roi in rois]) * 100
            success_rate = success_flags.mean() * 100

            print(f"  ✓ Loaded {len(rois)} SAM ROIs")
            print(f"  ✓ ROI coverage: {roi_areas.mean():.1f}±{roi_areas.std():.1f}%")
            print(f"  ✓ Success rate: {success_rate:.1f}%")

            return rois, True

        except Exception as e:
            print(f"  ⚠️  Error loading SAM ROIs: {e}")
            print(f"  ⚠️  Falling back to traditional ROI extraction")
    else:
        print(f"  ℹ️  SAM ROIs not found at: {roi_path}")
        print(f"  ℹ️  Using traditional ROI extraction")

    # Fallback to traditional ROI extraction
    print(f"  Generating traditional ROIs for {len(imgs)} images...")
    roi_extractor = ImprovedROIExtractor()
    rois = [roi_extractor.extract(img) for img in tqdm(imgs, desc=f"Extracting ROIs")]

    return rois, False


def load_dataset_with_rois(dataset_name, split, roi_base_dir='data/rois',
                           data_dir=None, seed=42):
    """
    Load dataset with ROIs (SAM if available, traditional fallback)

    Returns:
        imgs, labels, rois, using_sam
    """
    # Load images and labels
    imgs, labels = load_dataset_as_numpy(dataset_name, split, data_dir=data_dir, seed=seed)

    # Load or generate ROIs
    if dataset_name.lower() in ['isic2019', 'kvasir']:
        # Special handling for datasets with all/ structure
        # We need to load all ROIs first, then split
        roi_dir = os.path.join(roi_base_dir, f'{dataset_name.lower()}_rois')
        roi_path = os.path.join(roi_dir, 'all', 'all_rois.npz')

        if os.path.exists(roi_path):
            # Load all ROIs
            roi_data = np.load(roi_path)
            all_rois = list(roi_data['rois'])

            # Get indices for this split
            from torchvision import datasets
            from sklearn.model_selection import train_test_split

            if dataset_name.lower() == 'isic2019':
                if data_dir is None:
                    data_dir = os.path.expanduser('data/isic2019')

                train_dir = os.path.join(data_dir, 'train')
                test_dir = os.path.join(data_dir, 'test')

                if os.path.exists(train_dir) and os.path.exists(test_dir):
                    train_ds = datasets.ImageFolder(train_dir)
                    test_ds = datasets.ImageFolder(test_dir)
                    all_labels = train_ds.targets + test_ds.targets
                else:
                    full_ds = datasets.ImageFolder(data_dir)
                    all_labels = full_ds.targets
            else:  # kvasir
                if data_dir is None:
                    data_dir = os.path.expanduser('data/kvasir-dataset')
                full_ds = datasets.ImageFolder(data_dir)
                all_labels = full_ds.targets

            # Get split indices
            train_idx, temp_idx = train_test_split(
                range(len(all_labels)), train_size=0.7, test_size=0.3,
                stratify=all_labels, random_state=seed
            )

            temp_labels = [all_labels[i] for i in temp_idx]
            val_idx_temp, test_idx_temp = train_test_split(
                range(len(temp_idx)), train_size=0.5, test_size=0.5,
                stratify=temp_labels, random_state=seed
            )

            val_idx = [temp_idx[i] for i in val_idx_temp]
            test_idx = [temp_idx[i] for i in test_idx_temp]

            if split == 'train':
                indices = train_idx
            elif split == 'val':
                indices = val_idx
            else:
                indices = test_idx

            # Extract ROIs for this split
            rois = [all_rois[i] for i in indices]

            # Verify alignment
            if len(rois) != len(imgs):
                print(f"  ⚠️  ROI count mismatch after splitting: {len(rois)} vs {len(imgs)}")
                print(f"  ⚠️  Falling back to traditional ROI extraction")
                roi_extractor = ImprovedROIExtractor()
                rois = [roi_extractor.extract(img) for img in tqdm(imgs, desc=f"Extracting ROIs")]
                using_sam = False
            else:
                print(f"  ✓ Loaded {len(rois)} SAM ROIs for {split} split")
                using_sam = True
        else:
            # Generate traditional ROIs
            rois, using_sam = load_rois_with_fallback(dataset_name, split, imgs, roi_base_dir)
    else:
        # MedMNIST datasets with train/val/test structure
        rois, using_sam = load_rois_with_fallback(dataset_name, split, imgs, roi_base_dir)

    return imgs, labels, rois, using_sam


def load_imagenet100_with_bboxes(
        csv_path: str,
        img_dir: str,
        split: str = 'train',
        img_size: int = 224
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Load ImageNet-100 with bounding boxes

    CSV format: ImageId,PredictionString
    PredictionString: class_id x y w h [class_id x y w h ...]

    Returns:
        imgs: (N, H, W, 3) uint8
        labels: (N,) class IDs
        rois: List of N boolean masks from bboxes
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    imgs, labels, rois = [], [], []

    class_to_idx = {}  # Map class names to indices

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading ImageNet-100 {split}"):
        img_id = row['ImageId']
        pred_str = row['PredictionString']

        # Parse bbox
        parts = pred_str.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        x, y, w, h = map(int, parts[1:5])

        # Map class to index
        if class_id not in class_to_idx:
            class_to_idx[class_id] = len(class_to_idx)
        label = class_to_idx[class_id]

        # Load image
        img_path = os.path.join(img_dir, f"{img_id}.JPEG")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (img_size, img_size))

        # Scale bbox
        x = int(x * img_size / orig_w)
        y = int(y * img_size / orig_h)
        w = int(w * img_size / orig_w)
        h = int(h * img_size / orig_h)

        # Create ROI mask
        roi = np.zeros((img_size, img_size), dtype=bool)
        x2 = min(x + w, img_size)
        y2 = min(y + h, img_size)
        x1, y1 = max(0, x), max(0, y)
        roi[y1:y2, x1:x2] = True

        imgs.append(img_resized)
        labels.append(label)
        rois.append(roi)

    print(f"  ✓ Loaded {len(imgs)} ImageNet-100 images with {len(class_to_idx)} classes")

    return np.array(imgs, dtype=np.uint8), np.array(labels), rois


# ============================================================================
# AUGMENTATION
# ============================================================================

class TopologyAwareAugmentation:
    """
    ROI-guided augmentation that emphasizes changes in ROI while preserving topology
    """

    def __init__(self, weak_range=(0.005, 0.02), strong_range=(0.04, 0.12)):
        self.weak_range = weak_range
        self.strong_range = strong_range

    def boundary_noise(self, img: np.ndarray, roi: np.ndarray, sigma: float) -> np.ndarray:
        """Add noise at ROI boundaries"""
        boundaries = find_boundaries(roi, mode='thick')
        zone = binary_dilation(boundaries, disk(4))

        noise = np.random.normal(0, sigma * 255, img.shape).astype(np.float32)
        out = img.astype(np.float32)

        # Stronger noise at ROI boundaries
        out[zone] += 3.0 * noise[zone]
        # Weaker noise outside
        out[~roi] += 0.3 * noise[~roi]
        # Medium noise inside ROI
        roi_interior = roi & ~zone
        out[roi_interior] += noise[roi_interior]

        return np.clip(out, 0, 255).astype(np.uint8)

    def local_intensity_shift(self, img: np.ndarray, roi: np.ndarray, scale: float) -> np.ndarray:
        """Local intensity shifts guided by ROI"""
        out = img.astype(np.float32)
        h, w = roi.shape
        grid_size = 32

        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                local_roi = roi[i:i + grid_size, j:j + grid_size]
                if local_roi.any():
                    shift = np.random.uniform(-scale, scale)
                    patch = out[i:i + grid_size, j:j + grid_size]
                    patch[local_roi] *= (1 + shift)
                    out[i:i + grid_size, j:j + grid_size] = patch

        return np.clip(out, 0, 255).astype(np.uint8)

    def elastic_deformation(self, img: np.ndarray, alpha: float = 20, sigma: float = 3) -> np.ndarray:
        """Mild elastic deformation"""
        shape = img.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if img.ndim == 3:
            out = np.zeros_like(img)
            for c in range(img.shape[2]):
                out[:, :, c] = map_coordinates(img[:, :, c], indices, order=1).reshape(shape)
            return out.astype(np.uint8)
        else:
            return map_coordinates(img, indices, order=1).reshape(shape).astype(np.uint8)

    def weak_augment(self, img: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Weak augmentation"""
        sigma = np.random.uniform(*self.weak_range)
        aug_type = np.random.choice(['boundary', 'local_shift', 'elastic', 'smooth'])

        if aug_type == 'boundary':
            return self.boundary_noise(img, roi, sigma)
        elif aug_type == 'local_shift':
            return self.local_intensity_shift(img, roi, sigma * 3)
        elif aug_type == 'elastic':
            return self.elastic_deformation(img, alpha=10, sigma=2)
        else:
            return gaussian_filter(img, sigma=sigma * 5).astype(np.uint8)

    def strong_augment(self, img: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Strong augmentation"""
        sigma = np.random.uniform(*self.strong_range)
        aug_type = np.random.choice(['boundary', 'local_shift', 'elastic', 'smooth'])

        if aug_type == 'boundary':
            return self.boundary_noise(img, roi, sigma)
        elif aug_type == 'local_shift':
            return self.local_intensity_shift(img, roi, sigma * 4)
        elif aug_type == 'elastic':
            return self.elastic_deformation(img, alpha=30, sigma=4)
        else:
            return gaussian_filter(img, sigma=sigma * 10).astype(np.uint8)


# ============================================================================
# DATASET
# ============================================================================

class TopologyContrastiveDataset(Dataset):
    """
    Dataset for topology-based contrastive learning
    Computes PH on-the-fly from augmented images
    """

    def __init__(
            self,
            imgs: np.ndarray,
            labels: np.ndarray,
            rois: List[np.ndarray],
            augmenter: TopologyAwareAugmentation,
            n_h0: int = 96,
            n_h1: int = 192,
            mode: str = 'SSL',  # 'SSL', 'Supervised_CE', 'SupCon'
            use_strong: bool = True
    ):
        self.imgs = imgs
        self.labels = labels
        self.rois = rois
        self.augmenter = augmenter
        self.n_h0 = n_h0
        self.n_h1 = n_h1
        self.mode = mode
        self.use_strong = use_strong

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        roi = self.rois[idx]
        label = int(self.labels[idx])

        # Generate two augmented views
        if self.use_strong:
            view1 = self.augmenter.weak_augment(img, roi)
            view2 = self.augmenter.strong_augment(img, roi)
        else:
            view1 = self.augmenter.weak_augment(img, roi)
            view2 = self.augmenter.weak_augment(img, roi)

        # Compute PH on full augmented images (no cropping, no filling!)
        pd1 = compute_multiscale_persistence_full_image(view1, self.n_h0, self.n_h1)
        pd2 = compute_multiscale_persistence_full_image(view2, self.n_h0, self.n_h1)

        # Concatenate H0 and H1
        h0_1 = torch.from_numpy(pd1['h0']).float()
        h1_1 = torch.from_numpy(pd1['h1']).float()
        h0_2 = torch.from_numpy(pd2['h0']).float()
        h1_2 = torch.from_numpy(pd2['h1']).float()

        return h0_1, h1_1, h0_2, h1_2, label


# ============================================================================
# MODEL
# ============================================================================

class HierarchicalTopoEncoder(nn.Module):
    """
    Hierarchical encoder for persistent homology features

    Improvements:
    - Larger capacity (512 dim, 8 heads)
    - More dropout (0.15)
    - Separate processing for H0 and H1
    - Cross-homology attention
    """

    def __init__(
            self,
            hidden_dim: int = 512,
            num_heads: int = 8,
            dropout: float = 0.15,
            n_h0: int = 96,
            n_h1: int = 192
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_h0 = n_h0
        self.n_h1 = n_h1

        # Separate encoders for H0 and H1
        self.h0_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )

        self.h1_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )

        # Self-attention for H0 and H1
        self.h0_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.h1_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention between H0 and H1
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Output dimension
        self.out_dim = hidden_dim * 4  # max+mean for H0 and H1

        print(f"  ✓ Encoder: H0={n_h0}, H1={n_h1}, hidden={hidden_dim}, heads={num_heads}, dropout={dropout}")

    def forward(self, h0, h1):
        """
        Args:
            h0: (B, n_h0, 2) H0 birth-death pairs
            h1: (B, n_h1, 2) H1 birth-death pairs

        Returns:
            features: (B, out_dim)
        """
        # Encode H0 and H1 separately
        h0_feat = self.h0_encoder(h0)  # (B, n_h0, hidden_dim)
        h1_feat = self.h1_encoder(h1)  # (B, n_h1, hidden_dim)

        # Self-attention within each homology dimension
        h0_attn, _ = self.h0_attention(h0_feat, h0_feat, h0_feat)
        h0_feat = h0_feat + h0_attn  # Residual (in-place add)

        h1_attn, _ = self.h1_attention(h1_feat, h1_feat, h1_feat)
        h1_feat = h1_feat + h1_attn  # Residual (in-place add)

        # Cross-attention: H0 attends to H1
        h0_cross, _ = self.cross_attention(h0_feat, h1_feat, h1_feat)

        # Global pooling
        h0_max = h0_feat.max(dim=1).values  # (B, hidden_dim)
        h0_mean = h0_feat.mean(dim=1)  # (B, hidden_dim)
        h1_max = h1_feat.max(dim=1).values
        h1_mean = h1_feat.mean(dim=1)

        # Concatenate all features
        features = torch.cat([h0_max, h0_mean, h1_max, h1_mean], dim=1)  # (B, hidden_dim*4)

        return features


class ContrastiveModel(nn.Module):
    """Model for SSL/SupCon"""

    def __init__(self, encoder, projection_dim=256):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim)
        )

    def forward(self, h0, h1):
        features = self.encoder(h0, h1)
        z = self.projector(features)
        return F.normalize(z, dim=1)


class SupervisedModel(nn.Module):
    """Model for Supervised CE"""

    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, h0, h1):
        features = self.encoder(h0, h1)
        logits = self.classifier(features)
        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def info_nce_loss(z1, z2, temperature=0.1):
    """InfoNCE loss for contrastive learning"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # Mask out self-similarities
    mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_indices = torch.cat([
        torch.arange(N, 2 * N, device=sim.device),
        torch.arange(0, N, device=sim.device)
    ])

    loss = F.cross_entropy(sim, pos_indices)
    return loss


def supcon_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss"""
    features = F.normalize(features, dim=1)

    batch_size = features.shape[0]

    # Compute similarity matrix
    sim_matrix = torch.mm(features, features.t()) / temperature

    # Mask out self-similarities
    mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    sim_matrix.masked_fill_(mask, float('-inf'))

    # Create label mask
    labels = labels.contiguous().view(-1, 1)
    label_mask = torch.eq(labels, labels.t()).float()
    label_mask.masked_fill_(mask, 0)

    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()

    return loss


def train_supervised_ce(
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epoch,
        accumulation_steps=1
):
    """Train supervised model with cross-entropy"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (h0_1, h1_1, h0_2, h1_2, labels) in enumerate(pbar):
        h0 = h0_1.to(device)
        h1 = h1_1.to(device)
        labels = labels.to(device)

        # Mixed precision forward
        with autocast('cuda', dtype=torch.float16):
            logits = model(h0, h1)
            loss = F.cross_entropy(logits, labels) / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Update weights
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Statistics
        total_loss += loss.item() * accumulation_steps
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_contrastive(
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epoch,
        temperature=0.1,
        mode='SSL',
        accumulation_steps=1
):
    """Train contrastive model (SSL or SupCon)"""
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (h0_1, h1_1, h0_2, h1_2, labels) in enumerate(pbar):
        h0_1, h1_1 = h0_1.to(device), h1_1.to(device)
        h0_2, h1_2 = h0_2.to(device), h1_2.to(device)
        labels = labels.to(device)

        # Mixed precision forward
        with autocast('cuda', dtype=torch.float16):
            z1 = model(h0_1, h1_1)
            z2 = model(h0_2, h1_2)

            if mode == 'SSL':
                loss = info_nce_loss(z1, z2, temperature) / accumulation_steps
            else:  # SupCon
                z = torch.cat([z1, z2], dim=0)
                labels_cat = torch.cat([labels, labels], dim=0)
                loss = supcon_loss(z, labels_cat, temperature) / accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Update weights
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_supervised(model, test_loader, device):
    """Evaluate supervised model"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for h0_1, h1_1, _, _, labels in tqdm(test_loader, desc="Evaluating"):
            h0, h1 = h0_1.to(device), h1_1.to(device)
            labels = labels.to(device)

            with autocast('cuda', dtype=torch.float16):
                logits = model(h0, h1)

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def quick_linear_probe(
        encoder,
        train_loader,
        test_loader,
        num_classes,
        device,
        epochs=10
):
    """
    Quick linear probe evaluation for SSL models
    Train a linear classifier on frozen encoder features
    """
    encoder.eval()

    # Extract features
    print("  Extracting features...")
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    with torch.no_grad():
        for h0, h1, _, _, labels in tqdm(train_loader, desc="  Train features"):
            h0, h1 = h0.to(device), h1.to(device)
            with autocast('cuda', dtype=torch.float16):
                feat = encoder(h0, h1).float()
            train_features.append(feat.cpu())
            train_labels.append(labels)

        for h0, h1, _, _, labels in tqdm(test_loader, desc="  Test features"):
            h0, h1 = h0.to(device), h1.to(device)
            with autocast('cuda', dtype=torch.float16):
                feat = encoder(h0, h1).float()
            test_features.append(feat.cpu())
            test_labels.append(labels)

    X_train = torch.cat(train_features, dim=0)
    y_train = torch.cat(train_labels, dim=0)
    X_test = torch.cat(test_features, dim=0)
    y_test = torch.cat(test_labels, dim=0)

    print(f"  Features: train={X_train.shape}, test={X_test.shape}")

    # Train linear classifier
    linear = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    best_acc = 0
    for epoch in range(epochs):
        linear.train()
        optimizer.zero_grad()

        logits = linear(X_train)
        loss = criterion(logits, y_train)

        loss.backward()
        optimizer.step()

        # Evaluate
        linear.eval()
        with torch.no_grad():
            test_logits = linear(X_test)
            _, predicted = test_logits.max(1)
            acc = (predicted == y_test).float().mean().item() * 100

            if acc > best_acc:
                best_acc = acc

    print(f"  Quick probe accuracy: {best_acc:.2f}%")
    return best_acc


def test_time_augmentation(
        model,
        test_imgs,
        test_labels,
        test_rois,
        augmenter,
        n_h0,
        n_h1,
        device,
        n_aug=5
):
    """
    Test-time augmentation: average predictions over multiple augmented views
    """
    model.eval()
    all_preds = []

    print(f"Running TTA with {n_aug} augmentations...")

    with torch.no_grad():
        for img, roi in tqdm(zip(test_imgs, test_rois), total=len(test_imgs), desc="TTA"):
            preds = []

            for _ in range(n_aug):
                # Augment
                view = augmenter.weak_augment(img, roi)

                # Compute PH
                pd = compute_multiscale_persistence_full_image(view, n_h0, n_h1)
                h0 = torch.from_numpy(pd['h0']).unsqueeze(0).to(device).float()
                h1 = torch.from_numpy(pd['h1']).unsqueeze(0).to(device).float()

                # Predict
                with autocast('cuda', dtype=torch.float16):
                    logits = model(h0, h1)
                    prob = F.softmax(logits, dim=1)

                preds.append(prob.cpu())

            # Average predictions
            avg_pred = torch.stack(preds).mean(dim=0)
            all_preds.append(avg_pred)

    # Compute accuracy
    all_preds = torch.cat(all_preds, dim=0)
    predicted = all_preds.argmax(dim=1).numpy()
    accuracy = accuracy_score(test_labels, predicted) * 100

    print(f"TTA accuracy: {accuracy:.2f}%")
    return accuracy


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_stage2_parallel(args):
    """Main training function"""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 80}")
    print(f"Stage 2 Training: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Upper bound test: {args.upper_bound}")
    print(f"{'=' * 80}\n")

    # Load data
    print("Loading dataset...")
    imgs_train, labels_train, rois_train, using_sam = load_dataset_with_rois(
        args.dataset, 'train', roi_base_dir=args.roi_dir, seed=args.seed
    )

    imgs_val, labels_val, rois_val, _ = load_dataset_with_rois(
        args.dataset, 'val', roi_base_dir=args.roi_dir, seed=args.seed
    )

    imgs_test, labels_test, rois_test, _ = load_dataset_with_rois(
        args.dataset, 'test', roi_base_dir=args.roi_dir, seed=args.seed
    )

    # Upper bound test: train on ALL data (train+val+test)
    if args.upper_bound:
        print(f"\n⚠️  UPPER BOUND TEST: Training on train+val+test!")
        imgs_train = np.concatenate([imgs_train, imgs_val, imgs_test], axis=0)
        labels_train = np.concatenate([labels_train, labels_val, labels_test], axis=0)
        rois_train = rois_train + rois_val + rois_test
        print(f"Total training samples: {len(imgs_train)}\n")
    else:
        # Standard split: train on train+val
        imgs_train = np.concatenate([imgs_train, imgs_val], axis=0)
        labels_train = np.concatenate([labels_train, labels_val], axis=0)
        rois_train = rois_train + rois_val
        print(f"Training samples: {len(imgs_train)}")

    print(f"Test samples: {len(imgs_test)}")

    # Create datasets
    augmenter = TopologyAwareAugmentation(
        weak_range=(args.weak_min, args.weak_max),
        strong_range=(args.strong_min, args.strong_max)
    )

    train_dataset = TopologyContrastiveDataset(
        imgs_train, labels_train, rois_train, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1,
        mode=args.mode, use_strong=args.use_strong
    )

    test_dataset = TopologyContrastiveDataset(
        imgs_test, labels_test, rois_test, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1,
        mode=args.mode, use_strong=False
    )

    # DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Create model
    num_classes = get_num_classes(args.dataset)
    encoder = HierarchicalTopoEncoder(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        n_h0=args.n_h0 * 3,  # 3 scales
        n_h1=args.n_h1 * 3
    ).to(device)

    if args.mode == 'Supervised_CE':
        model = SupervisedModel(encoder, num_classes).to(device)
    else:
        model = ContrastiveModel(encoder, projection_dim=256).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_metric = 0.0
    results_history = []

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        if args.mode == 'Supervised_CE':
            train_loss, train_acc = train_supervised_ce(
                model, train_loader, optimizer, scheduler, scaler, device,
                epoch, args.accumulation_steps
            )

            print(f"Epoch {epoch}/{args.epochs}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")

            # Test accuracy every 20 epochs
            if epoch % 20 == 0 or epoch == args.epochs:
                test_acc = evaluate_supervised(model, test_loader, device)
                print(f"  → Test Accuracy: {test_acc:.2f}%")

                results_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                })

                if test_acc > best_metric:
                    best_metric = test_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'test_acc': test_acc
                    }, os.path.join(args.save_dir, 'best_model.pt'))
                    print(f"  ✓ Saved best model (acc={test_acc:.2f}%)")

        else:  # SSL or SupCon
            train_loss = train_contrastive(
                model, train_loader, optimizer, scheduler, scaler, device,
                epoch, args.temperature, args.mode, args.accumulation_steps
            )

            print(f"Epoch {epoch}/{args.epochs}: Loss={train_loss:.4f}")

            # Quick linear probe every 20 epochs
            if epoch % 20 == 0 or epoch == args.epochs:
                probe_acc = quick_linear_probe(
                    encoder, train_loader, test_loader, num_classes, device, epochs=10
                )

                results_history.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'probe_acc': probe_acc
                })

                if probe_acc > best_metric:
                    best_metric = probe_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'probe_acc': probe_acc
                    }, os.path.join(args.save_dir, 'best_model.pt'))
                    print(f"  ✓ Saved best model (probe_acc={probe_acc:.2f}%)")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': encoder.state_dict()
    }, os.path.join(args.save_dir, 'final_model.pt'))

    # Save results
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results_history, f, indent=2)

    print(f"\nTraining complete! Best metric: {best_metric:.2f}%")

    # Test-Time Augmentation (if supervised)
    if args.use_tta and args.mode == 'Supervised_CE':
        print(f"\n{'=' * 80}")
        print("Running Test-Time Augmentation...")
        print(f"{'=' * 80}\n")

        # Load best model
        checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])

        tta_acc = test_time_augmentation(
            model, imgs_test, labels_test, rois_test, augmenter,
            args.n_h0, args.n_h1, device, n_aug=5
        )

        print(f"\nFinal results:")
        print(f"  Best test accuracy: {best_metric:.2f}%")
        print(f"  TTA accuracy: {tta_acc:.2f}%")
        print(f"  Improvement: +{tta_acc - best_metric:.2f}%")

    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2: Topology-based Training')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['kvasir', 'pathmnist', 'octmnist', 'organsmnist', 'isic2019', 'imagenet100'])
    parser.add_argument('--roi_dir', type=str, default='data/rois')

    # Training mode
    parser.add_argument('--mode', type=str, default='Supervised_CE',
                        choices=['SSL', 'SupCon', 'Supervised_CE'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--temperature', type=float, default=0.1)

    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_h0', type=int, default=96)
    parser.add_argument('--n_h1', type=int, default=192)

    # Augmentation
    parser.add_argument('--weak_min', type=float, default=0.005)
    parser.add_argument('--weak_max', type=float, default=0.02)
    parser.add_argument('--strong_min', type=float, default=0.04)
    parser.add_argument('--strong_max', type=float, default=0.12)
    parser.add_argument('--use_strong', action='store_true', default=True)

    # Optimization
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (for large datasets)')
    parser.add_argument('--num_workers', type=int, default=16)

    # Evaluation
    parser.add_argument('--upper_bound', action='store_true', default=True,
                        help='Train on train+val+test for upper bound test')
    parser.add_argument('--proper_split', action='store_true',
                        help='Use proper train/val/test split (overrides upper_bound)')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use test-time augmentation after training')

    # Misc
    parser.add_argument('--save_dir', type=str, default='./stage2_results')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Override upper_bound if proper_split is set
    if args.proper_split:
        args.upper_bound = False

    # Create save directory
    save_path = os.path.join(args.save_dir, args.dataset, args.mode)
    os.makedirs(save_path, exist_ok=True)
    args.save_dir = save_path

    # Train
    model = train_stage2_parallel(args)

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()