#!/usr/bin/env python3
"""
Stage 2: Unified Topology Encoder Training
Supports: SimCLR, MoCoV3, BYOL, Barlow Twins
"""

import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

import cripser
from skimage import filters, feature, measure
from skimage.morphology import binary_dilation, binary_erosion, disk
from scipy.ndimage import gaussian_filter

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset Loading (Universal)
# ============================================================================

def load_dataset_as_numpy(dataset_name, split, data_dir=None, img_size=224, seed=42):
    """Load any dataset and return as numpy arrays"""
    from torchvision import datasets
    
    dataset_name_lower = dataset_name.lower()
    
    # MedMNIST datasets
    if dataset_name_lower in ['dermamnist', 'pathmnist', 'octmnist', 'organamnist', 
                              'pneumoniamnist', 'organcmnist', 'organsmnist']:
        import medmnist
        from medmnist import INFO
        
        info = INFO[dataset_name_lower]
        DataClass = getattr(medmnist, info['python_class'])
        ds = DataClass(split=split, download=True, size=img_size, as_rgb=True)
        
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
            raise ValueError("data_dir required for ISIC2019")
        
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        train_ds = datasets.ImageFolder(train_dir)
        test_ds = datasets.ImageFolder(test_dir)
        
        all_paths = [s[0] for s in train_ds.samples] + [s[0] for s in test_ds.samples]
        all_labels = train_ds.targets + test_ds.targets
        
        train_idx, test_idx = train_test_split(
            range(len(all_labels)), train_size=0.7, test_size=0.3,
            stratify=all_labels, random_state=seed
        )
        
        if split == 'train':
            paths = [all_paths[i] for i in train_idx]
            labels = [all_labels[i] for i in train_idx]
        else:
            paths = [all_paths[i] for i in test_idx]
            labels = [all_labels[i] for i in test_idx]
        
        # Load images
        print(f"Loading {len(paths)} images from ISIC2019...")
        imgs = []
        for path in tqdm(paths, desc=f"Loading {split} images"):
            img = Image.open(path).convert('RGB').resize((img_size, img_size))
            imgs.append(np.array(img))
        
        return np.array(imgs, dtype=np.uint8), np.array(labels)
    
    # Kvasir
    elif dataset_name_lower == 'kvasir':
        if data_dir is None:
            raise ValueError("data_dir required for Kvasir")
        
        full_ds = datasets.ImageFolder(data_dir)
        all_paths = [s[0] for s in full_ds.samples]
        all_labels = full_ds.targets
        
        train_idx, test_idx = train_test_split(
            range(len(all_labels)), train_size=0.7, test_size=0.3,
            stratify=all_labels, random_state=seed
        )
        
        if split == 'train':
            paths = [all_paths[i] for i in train_idx]
            labels = [all_labels[i] for i in train_idx]
        else:
            paths = [all_paths[i] for i in test_idx]
            labels = [all_labels[i] for i in test_idx]
        
        # Load images
        print(f"Loading {len(paths)} images from Kvasir...")
        imgs = []
        for path in tqdm(paths, desc=f"Loading {split} images"):
            img = Image.open(path).convert('RGB').resize((img_size, img_size))
            imgs.append(np.array(img))
        
        return np.array(imgs, dtype=np.uint8), np.array(labels)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_num_classes(dataset_name: str):
    num_classes_map = {
        'pathmnist': 9, 'octmnist': 4, 'pneumoniamnist': 2, 'dermamnist': 7,
        'organamnist': 11, 'organcmnist': 11, 'organsmnist': 11,
        'isic2019': 8, 'kvasir': 8
    }
    return num_classes_map[dataset_name.lower()]


# ============================================================================
# ROI Extraction
# ============================================================================

class ImprovedROIExtractor:
    """Enhanced ROI extraction with better boundary detection"""
    
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


# ============================================================================
# Topological Augmentations
# ============================================================================

class EnhancedTopologicalAugmentations:
    """Topology-aware augmentations that preserve or modify structure"""
    
    def __init__(self, weak_range=(0.02, 0.08), strong_range=(0.08, 0.15)):
        self.weak_range = weak_range
        self.strong_range = strong_range
    
    def boundary_noise(self, img, roi, sigma):
        """Add noise near ROI boundaries"""
        from scipy.ndimage import distance_transform_edt
        
        if not roi.any():
            return img
        
        dist = distance_transform_edt(roi)
        boundary_mask = (dist > 0) & (dist < 5)
        
        noise = np.random.randn(*img.shape) * sigma * 50
        noisy = img.astype(np.float32) + noise * boundary_mask[..., None]
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def local_intensity_shift(self, img, roi, intensity):
        """Shift intensity in local regions"""
        if not roi.any():
            return img
        
        labels = measure.label(roi)
        shifted = img.copy().astype(np.float32)
        
        for region_id in range(1, labels.max() + 1):
            mask = labels == region_id
            shift = np.random.uniform(-intensity, intensity)
            shifted[mask] += shift
        
        return np.clip(shifted, 0, 255).astype(np.uint8)
    
    def elastic_deformation(self, img, alpha=30, sigma=4):
        """Elastic deformation"""
        shape = img.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x_new = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
        
        from scipy.ndimage import map_coordinates
        if img.ndim == 3:
            return np.stack([
                map_coordinates(img[..., i], [y_new, x_new], order=1, mode='reflect')
                for i in range(img.shape[2])
            ], axis=2).astype(np.uint8)
        else:
            return map_coordinates(img, [y_new, x_new], order=1, mode='reflect').astype(np.uint8)
    
    def weak_augment(self, img, roi):
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
    
    def strong_augment(self, img, roi):
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
# Persistence Computation
# ============================================================================

def compute_multiscale_persistence(img, roi, n_h0=48, n_h1=96):
    """Enhanced persistence computation with multi-scale"""
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img.copy()
    
    # Flatten outside ROI to median
    if roi.any():
        median_val = np.median(gray[roi])
        gray[~roi] = median_val
    
    # Multi-scale computation
    scales = [1.0, 0.5, 0.25]  # Original, half, quarter resolution
    all_h0, all_h1 = [], []
    
    for scale in scales:
        if scale != 1.0:
            from skimage.transform import rescale
            gray_scaled = rescale(gray, scale, anti_aliasing=True, preserve_range=True)
        else:
            gray_scaled = gray
        
        # Normalize
        gray_norm = ((gray_scaled - gray_scaled.min()) / 
                     (gray_scaled.max() - gray_scaled.min() + 1e-8) * 255).astype(np.uint8)
        
        # Compute PH
        ph = cripser.computePH(gray_norm)
        
        for item in ph:
            if len(item) >= 3:
                dim, birth, death = int(item[0]), float(item[1]), float(item[2])
                if death >= 1e5:
                    continue
                
                birth, death = birth/255.0, death/255.0
                persistence = death - birth
                
                if dim == 0 and persistence > 0.005:
                    all_h0.append((birth, death, scale))
                elif dim == 1 and persistence > 0.005:
                    all_h1.append((birth, death, scale))
    
    # Sort by persistence
    all_h0.sort(key=lambda x: x[1]-x[0], reverse=True)
    all_h1.sort(key=lambda x: x[1]-x[0], reverse=True)
    
    # Convert to fixed size arrays
    def pad_points(pts, k):
        result = []
        for i in range(k):
            if i < len(pts):
                result.append([pts[i][0], pts[i][1]])  # birth, death
            else:
                result.append([0.0, 0.0])  # padding
        return np.array(result, dtype=np.float32)
    
    return {
        'h0': pad_points(all_h0, n_h0),
        'h1': pad_points(all_h1, n_h1)
    }


# ============================================================================
# Dataset
# ============================================================================

class TopologyContrastiveDataset(Dataset):
    """Universal dataset for all contrastive learning methods"""
    
    def __init__(self, imgs, labels, augmenter, roi_extractor, 
                 n_h0=48, n_h1=96, use_strong=False):
        self.imgs = imgs
        self.labels = labels
        self.augmenter = augmenter
        self.roi_extractor = roi_extractor
        self.n_h0 = n_h0
        self.n_h1 = n_h1
        self.use_strong = use_strong
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        # Extract ROI
        roi = self.roi_extractor.extract(img)
        
        # Create two augmented views
        if self.use_strong:
            view1 = self.augmenter.strong_augment(img, roi)
            view2 = self.augmenter.strong_augment(img, roi)
        else:
            view1 = self.augmenter.weak_augment(img, roi)
            view2 = self.augmenter.weak_augment(img, roi)
        
        # Compute persistence for both views
        pd1 = compute_multiscale_persistence(view1, roi, self.n_h0, self.n_h1)
        pd2 = compute_multiscale_persistence(view2, roi, self.n_h0, self.n_h1)
        
        # Return separate H0 and H1
        return (torch.from_numpy(pd1['h0']), torch.from_numpy(pd1['h1']),
                torch.from_numpy(pd2['h0']), torch.from_numpy(pd2['h1']),
                label)


# ============================================================================
# Base Encoder: Hierarchical Topological Encoder
# ============================================================================

class HierarchicalTopoEncoder(nn.Module):
    """
    Hierarchical encoder for H0/H1 with cross-attention
    Uses fixed homology dimension flags: H0=[1,0], H1=[0,1]
    """
    def __init__(self, input_dim=2, hidden_dim=256, num_heads=8, dropout=0.1, 
                 n_h0=48, n_h1=96):
        super().__init__()
        self.n_h0 = n_h0
        self.n_h1 = n_h1
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim  # For compatibility with wrappers
        
        # Input: [birth, death, h0_flag, h1_flag] = 4 dimensions
        actual_input_dim = input_dim + 2
        
        # Deeper encoders for better feature extraction
        self.h0_encoder = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.h1_encoder = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Self-attention with dropout
        self.h0_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.h1_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between H0 and H1
        self.h0_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.h1_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, h0_features, h1_features):
        """
        Args:
            h0_features: [batch, n_h0, 2] (birth, death)
            h1_features: [batch, n_h1, 2] (birth, death)
        Returns:
            features: [batch, hidden_dim]
        """
        B = h0_features.size(0)
        
        # Add fixed homology dimension flags
        # H0: [birth, death, 1, 0]
        h0_flags = torch.zeros(B, self.n_h0, 2, device=h0_features.device)
        h0_flags[:, :, 0] = 1.0  # H0 indicator
        h0_input = torch.cat([h0_features, h0_flags], dim=2)  # [B, n_h0, 4]
        
        # H1: [birth, death, 0, 1]
        h1_flags = torch.zeros(B, self.n_h1, 2, device=h1_features.device)
        h1_flags[:, :, 1] = 1.0  # H1 indicator
        h1_input = torch.cat([h1_features, h1_flags], dim=2)  # [B, n_h1, 4]
        
        # Encode
        h0 = self.h0_encoder(h0_input)  # [B, n_h0, hidden_dim]
        h1 = self.h1_encoder(h1_input)  # [B, n_h1, hidden_dim]
        
        # Self-attention with residual
        h0_self, _ = self.h0_self_attn(h0, h0, h0)
        h0 = h0 + h0_self  # Residual connection
        
        h1_self, _ = self.h1_self_attn(h1, h1, h1)
        h1 = h1 + h1_self  # Residual connection
        
        # Cross-attention
        h0_cross, _ = self.h0_cross_attn(h0, h1, h1)  # H0 attends to H1
        h1_cross, _ = self.h1_cross_attn(h1, h0, h0)  # H1 attends to H0
        
        # Dual pooling: max + mean (richer representation)
        h0_max = h0_cross.max(dim=1).values  # [B, hidden_dim]
        h0_mean = h0_cross.mean(dim=1)       # [B, hidden_dim]
        h1_max = h1_cross.max(dim=1).values  # [B, hidden_dim]
        h1_mean = h1_cross.mean(dim=1)       # [B, hidden_dim]
        
        # Combine all pooled features
        combined = torch.cat([h0_max, h0_mean, h1_max, h1_mean], dim=1)  # [B, hidden_dim*4]
        
        return combined


# ============================================================================
# Helper: MLP for projectors/predictors
# ============================================================================

def MLP(in_dim, hidden_dim, out_dim, dropout=0.0):
    """Helper to create MLP for projector/predictor"""
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
    return nn.Sequential(*layers)


# ============================================================================
# Method Wrappers
# ============================================================================

class SimCLRWrapper(nn.Module):
    """SimCLR: Simple contrastive learning"""
    def __init__(self, base_encoder, proj_hidden=512, proj_dim=256):
        super().__init__()
        self.encoder = base_encoder
        encoder_out_dim = base_encoder.out_dim * 4  # 4x due to pooling
        self.projector = MLP(encoder_out_dim, proj_hidden, proj_dim)
        
    def forward(self, h0_1, h1_1, h0_2, h1_2, temperature=0.1):
        # Get features from both views
        feat1 = self.encoder(h0_1, h1_1)
        feat2 = self.encoder(h0_2, h1_2)
        
        # Project
        z1 = self.projector(feat1)
        z2 = self.projector(feat2)
        
        # InfoNCE loss
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / temperature
        
        mask = torch.eye(2*N, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))
        
        pos_indices = torch.cat([
            torch.arange(N, 2*N, device=sim.device),
            torch.arange(0, N, device=sim.device)
        ])
        
        loss = F.cross_entropy(sim, pos_indices)
        return loss


class MoCoV3Wrapper(nn.Module):
    """MoCo v3: Momentum contrastive learning"""
    def __init__(self, base_encoder_cls, n_h0, n_h1, hidden_dim, 
                 proj_dim=256, proj_hidden=512, pred_hidden=512, m=0.99, dropout=0.1):
        super().__init__()
        self.m = m
        
        # Query and key encoders
        self.encoder_q = base_encoder_cls(n_h0=n_h0, n_h1=n_h1, hidden_dim=hidden_dim, dropout=dropout)
        self.encoder_k = base_encoder_cls(n_h0=n_h0, n_h1=n_h1, hidden_dim=hidden_dim, dropout=dropout)
        
        encoder_out_dim = self.encoder_q.out_dim * 4
        
        # Projector and predictor
        self.projector_q = MLP(encoder_out_dim, proj_hidden, proj_dim, dropout=dropout)
        self.projector_k = MLP(encoder_out_dim, proj_hidden, proj_dim)
        self.predictor = MLP(proj_dim, pred_hidden, proj_dim, dropout=dropout)
        
        # Initialize key encoder
        self._init_key_encoder()
        
    def _init_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def contrastive_loss(self, q, k, temperature):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        logits = torch.mm(q, k.t()) / temperature
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)
    
    def forward(self, h0_1, h1_1, h0_2, h1_2, temperature=0.1):
        # Query pipeline
        feat_q1 = self.encoder_q(h0_1, h1_1)
        proj_q1 = self.projector_q(feat_q1)
        q1 = self.predictor(proj_q1)
        
        feat_q2 = self.encoder_q(h0_2, h1_2)
        proj_q2 = self.projector_q(feat_q2)
        q2 = self.predictor(proj_q2)
        
        # Key pipeline (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k1 = self.projector_k(self.encoder_k(h0_1, h1_1))
            k2 = self.projector_k(self.encoder_k(h0_2, h1_2))
        
        # Symmetric contrastive loss
        loss = self.contrastive_loss(q1, k2, temperature) + self.contrastive_loss(q2, k1, temperature)
        return loss


class BYOLWrapper(nn.Module):
    """BYOL: Bootstrap your own latent"""
    def __init__(self, base_encoder_cls, n_h0, n_h1, hidden_dim,
                 proj_dim=256, proj_hidden=512, pred_hidden=512, m=0.99, dropout=0.1):
        super().__init__()
        self.m = m
        
        # Online and target encoders
        self.online_encoder = base_encoder_cls(n_h0=n_h0, n_h1=n_h1, hidden_dim=hidden_dim, dropout=dropout)
        self.target_encoder = base_encoder_cls(n_h0=n_h0, n_h1=n_h1, hidden_dim=hidden_dim, dropout=dropout)
        
        encoder_out_dim = self.online_encoder.out_dim * 4
        
        # Projector and predictor
        self.online_projector = MLP(encoder_out_dim, proj_hidden, proj_dim)
        self.predictor = MLP(proj_dim, pred_hidden, proj_dim)
        self.target_projector = MLP(encoder_out_dim, proj_hidden, proj_dim)
        
        self._init_target_network()
    
    @torch.no_grad()
    def _init_target_network(self):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_target_network(self):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def regression_loss(self, p, z):
        p_norm = F.normalize(p, dim=1)
        z_norm = F.normalize(z, dim=1)
        return 2 - 2 * (p_norm * z_norm).sum(dim=1)
    
    def forward(self, h0_1, h1_1, h0_2, h1_2, **kwargs):
        # Online predictions
        p1 = self.predictor(self.online_projector(self.online_encoder(h0_1, h1_1)))
        p2 = self.predictor(self.online_projector(self.online_encoder(h0_2, h1_2)))
        
        # Target projections
        with torch.no_grad():
            self._momentum_update_target_network()
            z1 = self.target_projector(self.target_encoder(h0_1, h1_1))
            z2 = self.target_projector(self.target_encoder(h0_2, h1_2))
        
        # Symmetric regression loss
        loss = self.regression_loss(p1, z2).mean() + self.regression_loss(p2, z1).mean()
        return loss


class BarlowTwinsWrapper(nn.Module):
    """Barlow Twins: Self-supervised learning via redundancy reduction"""
    def __init__(self, base_encoder, proj_hidden=512, proj_dim=256, lambd=0.01):
        super().__init__()
        self.encoder = base_encoder
        self.lambd = lambd
        
        encoder_out_dim = base_encoder.out_dim * 4
        
        # Barlow-style projector
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
        )
        
        # Batch norm before loss
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)
        
    def forward(self, h0_1, h1_1, h0_2, h1_2, **kwargs):
        # Get features from both views
        feat1 = self.encoder(h0_1, h1_1)
        feat2 = self.encoder(h0_2, h1_2)
        
        # Project and normalize
        z1 = self.bn(self.projector(feat1))
        z2 = self.bn(self.projector(feat2))
        
        # Barlow Twins loss
        batch_size = z1.size(0)
        feature_dim = z1.size(1)
        
        # Empirical cross-correlation matrix
        c = (z1.T @ z2) / batch_size
        
        # Loss: encourage identity matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(feature_dim - 1, feature_dim + 1)[:, :-1].pow_(2).sum()
        
        loss = on_diag + self.lambd * off_diag
        return loss


class DINOWrapper(nn.Module):
    """DINO: Self-distillation with no labels"""

    def __init__(self, base_encoder_cls, n_h0, n_h1, hidden_dim,
                 proj_dim=256, proj_hidden=2048, n_crops=2,
                 student_temp=0.1, teacher_temp=0.04, m=0.996, dropout=0.1):
        super().__init__()
        self.m = m
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.n_crops = n_crops

        # Student and teacher encoders
        self.student_encoder = base_encoder_cls(n_h0=n_h0, n_h1=n_h1, hidden_dim=hidden_dim, dropout=dropout)
        self.teacher_encoder = base_encoder_cls(n_h0=n_h0, n_h1=n_h1, hidden_dim=hidden_dim, dropout=dropout)

        encoder_out_dim = self.student_encoder.out_dim * 4

        # Projector (3 layers)
        self.student_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        self.teacher_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

        # Centering for teacher outputs
        self.register_buffer("center", torch.zeros(1, proj_dim))

        self._init_teacher()

    @torch.no_grad()
    def _init_teacher(self):
        for param_s, param_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_s, param_t in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_s, param_t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_t.data = param_t.data * self.m + param_s.data * (1. - self.m)
        for param_s, param_t in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            param_t.data = param_t.data * self.m + param_s.data * (1. - self.m)

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """Update center used for teacher output"""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # EMA update
        self.center = self.center * 0.9 + batch_center * 0.1

    def forward(self, h0_1, h1_1, h0_2, h1_2, **kwargs):
        # Student predictions
        s1 = self.student_projector(self.student_encoder(h0_1, h1_1))
        s2 = self.student_projector(self.student_encoder(h0_2, h1_2))

        # Normalize student outputs
        s1 = F.normalize(s1, dim=-1)
        s2 = F.normalize(s2, dim=-1)

        # Teacher predictions
        with torch.no_grad():
            self._momentum_update_teacher()
            t1 = self.teacher_projector(self.teacher_encoder(h0_1, h1_1))
            t2 = self.teacher_projector(self.teacher_encoder(h0_2, h1_2))

            # Center and normalize teacher outputs
            t1 = F.normalize(t1 - self.center, dim=-1)
            t2 = F.normalize(t2 - self.center, dim=-1)

            # Update center
            self._update_center(torch.cat([t1, t2]))

        # Compute cross-entropy loss (student predicts teacher)
        # Apply temperature
        s1_temp = s1 / self.student_temp
        s2_temp = s2 / self.student_temp
        t1_temp = t1 / self.teacher_temp
        t2_temp = t2 / self.teacher_temp

        # Cross-entropy: student1 -> teacher2, student2 -> teacher1
        loss1 = -torch.sum(t2_temp * F.log_softmax(s1_temp, dim=-1), dim=-1).mean()
        loss2 = -torch.sum(t1_temp * F.log_softmax(s2_temp, dim=-1), dim=-1).mean()

        loss = (loss1 + loss2) / 2
        return loss

# ============================================================================
# Training
# ============================================================================

def train_stage2(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Stage 2: Training Topological Encoder - {args.method.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*70}\n")
    
    # Determine data directory
    data_dir = None
    if args.dataset.lower() == 'isic2019':
        data_dir = os.path.expanduser(args.isic_root)
    elif args.dataset.lower() == 'kvasir':
        data_dir = os.path.expanduser(args.kvasir_root)
    
    # Load data
    print("Loading dataset...")
    imgs_train, labels_train = load_dataset_as_numpy(
        args.dataset, 'train', data_dir=data_dir, img_size=224, seed=args.seed
    )
    
    if args.n_samples:
        imgs_train = imgs_train[:args.n_samples]
        labels_train = labels_train[:args.n_samples]
    
    # Train/val split
    n_train = len(imgs_train)
    n_val = int(0.15 * n_train)
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    
    # Create datasets
    roi_extractor = ImprovedROIExtractor()
    augmenter = EnhancedTopologicalAugmentations(
        weak_range=(args.weak_min, args.weak_max),
        strong_range=(args.strong_min, args.strong_max)
    )
    
    full_dataset = TopologyContrastiveDataset(
        imgs_train, labels_train, augmenter, roi_extractor,
        n_h0=args.n_h0, n_h1=args.n_h1, use_strong=args.use_strong
    )
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"\nBatches per epoch: {len(train_loader)}")
    print(f"Total updates: {len(train_loader) * args.epochs}")
    
    # Create model based on method
    if args.method == 'simclr':
        base_encoder = HierarchicalTopoEncoder(
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            dropout=args.dropout, n_h0=args.n_h0, n_h1=args.n_h1
        )
        model = SimCLRWrapper(base_encoder, proj_hidden=args.proj_hidden, proj_dim=args.proj_dim).to(device)
        
    elif args.method == 'mocov3':
        model = MoCoV3Wrapper(
            HierarchicalTopoEncoder, n_h0=args.n_h0, n_h1=args.n_h1,
            hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
            proj_hidden=args.proj_hidden, pred_hidden=args.pred_hidden,
            m=args.momentum, dropout=args.dropout
        ).to(device)
        
    elif args.method == 'byol':
        model = BYOLWrapper(
            HierarchicalTopoEncoder, n_h0=args.n_h0, n_h1=args.n_h1,
            hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
            proj_hidden=args.proj_hidden, pred_hidden=args.pred_hidden,
            m=args.momentum, dropout=args.dropout
        ).to(device)

    elif args.method == 'dino':
        model = DINOWrapper(
            HierarchicalTopoEncoder, n_h0=args.n_h0, n_h1=args.n_h1,
            hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
            proj_hidden=args.proj_hidden,
            student_temp=args.student_temp, teacher_temp=args.teacher_temp,
            m=args.dino_momentum, dropout=args.dropout
        ).to(device)

        
    elif args.method == 'barlow':
        base_encoder = HierarchicalTopoEncoder(
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            dropout=args.dropout, n_h0=args.n_h0, n_h1=args.n_h1
        )
        model = BarlowTwinsWrapper(
            base_encoder, proj_hidden=args.proj_hidden,
            proj_dim=args.proj_dim, lambd=args.lambd
        ).to(device)
    
    print(f"\n[DEBUG] Created {args.method.upper()} model")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    
    # Cosine scheduler with warmup
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )
    
    # Mixed precision training - auto-detect FP16 or BF16
    use_amp = False
    amp_dtype = torch.float32
    scaler = None
    
    if args.use_amp and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
            print(f"\n✓ Using BF16 mixed precision training")
        else:
            use_amp = True
            amp_dtype = torch.float16
            scaler = GradScaler()
            print(f"\n✓ Using FP16 mixed precision training")
    else:
        print(f"\n✓ Using FP32 training")
    
    # Training config
    print(f"\nTraining config:")
    print(f"  Method: {args.method.upper()}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    if args.method in ['simclr', 'mocov3']:
        print(f"  Temperature: {args.temperature}")
    if args.method in ['mocov3', 'byol']:
        print(f"  Momentum: {args.momentum}")
    if args.method == 'dino':  # ADD THIS
        print(f"  Student temp: {args.student_temp}")
        print(f"  Teacher temp: {args.teacher_temp}")
        print(f"  Momentum: {args.dino_momentum}")

    if args.method == 'barlow':
        print(f"  Lambda: {args.lambd}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        # Warmup
        if epoch < warmup_epochs:
            lr = args.lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for h0_1, h1_1, h0_2, h1_2, _ in pbar:
            h0_1, h1_1 = h0_1.to(device).float(), h1_1.to(device).float()
            h0_2, h1_2 = h0_2.to(device).float(), h1_2.to(device).float()
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(dtype=amp_dtype):
                    loss = model(h0_1, h1_1, h0_2, h1_2, temperature=args.temperature)
                
                if scaler is not None:  # FP16
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:  # BF16
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                loss = model(h0_1, h1_1, h0_2, h1_2, temperature=args.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for h0_1, h1_1, h0_2, h1_2, _ in val_loader:
                h0_1, h1_1 = h0_1.to(device).float(), h1_1.to(device).float()
                h0_2, h1_2 = h0_2.to(device).float(), h1_2.to(device).float()
                
                loss = model(h0_1, h1_1, h0_2, h1_2, temperature=args.temperature)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save appropriate encoder based on method
            if args.method == 'simclr':
                encoder_state = model.encoder.state_dict()
            elif args.method == 'mocov3':
                encoder_state = model.encoder_q.state_dict()
            elif args.method == 'byol':
                encoder_state = model.online_encoder.state_dict()
            elif args.method == 'dino':
                encoder_state = model.student_encoder.state_dict()

            elif args.method == 'barlow':
                encoder_state = model.encoder.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'args': vars(args)
            }, os.path.join(args.out_dir, f'stage2_{args.dataset.lower()}_{args.method}_best.pt'))
            
            print(f"  → Saved best model (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"Stage 2 Training Complete - {args.dataset} - {args.method.upper()}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")
    
    # Save training history
    with open(os.path.join(args.out_dir, f'stage2_{args.dataset.lower()}_{args.method}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return best_val_loss


# ============================================================================
# Linear Probe Evaluation
# ============================================================================

def linear_probe_evaluation(args):
    """Evaluate learned topology features with linear probe"""
    print("\n" + "=" * 70)
    print(f"Linear Probe Evaluation - {args.method.upper()}")
    print("=" * 70 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint_path = os.path.join(args.out_dir, f'stage2_{args.dataset.lower()}_{args.method}_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load just the encoder
    encoder = HierarchicalTopoEncoder(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        n_h0=args.n_h0,
        n_h1=args.n_h1
    ).to(device)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    print("✓ Loaded encoder from:", checkpoint_path)

    # Determine data directory
    data_dir = None
    if args.dataset.lower() == 'isic2019':
        data_dir = os.path.expanduser(args.isic_root)
    elif args.dataset.lower() == 'kvasir':
        data_dir = os.path.expanduser(args.kvasir_root)

    # Load data
    print("Loading datasets...")
    imgs_train, labels_train = load_dataset_as_numpy(
        args.dataset, 'train', data_dir=data_dir, img_size=224, seed=args.seed
    )
    imgs_test, labels_test = load_dataset_as_numpy(
        args.dataset, 'test', data_dir=data_dir, img_size=224, seed=args.seed
    )

    roi_extractor = ImprovedROIExtractor()

    # Extract features
    def extract_features(imgs):
        features_list = []

        for img in tqdm(imgs, desc="Extracting features"):
            roi = roi_extractor.extract(img)
            pd = compute_multiscale_persistence(img, roi, args.n_h0, args.n_h1)

            h0 = torch.from_numpy(pd['h0']).unsqueeze(0).float().to(device)
            h1 = torch.from_numpy(pd['h1']).unsqueeze(0).float().to(device)

            with torch.no_grad():
                feat = encoder(h0, h1)  # [1, hidden_dim*4]
                features_list.append(feat.cpu().numpy())

        return np.concatenate(features_list, axis=0)

    print("\nExtracting training features...")
    X_train = extract_features(imgs_train)

    print("Extracting test features...")
    X_test = extract_features(imgs_test)

    # Train linear classifier
    print("\nTraining linear probe...")
    clf = LogisticRegression(max_iter=1000, random_state=args.seed, n_jobs=-1)
    clf.fit(X_train, labels_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = accuracy_score(labels_test, y_pred)
    f1 = f1_score(labels_test, y_pred, average='macro')

    num_classes = len(np.unique(labels_train))
    if num_classes == 2:
        auc = roc_auc_score(labels_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(labels_test, y_prob, multi_class='ovr', average='macro')

    print(f"\n{'=' * 70}")
    print(f"Linear Probe Results - {args.dataset} - {args.method.upper()}")
    print(f"{'=' * 70}")
    print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"AUC:       {auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'=' * 70}\n")

    # Save results
    results = {
        'dataset': args.dataset,
        'method': args.method,
        'accuracy': float(acc),
        'auc': float(auc),
        'f1': float(f1),
    }

    with open(os.path.join(args.out_dir, f'stage2_{args.dataset.lower()}_{args.method}_probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 2: Unified Topology Encoder Training')
    
    # Method selection
    parser.add_argument('--method', type=str, required=True,
                       choices=['simclr', 'mocov3', 'byol', 'barlow', 'dino'],
                       help='SSL method to use')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['dermamnist', 'pathmnist', 'octmnist', 'organamnist',
                               'isic2019', 'kvasir'])
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to .npz file (for MedMNIST datasets)')
    parser.add_argument('--isic_root', type=str, default='data/isic2019',
                       help='Path to ISIC2019 dataset folder')
    parser.add_argument('--kvasir_root', type=str, default='data/kvasir-dataset',
                       help='Path to Kvasir dataset folder')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Limit number of samples (for debugging)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Training or evaluation mode')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--n_h0', type=int, default=48,
                       help='Number of H0 points')
    parser.add_argument('--n_h1', type=int, default=96,
                       help='Number of H1 points')
    parser.add_argument('--proj_dim', type=int, default=256,
                       help='Projection dimension')
    parser.add_argument('--proj_hidden', type=int, default=512,
                       help='Projector hidden dimension')
    parser.add_argument('--pred_hidden', type=int, default=512,
                       help='Predictor hidden dimension (MoCo/BYOL)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-set if None)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use mixed precision (auto FP16/BF16)')
    parser.add_argument('--early_stop', type=int, default=20,
                       help='Early stopping patience (0 to disable)')
    
    # Method-specific hyperparameters
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for SimCLR/MoCoV3')
    parser.add_argument('--momentum', type=float, default=0.99,
                       help='Momentum for MoCoV3/BYOL')
    parser.add_argument('--lambd', type=float, default=0.01,
                       help='Lambda for Barlow Twins')
    parser.add_argument('--student_temp', type=float, default=0.1,
                        help='Student temperature for DINO')
    parser.add_argument('--teacher_temp', type=float, default=0.04,
                        help='Teacher temperature for DINO')
    parser.add_argument('--dino_momentum', type=float, default=0.996,
                        help='Momentum for DINO teacher (higher than MoCo/BYOL)')
    
    # Augmentation
    parser.add_argument('--weak_min', type=float, default=0.02)
    parser.add_argument('--weak_max', type=float, default=0.08)
    parser.add_argument('--strong_min', type=float, default=0.08)
    parser.add_argument('--strong_max', type=float, default=0.15)
    parser.add_argument('--use_strong', action='store_true',
                       help='Use strong augmentation')
    
    # Others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='./runs_stage2_unified')
    
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, args.method, args.dataset.lower())
    # Auto-set batch size based on dataset if not specified
    if args.batch_size is None:
        if args.dataset.lower() == 'kvasir':
            args.batch_size = 256  # Small dataset
        elif args.dataset.lower() == 'isic2019':
            args.batch_size = 512  # Medium dataset
        else:
            args.batch_size = 1024  # MedMNIST datasets
        print(f"Auto-set batch_size={args.batch_size} for {args.dataset}")

    if args.mode == 'train':
        train_stage2(args)
    elif args.mode == 'evaluate':
        linear_probe_evaluation(args)


if __name__ == "__main__":
    main()
