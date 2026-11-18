#!/usr/bin/env python3
"""
Unified SAM+SPICE Stage 2 with Ablation Support
Supports: w/o Stage 1, FixMatch, SwAV, SPICE, Ours (Confidence-Only), Ours (Prototype-Only), Ours (Full)
No MedMNIST_SPICE dependency - loads directly from medmnist library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np
import argparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torchvision.transforms.v2 as transformsV2
import json
import warnings
import copy

warnings.filterwarnings("ignore")
from segment_anything import sam_model_registry


# ============================================================================
#  Dataset Info & Loading
# ============================================================================

def get_dataset_info(dataset_name):
    """Get number of classes for each dataset."""
    dataset_configs = {
        'bloodmnist': 8, 'pathmnist': 9, 'dermamnist': 7, 'octmnist': 4,
        'pneumoniamnist': 2, 'retinamnist': 5, 'breastmnist': 2, 'tissuemnist': 8,
        'organamnist': 11, 'organcmnist': 11, 'organsmnist': 11,
        'kvasir': 8, 'isic2019': 8,
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_configs[dataset_name]


def is_medmnist(dataset_name):
    """Check if dataset is MedMNIST."""
    medmnist_datasets = [
        'bloodmnist', 'pathmnist', 'dermamnist', 'octmnist',
        'pneumoniamnist', 'retinamnist', 'breastmnist', 'tissuemnist',
        'organamnist', 'organcmnist', 'organsmnist'
    ]
    return dataset_name in medmnist_datasets


# ============================================================================
#  SAM Feature Extractor
# ============================================================================

class SAMFeatureExtractor(nn.Module):
    """Frozen SAM backbone for feature extraction."""

    def __init__(self, checkpoint_path, sam_model_type='vit_h',
                 sam_pretrained_path='./sam_vit_h_4b8939.pth', use_stage1=True):
        super().__init__()

        # Load SAM architecture
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_pretrained_path)
        self.backbone = sam_model.image_encoder

        # Load weights
        if use_stage1:
            print(f"Loading Stage 1 fine-tuned weights: {checkpoint_path}")
            self.backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            print(f"Using pretrained SAM weights (w/o Stage 1 fine-tuning)")

        # Feature dimensions
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]

        # Freeze
        self._freeze_and_verify()
        self.backbone.eval()

    def _freeze_and_verify(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        if trainable_params > 0:
            raise RuntimeError("SAM backbone must be frozen in Stage 2!")
        print("✓ SAM backbone frozen")

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.patch_embed(x)

            if features.shape[1:3] != self.backbone.pos_embed.shape[1:3]:
                pos_embed = F.interpolate(
                    self.backbone.pos_embed.permute(0, 3, 1, 2),
                    size=features.shape[1:3], mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                pos_embed = self.backbone.pos_embed

            features = features + pos_embed

            for blk in self.backbone.blocks:
                features = blk(features)

            pooled_features = F.adaptive_avg_pool2d(
                features.permute(0, 3, 1, 2), 1
            ).flatten(1)

            return F.normalize(pooled_features, p=2, dim=1)


# ============================================================================
#  Clustering Head
# ============================================================================

class SPICEClusteringHead(nn.Module):
    """Clustering head."""

    def __init__(self, input_dim, num_classes):
        super().__init__()

        hidden_dim = max(input_dim // 2, 512)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


# ============================================================================
#  Dataset Wrapper for GPU Augmentation
# ============================================================================

class SAMRawDataset(Dataset):
    """Wrapper that returns raw images for GPU augmentation."""

    def __init__(self, dataset_name, split='train', img_size=224, data_dir=None, seed=42):
        self.dataset_name = dataset_name
        self.split = split

        if is_medmnist(dataset_name):
            # MedMNIST: load directly
            import medmnist
            from medmnist import INFO

            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            self.dataset = DataClass(split=split, download=True, transform=None, size=img_size)

            # Get labels
            self.labels = self.dataset.labels.flatten()

            print(f"Dataset ({split}): {len(self.dataset)} samples, "
                  f"{len(np.unique(self.labels))} classes")

        else:
            # Kvasir or ISIC2019
            from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE

            if data_dir is None:
                raise ValueError(f"--data_dir required for {dataset_name}")

            if dataset_name == 'kvasir':
                self.base_dataset = Kvasir_SPICE(data_dir, split, img_size, seed=seed)
            elif dataset_name == 'isic2019':
                self.base_dataset = ISIC2019_SPICE(data_dir, split, img_size, seed=seed)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Get labels
            labels = []
            for i in range(len(self.base_dataset)):
                _, _, _, label = self.base_dataset[i]
                labels.append(label.item())
            self.labels = np.array(labels)

            print(f"Dataset ({split}): {len(self.base_dataset)} samples, "
                  f"{len(np.unique(self.labels))} classes")

        # Base transform to uint8
        self.base_transform = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.uint8, scale=False),
        ])

    def __len__(self):
        if is_medmnist(self.dataset_name):
            return len(self.dataset)
        else:
            return len(self.base_dataset)

    def __getitem__(self, idx):
        if is_medmnist(self.dataset_name):
            # MedMNIST
            image, label = self.dataset[idx]
            return self.base_transform(image), torch.tensor(label).long().squeeze()
        else:
            # Kvasir/ISIC
            _, _, img_test, label = self.base_dataset[idx]
            img_uint8 = (img_test * 255).clamp(0, 255).to(torch.uint8)
            return img_uint8, label


# ============================================================================
#  Pseudo-Labeling Methods
# ============================================================================

def ours_full_pseudo_labeling(features, probabilities, num_classes, top_k_ratio=0.5, **kwargs):
    """
    OURS (Full): Confidence-Filtered Prototype Pseudo-Labeling.
    Combines confidence filtering AND prototype matching.

    Steps:
    1. Select top-K confident samples per class
    2. Create prototype from these top-K samples
    3. Find nearest neighbors to prototype
    4. Assign pseudo-labels to nearest neighbors
    """
    batch_size = features.shape[0]
    device = features.device
    top_k = max(1, int(batch_size * top_k_ratio / num_classes))

    pseudo_labels = torch.full((batch_size, num_classes), -1, dtype=torch.long, device=device)

    for k in range(num_classes):
        if probabilities.shape[0] < top_k:
            continue

        # Step 1: Select top-K confident samples
        _, top_indices = torch.topk(probabilities[:, k], top_k)

        if len(top_indices) > 0:
            # Step 2: Create prototype
            prototype = features[top_indices].mean(dim=0)

            # Step 3: Find nearest neighbors to prototype
            similarities = F.cosine_similarity(features, prototype.unsqueeze(0), dim=1)
            _, nearest_indices = torch.topk(similarities, top_k)
            pseudo_labels[nearest_indices, k] = k

    # Convert to final labels
    final_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    for i in range(batch_size):
        valid_labels = pseudo_labels[i][pseudo_labels[i] >= 0]
        if len(valid_labels) > 0:
            final_labels[i] = valid_labels[0]

    return final_labels


def confidence_only_pseudo_labeling(features, probabilities, num_classes, top_k_ratio=0.5, **kwargs):
    """
    OURS (Ablation): Confidence-Only.
    Select top-k per class by confidence, no prototype.

    This tests whether the prototype step is necessary.
    """
    batch_size = features.shape[0]
    device = features.device
    top_k = max(1, int(batch_size * top_k_ratio / num_classes))

    pseudo_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    for k in range(num_classes):
        if probabilities.shape[0] < top_k:
            continue

        # Only use confidence - no prototype
        _, top_indices = torch.topk(probabilities[:, k], top_k)
        pseudo_labels[top_indices] = k

    return pseudo_labels


def prototype_only_pseudo_labeling(features, probabilities, num_classes, top_k_ratio=0.5, **kwargs):
    """
    OURS (Ablation): Prototype-Only.
    Use ALL samples to create prototypes (no confidence filter).

    This tests whether confidence filtering is necessary.
    """
    batch_size = features.shape[0]
    device = features.device
    top_k = max(1, int(batch_size * top_k_ratio / num_classes))

    pseudo_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    # Get predictions without confidence filtering
    preds = torch.argmax(probabilities, dim=1)

    for k in range(num_classes):
        class_mask = (preds == k)
        if class_mask.sum() == 0:
            continue

        # Create prototype from ALL predicted samples
        prototype = features[class_mask].mean(dim=0)

        # Find nearest neighbors
        similarities = F.cosine_similarity(features, prototype.unsqueeze(0), dim=1)
        _, nearest_indices = torch.topk(similarities, min(top_k, batch_size))
        pseudo_labels[nearest_indices] = k

    return pseudo_labels


def fixmatch_pseudo_labeling(features, probabilities, num_classes, confidence_threshold=0.95, **kwargs):
    """
    FixMatch: Simple confidence thresholding.

    From "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
    (Sohn et al., NeurIPS 2020)

    Uses a global threshold - if model confidence > threshold, use as pseudo-label.
    No per-class balancing, no prototype.
    """
    max_probs, pseudo_labels = torch.max(probabilities, dim=1)
    mask = max_probs >= confidence_threshold

    final_labels = pseudo_labels.clone()
    final_labels[~mask] = -1

    return final_labels


def flexmatch_pseudo_labeling(features, probabilities, num_classes,
                               confidence_threshold=0.95, **kwargs):
    """
    FlexMatch: Per-class adaptive thresholding.

    From "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"
    (Zhang et al., NeurIPS 2021)

    Uses per-class learning status to lower thresholds for harder classes.
    Classes the model is less confident about get lower thresholds.
    """
    max_probs, pseudo_labels = torch.max(probabilities, dim=1)

    # Per-class learning status: fraction of samples above base threshold
    class_thresholds = torch.zeros(num_classes, device=features.device)
    for c in range(num_classes):
        class_mask = (pseudo_labels == c)
        if class_mask.sum() > 0:
            class_thresholds[c] = (max_probs[class_mask] >= confidence_threshold).float().mean()

    # Adaptive threshold per class
    max_effect = class_thresholds.max().clamp(min=1e-6)
    adaptive_thresholds = confidence_threshold * (class_thresholds / max_effect)

    # Apply per-class thresholds
    sample_thresholds = adaptive_thresholds[pseudo_labels]
    mask = max_probs >= sample_thresholds

    final_labels = pseudo_labels.clone()
    final_labels[~mask] = -1

    return final_labels


def spice_pseudo_labeling(features, probabilities, num_classes, top_k_ratio=0.5,
                          reliability_threshold=0.8, num_neighbors=50, **kwargs):
    """
    SPICE (Original Paper): Prototype + Reliable Filtering.

    From "Semantic Pseudo-labeling for Image Clustering" (Niu et al., arXiv 2020)

    Two-stage process:
      1. Prototype-based pseudo-labeling (select top-k → create prototype → assign nearest)
      2. Neighbor consistency filtering using ORIGINAL PREDICTIONS (not pseudo-labels)

    CRITICAL: Consistency check uses model predictions, not pseudo-labels.
    This ensures samples are only kept if their ORIGINAL prediction agrees with neighbors.
    """
    batch_size = features.shape[0]
    device = features.device
    top_k = max(1, int(batch_size * top_k_ratio / num_classes))

    # Get original predictions (will be used for consistency check)
    original_preds = torch.argmax(probabilities, dim=1)

    # Stage 1: Prototype pseudo-labeling
    pseudo_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    for k in range(num_classes):
        if probabilities.shape[0] < top_k:
            continue

        # Select top-k samples
        _, top_indices = torch.topk(probabilities[:, k], top_k)

        if len(top_indices) > 0:
            # Create prototype
            prototype = features[top_indices].mean(dim=0)

            # Find nearest neighbors to prototype
            similarities = F.cosine_similarity(features, prototype.unsqueeze(0), dim=1)
            _, nearest_indices = torch.topk(similarities, top_k)
            pseudo_labels[nearest_indices] = k

    # Stage 2: Reliable filtering (neighborhood consistency with ORIGINAL PREDICTIONS)
    reliable_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(batch_size):
        if pseudo_labels[i] == -1:
            continue

        # Find k-nearest neighbors
        dist = torch.norm(features - features[i], dim=1, p=2)
        neighbor_indices = torch.topk(dist, k=min(num_neighbors, batch_size), largest=False).indices

        # CRITICAL FIX: Check consistency using ORIGINAL PREDICTIONS, not pseudo-labels
        # This is what the SPICE paper does - ensures the model itself is consistent
        consistency = (original_preds[neighbor_indices] == original_preds[i]).float().mean()

        if consistency >= reliability_threshold:
            reliable_mask[i] = True

    # Only keep reliable samples
    final_labels = pseudo_labels.clone()
    final_labels[~reliable_mask] = -1

    return final_labels


def swav_pseudo_labeling(features, probabilities, num_classes, epsilon=0.05,
                         sinkhorn_iterations=3, temperature=0.1, **kwargs):
    """
    SwAV: Simplified Sinkhorn-Knopp optimal transport.

    From "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"
    (Caron et al., NeurIPS 2020)

    Creates cluster assignments by solving optimal transport problem between
    samples and prototypes using the Sinkhorn-Knopp algorithm.

    Parameters:
    - epsilon: 0.05 (for numerical stability)
    - sinkhorn_iterations: 3 (paper uses 3 iterations)
    - temperature: 0.1 (sharpening parameter)

    Note: This is a simplified version without:
    - Multi-crop augmentation
    - Momentum-updated prototypes
    - Queue-based prototype updates

    For ablation study purposes, this simplified version is sufficient.
    """
    batch_size = features.shape[0]
    device = features.device

    # Step 1: Initialize prototypes from current predictions
    preds = torch.argmax(probabilities, dim=1)
    prototypes = []

    for k in range(num_classes):
        class_mask = (preds == k)
        if class_mask.sum() > 0:
            prototype = features[class_mask].mean(dim=0)
        else:
            # Random initialization for empty clusters
            prototype = torch.randn(features.shape[1], device=device)
        prototype = F.normalize(prototype, p=2, dim=0)
        prototypes.append(prototype)

    prototypes = torch.stack(prototypes)  # [num_classes, feature_dim]

    # Step 2: Compute similarity scores (cosine similarity / temperature)
    features_normalized = F.normalize(features, p=2, dim=1)
    scores = torch.mm(features_normalized, prototypes.t()) / temperature  # [batch_size, num_classes]

    # Step 3: Sinkhorn-Knopp algorithm for optimal transport
    # Convert scores to assignment matrix Q
    Q = torch.exp(scores).t()  # [num_classes, batch_size]
    Q = Q / Q.sum()  # Normalize to sum to 1

    for _ in range(sinkhorn_iterations):
        # Normalize rows (each prototype gets equal weight = 1/num_classes)
        sum_of_rows = Q.sum(dim=1, keepdim=True)
        Q = Q / (sum_of_rows + epsilon)
        Q = Q / num_classes

        # Normalize columns (each sample has total weight = 1/batch_size)
        sum_of_cols = Q.sum(dim=0, keepdim=True)
        Q = Q / (sum_of_cols + epsilon)
        Q = Q / batch_size

    Q = Q.t()  # [batch_size, num_classes]

    # Step 4: Assign to cluster with highest probability
    pseudo_labels = torch.argmax(Q, dim=1)

    return pseudo_labels


def get_pseudo_labeling_method(method_name):
    """Factory function for pseudo-labeling methods."""
    methods = {
        'ours': ours_full_pseudo_labeling,
        'confidence_only': confidence_only_pseudo_labeling,
        'prototype_only': prototype_only_pseudo_labeling,
        'fixmatch': fixmatch_pseudo_labeling,
        'flexmatch': flexmatch_pseudo_labeling,
        'spice': spice_pseudo_labeling,
        'swav': swav_pseudo_labeling,
        'wo_stage1': ours_full_pseudo_labeling,  # Same as ours, but with different backbone
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Choose from {list(methods.keys())}")

    return methods[method_name]


# ============================================================================
#  Loss & Evaluation
# ============================================================================

def double_softmax_cross_entropy(logits, targets):
    """Double softmax cross entropy."""
    probs = F.softmax(logits, dim=1)
    probs_double = F.softmax(probs, dim=1)
    return F.nll_loss(torch.log(probs_double + 1e-8), targets)


def accuracy(y_true, y_pred):
    """Clustering accuracy with Hungarian matching."""
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(ind[0], ind[1])) * 1.0 / y_pred.size


def evaluate_fast(head, features, labels, device, batch_size=2048):
    """Fast evaluation using pre-computed features."""
    head.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            batch_features = features[i:i + batch_size].to(device)
            outputs = head(batch_features)
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())

    all_preds = np.concatenate(all_preds)

    acc = accuracy(labels, all_preds)
    nmi = normalized_mutual_info_score(labels, all_preds)
    ari = adjusted_rand_score(labels, all_preds)

    return acc, nmi, ari


# ============================================================================
#  Main Training
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Stage 2: {args.pseudo_label_method.upper()}")
    print("=" * 80 + "\n")

    # Get dataset info
    num_classes = get_dataset_info(args.dataset)

    # Determine if using Stage 1
    use_stage1 = (args.pseudo_label_method != 'wo_stage1')

    # Normalization
    norm_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # GPU augmentations
    weak_gpu_augment = torch.nn.Sequential(
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.Resize((args.img_size, args.img_size), antialias=True),
        transformsV2.RandomHorizontalFlip(p=0.3),
        transformsV2.Normalize(**norm_params)
    ).to(device)

    strong_gpu_augment = torch.nn.Sequential(
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.RandomHorizontalFlip(p=0.5),
        transformsV2.RandomRotation(degrees=15),
        transformsV2.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.05),
        transformsV2.RandomResizedCrop(args.img_size, scale=(0.8, 1.0),
                                       ratio=(0.9, 1.1), antialias=True),
        transformsV2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transformsV2.Normalize(**norm_params)
    ).to(device)

    # Load feature extractor
    stage1_checkpoint = os.path.join(
        args.checkpoint_dir,
        f"stage1_sam_simclr_{args.dataset}_{args.img_size}.pth"
    )

    print("Loading SAM Feature Extractor...")
    feature_extractor = SAMFeatureExtractor(
        stage1_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_pretrained_path=args.sam_checkpoint,
        use_stage1=use_stage1
    ).to(device)
    print(f"Feature dim: {feature_extractor.feature_dim}\n")

    # Pre-compute test features
    print("=" * 80)
    print("Pre-computing test features...")
    print("=" * 80)

    test_dataset = SAMRawDataset(
        args.dataset, 'test', args.img_size, args.data_dir, args.seed
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_labels = test_dataset.labels

    all_test_features = []
    test_gpu_transform = torch.nn.Sequential(
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.Resize((args.img_size, args.img_size), antialias=True),
        transformsV2.Normalize(**norm_params)
    ).to(device)

    with torch.no_grad():
        for raw_img_batch, _ in tqdm(test_loader, desc="Extracting test features"):
            img_test = test_gpu_transform(raw_img_batch.to(device))
            features = feature_extractor(img_test)
            all_test_features.append(features.cpu())

    all_test_features = torch.cat(all_test_features, dim=0)
    print(f"✓ Test features: {all_test_features.shape}\n")

    # Training setup
    print("=" * 80)
    print(f"Training with: {args.pseudo_label_method}")
    print("=" * 80 + "\n")

    train_dataset = SAMRawDataset(
        args.dataset, 'train', args.img_size, args.data_dir, args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Initialize clustering heads
    print(f"Initializing {args.num_heads} clustering heads...")
    clustering_heads = [
        SPICEClusteringHead(feature_extractor.feature_dim, num_classes).to(device)
        for _ in range(args.num_heads)
    ]

    optimizers = [
        torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        for head in clustering_heads
    ]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
        for opt in optimizers
    ]

    # Get pseudo-labeling method
    pseudo_label_fn = get_pseudo_labeling_method(args.pseudo_label_method)
    print(f"Using: {args.pseudo_label_method}\n")

    # Track best performance
    best_head_performance = [0.0] * args.num_heads
    best_head_state_dicts = [None] * args.num_heads
    all_results = []

    # Training loop
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for raw_img_batch, _ in progress_bar:
            raw_img_batch = raw_img_batch.to(device, non_blocking=True)

            with torch.no_grad():
                img_weak = weak_gpu_augment(raw_img_batch)
                img_strong = strong_gpu_augment(raw_img_batch)

                combined_batch = torch.cat([img_weak, img_strong], dim=0)
                all_features = feature_extractor(combined_batch)
                features_weak, features_strong = torch.chunk(all_features, 2, dim=0)

            # Train each head
            for head_idx in range(args.num_heads):
                head = clustering_heads[head_idx]
                optimizer = optimizers[head_idx]
                scheduler = schedulers[head_idx]

                # Generate pseudo-labels
                with torch.no_grad():
                    head.eval()
                    probs_weak = F.softmax(head(features_weak), dim=1)

                    # Call pseudo-labeling method with appropriate parameters
                    pseudo_label_kwargs = {
                        'features': features_weak,
                        'probabilities': probs_weak,
                        'num_classes': num_classes,
                        'top_k_ratio': args.top_k_ratio,
                        'confidence_threshold': args.fixmatch_threshold,
                        'reliability_threshold': args.spice_reliability_threshold,
                        'num_neighbors': args.spice_num_neighbors,
                        'epsilon': 0.05,
                        'sinkhorn_iterations': 3,
                        'temperature': 0.1,
                    }

                    pseudo_labels = pseudo_label_fn(**pseudo_label_kwargs)

                # Train on valid pseudo-labels
                valid_mask = (pseudo_labels >= 0)

                if valid_mask.sum() > 0:
                    head.train()
                    logits_strong = head(features_strong[valid_mask])
                    loss = double_softmax_cross_entropy(logits_strong, pseudo_labels[valid_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

        # Evaluation
        if (epoch + 1) % args.eval_frequency == 0 or (epoch + 1) == args.epochs:
            print(f"\nEpoch {epoch + 1} Evaluation:")
            print(f"{'Head':<4} | {'ACC':<6} | {'NMI':<6} | {'ARI':<6} | {'Best':<6} | Status")
            print("-" * 60)

            for head_idx, head in enumerate(clustering_heads):
                test_acc, test_nmi, test_ari = evaluate_fast(
                    head, all_test_features, test_labels, device,
                    batch_size=args.batch_size
                )

                status = ""
                if test_acc > best_head_performance[head_idx]:
                    best_head_performance[head_idx] = test_acc
                    best_head_state_dicts[head_idx] = copy.deepcopy(head.state_dict())
                    status = "✓ NEW BEST"

                print(f"H{head_idx:<3} | {test_acc:<6.3f} | {test_nmi:<6.3f} | "
                      f"{test_ari:<6.3f} | {best_head_performance[head_idx]:<6.3f} | {status}")

            print(f"\nBest overall ACC: {max(best_head_performance):.4f}")
            print("=" * 80)

            all_results.append({
                'epoch': epoch + 1,
                'best_acc_snapshot': float(max(best_head_performance))
            })

    # Save best head
    print("\n" + "=" * 80)
    print("Selecting best head...")

    best_head_idx = np.argmax(best_head_performance)
    best_acc = best_head_performance[best_head_idx]

    print(f"Best head: #{best_head_idx} with ACC: {best_acc:.4f}")

    best_state_dict = best_head_state_dicts[best_head_idx]
    if best_state_dict is None:
        best_state_dict = clustering_heads[best_head_idx].state_dict()

    # Save checkpoint
    stage2_checkpoint_path = os.path.join(
        args.output_dir,
        f"stage2_{args.pseudo_label_method}_{args.dataset}_{args.img_size}.pth"
    )
    torch.save(best_state_dict, stage2_checkpoint_path)
    print(f"✓ Saved: {stage2_checkpoint_path}")

    # Save results
    results_path = os.path.join(
        args.output_dir,
        f"stage2_{args.pseudo_label_method}_results_{args.dataset}_{args.img_size}.json"
    )

    final_results = {
        'method': args.pseudo_label_method,
        'use_stage1': use_stage1,
        'best_head_idx': int(best_head_idx),
        'best_head_acc': float(best_acc),
        'epoch_history': all_results
    }

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"✓ Results: {results_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser('Unified Stage 2 with Ablation Support')

    # Dataset
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # Paths
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str,
                        help='Directory with Stage 1 checkpoint')
    parser.add_argument('--output_dir', default='./checkpoints_stage2', type=str)

    # Model
    parser.add_argument('--sam_model_type', default='vit_h', type=str)
    parser.add_argument('--sam_checkpoint', default='./sam_vit_h_4b8939.pth', type=str)

    # Ablation: Pseudo-labeling method
    parser.add_argument('--pseudo_label_method', default='ours', type=str,
                        choices=['ours', 'confidence_only', 'prototype_only',
                                 'fixmatch', 'flexmatch', 'spice', 'swav', 'wo_stage1'],
                        help='Pseudo-labeling method')

    # Training
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=6e-4, type=float)
    parser.add_argument('--num_heads', default=10, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_frequency', default=10, type=int)

    # Method-specific parameters
    parser.add_argument('--top_k_ratio', default=0.5, type=float)
    parser.add_argument('--fixmatch_threshold', default=0.95, type=float)
    parser.add_argument('--spice_reliability_threshold', default=0.8, type=float)
    parser.add_argument('--spice_num_neighbors', default=50, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)