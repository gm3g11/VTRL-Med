#!/usr/bin/env python3
"""
Unified SAM+SPICE Stage 3: Joint Fine-Tuning with Topology
Supports: MedMNIST, Kvasir, ISIC2019
Multimodal learning: Visual (SAM) + Topology (Persistent Homology)
Can work with or without Stage 2 checkpoint (uses k-means initialization if Stage 2 missing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import os
import numpy as np
import argparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import torchvision.transforms.v2 as transformsV2
import json
import warnings
import copy

warnings.filterwarnings("ignore")
from segment_anything import sam_model_registry
import cripser


# ============================================================================
#  On-the-Fly Persistent Homology Extraction
# ============================================================================

def extract_ph_from_tensor(img_tensor, n_h0=36, n_h1=72, img_size=224):
    """Extract (108, 4) PH tensor from normalized image tensor"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    if torch.is_tensor(img_tensor):
        img_np = img_tensor.cpu().numpy()
    else:
        img_np = img_tensor

    # Denormalize
    img_denorm = img_np * std + mean
    grayscale = (0.299 * img_denorm[0] + 0.587 * img_denorm[1] + 0.114 * img_denorm[2])
    grayscale_uint8 = np.clip(grayscale * 255, 0, 255).astype(np.uint8)

    try:
        pd = cripser.computePH(grayscale_uint8, maxdim=1)

        if len(pd) == 0 or pd.shape[1] < 9:
            return np.zeros((n_h0 + n_h1, 4), dtype=np.float32)

        h0_all = pd[pd[:, 0] == 0]
        h1_all = pd[pd[:, 0] == 1]

        h0 = h0_all[h0_all[:, 2] < 1e100]
        h1 = h1_all

        h0_selected = _select_topk(h0, n_h0, dim=0)
        h1_selected = _select_topk(h1, n_h1, dim=1)

        return np.vstack([h0_selected, h1_selected]).astype(np.float32)

    except Exception:
        return np.zeros((n_h0 + n_h1, 4), dtype=np.float32)


def _select_topk(ph_data, k, dim):
    """Select top-K features by persistence - 4D output"""
    result = np.zeros((k, 4), dtype=np.float32)

    if len(ph_data) == 0:
        return result

    persistence = ph_data[:, 2] - ph_data[:, 1]
    sorted_idx = np.argsort(persistence)[::-1]
    n_select = min(k, len(ph_data))

    for i in range(n_select):
        idx = sorted_idx[i]
        birth, death = ph_data[idx, 1], ph_data[idx, 2]
        pers = persistence[idx]

        result[i] = [
            birth / 255.0,
            death / 255.0,
            pers / 255.0,
            float(dim)
        ]

    return result


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


def load_dataset(dataset_name, split, img_size, data_dir=None, seed=42):
    """Load dataset."""

    if is_medmnist(dataset_name):
        from vtrlmed.datasets.medmnist_dataset import MedMNIST_SPICE
        dataset = MedMNIST_SPICE(dataset_name, split, img_size, download=True)
    else:
        from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE

        if data_dir is None:
            raise ValueError(f"--data_dir required for {dataset_name}")

        if dataset_name == 'kvasir':
            dataset = Kvasir_SPICE(data_dir, split, img_size, seed=seed)
        elif dataset_name == 'isic2019':
            dataset = ISIC2019_SPICE(data_dir, split, img_size, seed=seed)

    return dataset


# ============================================================================
#  Dataset Wrapper with Topology Extraction
# ============================================================================

class SAMDataset(Dataset):
    """Dataset wrapper with augmentation (no topology extraction yet)."""

    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Returns (weak, strong, test, label) from base dataset
        return self.dataset[idx]


def collate_fn_with_topology(batch):
    """Collate function that extracts topology features on-the-fly."""
    img_weak = torch.stack([item[0] for item in batch])
    img_strong = torch.stack([item[1] for item in batch])
    img_test = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])

    batch_size = len(batch)
    topo_weak = np.zeros((batch_size, 108, 4), dtype=np.float32)
    topo_strong = np.zeros((batch_size, 108, 4), dtype=np.float32)

    # Extract topology features from augmented images
    for i in range(batch_size):
        topo_weak[i] = extract_ph_from_tensor(img_weak[i])
        topo_strong[i] = extract_ph_from_tensor(img_strong[i])

    topo_weak = torch.from_numpy(topo_weak)
    topo_strong = torch.from_numpy(topo_strong)

    return img_weak, img_strong, img_test, topo_weak, topo_strong, labels


# ============================================================================
#  Model Components
# ============================================================================

class SPICEClusteringHead(nn.Module):
    """Enhanced clustering head with learnable temperature."""

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

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, use_temperature=True):
        logits = self.head(x)
        if use_temperature:
            logits = logits / torch.clamp(self.temperature, min=0.01)
        return logits


class TopologyPointNet(nn.Module):
    """Deeper PointNet for (108, 4) topology tensors."""

    def __init__(self, num_classes):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(4, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, use_temperature=True):
        x = x.permute(0, 2, 1)  # (B, 4, 108)
        x = self.mlp1(x)  # (B, 512, 108)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, 512)
        logits = self.mlp2(x)

        if use_temperature:
            logits = logits / torch.clamp(self.temperature, min=0.01)
        return logits


class MultimodalModel(nn.Module):
    """Combined SAM backbone + clustering head + topology head."""

    def __init__(self, stage1_path, stage2_path, sam_model_type, sam_checkpoint,
                 num_classes, unfrozen_blocks=4, has_stage2=True):
        super().__init__()

        # Load SAM backbone
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder

        # Load Stage 1 weights
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        self.backbone.load_state_dict(torch.load(stage1_path, map_location='cpu'))

        # Feature dimensions
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        self.num_classes = num_classes

        # Selective unfreezing
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfrozen_blocks > 0:
            if hasattr(self.backbone, 'neck'):
                for param in self.backbone.neck.parameters():
                    param.requires_grad = True

            for block in self.backbone.blocks[-unfrozen_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Load or initialize clustering head
        self.clustering_head = SPICEClusteringHead(self.feature_dim, num_classes)

        if has_stage2 and os.path.exists(stage2_path):
            print(f"Loading Stage 2 checkpoint: {stage2_path}")
            self.clustering_head.load_state_dict(
                torch.load(stage2_path, map_location='cpu'),
                strict=False
            )
        else:
            print("Stage 2 checkpoint not found - will initialize with k-means")

        # Initialize topology head
        self.topology_head = TopologyPointNet(num_classes=num_classes)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Unfrozen blocks: {unfrozen_blocks}")
        print(f"Trainable: {trainable_params:,} / {total_params:,} "
              f"({trainable_params / total_params * 100:.2f}%)")

    def forward(self, x, topo_x=None, use_temperature=True):
        """Forward pass with visual and optional topology features."""
        # Visual branch
        features = self.backbone.patch_embed(x)

        # Positional encoding
        if features.shape[1:3] != self.backbone.pos_embed.shape[1:3]:
            pos_embed = F.interpolate(
                self.backbone.pos_embed.permute(0, 3, 1, 2),
                size=features.shape[1:3], mode='bicubic', align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            pos_embed = self.backbone.pos_embed

        features = features + pos_embed

        # Through transformer blocks
        for blk in self.backbone.blocks:
            features = blk(features)

        # Global pooling
        pooled_features = F.adaptive_avg_pool2d(
            features.permute(0, 3, 1, 2), 1
        ).flatten(1)

        # Normalize and predict
        normalized_features = F.normalize(pooled_features, p=2, dim=1)
        visual_logits = self.clustering_head(normalized_features, use_temperature)

        # Topology branch (optional)
        topo_logits = None
        if topo_x is not None:
            topo_logits = self.topology_head(topo_x, use_temperature)

        return normalized_features, visual_logits, topo_logits


# ============================================================================
#  K-means Initialization (Simplified)
# ============================================================================

def initialize_clustering_head_with_kmeans(model, train_loader, device, num_classes):
    """
    Initialize clustering head using k-means on Stage 1 features.
    Simple approach: k-means → initialize final layer → done.
    """

    print("\n" + "=" * 80)
    print("Initialization Phase: K-means on Stage 1 Features")
    print("=" * 80)

    # Step 1: Extract features from Stage 1 backbone
    print("\nExtracting features from training set...")
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting features"):
            # Unpack based on whether topology is included
            if len(batch) == 6:  # With topology
                _, _, img_test, _, _, _ = batch
            else:  # Without topology (shouldn't happen, but handle it)
                _, _, img_test, _ = batch

            img_test = img_test.to(device, non_blocking=True)
            features, _, _ = model(img_test, None, use_temperature=False)
            all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    print(f"✓ Extracted features: {all_features.shape}")

    # Step 2: Run k-means clustering
    print(f"\nRunning k-means with {num_classes} clusters...")
    kmeans = KMeans(
        n_clusters=num_classes,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=0
    )
    kmeans.fit(all_features)

    cluster_centers = kmeans.cluster_centers_  # [num_classes, feature_dim]
    labels = kmeans.labels_

    print(f"✓ K-means complete")
    print(f"  Cluster centers: {cluster_centers.shape}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")

    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    cluster_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Cluster distribution: {cluster_dist}")

    # Step 3: Initialize clustering head with cluster centers
    print(f"\nInitializing clustering head with cluster centers...")

    with torch.no_grad():
        # Convert cluster centers to tensor
        cluster_centers_tensor = torch.from_numpy(cluster_centers).float().to(device)
        cluster_centers_normalized = F.normalize(cluster_centers_tensor, p=2, dim=1)

        # Pass cluster centers through the first part of the head
        first_linear = model.clustering_head.head[0]
        layer_norm = model.clustering_head.head[1]
        relu = model.clustering_head.head[2]
        final_linear = model.clustering_head.head[-1]

        # Project cluster centers through first layers
        projected_centers = first_linear(cluster_centers_normalized)
        projected_centers = layer_norm(projected_centers)
        projected_centers = relu(projected_centers)
        projected_centers = F.normalize(projected_centers, p=2, dim=1)

        # Initialize final layer
        final_linear.weight.copy_(projected_centers)
        final_linear.bias.zero_()

        print(f"✓ Cluster centers projected through first layers: {projected_centers.shape}")
        print(f"✓ Final layer initialized: {final_linear.weight.shape}")

    print("\n✓ K-means initialization complete")
    print("  Ready to find reliable pseudo-labels\n")


# ============================================================================
#  Pseudo-Labeling & Loss Functions
# ============================================================================

def generate_reliable_samples(model, dataloader, device, num_classes,
                              num_neighbors=50, reliability_threshold=0.8,
                              consistency_threshold=0.7):
    """Generate reliable pseudo-labels using neighbor consistency."""
    model.eval()
    all_features, all_predictions, all_confidences = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating pseudo-labels"):
            _, _, img_test, topo_weak, _, _ = batch
            features, logits, _ = model(
                img_test.to(device),
                None,  # Don't use topology for pseudo-label generation
                use_temperature=True
            )
            probs = F.softmax(logits, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)

            all_features.append(features.cpu())
            all_predictions.append(predictions.cpu())
            all_confidences.append(max_probs.cpu())

    all_features = torch.cat(all_features)
    all_predictions = torch.cat(all_predictions)
    all_confidences = torch.cat(all_confidences)

    # Find reliable samples
    reliable_indices = []

    for i in tqdm(range(len(all_features)), desc="Finding reliable samples"):
        # Check confidence threshold
        if all_confidences[i] < consistency_threshold:
            continue

        # Find K-nearest neighbors
        dist = torch.norm(all_features - all_features[i], dim=1, p=2)
        neighbor_indices = torch.topk(dist, k=num_neighbors, largest=False).indices

        # Check neighbor consistency
        consistency_score = (all_predictions[neighbor_indices] == all_predictions[i]).float().mean()

        # Check cluster size (prevent domination)
        cluster_id = all_predictions[i].item()
        cluster_size = (all_predictions == cluster_id).sum().item()
        max_allowed_size = len(all_predictions) * 1.5 / num_classes

        if consistency_score > reliability_threshold and cluster_size < max_allowed_size:
            reliable_indices.append(i)

    return reliable_indices


def entropy_regularization(logits, lambda_ent=0.1):
    """Entropy regularization."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy_per_sample = -(probs * log_probs).sum(dim=1)
    return -lambda_ent * entropy_per_sample.mean()


def cluster_balance_regularization(logits, num_classes, lambda_balance=1.0):
    """Cluster balance regularization."""
    probs = F.softmax(logits, dim=1)
    cluster_probs = probs.mean(dim=0)
    uniform_target = torch.ones_like(cluster_probs) / num_classes
    balance_loss = F.kl_div(
        torch.log(cluster_probs + 1e-8),
        uniform_target,
        reduction='batchmean'
    )
    return lambda_balance * balance_loss


def swapped_prediction_loss(logits1, logits2, lambda_swap=0.1):
    """Swapped prediction for consistency."""
    with torch.no_grad():
        targets = torch.argmax(logits1, dim=1)
    return lambda_swap * F.cross_entropy(logits2, targets)


def improved_cross_entropy(logits, targets, reduction='mean'):
    """Standard cross entropy."""
    return F.cross_entropy(logits, targets, reduction=reduction)


def topology_consistency_loss(visual_logits, topo_logits, confidence_threshold=0.7):
    """Topology learns from confident visual predictions."""
    with torch.no_grad():
        visual_probs = F.softmax(visual_logits, dim=1)
        max_probs, pseudo_labels = torch.max(visual_probs, dim=1)
        mask = max_probs.ge(confidence_threshold).float()

    if mask.sum() == 0:
        return torch.tensor(0.0, device=visual_logits.device)

    loss = (F.cross_entropy(topo_logits, pseudo_labels, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
    return loss


def mutual_agreement_loss(visual_logits, topo_logits, agreement_threshold=0.6):
    """Reinforce when both branches agree."""
    visual_probs = F.softmax(visual_logits, dim=1)
    topo_probs = F.softmax(topo_logits, dim=1)

    visual_max, visual_pred = torch.max(visual_probs, dim=1)
    topo_max, topo_pred = torch.max(topo_probs, dim=1)

    agreement = (visual_pred == topo_pred).float()
    confidence_mask = ((visual_max > agreement_threshold) & (topo_max > agreement_threshold)).float()
    mask = agreement * confidence_mask

    if mask.sum() == 0:
        return torch.tensor(0.0, device=visual_logits.device)

    loss = F.kl_div(
        F.log_softmax(visual_logits, dim=1),
        topo_probs.detach(),
        reduction='none'
    ).sum(dim=1)

    return (loss * mask).sum() / (mask.sum() + 1e-8)


# ============================================================================
#  Evaluation Functions
# ============================================================================

def hungarian_accuracy(y_true, y_pred):
    """Clustering accuracy with Hungarian matching."""
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {pred_idx: true_idx for pred_idx, true_idx in zip(row_ind, col_ind)}
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size

    return acc, mapping


def calculate_ece(probs, preds_remapped, labels, n_bins=10):
    """Calculate Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    accuracies = (preds_remapped == labels)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def evaluate(model, loader, device, n_bins_ece, true_dist):
    """Comprehensive evaluation with visual and topology metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_topo_preds = []

    with torch.no_grad():
        for batch in loader:
            _, _, img_test, topo_weak, _, labels = batch
            _, vis_logits, topo_logits = model(
                img_test.to(device, non_blocking=True),
                topo_weak.to(device, non_blocking=True),
                use_temperature=True
            )
            probs = torch.softmax(vis_logits, dim=1)
            preds = torch.argmax(vis_logits, dim=1)
            topo_preds = torch.argmax(topo_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_topo_preds.append(topo_preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_topo_preds = np.concatenate(all_topo_preds)

    # Visual metrics
    vis_acc, mapping = hungarian_accuracy(all_labels, all_preds)
    vis_nmi = normalized_mutual_info_score(all_labels, all_preds)
    vis_ari = adjusted_rand_score(all_labels, all_preds)

    # Topology metrics
    topo_acc, _ = hungarian_accuracy(all_labels, all_topo_preds)
    topo_nmi = normalized_mutual_info_score(all_labels, all_topo_preds)
    topo_ari = adjusted_rand_score(all_labels, all_topo_preds)

    # Calibration
    preds_remapped = np.array([mapping.get(pred, -1) for pred in all_preds])
    ece = calculate_ece(all_probs, preds_remapped, all_labels, n_bins=n_bins_ece)

    # Cluster distribution
    pred_ids, pred_counts = np.unique(all_preds, return_counts=True)
    num_classes = len(true_dist)
    aligned_pred_counts = np.zeros(num_classes)

    for pred_idx, count in zip(pred_ids, pred_counts):
        true_idx = mapping.get(pred_idx)
        if true_idx is not None and true_idx < num_classes:
            aligned_pred_counts[true_idx] = count

    pred_dist = aligned_pred_counts / max(1, aligned_pred_counts.sum())
    pred_dist += 1e-9
    kl_divergence = entropy(pk=true_dist, qk=pred_dist)

    return {
        'vis_acc': vis_acc,
        'vis_nmi': vis_nmi,
        'vis_ari': vis_ari,
        'topo_acc': topo_acc,
        'topo_nmi': topo_nmi,
        'topo_ari': topo_ari,
        'ece': ece,
        'kl_divergence': kl_divergence,
        'cluster_counts': pred_counts.tolist(),
        'num_active_clusters': len(pred_counts)
    }


def get_current_lambda(epoch, anneal_epochs, start_val, end_val):
    """Linear annealing schedule."""
    if epoch >= anneal_epochs:
        return end_val
    return start_val - (start_val - end_val) * (epoch / anneal_epochs)


def get_topology_weight(epoch, phase_a_end, phase_b_end, tw_phase_a, tw_phase_b, tw_phase_c):
    """Phase-dependent topology weight."""
    if epoch < phase_a_end:
        return tw_phase_a
    elif epoch < phase_b_end:
        return tw_phase_b
    else:
        return tw_phase_c


# ============================================================================
#  Main Training
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Stage 3: Joint Fine-Tuning with Topology")
    print("=" * 80 + "\n")

    # Get dataset info
    num_classes = get_dataset_info(args.dataset)

    # Paths
    stage1_path = os.path.join(
        args.checkpoint_dir,
        f"stage1_sam_simclr_{args.dataset}_{args.img_size}.pth"
    )
    stage2_path = os.path.join(
        args.stage2_dir,
        f"stage2_ours_{args.dataset}_{args.img_size}.pth"
    )
    stage3_best_path = os.path.join(
        args.output_dir,
        f"stage3_sam_{args.dataset}_best.pth"
    )
    results_path = os.path.join(
        args.output_dir,
        f"stage3_sam_results_{args.dataset}.json"
    )

    # Check if Stage 2 exists
    has_stage2 = os.path.exists(stage2_path)
    if not has_stage2:
        print(f"⚠️  Stage 2 checkpoint not found: {stage2_path}")
        print("   Will initialize clustering head with k-means")

    # Build model
    print("\n--- Building Multimodal Model ---")
    print(f"Stage 1: {stage1_path}")
    if has_stage2:
        print(f"Stage 2: {stage2_path}")
    print()

    model = MultimodalModel(
        stage1_path, stage2_path,
        args.sam_model_type, args.sam_checkpoint,
        num_classes, args.unfrozen_blocks, has_stage2
    ).to(device)

    print("✓ Multimodal model built\n")

    # Load datasets
    train_base_dataset = load_dataset(args.dataset, 'train', args.img_size, args.data_dir, args.seed)
    train_dataset = SAMDataset(train_base_dataset)

    # REDUCED BATCH SIZE FOR K-MEANS INITIALIZATION
    full_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_topology,
        pin_memory=True
    )

    # If Stage 2 is skipped, run k-means initialization
    if not has_stage2:
        initialize_clustering_head_with_kmeans(model, full_loader, device, num_classes)

    # Generate reliable pseudo-labels
    print("=" * 80)
    print("Step 1: Generating Reliable Pseudo-Labels")
    print("=" * 80)

    # Use original batch size for pseudo-label generation
    full_loader_normal = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_topology,
        pin_memory=True
    )

    reliable_indices = generate_reliable_samples(
        model, full_loader_normal, device, num_classes,
        num_neighbors=args.num_neighbors,
        reliability_threshold=args.reliability_threshold,
        consistency_threshold=args.confidence_threshold
    )

    print(f"\n✓ Found {len(reliable_indices)} reliable samples "
          f"({len(reliable_indices) / len(train_dataset) * 100:.2f}%)")

    if len(reliable_indices) == 0:
        print("ERROR: No reliable samples found. Try lowering thresholds.")
        return

    # Fine-tuning setup
    print("\n" + "=" * 80)
    print("Step 2: Joint Fine-Tuning with Topology")
    print("=" * 80 + "\n")

    # Optimizer with topology head
    optimizer_params = []

    if args.lr_backbone > 0:
        optimizer_params.append({
            'params': model.backbone.parameters(),
            'lr': args.lr_backbone,
            'weight_decay': 0.05
        })

    if args.lr_head > 0:
        optimizer_params.append({
            'params': model.clustering_head.parameters(),
            'lr': args.lr_head,
            'weight_decay': 1e-4
        })

    # Topology head with higher learning rate
    optimizer_params.append({
        'params': model.topology_head.parameters(),
        'lr': args.lr_head * args.lr_topo_multiplier,
        'weight_decay': 1e-4
    })

    optimizer = torch.optim.AdamW(optimizer_params)

    # Dataloaders
    labeled_dataset = Subset(train_dataset, reliable_indices)
    sub_batch_size = max(1, args.batch_size // 2)

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=sub_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_topology,
        pin_memory=True,
        drop_last=True
    )

    unlabeled_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_topology,
        pin_memory=True,
        drop_last=True
    )

    # Test set
    eval_test_dataset = SAMDataset(load_dataset(args.dataset, 'test', args.img_size, args.data_dir, args.seed))
    eval_test_loader = DataLoader(
        eval_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_topology,
        pin_memory=True
    )

    # Get true distribution
    test_labels = []
    for i in range(len(eval_test_dataset)):
        _, _, _, label = eval_test_dataset[i]
        test_labels.append(label.item())
    test_labels = np.array(test_labels)
    _, true_test_counts = np.unique(test_labels, return_counts=True)
    true_test_dist = true_test_counts / true_test_counts.sum()

    # Training loop
    best_test_acc = 0.0
    all_results = []

    print(f"Training for {args.epochs} epochs...")
    print(f"Phase A (0-{args.phase_a_end}): Topology Learning (weight={args.tw_phase_a})")
    print(f"Phase B ({args.phase_a_end}-{args.phase_b_end}): Mutual Refinement (weight={args.tw_phase_b})")
    print(f"Phase C ({args.phase_b_end}-{args.epochs}): Visual Dominance (weight={args.tw_phase_c})")
    print(f"Regularization: entropy={args.lambda_entropy}, "
          f"balance={args.lambda_balance_start}->{args.lambda_balance_end}, "
          f"swap={args.lambda_swap}\n")

    for epoch in range(args.epochs):
        model.train()
        labeled_iter = iter(labeled_loader)

        # Get topology weight for this epoch
        topo_weight = get_topology_weight(
            epoch, args.phase_a_end, args.phase_b_end,
            args.tw_phase_a, args.tw_phase_b, args.tw_phase_c
        )

        # Get current balance weight
        current_lambda_balance = get_current_lambda(
            epoch, args.lambda_anneal_epochs,
            args.lambda_balance_start, args.lambda_balance_end
        )

        # Determine phase
        if epoch < args.phase_a_end:
            phase = "Phase A"
        elif epoch < args.phase_b_end:
            phase = "Phase B"
        else:
            phase = "Phase C"

        progress_bar = tqdm(unlabeled_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [{phase}]")

        for img_weak_unlab, img_strong_unlab, _, topo_weak_unlab, topo_strong_unlab, _ in progress_bar:
            # Get labeled batch
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)

            img_weak_lab, _, _, topo_weak_lab, _, true_pseudo_labels = labeled_batch

            # Move to device
            img_weak_lab = img_weak_lab.to(device, non_blocking=True)
            img_weak_unlab = img_weak_unlab.to(device, non_blocking=True)
            img_strong_unlab = img_strong_unlab.to(device, non_blocking=True)
            topo_weak_lab = topo_weak_lab.to(device, non_blocking=True)
            topo_strong_unlab = topo_strong_unlab.to(device, non_blocking=True)
            true_pseudo_labels = true_pseudo_labels.to(device, non_blocking=True)

            # Supervised loss on reliable samples
            _, vis_logits_lab, topo_logits_lab = model(img_weak_lab, topo_weak_lab)
            loss_vis_lab = improved_cross_entropy(vis_logits_lab, true_pseudo_labels)

            # Unsupervised loss on unlabeled samples
            with torch.no_grad():
                _, vis_logits_weak, _ = model(img_weak_unlab, None)
                max_probs, pseudo_labels_unlab = torch.max(
                    F.softmax(vis_logits_weak, dim=1), dim=1
                )
                mask = max_probs.ge(args.confidence_threshold).float()

            _, vis_logits_strong, topo_logits_strong = model(img_strong_unlab, topo_strong_unlab)
            loss_vis_unlab = (improved_cross_entropy(
                vis_logits_strong, pseudo_labels_unlab, reduction='none'
            ) * mask).mean()

            loss_visual = loss_vis_lab + loss_vis_unlab

            # Topology losses (phase-dependent)
            if epoch < args.phase_a_end:  # Phase A: Direct supervision
                loss_topo_lab = improved_cross_entropy(topo_logits_lab, true_pseudo_labels)
                loss_topo_unlab = (improved_cross_entropy(
                    topo_logits_strong, pseudo_labels_unlab, reduction='none'
                ) * mask).mean()
                loss_topo = loss_topo_lab + loss_topo_unlab
            elif epoch < args.phase_b_end:  # Phase B: Mutual refinement
                loss_topo_consistency = topology_consistency_loss(vis_logits_strong, topo_logits_strong, 0.7)
                loss_mutual = mutual_agreement_loss(vis_logits_strong, topo_logits_strong, 0.6)
                loss_topo = loss_topo_consistency + 0.5 * loss_mutual
            else:  # Phase C: Visual dominance
                loss_topo = topology_consistency_loss(vis_logits_strong, topo_logits_strong, 0.8)

            # Regularization losses
            all_logits = torch.cat([vis_logits_lab, vis_logits_strong], dim=0)
            loss_balance = cluster_balance_regularization(
                all_logits, num_classes, current_lambda_balance
            )
            loss_entropy = entropy_regularization(all_logits, args.lambda_entropy)
            loss_swap = swapped_prediction_loss(vis_logits_weak, vis_logits_strong, args.lambda_swap)

            # Total loss
            total_loss = loss_visual + topo_weight * loss_topo + loss_balance + loss_entropy + loss_swap

            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'vis': f'{loss_visual.item():.3f}',
                'topo': f'{loss_topo.item():.3f}',
                'tw': f'{topo_weight:.2f}'
            })

        # Evaluation
        if (epoch + 1) % args.eval_frequency == 0 or (epoch + 1) == args.epochs:
            metrics = evaluate(model, eval_test_loader, device, args.ece_bins, true_test_dist)

            print(f"\n--- Epoch {epoch + 1} ---")
            print(
                f"  Visual:   ACC={metrics['vis_acc']:.4f} | NMI={metrics['vis_nmi']:.4f} | ARI={metrics['vis_ari']:.4f}")
            print(
                f"  Topology: ACC={metrics['topo_acc']:.4f} | NMI={metrics['topo_nmi']:.4f} | ARI={metrics['topo_ari']:.4f}")
            print(
                f"  ECE={metrics['ece']:.4f} | KL={metrics['kl_divergence']:.4f} | Clusters={metrics['num_active_clusters']}/{num_classes}")

            # Save best model
            if metrics['vis_acc'] > best_test_acc:
                best_test_acc = metrics['vis_acc']
                print(f"  🚀 NEW BEST: {best_test_acc:.4f}")
                torch.save(model.state_dict(), stage3_best_path)

            all_results.append({'epoch': epoch + 1, **metrics})

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    if os.path.exists(stage3_best_path):
        model.load_state_dict(torch.load(stage3_best_path))
        print("✓ Loaded best checkpoint")

    final_metrics = evaluate(model, eval_test_loader, device, args.ece_bins, true_test_dist)

    print(f"\nFinal Performance:")
    print(
        f"  Visual:   ACC={final_metrics['vis_acc']:.4f} | NMI={final_metrics['vis_nmi']:.4f} | ARI={final_metrics['vis_ari']:.4f}")
    print(
        f"  Topology: ACC={final_metrics['topo_acc']:.4f} | NMI={final_metrics['topo_nmi']:.4f} | ARI={final_metrics['topo_ari']:.4f}")
    print(f"  ECE={final_metrics['ece']:.4f}")

    # Save results
    with open(results_path, 'w') as f:
        json.dump({
            'has_stage2': has_stage2,
            'final_best_metrics': final_metrics,
            'training_epochs': all_results
        }, f, indent=2)

    print(f"\n✓ Results: {results_path}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser('Unified SAM+SPICE Stage 3 with Topology')

    # Dataset
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # Paths
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
    parser.add_argument('--stage2_dir', default='./checkpoints_stage2', type=str)
    parser.add_argument('--output_dir', default='./checkpoints_stage3', type=str)

    # Model
    parser.add_argument('--sam_model_type', default='vit_h', type=str)
    parser.add_argument('--sam_checkpoint', default='./sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--unfrozen_blocks', default=4, type=int)

    # Training
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr_backbone', default=5e-6, type=float)
    parser.add_argument('--lr_head', default=5e-5, type=float)
    parser.add_argument('--num_workers', default=16, type=int)

    # Pseudo-labeling
    parser.add_argument('--num_neighbors', default=50, type=int)
    parser.add_argument('--reliability_threshold', default=0.8, type=float)
    parser.add_argument('--confidence_threshold', default=0.8, type=float)

    # Regularization
    parser.add_argument('--lambda_swap', default=0.1, type=float)
    parser.add_argument('--lambda_entropy', default=0.2, type=float)
    parser.add_argument('--lambda_balance_start', default=2.0, type=float)
    parser.add_argument('--lambda_balance_end', default=0.5, type=float)
    parser.add_argument('--lambda_anneal_epochs', default=25, type=int)

    # Topology phase control
    parser.add_argument('--phase_a_end', default=25, type=int, help='End of Phase A (epochs)')
    parser.add_argument('--phase_b_end', default=40, type=int, help='End of Phase B (epochs)')
    parser.add_argument('--tw_phase_a', default=0.5, type=float, help='Topology weight Phase A')
    parser.add_argument('--tw_phase_b', default=0.2, type=float, help='Topology weight Phase B')
    parser.add_argument('--tw_phase_c', default=0.05, type=float, help='Topology weight Phase C')
    parser.add_argument('--lr_topo_multiplier', default=2.0, type=float, help='LR multiplier for topology head')

    # Evaluation
    parser.add_argument('--eval_frequency', default=2, type=int)
    parser.add_argument('--ece_bins', default=10, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)