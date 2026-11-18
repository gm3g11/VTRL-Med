#!/usr/bin/env python3
"""
K-Means Backbone Comparison Experiment
Compare feature representations for k-means clustering on 3 medical imaging datasets.

Backbones:
  1. Raw Pixels (flattened, no normalization)
  2. ResNet-50 (ImageNet pretrained)
  3. SAM ViT-H (frozen, pretrained)
  4. SAM ViT-H (adapted, Stage 1 SimCLR checkpoint)

Datasets: PathMNIST (9 cls), Kvasir (8 cls), ISIC2019 (8 cls)
K-Means: GPU-accelerated via PyTorch
Metrics: ACC (Hungarian), NMI, ARI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from PIL import Image
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from segment_anything import sam_model_registry
from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE


# ============================================================================
# Configuration
# ============================================================================

STAGE1_PATHS = {
    'pathmnist': 'checkpoints/sam_spice/stage1/pathmnist/stage1_sam_simclr_pathmnist_224.pth',
    'kvasir': 'checkpoints/sam_spice_kvasir_isic/stage1/kvasir/stage1_sam_simclr_kvasir_224.pth',
    'isic2019': 'checkpoints/sam_spice_kvasir_isic/stage1/isic2019/stage1_sam_simclr_isic2019_224.pth',
}

DATASET_CONFIG = {
    'pathmnist': {'num_classes': 9, 'data_dir': None},
    'kvasir': {'num_classes': 8, 'data_dir': os.path.expanduser('data/kvasir-dataset')},
    'isic2019': {'num_classes': 8, 'data_dir': os.path.expanduser('data/isic2019')},
}

BACKBONES = ['raw_pixels', 'resnet50', 'sam_frozen', 'sam_adapted']

SAM_CHECKPOINT = 'weights/sam_vit_h_4b8939.pth'
SAM_MODEL_TYPE = 'vit_h'
IMG_SIZE = 224
SEED = 42


# ============================================================================
# Metrics
# ============================================================================

def hungarian_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size
    return acc


def gpu_kmeans(features, n_clusters, n_init=20, seed=42, max_iter=300):
    """Run k-means clustering on GPU via PyTorch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    n, d = x.shape

    best_inertia = float('inf')
    best_labels = None

    for run in range(n_init):
        torch.manual_seed(seed + run)
        # Random initialization (k-means++)
        indices = torch.randperm(n, device=device)[:n_clusters]
        centroids = x[indices].clone()

        for _ in range(max_iter):
            # Assign: compute distances to centroids
            # (n, d) vs (k, d) -> (n, k) distances
            dists = torch.cdist(x, centroids)  # [n, k]
            labels = dists.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=device)
            new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), x)
            counts.scatter_add_(0, labels, torch.ones(n, device=device))

            # Handle empty clusters
            empty = counts == 0
            counts[empty] = 1
            new_centroids /= counts.unsqueeze(1)

            # Re-init empty clusters from random points
            if empty.any():
                new_centroids[empty] = x[torch.randperm(n, device=device)[:empty.sum()]]

            # Check convergence
            shift = (new_centroids - centroids).norm()
            centroids = new_centroids
            if shift < 1e-6:
                break

        # Compute inertia
        dists = torch.cdist(x, centroids)
        labels = dists.argmin(dim=1)
        inertia = dists.gather(1, labels.unsqueeze(1)).sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.cpu().numpy()

    return best_labels


# ============================================================================
# Dataset Loading
# ============================================================================

class UnifiedTestDataset(Dataset):
    """
    Unified test dataset for all 3 datasets.
    normalize=False for raw pixels, True for backbone-based extractors.
    """

    def __init__(self, dataset_name, normalize=True, img_size=224, seed=42):
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.img_size = img_size
        self.samples = []  # list of (image_path_or_index, label)
        self._medmnist_dataset = None

        if dataset_name == 'pathmnist':
            self._init_medmnist(dataset_name, img_size)
        elif dataset_name == 'kvasir':
            self._init_kvasir(img_size, seed)
        elif dataset_name == 'isic2019':
            self._init_isic2019(img_size, seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Build transforms
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def _init_medmnist(self, dataset_name, img_size):
        import medmnist
        from medmnist import INFO
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        self._medmnist_dataset = DataClass(split='test', download=True,
                                           transform=None, size=img_size)
        print(f"{dataset_name} test: {len(self._medmnist_dataset)} images")

    def _init_kvasir(self, img_size, seed):
        ds = Kvasir_SPICE(
            data_dir=DATASET_CONFIG['kvasir']['data_dir'],
            img_size=img_size, split='test', seed=seed
        )
        self.samples = ds.samples  # list of (path, label)

    def _init_isic2019(self, img_size, seed):
        ds = ISIC2019_SPICE(
            data_dir=DATASET_CONFIG['isic2019']['data_dir'],
            img_size=img_size, split='test', seed=seed
        )
        self.samples = ds.samples  # list of (path, label)

    def __len__(self):
        if self._medmnist_dataset is not None:
            return len(self._medmnist_dataset)
        return len(self.samples)

    def __getitem__(self, idx):
        if self._medmnist_dataset is not None:
            img, label = self._medmnist_dataset[idx]
            # img is PIL Image
            img = img.convert('RGB')
            label = int(label.squeeze())
        else:
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            label = int(label)

        img = self.transform(img)
        return img, label


# ============================================================================
# Feature Extractors
# ============================================================================

def extract_raw_pixels(loader):
    """Flatten images to raw pixel vectors."""
    all_features = []
    all_labels = []
    for imgs, labels in tqdm(loader, desc="Raw pixels"):
        # imgs: [B, 3, 224, 224] in [0,1]
        flat = imgs.view(imgs.size(0), -1).numpy()
        all_features.append(flat)
        all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def extract_resnet50_features(loader, device):
    """Extract features from ImageNet-pretrained ResNet-50 (avgpool layer)."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove the final fc layer, keep up to avgpool
    model.fc = nn.Identity()
    model = model.to(device).eval()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="ResNet-50"):
            imgs = imgs.to(device)
            features = model(imgs)  # [B, 2048]
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_features), np.concatenate(all_labels)


class SAMFeatureExtractor(nn.Module):
    """SAM ViT-H backbone for feature extraction."""

    def __init__(self, sam_checkpoint, sam_model_type='vit_h',
                 stage1_checkpoint=None):
        super().__init__()
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder

        # Load Stage 1 weights if provided (adapted)
        if stage1_checkpoint is not None:
            print(f"  Loading Stage 1 weights: {stage1_checkpoint}")
            state_dict = torch.load(stage1_checkpoint, map_location='cpu')
            self.backbone.load_state_dict(state_dict)

        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.patch_embed(x)

            if features.shape[1:3] != self.backbone.pos_embed.shape[1:3]:
                pos_embed = F.interpolate(
                    self.backbone.pos_embed.permute(0, 3, 1, 2),
                    size=features.shape[1:3],
                    mode='bicubic', align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                pos_embed = self.backbone.pos_embed

            features = features + pos_embed

            for blk in self.backbone.blocks:
                features = blk(features)

            pooled = F.adaptive_avg_pool2d(
                features.permute(0, 3, 1, 2), 1
            ).flatten(1)

            return F.normalize(pooled, p=2, dim=1)


def extract_sam_features(loader, device, sam_checkpoint, stage1_checkpoint=None):
    """Extract features from SAM ViT-H backbone."""
    model = SAMFeatureExtractor(
        sam_checkpoint, SAM_MODEL_TYPE, stage1_checkpoint
    ).to(device)

    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="SAM ViT-H"):
            imgs = imgs.to(device)
            features = model(imgs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_features), np.concatenate(all_labels)


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_single(dataset_name, backbone_name, device, project_root, batch_size=256):
    """Run a single dataset x backbone evaluation."""

    num_classes = DATASET_CONFIG[dataset_name]['num_classes']
    needs_normalize = (backbone_name != 'raw_pixels')

    print(f"\n--- {dataset_name} / {backbone_name} ---")

    # Load dataset
    dataset = UnifiedTestDataset(dataset_name, normalize=needs_normalize,
                                 img_size=IMG_SIZE, seed=SEED)
    num_workers = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"  Samples: {len(dataset)}")

    # Extract features
    sam_ckpt = os.path.join(project_root, SAM_CHECKPOINT)

    if backbone_name == 'raw_pixels':
        features, labels = extract_raw_pixels(loader)
    elif backbone_name == 'resnet50':
        features, labels = extract_resnet50_features(loader, device)
    elif backbone_name == 'sam_frozen':
        features, labels = extract_sam_features(loader, device, sam_ckpt,
                                                stage1_checkpoint=None)
    elif backbone_name == 'sam_adapted':
        stage1_path = os.path.join(project_root, STAGE1_PATHS[dataset_name])
        if not os.path.exists(stage1_path):
            print(f"  Stage 1 checkpoint not found: {stage1_path}")
            return None
        features, labels = extract_sam_features(loader, device, sam_ckpt,
                                                stage1_checkpoint=stage1_path)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    print(f"  Features shape: {features.shape}")

    # GPU k-means
    predictions = gpu_kmeans(features, num_classes, n_init=20, seed=SEED)

    # Metrics
    acc = hungarian_accuracy(labels, predictions)
    nmi = normalized_mutual_info_score(labels, predictions)
    ari = adjusted_rand_score(labels, predictions)

    print(f"  ACC: {acc*100:.2f}%  NMI: {nmi*100:.2f}%  ARI: {ari*100:.2f}%")

    return {'acc': acc, 'nmi': nmi, 'ari': ari}


def print_summary_table(results):
    """Print human-readable summary table."""
    datasets = ['pathmnist', 'kvasir', 'isic2019']
    dataset_labels = {'pathmnist': 'PathMNIST', 'kvasir': 'Kvasir', 'isic2019': 'ISIC2019'}
    backbone_labels = {
        'raw_pixels': 'Raw Pixels',
        'resnet50': 'ResNet-50',
        'sam_frozen': 'SAM Frozen',
        'sam_adapted': 'SAM Adapted',
    }

    print(f"\n{'='*100}")
    print("K-MEANS BACKBONE COMPARISON - SUMMARY")
    print(f"{'='*100}")

    # Header
    header = f"{'Dataset':<12}"
    for bb in BACKBONES:
        header += f" | {backbone_labels[bb]:>30}"
    print(header)
    print("-" * 100)

    for ds in datasets:
        row = f"{dataset_labels[ds]:<12}"
        for bb in BACKBONES:
            key = f"{ds}_{bb}"
            if key in results and results[key] is not None:
                m = results[key]
                cell = f"A={m['acc']*100:.1f} N={m['nmi']*100:.1f} R={m['ari']*100:.1f}"
            else:
                cell = "---"
            row += f" | {cell:>30}"
        print(row)

    print(f"{'='*100}")


def print_latex_table(results):
    """Print LaTeX-formatted table."""
    datasets = ['pathmnist', 'kvasir', 'isic2019']
    dataset_labels = {'pathmnist': 'PathMNIST', 'kvasir': 'Kvasir', 'isic2019': 'ISIC2019'}

    print(f"\n{'='*100}")
    print("LATEX TABLE")
    print(f"{'='*100}")
    print(r"\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c|}{Raw Pixels} & \multicolumn{3}{c|}{ResNet-50} & \multicolumn{3}{c|}{SAM Frozen} & \multicolumn{3}{c}{SAM Adapted} \\")
    print(r" & ACC & NMI & ARI & ACC & NMI & ARI & ACC & NMI & ARI & ACC & NMI & ARI \\")
    print(r"\midrule")

    for ds in datasets:
        row = dataset_labels[ds]
        for bb in BACKBONES:
            key = f"{ds}_{bb}"
            if key in results and results[key] is not None:
                m = results[key]
                row += f" & {m['acc']*100:.2f} & {m['nmi']*100:.2f} & {m['ari']*100:.2f}"
            else:
                row += " & -- & -- & --"
        row += r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser('K-Means Backbone Comparison')
    parser.add_argument('--project_root', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--datasets', nargs='+',
                        default=['pathmnist', 'kvasir', 'isic2019'])
    parser.add_argument('--backbones', nargs='+', default=BACKBONES)
    parser.add_argument('--output', type=str,
                        default='results/kmeans_backbone_comparison.json')
    parser.add_argument('--stage1_override', type=str, default=None,
                        help='Override Stage 1 checkpoint path for sam_adapted '
                             '(format: dataset:path, e.g. isic2019:checkpoints/...)')
    args = parser.parse_args()

    # Apply stage1 override
    if args.stage1_override:
        ds_name, ckpt_path = args.stage1_override.split(':', 1)
        STAGE1_PATHS[ds_name] = ckpt_path
        print(f"Stage 1 override: {ds_name} -> {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Backbones: {args.backbones}")
    print(f"{'='*80}\n")

    # Load existing results if any
    output_path = os.path.join(args.project_root, args.output)
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {output_path}")
    else:
        results = {}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run evaluations
    for ds in args.datasets:
        for bb in args.backbones:
            key = f"{ds}_{bb}"

            # Skip if already computed
            if key in results and results[key] is not None:
                print(f"\nSkipping {key} (already computed)")
                continue

            try:
                metrics = evaluate_single(ds, bb, device, args.project_root,
                                          args.batch_size)
                results[key] = metrics
            except Exception as e:
                print(f"\nERROR on {key}: {e}")
                import traceback
                traceback.print_exc()
                results[key] = None

            # Save after each evaluation
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Final output
    print_summary_table(results)
    print_latex_table(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
