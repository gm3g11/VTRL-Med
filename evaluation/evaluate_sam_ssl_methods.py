#!/usr/bin/env python3
"""
Evaluate SAM + SSL Methods (BYOL, DINO, MoCoV3)
K-Means clustering evaluation on complete test datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import json
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings('ignore')

from segment_anything import sam_model_registry
from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE
from vtrlmed.datasets.medmnist_dataset import MedMNIST_SPICE


# ============================================================================
# Model Components
# ============================================================================

class SAMFeatureExtractor(nn.Module):
    """SAM ViT backbone for feature extraction."""
    
    def __init__(self, stage1_checkpoint_path, sam_model_type='vit_h', 
                 sam_checkpoint='./sam_vit_h_4b8939.pth'):
        super().__init__()
        
        print(f"Loading SAM model...")
        print(f"  SAM checkpoint: {sam_checkpoint}")
        print(f"  Stage 1 checkpoint: {stage1_checkpoint_path}")
        
        # Load base SAM model
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder
        
        # Load Stage 1 trained weights
        if os.path.exists(stage1_checkpoint_path):
            print(f"  Loading Stage 1 weights...")
            state_dict = torch.load(stage1_checkpoint_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print("  ✓ Stage 1 weights loaded")
        else:
            print(f"  ⚠ Stage 1 checkpoint not found, using base SAM only")
        
        # Feature dimension
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        
        # Freeze all parameters for evaluation
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        print(f"  Feature dimension: {self.feature_dim}")
        print("✓ SAM model loaded for evaluation")
    
    def forward(self, x):
        """
        Forward pass through SAM backbone.
        Returns L2-normalized pooled features [B, feature_dim].
        """
        with torch.no_grad():
            # Patch embedding
            features = self.backbone.patch_embed(x)
            
            # Add positional encoding
            if features.shape[1:3] != self.backbone.pos_embed.shape[1:3]:
                pos_embed = F.interpolate(
                    self.backbone.pos_embed.permute(0, 3, 1, 2),
                    size=features.shape[1:3],
                    mode='bicubic',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                pos_embed = self.backbone.pos_embed
            
            features = features + pos_embed
            
            # Through transformer blocks
            for blk in self.backbone.blocks:
                features = blk(features)
            
            # Global average pooling
            pooled_features = F.adaptive_avg_pool2d(
                features.permute(0, 3, 1, 2), 1
            ).flatten(1)
            
            # L2 normalize (consistent with contrastive learning)
            return F.normalize(pooled_features, p=2, dim=1)


class TestDatasetWrapper:
    """Wrapper for test set evaluation - returns only test images."""
    def __init__(self, dataset_name, data_dir=None, img_size=224, seed=42):
        self.dataset_name = dataset_name
        
        if dataset_name == 'kvasir':
            if data_dir is None:
                data_dir = os.path.expanduser('data/kvasir-dataset')
            self.dataset = Kvasir_SPICE(
                data_dir=data_dir,
                img_size=img_size,
                split='test',
                seed=seed
            )
        elif dataset_name == 'isic2019':
            if data_dir is None:
                data_dir = os.path.expanduser('data/isic2019')
            self.dataset = ISIC2019_SPICE(
                data_dir=data_dir,
                img_size=img_size,
                split='test',
                seed=seed
            )
        elif dataset_name == 'pathmnist':
            self.dataset = MedMNIST_SPICE(
                'pathmnist',
                split='test',
                img_size=img_size,
                download=True
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"✓ {dataset_name} test set loaded: {len(self.dataset)} images")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # SPICE dataset returns: (weak_aug, strong_aug, test_img, label)
        _, _, test_img, label = self.dataset[idx]
        return test_img, label


# ============================================================================
# Evaluation Functions
# ============================================================================

def hungarian_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size
    return acc


def extract_features(feature_extractor, loader, device):
    """Extract features from entire test dataset."""
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features from test set...")
    with torch.no_grad():
        for img_batch, labels in tqdm(loader, desc="Feature extraction"):
            # Images are already normalized from dataset
            img_batch = img_batch.to(device, non_blocking=True)
            features = feature_extractor(img_batch)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"✓ Features extracted: {all_features.shape}")
    print(f"✓ Labels extracted: {all_labels.shape}")
    
    return all_features, all_labels


def evaluate_kmeans_clustering(features, labels, num_classes, n_init=20):
    """Run k-means and evaluate clustering metrics."""
    print(f"\nRunning k-means clustering...")
    print(f"  Number of clusters: {num_classes}")
    print(f"  Number of initializations: {n_init}")
    print(f"  Max iterations: 300")
    
    # Run k-means
    kmeans = KMeans(
        n_clusters=num_classes, 
        n_init=n_init, 
        random_state=42, 
        max_iter=300
    )
    predictions = kmeans.fit_predict(features)
    
    # Calculate metrics
    acc = hungarian_accuracy(labels, predictions)
    nmi = normalized_mutual_info_score(labels, predictions)
    ari = adjusted_rand_score(labels, predictions)
    
    print(f"✓ Clustering complete")
    
    return {
        'acc': acc,
        'nmi': nmi,
        'ari': ari,
        'predictions': predictions
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_single_model(args):
    """Evaluate a single SSL method on a single dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n{'='*80}")
    print(f"Evaluating: SAM+{args.ssl_method.upper()} on {args.dataset.upper()}")
    print(f"{'='*80}\n")
    
    # Build Stage 1 checkpoint path
    stage1_checkpoint = os.path.join(
        args.checkpoint_dir,
        args.ssl_method,
        f"stage1_sam_{args.ssl_method}_{args.dataset}_{args.img_size}.pth"
    )
    
    if not os.path.exists(stage1_checkpoint):
        print(f"✗ Stage 1 checkpoint not found: {stage1_checkpoint}")
        return None
    
    print(f"✓ Found checkpoint: {stage1_checkpoint}")
    
    # Load feature extractor
    feature_extractor = SAMFeatureExtractor(
        stage1_checkpoint_path=stage1_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint
    ).to(device)
    
    # Load test dataset
    test_dataset = TestDatasetWrapper(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        img_size=args.img_size,
        seed=args.seed
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Total test samples: {len(test_dataset)}")
    
    # Extract features from entire test set
    features, labels = extract_features(feature_extractor, test_loader, device)
    
    # Determine number of classes
    num_classes = len(np.unique(labels))
    print(f"Number of classes: {num_classes}")
    
    # K-means clustering evaluation
    metrics = evaluate_kmeans_clustering(features, labels, num_classes, n_init=20)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Method: SAM+{args.ssl_method.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Test samples: {len(labels)}")
    print(f"Number of clusters: {num_classes}")
    print(f"{'-'*80}")
    print(f"ACC: {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
    print(f"NMI: {metrics['nmi']:.4f} ({metrics['nmi']*100:.2f}%)")
    print(f"ARI: {metrics['ari']:.4f} ({metrics['ari']*100:.2f}%)")
    print(f"{'='*80}")
    
    # Clean up
    del feature_extractor
    torch.cuda.empty_cache()
    
    return {
        'ssl_method': args.ssl_method,
        'dataset': args.dataset,
        'acc': metrics['acc'],
        'nmi': metrics['nmi'],
        'ari': metrics['ari'],
        'num_samples': len(labels),
        'num_classes': num_classes
    }


def evaluate_all_methods(args):
    """Evaluate all SSL methods on all datasets."""
    
    ssl_methods = ['byol', 'dino', 'mocov3']
    datasets = ['kvasir', 'isic2019', 'pathmnist']
    
    print(f"{'='*80}")
    print("BATCH EVALUATION - All SSL Methods on All Datasets")
    print(f"{'='*80}\n")
    
    results = []
    
    for ssl_method in ssl_methods:
        for dataset in datasets:
            print(f"\n{'#'*80}")
            print(f"# SAM+{ssl_method.upper()} on {dataset.upper()}")
            print(f"{'#'*80}\n")
            
            # Create args for single evaluation
            single_args = argparse.Namespace(
                ssl_method=ssl_method,
                dataset=dataset,
                checkpoint_dir=args.checkpoint_dir,
                sam_model_type=args.sam_model_type,
                sam_checkpoint=args.sam_checkpoint,
                data_dir=args.data_dir if dataset != 'pathmnist' else None,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed
            )
            
            # Set correct data_dir for each dataset
            if dataset == 'kvasir':
                single_args.data_dir = args.kvasir_dir
            elif dataset == 'isic2019':
                single_args.data_dir = args.isic_dir
            
            try:
                result = evaluate_single_model(single_args)
                if result:
                    results.append(result)
                else:
                    print(f"✗ Skipped SAM+{ssl_method} on {dataset}")
            except Exception as e:
                print(f"✗ Error evaluating SAM+{ssl_method} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results to JSON
    output_file = 'stage1_sam_ssl_kmeans_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - Stage 1 K-Means Clustering")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Dataset':<15} {'ACC':>10} {'NMI':>10} {'ARI':>10}")
    print(f"{'-'*70}")
    
    for result in results:
        method_name = f"SAM+{result['ssl_method'].upper()}"
        print(f"{method_name:<20} {result['dataset']:<15} "
              f"{result['acc']*100:>9.2f}% {result['nmi']*100:>9.2f}% {result['ari']*100:>9.2f}%")
    
    # Calculate averages per method
    print(f"\n{'-'*70}")
    print("AVERAGES PER METHOD")
    print(f"{'-'*70}")
    
    for ssl_method in ssl_methods:
        method_results = [r for r in results if r['ssl_method'] == ssl_method]
        if method_results:
            avg_acc = np.mean([r['acc'] for r in method_results])
            avg_nmi = np.mean([r['nmi'] for r in method_results])
            avg_ari = np.mean([r['ari'] for r in method_results])
            method_name = f"SAM+{ssl_method.upper()}"
            print(f"{method_name:<20} {'AVG':<15} "
                  f"{avg_acc*100:>9.2f}% {avg_nmi*100:>9.2f}% {avg_ari*100:>9.2f}%")
    
    # Calculate averages per dataset
    print(f"\n{'-'*70}")
    print("AVERAGES PER DATASET")
    print(f"{'-'*70}")
    
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if dataset_results:
            avg_acc = np.mean([r['acc'] for r in dataset_results])
            avg_nmi = np.mean([r['nmi'] for r in dataset_results])
            avg_ari = np.mean([r['ari'] for r in dataset_results])
            print(f"{'ALL':<20} {dataset.upper():<15} "
                  f"{avg_acc*100:>9.2f}% {avg_nmi*100:>9.2f}% {avg_ari*100:>9.2f}%")
    
    # Print LaTeX format
    print(f"\n{'='*80}")
    print("LATEX FORMAT")
    print(f"{'='*80}")
    
    for ssl_method in ssl_methods:
        print(f"\n% SAM+{ssl_method.upper()}")
        for dataset in datasets:
            result = next((r for r in results if r['ssl_method'] == ssl_method and r['dataset'] == dataset), None)
            if result:
                print(f"{dataset}: &{result['acc']*100:.2f} &{result['nmi']*100:.2f} &{result['ari']*100:.2f} \\\\")
            else:
                print(f"{dataset}: &-- &-- &-- \\\\")
    
    print(f"\n{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser('SAM+SSL Methods K-Means Evaluation')
    
    # Mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch'],
                       help='Evaluation mode: single or batch')
    
    # Model
    parser.add_argument('--ssl_method', type=str, default='byol',
                       choices=['byol', 'dino', 'mocov3'],
                       help='SSL method (required for single mode)')
    parser.add_argument('--dataset', type=str, default='kvasir',
                       choices=['kvasir', 'isic2019', 'pathmnist'],
                       help='Dataset name (required for single mode)')
    
    # SAM model
    parser.add_argument('--sam_model_type', type=str, default='vit_h',
                       choices=['vit_b', 'vit_l', 'vit_h'],
                       help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, default='./sam_vit_h_4b8939.pth',
                       help='Base SAM checkpoint path')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Base directory for checkpoints')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Dataset directory (for single mode)')
    parser.add_argument('--kvasir_dir', type=str, 
                       default=os.path.expanduser('data/kvasir-dataset'),
                       help='Kvasir dataset directory (for batch mode)')
    parser.add_argument('--isic_dir', type=str,
                       default=os.path.expanduser('data/isic2019'),
                       help='ISIC2019 dataset directory (for batch mode)')
    
    # Evaluation settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'single':
        # Evaluate single method
        if args.dataset in ['kvasir', 'isic2019'] and args.data_dir is None:
            raise ValueError(f"--data_dir is required for {args.dataset} in single mode")
        
        result = evaluate_single_model(args)
        
        if result:
            # Save single result
            output_file = f"stage1_sam_{args.ssl_method}_{args.dataset}_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {output_file}")
    
    else:
        # Batch evaluation
        evaluate_all_methods(args)
