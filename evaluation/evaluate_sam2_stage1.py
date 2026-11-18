#!/usr/bin/env python3
"""
Evaluate Stage 1 SAM2 Models - Full Test Dataset
K-Means clustering evaluation on complete test set
Supports: SAM2, Medical-SAM2, MedSAM-2
"""

import sys
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

# Add SAM2 to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'tools', 'segment_anything_2'))

from sam2.build_sam import build_sam2
from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE
from vtrlmed.datasets.medmnist_dataset import MedMNIST_SPICE


# ============================================================================
# Model Components
# ============================================================================

class SAM2FeatureExtractor(nn.Module):
    """SAM2 backbone for feature extraction."""
    
    def __init__(self, config_name, base_checkpoint, stage1_checkpoint=None, model_type='sam2'):
        super().__init__()
        
        print(f"Loading {model_type.upper()} model for evaluation...")
        print(f"  Config: {config_name}")
        print(f"  Base checkpoint: {base_checkpoint}")
        
        # Load full SAM2 model with base weights
        sam2_model = build_sam2(config_name, base_checkpoint)
        
        # Extract image encoder (backbone)
        self.backbone = sam2_model.image_encoder
        
        # Load Stage 1 trained weights if provided
        if stage1_checkpoint and os.path.exists(stage1_checkpoint):
            print(f"  Loading Stage 1 weights: {stage1_checkpoint}")
            state_dict = torch.load(stage1_checkpoint, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print("  ✓ Stage 1 weights loaded")
        else:
            print("  ⚠ Using base model only (no Stage 1 weights)")
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.backbone.cuda()
            
            test_output = self.backbone(dummy_input)
            features = self._extract_features(test_output)
            
            if len(features.shape) == 4:  # [B, C, H, W]
                self.feature_dim = features.shape[1]
            elif len(features.shape) == 3:  # [B, N, C]
                self.feature_dim = features.shape[2]
            elif len(features.shape) == 2:  # [B, C]
                self.feature_dim = features.shape[1]
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
            
            print(f"  Feature dimension: {self.feature_dim}")
    
    def _extract_features(self, output):
        """Extract features from SAM2 output."""
        if isinstance(output, dict):
            # Try common keys
            for key in ['vision_features', 'image_embed', 'backbone_fpn', 'features']:
                if key in output:
                    features = output[key]
                    if isinstance(features, (list, tuple)):
                        return features[-1]
                    return features
            
            # Get any tensor
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
                elif isinstance(value, (list, tuple)):
                    return value[-1]
        elif isinstance(output, (list, tuple)):
            return output[-1]
        else:
            return output
        
        raise ValueError(f"Could not extract features from output")
    
    def forward(self, x):
        """Forward pass returning pooled features [B, feature_dim]."""
        with torch.no_grad():
            output = self.backbone(x)
            features = self._extract_features(output)
            
            # Global average pooling
            if len(features.shape) == 4:  # [B, C, H, W]
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.flatten(1)
            elif len(features.shape) == 3:  # [B, N, C]
                features = features.mean(dim=1)
            
            # L2 normalize
            return F.normalize(features, p=2, dim=1)


class TestDatasetWrapper:
    """Wrapper for test set evaluation - returns only test images"""
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
    """Evaluate a single model on a single dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {args.model_type.upper()} on {args.dataset.upper()}")
    print(f"{'='*80}\n")
    
    # Determine config and base checkpoint based on model type
    if args.model_type == 'medical_sam2':
        config_name = 'sam2_hiera_t'
        base_checkpoint = 'weights/MedSAM2_pretrain.pth'
    elif args.model_type == 'medsam2':
        config_name = 'sam2_hiera_t'
        base_checkpoint = 'weights/MedSAM2_latest.pt'
    else:  # sam2
        config_name = 'sam2_hiera_l'
        base_checkpoint = 'weights/sam2_hiera_large.pt'
    
    # Build Stage 1 checkpoint path
    stage1_checkpoint = os.path.join(
        args.checkpoint_dir,
        f"{args.model_type}_simclr",
        f"stage1_{args.model_type}_simclr_{args.dataset}_{args.img_size}.pth"
    )
    
    if not os.path.exists(stage1_checkpoint):
        print(f"✗ Stage 1 checkpoint not found: {stage1_checkpoint}")
        return None
    
    print(f"✓ Found checkpoint: {stage1_checkpoint}")
    
    # Load feature extractor
    feature_extractor = SAM2FeatureExtractor(
        config_name=config_name,
        base_checkpoint=base_checkpoint,
        stage1_checkpoint=stage1_checkpoint,
        model_type=args.model_type
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
    print(f"Model: {args.model_type.upper()}")
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
        'model_type': args.model_type,
        'dataset': args.dataset,
        'acc': metrics['acc'],
        'nmi': metrics['nmi'],
        'ari': metrics['ari'],
        'num_samples': len(labels),
        'num_classes': num_classes
    }


def evaluate_all_models(args):
    """Evaluate all model-dataset combinations."""
    
    model_types = ['medical_sam2', 'medsam2', 'sam2']
    datasets = ['kvasir', 'isic2019', 'pathmnist']
    
    print(f"{'='*80}")
    print("BATCH EVALUATION - All Models on All Datasets")
    print(f"{'='*80}\n")
    
    results = []
    
    for model_type in model_types:
        for dataset in datasets:
            print(f"\n{'#'*80}")
            print(f"# {model_type.upper()} on {dataset.upper()}")
            print(f"{'#'*80}\n")
            
            # Create args for single evaluation
            single_args = argparse.Namespace(
                model_type=model_type,
                dataset=dataset,
                checkpoint_dir=args.checkpoint_dir,
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
                    print(f"✗ Skipped {model_type} on {dataset}")
            except Exception as e:
                print(f"✗ Error evaluating {model_type} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results to JSON
    output_file = 'stage1_sam2_kmeans_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - Stage 1 K-Means Clustering")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Dataset':<15} {'ACC':>10} {'NMI':>10} {'ARI':>10}")
    print(f"{'-'*70}")
    
    for result in results:
        print(f"{result['model_type']:<20} {result['dataset']:<15} "
              f"{result['acc']*100:>9.2f}% {result['nmi']*100:>9.2f}% {result['ari']*100:>9.2f}%")
    
    # Calculate averages per model
    print(f"\n{'-'*70}")
    print("AVERAGES PER MODEL")
    print(f"{'-'*70}")
    
    for model_type in model_types:
        model_results = [r for r in results if r['model_type'] == model_type]
        if model_results:
            avg_acc = np.mean([r['acc'] for r in model_results])
            avg_nmi = np.mean([r['nmi'] for r in model_results])
            avg_ari = np.mean([r['ari'] for r in model_results])
            print(f"{model_type:<20} {'AVG':<15} "
                  f"{avg_acc*100:>9.2f}% {avg_nmi*100:>9.2f}% {avg_ari*100:>9.2f}%")
    
    # Print LaTeX format
    print(f"\n{'='*80}")
    print("LATEX FORMAT (multiply by 100)")
    print(f"{'='*80}")
    
    for model_type in model_types:
        print(f"\n% {model_type.upper()}")
        for dataset in datasets:
            result = next((r for r in results if r['model_type'] == model_type and r['dataset'] == dataset), None)
            if result:
                print(f"{dataset}: &{result['acc']*100:.2f} &{result['nmi']*100:.2f} &{result['ari']*100:.2f} \\\\")
            else:
                print(f"{dataset}: &-- &-- &-- \\\\")
    
    print(f"\n{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser('SAM2 Stage 1 K-Means Evaluation')
    
    # Mode
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch'],
                       help='Evaluation mode: single or batch')
    
    # Model
    parser.add_argument('--model_type', type=str, default='medical_sam2',
                       choices=['sam2', 'medical_sam2', 'medsam2'],
                       help='Model type (required for single mode)')
    parser.add_argument('--dataset', type=str, default='kvasir',
                       choices=['kvasir', 'isic2019', 'pathmnist'],
                       help='Dataset name (required for single mode)')
    
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
        # Evaluate single model
        if args.dataset in ['kvasir', 'isic2019'] and args.data_dir is None:
            raise ValueError(f"--data_dir is required for {args.dataset} in single mode")
        
        result = evaluate_single_model(args)
        
        if result:
            # Save single result
            output_file = f"stage1_{args.model_type}_{args.dataset}_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {output_file}")
    
    else:
        # Batch evaluation
        evaluate_all_models(args)
