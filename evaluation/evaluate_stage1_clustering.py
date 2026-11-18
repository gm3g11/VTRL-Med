#!/usr/bin/env python3
"""
Evaluate Stage 1 Clustering Results (K-Means on SAM Features)
Generate ACC, NMI, ARI metrics using k-means clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torchvision.transforms.v2 as transformsV2
import json
from pathlib import Path

from segment_anything import sam_model_registry

# ============================================================================
# Model Components
# ============================================================================

class SAMFeatureExtractor(nn.Module):
    def __init__(self, stage1_checkpoint_path, sam_model_type='vit_h', 
                 sam_checkpoint='./sam_vit_h_4b8939.pth'):
        super().__init__()
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder
        
        # Load Stage 1 weights
        print(f"Loading Stage 1 weights from: {stage1_checkpoint_path}")
        self.backbone.load_state_dict(torch.load(stage1_checkpoint_path, map_location='cpu'))
        
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        print(f"✓ SAM backbone loaded with feature dim: {self.feature_dim}")
    
    def forward(self, x):
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
            
            # L2 normalize
            return F.normalize(pooled_features, p=2, dim=1)


class SAMRawDataset(Dataset):
    def __init__(self, dataset_name, split='train'):
        import medmnist
        from medmnist import INFO
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(split=split, download=True, transform=None, size=224)
        self.base_transform = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.uint8, scale=False),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self.base_transform(image), torch.tensor(label).long().squeeze()


# ============================================================================
# Evaluation Functions
# ============================================================================

def hungarian_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size
    return acc


def extract_features(feature_extractor, loader, device):
    """Extract features from dataset"""
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    # Test transform
    test_transform = torch.nn.Sequential(
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.Resize((224, 224), antialias=True),
        transformsV2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ).to(device)
    
    with torch.no_grad():
        for raw_img_batch, labels in tqdm(loader, desc="Extracting features"):
            # Transform and extract features
            img_batch = test_transform(raw_img_batch.to(device))
            features = feature_extractor(img_batch)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels


def evaluate_kmeans_clustering(features, labels, num_classes, n_init=20):
    """Run k-means and evaluate clustering metrics"""
    print(f"Running k-means with {num_classes} clusters (n_init={n_init})...")
    
    # Run k-means
    kmeans = KMeans(n_clusters=num_classes, n_init=n_init, random_state=42, max_iter=300)
    predictions = kmeans.fit_predict(features)
    
    # Calculate metrics
    acc = hungarian_accuracy(labels, predictions)
    nmi = normalized_mutual_info_score(labels, predictions)
    ari = adjusted_rand_score(labels, predictions)
    
    return {
        'acc': acc,
        'nmi': nmi,
        'ari': ari
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    # Configuration
    datasets = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 
                'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
                'organamnist', 'organcmnist', 'organsmnist']
    
    dataset_num_classes = {
        'bloodmnist': 8, 'pathmnist': 9, 'dermamnist': 7, 'octmnist': 4,
        'pneumoniamnist': 2, 'retinamnist': 5, 'breastmnist': 2, 'tissuemnist': 8,
        'organamnist': 11, 'organcmnist': 11, 'organsmnist': 11,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    stage1_base = Path('SAM_SPICE_checkpoints/stage1')
    sam_checkpoint = './sam_vit_h_4b8939.pth'
    sam_model_type = 'vit_h'
    img_size = 224
    batch_size = 512
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Evaluating: {dataset}")
        print(f"{'='*80}")
        
        # Path
        stage1_path = stage1_base / dataset / f"stage1_sam_simclr_{dataset}_{img_size}.pth"
        
        if not stage1_path.exists():
            print(f"✗ Stage 1 checkpoint not found: {stage1_path}")
            results[dataset] = None
            continue
        
        print(f"✓ Found checkpoint")
        
        try:
            # Load feature extractor
            num_classes = dataset_num_classes[dataset]
            
            feature_extractor = SAMFeatureExtractor(
                str(stage1_path), sam_model_type, sam_checkpoint
            ).to(device)
            
            # Load test data
            test_dataset = SAMRawDataset(dataset, 'test')
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=8
            )
            print(f"✓ Test data loaded: {len(test_dataset)} samples")
            
            # Extract features
            features, labels = extract_features(feature_extractor, test_loader, device)
            print(f"✓ Features extracted: {features.shape}")
            
            # K-means clustering
            metrics = evaluate_kmeans_clustering(features, labels, num_classes)
            
            print(f"\nResults:")
            print(f"  ACC: {metrics['acc']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            print(f"  ARI: {metrics['ari']:.4f}")
            
            results[dataset] = metrics
            
            # Clean up
            del feature_extractor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset] = None
    
    # Save results
    output_file = 'stage1_kmeans_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE - Stage 1 K-Means Clustering")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'ACC':<10} {'NMI':<10} {'ARI':<10}")
    print("-" * 50)
    
    acc_sum, nmi_sum, ari_sum = 0, 0, 0
    count = 0
    
    for dataset in datasets:
        if results[dataset] is not None:
            metrics = results[dataset]
            print(f"{dataset:<20} {metrics['acc']*100:>8.2f}% {metrics['nmi']*100:>8.2f}% {metrics['ari']*100:>8.2f}%")
            acc_sum += metrics['acc']
            nmi_sum += metrics['nmi']
            ari_sum += metrics['ari']
            count += 1
        else:
            print(f"{dataset:<20} {'MISSING'}")
    
    if count > 0:
        print("-" * 50)
        print(f"{'AVERAGE':<20} {acc_sum/count*100:>8.2f}% {nmi_sum/count*100:>8.2f}% {ari_sum/count*100:>8.2f}%")
    
    # Print LaTeX format
    print(f"\n{'='*80}")
    print("LATEX FORMAT (multiply by 100)")
    print(f"{'='*80}")
    
    row1_datasets = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 
                     'retinamnist', 'breastmnist', 'bloodmnist']
    row2_datasets = ['tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
    
    def format_latex(dataset_list):
        output = []
        for d in dataset_list:
            if results[d] is not None:
                m = results[d]
                output.append(f"&{m['acc']*100:.2f} &{m['nmi']*100:.2f} &{m['ari']*100:.2f}")
            else:
                output.append("&-- &-- &--")
        return " ".join(output)
    
    print("\n% Row 1")
    print(format_latex(row1_datasets) + " \\\\")
    
    print("\n% Row 2 (with Average)")
    row2_str = format_latex(row2_datasets)
    if count > 0:
        avg_str = f"&{acc_sum/count*100:.2f} &{nmi_sum/count*100:.2f} &{ari_sum/count*100:.2f}"
        print(f"{row2_str} {avg_str} \\\\")
    else:
        print(f"{row2_str} \\\\")
    
    print(f"\n{'='*80}")
    print("Note: Stage 1 uses k-means on SAM+SimCLR features (no clustering head)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
