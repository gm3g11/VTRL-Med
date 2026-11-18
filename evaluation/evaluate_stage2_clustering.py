#!/usr/bin/env python3
"""
Evaluate Stage 2 Clustering Results
Generate ACC, NMI, ARI metrics from saved Stage 2 weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import torchvision.transforms.v2 as transformsV2
import json
from pathlib import Path

from segment_anything import sam_model_registry

# ============================================================================
# Model Components (from Stage 2)
# ============================================================================

class SAMFeatureExtractor(nn.Module):
    def __init__(self, stage1_checkpoint_path, sam_model_type='vit_h', 
                 sam_checkpoint='./sam_vit_h_4b8939.pth'):
        super().__init__()
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder
        self.backbone.load_state_dict(torch.load(stage1_checkpoint_path, map_location='cpu'))
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
                    mode='bicubic',
                    align_corners=False
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


class SPICEClusteringHead(nn.Module):
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
    
    def forward(self, x):
        return self.head(x)


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


def evaluate_clustering(model, feature_extractor, test_loader, device):
    """Evaluate clustering metrics"""
    model.eval()
    feature_extractor.eval()
    
    all_preds = []
    all_labels = []
    
    # Test transform
    test_transform = torch.nn.Sequential(
        transformsV2.ToDtype(torch.float32, scale=True),
        transformsV2.Resize((224, 224), antialias=True),
        transformsV2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ).to(device)
    
    with torch.no_grad():
        for raw_img_batch, labels in tqdm(test_loader, desc="Evaluating"):
            # Transform and extract features
            img_batch = test_transform(raw_img_batch.to(device))
            features = feature_extractor(img_batch)
            
            # Get predictions
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    acc = hungarian_accuracy(all_labels, all_preds)
    nmi = normalized_mutual_info_score(all_labels, all_preds)
    ari = adjusted_rand_score(all_labels, all_preds)
    
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
    stage2_base = Path('SAM_SPICE_checkpoints/stage2')
    sam_checkpoint = './sam_vit_h_4b8939.pth'
    sam_model_type = 'vit_h'
    img_size = 224
    batch_size = 512
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Evaluating: {dataset}")
        print(f"{'='*80}")
        
        # Paths
        stage1_path = stage1_base / dataset / f"stage1_sam_simclr_{dataset}_{img_size}.pth"
        stage2_path = stage2_base / dataset / f"stage2_sam_simclr_{dataset}_{img_size}.pth"
        
        if not stage1_path.exists():
            print(f"✗ Stage 1 checkpoint not found: {stage1_path}")
            results[dataset] = None
            continue
        
        if not stage2_path.exists():
            print(f"✗ Stage 2 checkpoint not found: {stage2_path}")
            results[dataset] = None
            continue
        
        print(f"✓ Found checkpoints")
        
        try:
            # Load models
            num_classes = dataset_num_classes[dataset]
            
            feature_extractor = SAMFeatureExtractor(
                str(stage1_path), sam_model_type, sam_checkpoint
            ).to(device)
            
            clustering_head = SPICEClusteringHead(
                feature_extractor.feature_dim, num_classes
            ).to(device)
            
            clustering_head.load_state_dict(torch.load(stage2_path, map_location='cpu'))
            print(f"✓ Models loaded")
            
            # Load test data
            test_dataset = SAMRawDataset(dataset, 'test')
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=8
            )
            print(f"✓ Test data loaded: {len(test_dataset)} samples")
            
            # Evaluate
            metrics = evaluate_clustering(
                clustering_head, feature_extractor, test_loader, device
            )
            
            print(f"\nResults:")
            print(f"  ACC: {metrics['acc']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            print(f"  ARI: {metrics['ari']:.4f}")
            
            results[dataset] = metrics
            
        except Exception as e:
            print(f"✗ Error processing {dataset}: {e}")
            results[dataset] = None
    
    # Save results
    output_file = 'stage2_clustering_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
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


if __name__ == '__main__':
    main()
