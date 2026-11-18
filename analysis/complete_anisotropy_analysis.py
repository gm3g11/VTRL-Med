#!/usr/bin/env python3
"""
Complete anisotropy analysis with dataset loading for DINO+SAM features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import stats
import warnings
from tqdm import tqdm
import argparse
from collections import Counter, OrderedDict
import medmnist
from medmnist import INFO
from medmnist.dataset import DermaMNIST
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")

# Import required components
from segment_anything import sam_model_registry

class MedMNISTDataset(torch.utils.data.Dataset):
    """Dataset wrapper for MedMNIST with proper transforms - based on working SimCLR code."""
    
    def __init__(self, dataset_name, split='test', target_size=1024, download=True):
        self.dataset_name = dataset_name
        self.split = split
        
        # Use the proven approach from the SimCLR code, but handle size properly
        if dataset_name == 'dermamnist':
            # First try to create dataset to check available sizes
            try:
                temp_ds = DermaMNIST(split=split, download=download, transform=None, size=224)
                available_sizes = temp_ds.available_sizes
                print(f"Available sizes for DermaMNIST: {available_sizes}")
                
                # Use the largest available size, or 224 as fallback
                if 224 in available_sizes:
                    native_size = 224
                elif 64 in available_sizes:
                    native_size = 64
                elif 28 in available_sizes:
                    native_size = 28
                else:
                    native_size = available_sizes[0] if available_sizes else 224
                
                print(f"Using native size: {native_size}, will resize to {target_size}")
                self.dataset = DermaMNIST(split=split, download=download, transform=None, size=native_size)
                
            except Exception as e:
                print(f"Error with size parameter: {e}")
                # Fallback without size parameter
                self.dataset = DermaMNIST(split=split, download=download, transform=None)
        else:
            # Fallback for other datasets
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            try:
                self.dataset = DataClass(split=split, download=download, transform=None, size=224)
            except TypeError:
                # Try without size parameter
                self.dataset = DataClass(split=split, download=download, transform=None)
        
        # SAM expects target_size x target_size RGB images
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {dataset_name} {split} split: {len(self.dataset)} samples")
        
        # Test first sample to understand format
        try:
            sample_image, sample_label = self.dataset[0]
            print(f"Sample image type: {type(sample_image)}, shape/size: {getattr(sample_image, 'shape', getattr(sample_image, 'size', 'unknown'))}")
            print(f"Sample label: {sample_label} (type: {type(sample_label)})")
        except Exception as e:
            print(f"Could not inspect sample: {e}")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # Convert to PIL Image using the proven SimCLR approach
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                # Grayscale 2D -> RGB
                img = np.stack([img]*3, axis=-1)
            elif img.shape[-1] == 1:
                # Grayscale 3D -> RGB
                img = np.repeat(img, 3, axis=-1)
            img = Image.fromarray(img.astype(np.uint8))
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        
        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transforms
        img = self.transform(img)
        
        # Robust label conversion (from SimCLR code)
        label = int(np.array(target).item())
        
        return img, torch.tensor(label).long()

class AnisotropyAnalyzer:
    """Analyze and fix anisotropic embeddings."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.features = None
        self.labels = None
        
    def load_backbone(self, sam_model_type='vit_h'):
        """Load SAM backbone with trained weights."""
        print(f"Loading SAM {sam_model_type} backbone...")
        
        # Load base SAM model
        backbone = sam_model_registry[sam_model_type](checkpoint='./sam_vit_h_4b8939.pth').image_encoder
        
        # Load your checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        if 'student' in ckpt:
            state_dict = ckpt['student']
        else:
            state_dict = ckpt
            
        print(f"Checkpoint keys: {list(state_dict.keys())[:10]}...")
        
        # Extract backbone weights
        backbone_state = OrderedDict()
        loaded_keys = 0
        
        for k, v in state_dict.items():
            if k.startswith('module.backbone.'):
                new_key = k[16:]  # Remove 'module.backbone.'
                backbone_state[new_key] = v
                loaded_keys += 1
            elif k.startswith('backbone.'):
                new_key = k[9:]  # Remove 'backbone.'
                backbone_state[new_key] = v
                loaded_keys += 1
            # Also check for direct matches
            elif k in backbone.state_dict():
                backbone_state[k] = v
                loaded_keys += 1
        
        print(f"Loaded {loaded_keys} backbone parameters")
        print(f"Expected backbone parameters: {len(backbone.state_dict())}")
        
        # Load state dict
        missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        backbone.eval()
        backbone = backbone.to(self.device)
        return backbone
    
    def extract_features(self, backbone, dataloader, max_samples=1000):
        """Extract features from the backbone."""
        all_features = []
        all_labels = []
        
        print(f"Extracting features (max {max_samples} samples)...")
        
        with torch.no_grad():
            sample_count = 0
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
                if sample_count >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # Manual forward pass (matching Stage 1 DINO forward)
                B, C, H, W = images.shape
                
                # Patch embedding
                features = backbone.patch_embed(images)  # [B, H/16, W/16, 1280]
                
                # Add positional encoding
                if features.shape[1:3] != backbone.pos_embed.shape[1:3]:
                    pos_embed = F.interpolate(
                        backbone.pos_embed.permute(0, 3, 1, 2),
                        size=features.shape[1:3], mode='bicubic', align_corners=False
                    ).permute(0, 2, 3, 1)
                else:
                    pos_embed = backbone.pos_embed
                
                features = features + pos_embed
                
                # Through transformer blocks
                for blk in backbone.blocks:
                    features = blk(features)
                
                # Global average pooling to get 1280-dim features
                pooled = F.adaptive_avg_pool2d(features.permute(0, 3, 1, 2), 1).flatten(1)
                
                all_features.append(pooled.cpu())
                all_labels.append(labels)
                
                sample_count += images.shape[0]
        
        self.features = torch.cat(all_features, dim=0).numpy()
        self.labels = torch.cat(all_labels, dim=0).numpy()
        
        print(f"Extracted features shape: {self.features.shape}")
        print(f"Label distribution: {Counter(self.labels)}")
        
        return self.features, self.labels
    
    def comprehensive_anisotropy_check(self):
        """Most comprehensive anisotropy analysis."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANISOTROPY ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Mean vector dominance
        print("\n1. MEAN VECTOR DOMINANCE")
        print("-"*50)
        
        mean_vector = np.mean(self.features, axis=0, keepdims=True)
        mean_norm = np.linalg.norm(mean_vector)
        feature_norms = np.linalg.norm(self.features, axis=1)
        
        print(f"Mean vector norm: {mean_norm:.4f}")
        print(f"Average feature norm: {np.mean(feature_norms):.4f}")
        print(f"Feature norm std: {np.std(feature_norms):.4f}")
        
        # Cosine similarity to mean
        features_normalized = self.features / (np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-8)
        mean_normalized = mean_vector / (mean_norm + 1e-8)
        cos_to_mean = np.dot(features_normalized, mean_normalized.T).squeeze()
        
        results['mean_cos_sim'] = {
            'mean': np.mean(cos_to_mean),
            'std': np.std(cos_to_mean),
            'min': np.min(cos_to_mean),
            'max': np.max(cos_to_mean)
        }
        
        print(f"Cosine similarity to mean:")
        print(f"  - Average: {results['mean_cos_sim']['mean']:.4f}")
        print(f"  - Std: {results['mean_cos_sim']['std']:.4f}")
        print(f"  - Range: [{results['mean_cos_sim']['min']:.4f}, {results['mean_cos_sim']['max']:.4f}]")
        
        # Severity assessment
        if results['mean_cos_sim']['mean'] > 0.95:
            print("  🚨 SEVERE ANISOTROPY: Features extremely aligned with mean!")
            anisotropy_level = "SEVERE"
        elif results['mean_cos_sim']['mean'] > 0.9:
            print("  ⚠️ HIGH ANISOTROPY: Features strongly aligned with mean!")
            anisotropy_level = "HIGH"
        elif results['mean_cos_sim']['mean'] > 0.7:
            print("  ⚠️ MODERATE ANISOTROPY: Features moderately aligned with mean")
            anisotropy_level = "MODERATE"
        else:
            print("  ✓ LOW ANISOTROPY")
            anisotropy_level = "LOW"
        
        results['anisotropy_level'] = anisotropy_level
        
        # 2. Isotropy measures
        print("\n2. ISOTROPY MEASURES")
        print("-"*50)
        
        # Center features
        features_centered = self.features - mean_vector
        
        # Covariance analysis
        cov = np.cov(features_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Multiple isotropy metrics
        isotropy_score = eigenvalues[-1] / (eigenvalues[0] + 1e-8)
        effective_rank = np.exp(-np.sum((eigenvalues / np.sum(eigenvalues)) * np.log(eigenvalues / np.sum(eigenvalues) + 1e-10)))
        
        results['isotropy'] = {
            'score': isotropy_score,
            'effective_rank': effective_rank,
            'total_components': len(eigenvalues)
        }
        
        print(f"Isotropy score: {isotropy_score:.6f}")
        print(f"  (1.0 = perfectly isotropic, 0.0 = completely anisotropic)")
        print(f"Effective rank: {effective_rank:.1f} / {len(eigenvalues)}")
        
        if isotropy_score < 0.001:
            print("  🚨 SEVERE isotropy loss!")
        elif isotropy_score < 0.01:
            print("  ⚠️ HIGH isotropy loss")
        elif isotropy_score < 0.1:
            print("  ⚠️ MODERATE isotropy loss")
        else:
            print("  ✓ Reasonable isotropy")
        
        # 3. Principal component dominance
        print("\n3. PRINCIPAL COMPONENT ANALYSIS")
        print("-"*50)
        
        pca = PCA(n_components=min(50, self.features.shape[1]))
        pca.fit(self.features)
        
        var_explained = pca.explained_variance_ratio_
        results['pca'] = {
            'pc1_variance': var_explained[0],
            'pc1_3_variance': np.sum(var_explained[:3]),
            'pc1_10_variance': np.sum(var_explained[:10])
        }
        
        print(f"Variance explained:")
        print(f"  - PC1: {var_explained[0]:.4f} ({var_explained[0]*100:.1f}%)")
        print(f"  - PC1-3: {np.sum(var_explained[:3]):.4f} ({np.sum(var_explained[:3])*100:.1f}%)")
        print(f"  - PC1-10: {np.sum(var_explained[:10]):.4f} ({np.sum(var_explained[:10])*100:.1f}%)")
        
        if var_explained[0] > 0.7:
            print("  🚨 PC1 dominates completely - severe anisotropy!")
        elif var_explained[0] > 0.5:
            print("  ⚠️ PC1 is very dominant - high anisotropy")
        elif var_explained[0] > 0.3:
            print("  ⚠️ PC1 is somewhat dominant - moderate anisotropy")
        else:
            print("  ✓ Reasonable PC distribution")
        
        # 4. Hubness analysis
        print("\n4. HUBNESS ANALYSIS")
        print("-"*50)
        
        # Compute hubness for subset (computationally expensive)
        n_sample = min(500, len(self.features))
        sample_idx = np.random.choice(len(self.features), n_sample, replace=False)
        sample_features = self.features[sample_idx]
        sample_labels = self.labels[sample_idx]
        
        # Cosine similarity
        similarities = cosine_similarity(sample_features)
        np.fill_diagonal(similarities, -np.inf)
        
        k = 10
        hub_scores = np.zeros(n_sample)
        
        for i in range(n_sample):
            # Find k nearest neighbors
            neighbors = np.argpartition(similarities[i], -k)[-k:]
            hub_scores[neighbors] += 1
        
        results['hubness'] = {
            'mean_hub_score': np.mean(hub_scores),
            'std_hub_score': np.std(hub_scores),
            'max_hub_score': np.max(hub_scores),
            'skewness': stats.skew(hub_scores)
        }
        
        print(f"Hub occurrence statistics (k={k}, n={n_sample}):")
        print(f"  - Mean: {np.mean(hub_scores):.2f}")
        print(f"  - Std: {np.std(hub_scores):.2f}")
        print(f"  - Max: {np.max(hub_scores):.0f}")
        print(f"  - Skewness: {stats.skew(hub_scores):.4f}")
        
        # Identify major hubs
        hub_threshold = np.mean(hub_scores) + 2 * np.std(hub_scores)
        n_hubs = np.sum(hub_scores > hub_threshold)
        
        print(f"  - Number of major hubs: {n_hubs}")
        
        if n_hubs > 0:
            hub_indices = np.where(hub_scores > hub_threshold)[0]
            hub_labels = sample_labels[hub_indices]
            hub_label_counts = Counter(hub_labels)
            
            print(f"  - Hub label distribution: {dict(hub_label_counts)}")
            
            # Check if hubs are class-imbalanced
            total_label_counts = Counter(sample_labels)
            print(f"  - Overall label distribution: {dict(total_label_counts)}")
        
        # 5. Class separability analysis
        print("\n5. CLASS SEPARABILITY ANALYSIS")
        print("-"*50)
        
        unique_labels = np.unique(self.labels)
        intra_distances = []
        inter_distances = []
        
        # Sample for computational efficiency
        for label in unique_labels:
            class_mask = self.labels == label
            class_features = self.features[class_mask]
            
            if len(class_features) < 2:
                continue
                
            # Sample within class
            n_intra_samples = min(50, len(class_features))
            if n_intra_samples > 1:
                sample_idx = np.random.choice(len(class_features), n_intra_samples, replace=False)
                sampled_class_features = class_features[sample_idx]
                
                # Intra-class distances
                for i in range(len(sampled_class_features)):
                    for j in range(i+1, len(sampled_class_features)):
                        dist = np.linalg.norm(sampled_class_features[i] - sampled_class_features[j])
                        intra_distances.append(dist)
                
                # Inter-class distances (to other classes)
                other_features = self.features[~class_mask]
                n_inter_samples = min(100, len(other_features))
                if n_inter_samples > 0:
                    other_sample_idx = np.random.choice(len(other_features), n_inter_samples, replace=False)
                    sampled_other_features = other_features[other_sample_idx]
                    
                    for i in range(min(10, len(sampled_class_features))):
                        for j in range(min(20, len(sampled_other_features))):
                            dist = np.linalg.norm(sampled_class_features[i] - sampled_other_features[j])
                            inter_distances.append(dist)
        
        if len(intra_distances) > 0 and len(inter_distances) > 0:
            mean_intra = np.mean(intra_distances)
            mean_inter = np.mean(inter_distances)
            separability = mean_inter / (mean_intra + 1e-8)
            
            results['separability'] = {
                'intra_class_distance': mean_intra,
                'inter_class_distance': mean_inter,
                'separability_ratio': separability
            }
            
            print(f"Distance analysis:")
            print(f"  - Mean intra-class distance: {mean_intra:.4f}")
            print(f"  - Mean inter-class distance: {mean_inter:.4f}")
            print(f"  - Separability ratio: {separability:.4f}")
            
            if separability < 1.1:
                print("  🚨 POOR separability - classes heavily overlapping!")
            elif separability < 1.5:
                print("  ⚠️ MODERATE separability - classes somewhat overlapping")
            else:
                print("  ✓ GOOD separability")
        
        # 6. Overall assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        issues = []
        if results['mean_cos_sim']['mean'] > 0.9:
            issues.append(f"High mean alignment ({results['mean_cos_sim']['mean']:.3f})")
        if results['isotropy']['score'] < 0.01:
            issues.append(f"Low isotropy ({results['isotropy']['score']:.6f})")
        if results['pca']['pc1_variance'] > 0.5:
            issues.append(f"PC1 dominance ({results['pca']['pc1_variance']:.3f})")
        if results['hubness']['skewness'] > 2:
            issues.append(f"High hubness ({results['hubness']['skewness']:.2f})")
        
        if len(issues) > 0:
            print(f"🚨 ANISOTROPY DETECTED - {len(issues)} major issues:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n✅ RECOMMENDATION: Apply anisotropy corrections before Stage 2/3!")
        else:
            print("✅ Features appear reasonably isotropic")
        
        return results

def test_dataset_loading(dataset_name):
    """Test dataset loading to debug issues - using proven SimCLR approach."""
    print(f"Testing {dataset_name} dataset loading...")
    
    try:
        # Test dataset creation using the corrected approach
        dataset = MedMNISTDataset(dataset_name, split='test', target_size=1024)
        print(f"✓ Dataset created successfully: {len(dataset)} samples")
        
        # Test first sample
        image, label = dataset[0]
        print(f"✓ First sample loaded: image shape {image.shape}, label {label}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch_images, batch_labels = next(iter(dataloader))
        print(f"✓ DataLoader works: batch shape {batch_images.shape}, labels {batch_labels}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser('Complete Anisotropy Analysis')
    parser.add_argument('--checkpoint', required=True, type=str,
                       help='Path to DINO+SAM checkpoint')
    parser.add_argument('--dataset', required=True, type=str,
                       help='MedMNIST dataset name (e.g., dermamnist)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_dataset_only', action='store_true',
                       help='Only test dataset loading')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SPICE+SAM ANISOTROPY ANALYSIS")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    
    # Test dataset loading first
    if args.test_dataset_only or not test_dataset_loading(args.dataset):
        if args.test_dataset_only:
            print("Dataset test completed.")
        else:
            print("Please fix dataset loading issues before proceeding.")
        return
    
    # Initialize analyzer
    analyzer = AnisotropyAnalyzer(args.checkpoint, device=args.device)
    
    # Load backbone
    backbone = analyzer.load_backbone()
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    dataset = MedMNISTDataset(args.dataset, split='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Extract features
    features, labels = analyzer.extract_features(backbone, dataloader, args.max_samples)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_anisotropy_check()
    
    # Save results
    import json
    output_file = f"anisotropy_analysis_{args.dataset}.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_serializable[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                results_serializable[key] = convert_numpy(value)
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to {output_file}")

if __name__ == '__main__':
    main()
