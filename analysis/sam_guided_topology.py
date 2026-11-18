#!/usr/bin/env python3
"""
SAM-Guided Topology Feature Extraction
- Uses fine-tuned SAM to identify better ROIs/boundaries
- Computes proper persistent homology with H0/H1 separation
- Multi-scale analysis like phase2_v2.py
"""

import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import warnings

warnings.filterwarnings("ignore")

# Import required libraries
try:
    import cripser
    print("✓ cripser available for persistent homology")
except ImportError:
    print("❌ cripser not found. Install with: pip install cripser")
    exit(1)

try:
    from segment_anything import sam_model_registry
    print("✓ SAM available")
except ImportError:
    print("❌ SAM not found. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)

from skimage import filters, measure, morphology
from skimage.segmentation import find_boundaries
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter

class SAMBoundaryExtractor(nn.Module):
    """Use fine-tuned SAM to extract better ROIs and boundaries"""
    def __init__(self, stage1_checkpoint_path, sam_model_type='vit_h'):
        super().__init__()
        # Load SAM backbone
        sam_model = sam_model_registry[sam_model_type](checkpoint=None)
        self.backbone = sam_model.image_encoder
        
        # Load fine-tuned weights
        print(f"Loading fine-tuned SAM for boundary extraction: {stage1_checkpoint_path}")
        self.backbone.load_state_dict(torch.load(stage1_checkpoint_path, map_location='cpu'))
        
        # Freeze for feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Feature dimensions
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        
    def extract_boundary_features(self, x):
        """Extract spatial features for boundary detection"""
        with torch.no_grad():
            # Patch embedding
            features = self.backbone.patch_embed(x)
            
            # Add positional encoding
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
            
            # Return spatial features (B, H, W, D) - preserve spatial structure
            return features

def sam_guided_roi_extraction(image, sam_extractor, device, 
                              edge_threshold_percentile=85, coverage_target=0.25):
    """Extract ROI using SAM boundary detection capabilities"""
    # Preprocess image for SAM
    if isinstance(image, np.ndarray):
        # Convert to tensor and normalize
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if image.dtype == np.uint8:
            from PIL import Image
            image_pil = Image.fromarray(image)
            image_tensor = transform(image_pil).unsqueeze(0).to(device)
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    else:
        image_tensor = image.to(device)
    
    # Extract SAM spatial features
    sam_features = sam_extractor.extract_boundary_features(image_tensor)  # (1, H, W, D)
    
    # Convert to boundary map using multiple strategies
    sam_features_np = sam_features.squeeze(0).cpu().numpy()  # (H, W, D)
    
    # Strategy 1: L2 norm of features (magnitude indicates importance)
    boundary_map1 = np.linalg.norm(sam_features_np, axis=2)
    
    # Strategy 2: Feature variance (high variance indicates boundaries)
    boundary_map2 = np.var(sam_features_np, axis=2)
    
    # Strategy 3: Local gradient magnitude in feature space
    grad_x = np.gradient(sam_features_np, axis=1)
    grad_y = np.gradient(sam_features_np, axis=0)
    boundary_map3 = np.sqrt(np.sum(grad_x**2, axis=2) + np.sum(grad_y**2, axis=2))
    
    # Combine boundary maps
    boundary_combined = (boundary_map1 + boundary_map2 + boundary_map3) / 3.0
    
    # Resize to original image size
    original_h, original_w = image.shape[:2] if isinstance(image, np.ndarray) else (224, 224)
    from skimage.transform import resize
    boundary_resized = resize(boundary_combined, (original_h, original_w), 
                            mode='constant', preserve_range=True, anti_aliasing=True)
    
    # Threshold to get ROI
    threshold = np.percentile(boundary_resized, edge_threshold_percentile)
    roi_initial = boundary_resized > threshold
    
    # Morphological cleanup
    roi_cleaned = morphology.binary_closing(roi_initial, morphology.disk(3))
    roi_cleaned = morphology.binary_opening(roi_cleaned, morphology.disk(2))
    
    # Ensure reasonable coverage
    current_coverage = roi_cleaned.sum() / roi_cleaned.size
    
    # Adjust threshold if coverage is too low/high
    if current_coverage < 0.1:  # Too little ROI
        threshold = np.percentile(boundary_resized, 70)
        roi_cleaned = boundary_resized > threshold
        roi_cleaned = morphology.binary_closing(roi_cleaned, morphology.disk(3))
    elif current_coverage > 0.5:  # Too much ROI
        threshold = np.percentile(boundary_resized, 92)
        roi_cleaned = boundary_resized > threshold
        roi_cleaned = morphology.binary_opening(roi_cleaned, morphology.disk(1))
    
    return roi_cleaned.astype(bool)

def compute_sam_guided_persistence(img, roi, n_h0=48, n_h1=48):
    """Compute persistent homology with SAM-guided preprocessing"""
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img.copy()
    
    # Apply ROI masking - set background to median of ROI
    if roi.any():
        median_val = np.median(gray[roi])
        gray[~roi] = median_val
    
    # Multi-scale computation (like phase2_v2.py)
    scales = [1.0, 0.75, 0.5]  # Slightly different scales for SAM-guided approach
    all_h0, all_h1 = [], []
    
    for scale in scales:
        if scale != 1.0:
            gray_scaled = rescale(gray, scale, anti_aliasing=True, 
                                preserve_range=True, channel_axis=None)
        else:
            gray_scaled = gray
        
        # Normalize to [0, 255]
        gray_norm = ((gray_scaled - gray_scaled.min()) / 
                     (gray_scaled.max() - gray_scaled.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply slight Gaussian smoothing to reduce noise
        gray_smooth = gaussian_filter(gray_norm.astype(np.float32), sigma=0.5)
        gray_smooth = np.clip(gray_smooth, 0, 255).astype(np.uint8)
        
        try:
            # Compute persistent homology
            ph = cripser.computePH(gray_smooth)
            
            for item in ph:
                if len(item) >= 3:
                    dim, birth, death = int(item[0]), float(item[1]), float(item[2])
                    
                    # Skip infinite bars
                    if death >= 1e5:
                        continue
                    
                    # Normalize
                    birth, death = birth/255.0, death/255.0
                    persistence = death - birth
                    
                    # Filter by minimum persistence threshold
                    min_persistence = 0.01 if scale == 1.0 else 0.005
                    
                    if dim == 0 and persistence > min_persistence:
                        # H0: connected components
                        all_h0.append((birth, death, persistence, scale))
                    elif dim == 1 and persistence > min_persistence:
                        # H1: loops/holes
                        all_h1.append((birth, death, persistence, scale))
        
        except Exception as e:
            print(f"Warning: PH computation failed for scale {scale}: {e}")
            continue
    
    # Sort by persistence (most persistent first)
    all_h0.sort(key=lambda x: x[2], reverse=True)
    all_h1.sort(key=lambda x: x[2], reverse=True)
    
    # Convert to fixed-size arrays
    def create_persistence_features(points, k):
        features = []
        for i in range(k):
            if i < len(points):
                birth, death, persistence, scale = points[i]
                features.extend([birth, death, persistence])
            else:
                features.extend([0.0, 0.0, 0.0])  # Padding
        return features
    
    h0_features = create_persistence_features(all_h0, n_h0)  # n_h0 * 3 features
    h1_features = create_persistence_features(all_h1, n_h1)  # n_h1 * 3 features
    
    return np.array(h0_features + h1_features, dtype=np.float32)

class SAMGuidedDataset(Dataset):
    """Dataset for extracting SAM-guided topology features"""
    def __init__(self, dataset_name, split='train'):
        import medmnist
        from medmnist import INFO
        
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(split=split, download=True, transform=None, size=224)
        
        print(f"SAM-guided dataset loaded: {len(self.dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert to numpy array
        if hasattr(image, 'numpy'):
            image_np = image.numpy()
        else:
            image_np = np.array(image)
        
        # Ensure 3 channels
        if image_np.ndim == 2:
            image_np = np.stack([image_np, image_np, image_np], axis=2)
        elif image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
        
        return image_np, label.item(), idx

def extract_sam_guided_topology_features(dataset_name, stage1_checkpoint_path, 
                                       output_dir, sam_model_type='vit_h', 
                                       n_h0=48, n_h1=48, batch_size=32):
    """Extract SAM-guided topology features for entire dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize SAM boundary extractor
    sam_extractor = SAMBoundaryExtractor(stage1_checkpoint_path, sam_model_type).to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        print(f"\n=== Processing {split} set ===")
        
        # Load dataset
        dataset = SAMGuidedDataset(dataset_name, split)
        
        # Extract features
        all_topology_features = []
        all_labels = []
        
        print("Extracting SAM-guided topology features...")
        for i in tqdm(range(len(dataset)), desc=f"Processing {split}"):
            image_np, label, idx = dataset[i]
            
            try:
                # Extract SAM-guided ROI
                roi = sam_guided_roi_extraction(image_np, sam_extractor, device)
                
                # Compute persistent homology
                topo_features = compute_sam_guided_persistence(image_np, roi, n_h0, n_h1)
                
                all_topology_features.append(topo_features)
                all_labels.append(label)
                
            except Exception as e:
                print(f"Warning: Failed to process sample {i}: {e}")
                # Use zero features as fallback
                fallback_features = np.zeros(3 * (n_h0 + n_h1), dtype=np.float32)
                all_topology_features.append(fallback_features)
                all_labels.append(label)
        
        # Convert to arrays
        topology_features = np.array(all_topology_features)
        labels = np.array(all_labels)
        
        print(f"SAM-guided topology features shape: {topology_features.shape}")
        
        # Handle NaN/Inf values
        topology_features = np.nan_to_num(topology_features, nan=0, posinf=0, neginf=0)
        
        # Save features
        features_path = os.path.join(output_dir, f'sam_topology_features_{dataset_name}_{split}.npy')
        labels_path = os.path.join(output_dir, f'sam_topology_labels_{dataset_name}_{split}.npy')
        
        np.save(features_path, topology_features)
        np.save(labels_path, labels)
        
        print(f"Saved SAM-guided topology features: {features_path}")
        print(f"Feature statistics: mean={topology_features.mean():.4f}, std={topology_features.std():.4f}")
    
    total_dims = 3 * (n_h0 + n_h1)
    print(f"\nSAM-guided topology extraction complete!")
    print(f"Feature dimensions: {total_dims} (H0: {n_h0*3}, H1: {n_h1*3})")
    print(f"Features saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='SAM-Guided Topology Feature Extraction')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['dermamnist', 'bloodmnist', 'pathmnist', 'octmnist', 'pneumoniamnist',
                               'retinamnist', 'breastmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist'])
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 SAM checkpoint')
    parser.add_argument('--output_dir', type=str, default='./sam_topology_features')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--n_h0', type=int, default=48, help='Number of H0 persistence points')
    parser.add_argument('--n_h1', type=int, default=48, help='Number of H1 persistence points')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    print("="*70)
    print("SAM-GUIDED PERSISTENT HOMOLOGY FEATURE EXTRACTION")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"SAM checkpoint: {args.stage1_checkpoint}")
    print(f"H0 features: {args.n_h0}, H1 features: {args.n_h1}")
    print(f"Total dimensions: {3 * (args.n_h0 + args.n_h1)}")
    
    # Extract features
    extract_sam_guided_topology_features(
        dataset_name=args.dataset,
        stage1_checkpoint_path=args.stage1_checkpoint,
        output_dir=args.output_dir,
        sam_model_type=args.sam_model_type,
        n_h0=args.n_h0,
        n_h1=args.n_h1,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
