#!/usr/bin/env python3
"""
Topological feature analysis using fine-tuned SAM features
Compares raw image topology vs SAM feature topology
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import filters, measure, morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
from PIL import Image
import medmnist
from medmnist import INFO

# Import SAM
try:
    from segment_anything import sam_model_registry
except ImportError:
    print("Warning: segment_anything not found. Please install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)

class SAMFeatureExtractor(nn.Module):
    """Extract spatial features from fine-tuned SAM"""
    def __init__(self, checkpoint_path, sam_model_type='vit_h'):
        super().__init__()
        # Load SAM backbone
        sam_model = sam_model_registry[sam_model_type](checkpoint=None)
        self.backbone = sam_model.image_encoder
        
        # Load fine-tuned weights
        print(f"Loading fine-tuned SAM from: {checkpoint_path}")
        self.backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        # Freeze for feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Feature dimensions
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        
    def forward(self, x):
        """Extract spatial features (before global pooling)"""
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
            
            # Return spatial features (B, H, W, D) - do NOT pool globally
            return features

class DermaMNISTDataset(Dataset):
    """Dataset wrapper for DermaMNIST"""
    def __init__(self, split='train'):
        info = INFO['dermamnist']
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(split=split, download=True, transform=None, size=224)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert PIL to numpy array and then to tensor
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        
        # Convert HWC to CHW
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Normalize with ImageNet stats (manual normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor, label.item()

def extract_sam_features(sam_extractor, dataset, device, batch_size=32):
    """Extract SAM features for all images"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    all_labels = []
    
    print("Extracting SAM features...")
    sam_extractor = sam_extractor.to(device)
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            # Extract features: (B, H, W, feature_dim)
            features = sam_extractor(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.tolist())
    
    # Concatenate all features
    features_array = np.concatenate(all_features, axis=0)  # (N, H, W, feature_dim)
    labels_array = np.array(all_labels)
    
    print(f"SAM features shape: {features_array.shape}")
    return features_array, labels_array

def sam_features_to_2d(features, method='l2_norm'):
    """Convert 3D SAM features (H, W, D) to 2D for topology analysis"""
    if method == 'l2_norm':
        # L2 norm across feature dimension
        return np.linalg.norm(features, axis=2)
    elif method == 'mean':
        # Mean across feature dimension
        return np.mean(features, axis=2)
    elif method == 'max':
        # Max across feature dimension
        return np.max(features, axis=2)
    elif method == 'pca':
        # First principal component
        H, W, D = features.shape
        features_flat = features.reshape(-1, D)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(features_flat)
        return pc1.reshape(H, W)
    else:
        raise ValueError(f"Unknown method: {method}")

def simple_roi_extraction(img, threshold_percentile=90):
    """Simple ROI extraction using edge detection"""
    # Edge detection
    edges = filters.sobel(img)
    
    # Threshold to get binary mask
    threshold = np.percentile(edges, threshold_percentile)
    roi = edges > threshold
    
    # Clean up with morphology
    roi = morphology.binary_closing(roi, morphology.disk(2))
    roi = morphology.binary_opening(roi, morphology.disk(1))
    
    return roi.astype(bool)

def extract_basic_topological_features(img_2d, roi=None):
    """Extract basic topological features from 2D image"""
    # Apply ROI mask if provided
    if roi is not None:
        img_masked = img_2d.copy()
        img_masked[~roi] = img_2d[roi].mean() if roi.any() else img_2d.mean()
        img_2d = img_masked
    
    # Normalize to [0, 1]
    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
    
    features = {}
    thresholds = [0.3, 0.5, 0.7]
    
    for i, thresh in enumerate(thresholds):
        # Create binary image
        binary = img_2d > thresh
        
        # Connected components
        labeled_img = measure.label(binary)
        num_components = len(np.unique(labeled_img)) - 1
        
        # Component properties
        if num_components > 0:
            component_props = measure.regionprops(labeled_img)
            component_areas = [prop.area for prop in component_props]
            largest_component_ratio = max(component_areas) / img_2d.size
            area_std = np.std(component_areas) if len(component_areas) > 1 else 0
        else:
            largest_component_ratio = 0
            area_std = 0
        
        # Euler characteristic
        euler_char = num_components
        
        # Hole count
        filled = binary_fill_holes(binary)
        holes = filled.astype(int) - binary.astype(int)
        num_holes = measure.label(holes).max()
        
        # Boundary features
        if num_components > 0:
            boundaries = find_boundaries(binary, mode='thick')
            boundary_length = boundaries.sum()
            total_area = binary.sum()
            if boundary_length > 0:
                circularity = 4 * np.pi * total_area / (boundary_length ** 2)
            else:
                circularity = 0
        else:
            boundary_length = 0
            circularity = 0
        
        # Store features
        prefix = f't{i}_'
        features.update({
            f'{prefix}num_components': num_components,
            f'{prefix}euler_char': euler_char,
            f'{prefix}num_holes': num_holes,
            f'{prefix}largest_comp_ratio': largest_component_ratio,
            f'{prefix}area_std': area_std,
            f'{prefix}boundary_length': boundary_length,
            f'{prefix}circularity': circularity,
        })
    
    # Global features
    features['total_boundary_complexity'] = sum([features[f't{i}_boundary_length'] for i in range(len(thresholds))])
    features['avg_components'] = np.mean([features[f't{i}_num_components'] for i in range(len(thresholds))])
    features['max_holes'] = max([features[f't{i}_num_holes'] for i in range(len(thresholds))])
    
    return features

def analyze_sam_topology_features(sam_checkpoint_path):
    """Analyze topological features extracted from SAM features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = DermaMNISTDataset('train')
    test_dataset = DermaMNISTDataset('test')
    
    # Load SAM feature extractor
    sam_extractor = SAMFeatureExtractor(sam_checkpoint_path, sam_model_type='vit_h')
    
    # Extract SAM features
    train_sam_features, train_labels = extract_sam_features(sam_extractor, train_dataset, device)
    test_sam_features, test_labels = extract_sam_features(sam_extractor, test_dataset, device)
    
    # Test different methods to convert SAM features to 2D
    methods = ['l2_norm', 'mean', 'pca']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing method: {method} ---")
        
        # Convert SAM features to 2D
        train_features_list = []
        test_features_list = []
        
        print("Converting SAM features to 2D and extracting topology features...")
        
        # Process training data
        for i in tqdm(range(len(train_sam_features)), desc=f"Train {method}"):
            sam_feat = train_sam_features[i]  # (H, W, D)
            img_2d = sam_features_to_2d(sam_feat, method)
            roi = simple_roi_extraction(img_2d)
            topo_features = extract_basic_topological_features(img_2d, roi)
            train_features_list.append(list(topo_features.values()))
        
        # Process test data
        for i in tqdm(range(len(test_sam_features)), desc=f"Test {method}"):
            sam_feat = test_sam_features[i]  # (H, W, D)
            img_2d = sam_features_to_2d(sam_feat, method)
            roi = simple_roi_extraction(img_2d)
            topo_features = extract_basic_topological_features(img_2d, roi)
            test_features_list.append(list(topo_features.values()))
        
        # Convert to arrays
        X_train = np.array(train_features_list)
        X_test = np.array(test_features_list)
        
        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        method_results = {}
        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, train_labels)
            test_pred = clf.predict(X_test_scaled)
            test_acc = accuracy_score(test_labels, test_pred)
            method_results[name] = test_acc
            print(f"{name} - Test Acc: {test_acc:.4f}")
        
        results[method] = method_results
    
    return results

def compare_raw_vs_sam():
    """Compare raw image topology vs SAM feature topology"""
    print("="*60)
    print("COMPARISON: Raw Image vs SAM Feature Topology")
    print("="*60)
    
    sam_checkpoint_path = "./checkpoints/stage1_sam_simclr_dermamnist_224.pth"
    
    try:
        # Test SAM feature topology
        sam_results = analyze_sam_topology_features(sam_checkpoint_path)
        
        # Print results
        print("\n" + "="*50)
        print("SAM FEATURE TOPOLOGY RESULTS")
        print("="*50)
        
        for method, method_results in sam_results.items():
            print(f"\nMethod: {method}")
            for clf_name, acc in method_results.items():
                print(f"  {clf_name}: {acc:.4f}")
        
        # Find best SAM result
        best_sam_acc = 0
        best_sam_method = ""
        best_sam_clf = ""
        
        for method, method_results in sam_results.items():
            for clf_name, acc in method_results.items():
                if acc > best_sam_acc:
                    best_sam_acc = acc
                    best_sam_method = method
                    best_sam_clf = clf_name
        
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(f"Raw Image Topology (from previous run): 0.6763")
        print(f"SAM Feature Topology (best): {best_sam_acc:.4f} ({best_sam_method}, {best_sam_clf})")
        
        improvement = best_sam_acc - 0.6763
        print(f"Difference: {improvement:+.4f}")
        
        if improvement > 0.01:
            print("✅ SAM features help topology extraction")
        elif improvement < -0.01:
            print("❌ SAM features hurt topology extraction")  
        else:
            print("➡️  SAM features have minimal impact on topology")
            
    except FileNotFoundError:
        print(f"❌ SAM checkpoint not found at: {sam_checkpoint_path}")
        print("Please ensure you have run Stage 1 training and the checkpoint exists.")
        return
    except Exception as e:
        print(f"❌ Error during SAM analysis: {e}")
        return

if __name__ == "__main__":
    compare_raw_vs_sam()
