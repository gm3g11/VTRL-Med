#!/usr/bin/env python3
"""
Basic topological feature analysis for DermaMNIST
Tests discriminative power of simple topological features
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage import filters, measure, morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import medmnist
from medmnist import INFO

def load_dermamnist():
    """Load DermaMNIST dataset"""
    info = INFO['dermamnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', download=True, transform=None, size=224)
    test_dataset = DataClass(split='test', download=True, transform=None, size=224)
    
    # Extract images and labels
    train_images = np.array([np.array(train_dataset[i][0]) for i in range(len(train_dataset))])
    train_labels = np.array([train_dataset[i][1].item() for i in range(len(train_dataset))])
    
    test_images = np.array([np.array(test_dataset[i][0]) for i in range(len(test_dataset))])
    test_labels = np.array([test_dataset[i][1].item() for i in range(len(test_dataset))])
    
    print(f"Train: {train_images.shape}, Labels: {len(np.unique(train_labels))} classes")
    print(f"Test: {test_images.shape}, Labels: {len(np.unique(test_labels))} classes")
    
    return train_images, train_labels, test_images, test_labels

def simple_roi_extraction(img, threshold_percentile=90):
    """Simple ROI extraction using edge detection"""
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img.copy()
    
    # Edge detection
    edges = filters.sobel(gray)
    
    # Threshold to get binary mask
    threshold = np.percentile(edges, threshold_percentile)
    roi = edges > threshold
    
    # Clean up with morphology
    roi = morphology.binary_closing(roi, morphology.disk(2))
    roi = morphology.binary_opening(roi, morphology.disk(1))
    
    return roi.astype(bool)

def extract_basic_topological_features(img, roi=None):
    """Extract basic topological features from image"""
    # Convert to grayscale
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img.copy()
    
    # Apply ROI mask if provided
    if roi is not None:
        gray_masked = gray.copy()
        gray_masked[~roi] = gray[roi].mean()  # Set background to mean of ROI
        gray = gray_masked
    
    # Binarize image using multiple thresholds
    features = {}
    
    # Multiple threshold levels to capture different structures
    thresholds = [0.3, 0.5, 0.7]
    
    for i, thresh in enumerate(thresholds):
        # Create binary image
        binary = gray > (thresh * gray.max())
        
        # Connected components (H0 features)
        labeled_img = measure.label(binary)
        num_components = len(np.unique(labeled_img)) - 1  # Subtract background
        
        # Component sizes
        if num_components > 0:
            component_props = measure.regionprops(labeled_img)
            component_areas = [prop.area for prop in component_props]
            largest_component_ratio = max(component_areas) / (gray.shape[0] * gray.shape[1])
            area_std = np.std(component_areas) if len(component_areas) > 1 else 0
        else:
            largest_component_ratio = 0
            area_std = 0
        
        # Euler characteristic (components - holes)
        euler_char = num_components
        
        # Approximate hole count using morphological operations
        # Fill holes and compare
        filled = binary_fill_holes(binary)
        holes = filled.astype(int) - binary.astype(int)
        num_holes = measure.label(holes).max()
        
        # Boundary complexity
        if num_components > 0:
            boundaries = find_boundaries(binary, mode='thick')
            boundary_length = boundaries.sum()
            # Approximate circularity (4π * area / perimeter²)
            total_area = binary.sum()
            if boundary_length > 0:
                circularity = 4 * np.pi * total_area / (boundary_length ** 2)
            else:
                circularity = 0
        else:
            boundary_length = 0
            circularity = 0
        
        # Store features for this threshold
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
    
    # Global features across all thresholds
    features['total_boundary_complexity'] = sum([features[f't{i}_boundary_length'] for i in range(len(thresholds))])
    features['avg_components'] = np.mean([features[f't{i}_num_components'] for i in range(len(thresholds))])
    features['max_holes'] = max([features[f't{i}_num_holes'] for i in range(len(thresholds))])
    
    return features

def analyze_topological_discriminability(train_images, train_labels, test_images, test_labels):
    """Analyze discriminative power of topological features"""
    print("Extracting topological features...")
    
    # Extract ROIs for all images
    print("Computing ROIs...")
    train_rois = [simple_roi_extraction(img) for img in tqdm(train_images, desc="Train ROIs")]
    test_rois = [simple_roi_extraction(img) for img in tqdm(test_images, desc="Test ROIs")]
    
    # Extract topological features
    print("Extracting topological features...")
    train_features = []
    for img, roi in tqdm(zip(train_images, train_rois), desc="Train features", total=len(train_images)):
        features = extract_basic_topological_features(img, roi)
        train_features.append(list(features.values()))
    
    test_features = []
    for img, roi in tqdm(zip(test_images, test_rois), desc="Test features", total=len(test_images)):
        features = extract_basic_topological_features(img, roi)
        test_features.append(list(features.values()))
    
    # Convert to numpy arrays
    X_train = np.array(train_features)
    X_test = np.array(test_features)
    
    print(f"Feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")
    
    # Handle NaN/Inf values
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test with multiple classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, train_labels)
        
        train_pred = clf.predict(X_train_scaled)
        test_pred = clf.predict(X_test_scaled)
        
        train_acc = accuracy_score(train_labels, train_pred)
        test_acc = accuracy_score(test_labels, test_pred)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"{name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Feature importance analysis (using Random Forest)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, train_labels)
    
    # Get feature names
    feature_names = []
    thresholds = [0.3, 0.5, 0.7]
    base_features = ['num_components', 'euler_char', 'num_holes', 'largest_comp_ratio', 
                    'area_std', 'boundary_length', 'circularity']
    
    for i in range(len(thresholds)):
        for feat in base_features:
            feature_names.append(f't{i}_{feat}')
    
    feature_names.extend(['total_boundary_complexity', 'avg_components', 'max_holes'])
    
    # Plot feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance of Basic Topological Features")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 most important features:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:<25} ({importances[idx]:.4f})")
    
    # Class-wise analysis
    print(f"\nClass distribution:")
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples")
    
    return results, X_train_scaled, X_test_scaled, train_labels, test_labels

def main():
    print("Basic Topological Feature Analysis for DermaMNIST")
    print("=" * 50)
    
    # Load data
    train_images, train_labels, test_images, test_labels = load_dermamnist()
    
    # Analyze discriminative power
    results, X_train, X_test, y_train, y_test = analyze_topological_discriminability(
        train_images, train_labels, test_images, test_labels
    )
    
    print("\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)
    
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print()
    
    # Compare with random baseline
    random_acc = 1.0 / len(np.unique(y_train))
    print(f"Random baseline: {random_acc:.4f}")
    
    best_acc = max([results[name]['test_accuracy'] for name in results])
    improvement = best_acc - random_acc
    print(f"Best improvement over random: {improvement:.4f}")
    
    if best_acc > 0.4:  # Reasonable threshold for medical data
        print("\n✅ CONCLUSION: Basic topological features show discriminative power")
        print("   This suggests more sophisticated topology integration could be beneficial")
    else:
        print("\n❌ CONCLUSION: Limited discriminative power from basic topological features")
        print("   Topology integration may not provide substantial benefits")

if __name__ == "__main__":
    main()
