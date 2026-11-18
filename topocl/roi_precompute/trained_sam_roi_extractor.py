#!/usr/bin/env python3
"""
Trained SAM ROI Extractor - Uses your Stage 1 fine-tuned SAM weights
"""

import os
import warnings
import numpy as np
import torch
from PIL import Image
from collections import OrderedDict
from skimage import measure
from skimage.morphology import binary_dilation, binary_erosion, disk
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Suppress FutureWarning from torch.load in SAM library
warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.load.*weights_only.*')


def load_trained_sam(trained_checkpoint_path, base_sam_path, model_type='vit_h', device='cuda'):
    """
    Load SAM with trained image encoder weights from Stage 1
    
    Args:
        trained_checkpoint_path: Path to stage1_sam_simclr_dermamnist_224.pth
        base_sam_path: Path to base sam_vit_h_4b8939.pth
        model_type: SAM model type (vit_h, vit_l, vit_b)
        device: Device to load model on
    
    Returns:
        SAM model with trained encoder
    """
    print(f"Loading trained SAM from: {trained_checkpoint_path}")
    
    # Step 1: Load base SAM model structure
    print("  1. Loading base SAM structure...")
    sam = sam_model_registry[model_type](checkpoint=base_sam_path)
    
    # Step 2: Load trained encoder weights
    print("  2. Loading trained encoder weights...")
    trained_state_dict = torch.load(trained_checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"     Found {len(trained_state_dict)} trained parameters")
    
    # Step 3: Add 'image_encoder.' prefix to match SAM's structure
    print("  3. Adding 'image_encoder.' prefix...")
    encoder_state_dict = OrderedDict()
    for key, value in trained_state_dict.items():
        new_key = f'image_encoder.{key}'
        encoder_state_dict[new_key] = value
    
    # Step 4: Load into SAM (only encoder weights, keep mask_decoder and prompt_encoder from base)
    print("  4. Loading weights into SAM...")
    missing_keys, unexpected_keys = sam.load_state_dict(encoder_state_dict, strict=False)
    
    # Verify loading
    expected_missing = ['mask_decoder', 'prompt_encoder']  # These should be missing (from base model)
    actual_missing_prefixes = set([k.split('.')[0] for k in missing_keys])
    
    print(f"\n  📊 Loading Results:")
    print(f"     - Loaded: {len(encoder_state_dict)} image_encoder parameters")
    print(f"     - Missing: {len(missing_keys)} parameters (expected: mask_decoder + prompt_encoder)")
    print(f"     - Unexpected: {len(unexpected_keys)} parameters")
    
    if len(unexpected_keys) > 0:
        print(f"\n  ⚠️  Warning: {len(unexpected_keys)} unexpected keys!")
        for key in list(unexpected_keys)[:5]:
            print(f"       - {key}")
    
    # Move to device and set to eval mode
    sam.to(device)
    sam.eval()
    
    print(f"\n  ✅ Trained SAM loaded successfully on {device}!")
    return sam


class TrainedSAMROIExtractor:
    """
    ROI extraction using Stage 1 fine-tuned SAM
    This should give much better results than pretrained SAM for medical images!
    """
    
    def __init__(self, trained_checkpoint_path, base_sam_path, model_type='vit_h', device='cuda', dataset_name='DermaMNIST'):
        """
        Args:
            trained_checkpoint_path: Path to stage1_sam_simclr_dermamnist_224.pth
            base_sam_path: Path to sam_vit_h_4b8939.pth
            model_type: 'vit_h', 'vit_l', or 'vit_b'
            device: 'cuda' or 'cpu'
            dataset_name: Dataset name for parameter tuning
        """
        self.device = device
        self.dataset_name = dataset_name
        
        # Load trained SAM
        sam = load_trained_sam(trained_checkpoint_path, base_sam_path, model_type, device)
        
        # Dataset-specific parameters
        # PathMNIST (histopathology) needs more lenient settings
        if 'path' in dataset_name.lower():
            print("  Using relaxed parameters for PathMNIST (histopathology)")
            points_per_side = 32
            pred_iou_thresh = 0.80  # Lower threshold
            stability_score_thresh = 0.85  # Lower threshold
            min_mask_region_area = 150  # Smaller regions OK
        else:
            # Default for dermatology, organs, etc.
            points_per_side = 32
            pred_iou_thresh = 0.88
            stability_score_thresh = 0.92
            min_mask_region_area = 200
        
        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
        )
        
        print("Trained SAM ROI extractor ready!\n")
    
    def _score_mask(self, mask_data, img_area):
        """Score mask for medical image relevance"""
        mask = mask_data['segmentation']
        area = mask.sum()
        coverage = area / img_area
        
        # Dataset-specific coverage thresholds
        if 'path' in self.dataset_name.lower():
            # PathMNIST: histopathology can have varied coverage
            min_cov, max_cov = 0.05, 0.75  # More permissive
            ideal_min, ideal_max = 0.15, 0.55
        else:
            # Default for dermatology, organs, etc.
            min_cov, max_cov = 0.08, 0.65
            ideal_min, ideal_max = 0.20, 0.45
        
        # Filter out bad masks
        if coverage < min_cov:  # Too small
            return 0.0
        if coverage > max_cov:  # Too large
            return 0.0
        
        # Prefer moderate coverage
        if ideal_min <= coverage <= ideal_max:
            coverage_score = 1.0
        elif coverage < ideal_min:
            coverage_score = coverage / ideal_min
        else:
            coverage_score = max(0, 1.0 - (coverage - ideal_max) / (max_cov - ideal_max))
        
        # Combine with SAM's quality scores
        iou_score = mask_data['predicted_iou']
        stability_score = mask_data['stability_score']
        
        # Weighted combination
        final_score = (
            0.4 * coverage_score +
            0.3 * iou_score +
            0.3 * stability_score
        )
        
        return final_score
    
    def _refine_roi(self, roi):
        """Post-process ROI to clean up artifacts"""
        # Remove small components
        labeled = measure.label(roi)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            return roi
        
        # Keep significant regions (>3% of image)
        img_area = roi.size
        min_area = 0.03 * img_area
        
        roi_refined = np.zeros_like(roi)
        for region in regions:
            if region.area >= min_area:
                roi_refined[labeled == region.label] = True
        
        # Smooth boundaries
        if roi_refined.sum() > 0:
            roi_refined = binary_erosion(roi_refined, disk(1))
            roi_refined = binary_dilation(roi_refined, disk(2))
        
        return roi_refined
    
    def _fallback_roi(self, img_shape):
        """Fallback if SAM fails"""
        roi = np.zeros(img_shape[:2], dtype=bool)
        h, w = img_shape[:2]
        center_h, center_w = h // 2, w // 2
        size_h, size_w = h // 3, w // 3
        roi[center_h - size_h:center_h + size_h,
            center_w - size_w:center_w + size_w] = True
        return roi
    
    def extract(self, img):
        """
        Extract ROI using trained SAM
        
        Args:
            img: RGB or grayscale image array (H, W, 3) or (H, W) with values 0-255
            
        Returns:
            roi: Binary mask (H, W) indicating region of interest
        """
        # Ensure RGB format for SAM
        if img.ndim == 2:
            # Pure grayscale (H, W) -> (H, W, 3)
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            # Grayscale with channel (H, W, 1) -> (H, W, 3)
            img = np.repeat(img, 3, axis=-1)
        
        # Ensure contiguous array for SAM
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # Generate masks using trained SAM
        try:
            masks = self.mask_generator.generate(img)
        except Exception as e:
            # Uncomment for debugging:
            # print(f"SAM failed: {e} | img shape: {img.shape}, dtype: {img.dtype}, contiguous: {img.flags['C_CONTIGUOUS']}")
            print(f"SAM failed: {e}, using fallback")
            return self._fallback_roi(img.shape)
        
        if len(masks) == 0:
            return self._fallback_roi(img.shape)
        
        # Score and filter masks
        img_area = img.shape[0] * img.shape[1]
        scored_masks = []
        for mask_data in masks:
            score = self._score_mask(mask_data, img_area)
            if score > 0.0:
                scored_masks.append((score, mask_data))
        
        if len(scored_masks) == 0:
            return self._fallback_roi(img.shape)
        
        # Sort by score and take best
        scored_masks.sort(key=lambda x: x[0], reverse=True)
        combined_roi = scored_masks[0][1]['segmentation'].copy()
        
        # Optionally combine with second-best if complementary
        if len(scored_masks) > 1:
            second_mask = scored_masks[1][1]['segmentation']
            overlap = (combined_roi & second_mask).sum()
            second_area = second_mask.sum()
            
            # Add if low overlap (< 30%)
            if overlap < 0.3 * second_area:
                combined_roi = combined_roi | second_mask
        
        # Refine ROI
        combined_roi = self._refine_roi(combined_roi)
        
        # Final sanity check - dataset-specific
        final_coverage = combined_roi.sum() / img_area
        if 'path' in self.dataset_name.lower():
            # PathMNIST: more lenient
            if final_coverage < 0.03 or final_coverage > 0.80:
                return self._fallback_roi(img.shape)
        else:
            # Default
            if final_coverage < 0.05 or final_coverage > 0.70:
                return self._fallback_roi(img.shape)
        
        return combined_roi


def load_external_dataset(dataset_name, data_dir=None, n_samples=None):
    """
    Load ISIC2019 or Kvasir datasets from filesystem
    
    Args:
        dataset_name: 'ISIC2019' or 'Kvasir'
        data_dir: Root directory containing the dataset
        n_samples: Number of samples to load (None = all, for testing use small number)
        
    Returns:
        imgs: Array of images (N, H, W, 3)
        labels: Array of labels (N,)
    """
    import os
    from PIL import Image
    from torchvision import datasets
    
    dataset_lower = dataset_name.lower()
    
    # Default data paths
    default_paths = {
        'isic2019': 'data/isic2019',
        'kvasir': 'data/kvasir-dataset'
    }
    
    # Try default path first
    if data_dir is None:
        if dataset_lower in default_paths:
            data_dir = os.path.expanduser(default_paths[dataset_lower])
            if os.path.exists(data_dir):
                print(f"Using default path: {data_dir}")
            else:
                raise FileNotFoundError(f"Default path not found: {data_dir}")
        else:
            raise ValueError(f"No default path for {dataset_name}")
    else:
        data_dir = os.path.expanduser(data_dir)
    
    # Load all images from dataset (no train/test split - will be done in Stage 2)
    if dataset_name == 'ISIC2019':
        # Try train/ and test/ structure first
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_ds = datasets.ImageFolder(train_dir)
            test_ds = datasets.ImageFolder(test_dir)
            
            # Combine ALL images (no splitting)
            all_paths = [s[0] for s in train_ds.samples] + [s[0] for s in test_ds.samples]
            all_labels = train_ds.targets + test_ds.targets
        else:
            # Single folder with class subfolders
            full_ds = datasets.ImageFolder(data_dir)
            all_paths = [s[0] for s in full_ds.samples]
            all_labels = full_ds.targets
    
    elif dataset_name == 'Kvasir':
        # Single folder with class subfolders
        full_ds = datasets.ImageFolder(data_dir)
        all_paths = [s[0] for s in full_ds.samples]
        all_labels = full_ds.targets
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Found {len(all_paths)} total images")
    
    # For testing, sample n_samples; for production, use all
    if n_samples is not None and n_samples < len(all_paths):
        indices = np.linspace(0, len(all_paths)-1, n_samples, dtype=int)
        paths = [all_paths[i] for i in indices]
        labels = [all_labels[i] for i in indices]
        print(f"Testing with {n_samples} sampled images")
    else:
        paths = all_paths
        labels = all_labels
        print(f"Loading all {len(paths)} images")
    
    # Load images
    imgs = []
    final_labels = []
    
    print("Loading images...")
    from tqdm import tqdm
    for img_path, label in tqdm(zip(paths, labels)):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)
            img = np.array(img, dtype=np.uint8)
            imgs.append(img)
            final_labels.append(label)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue
    
    imgs = np.array(imgs)
    final_labels = np.array(final_labels)
    
    print(f"Successfully loaded {len(imgs)} images")
    
    return imgs, final_labels


# Test the extractor
if __name__ == "__main__":
    import medmnist
    from tqdm import tqdm
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Trained SAM ROI Extractor")
    parser.add_argument('--dataset', type=str, default='DermaMNIST',
                       choices=['DermaMNIST', 'OCTMNIST', 'OrganSMNIST', 'TissueMNIST', 
                               'PathMNIST', 'BloodMNIST', 'BreastMNIST', 'RetinaMNIST',
                               'PneumoniaMNIST', 'OrganAMNIST', 'OrganCMNIST',
                               'ISIC2019', 'Kvasir'])
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'val', 'test'],
                       help="Dataset split (MedMNIST only; ISIC2019/Kvasir load all images)")
    parser.add_argument('--n_samples', type=int, default=999999, 
                       help="Number of samples to process (default: all images)")
    parser.add_argument('--data_dir', type=str, default=None, 
                       help="Data directory for ISIC2019/Kvasir (optional, defaults: data/isic2019, data/kvasir-dataset)")
    args = parser.parse_args()
    
    dataset_lower = args.dataset.lower()
    
    # Detect dataset type and set checkpoint paths
    is_medmnist = args.dataset not in ['ISIC2019', 'Kvasir']
    
    if is_medmnist:
        # MedMNIST datasets
        trained_checkpoint = f'checkpoints/sam_spice/stage1/{dataset_lower}/stage1_sam_simclr_{dataset_lower}_224.pth'
    else:
        # ISIC2019 and Kvasir
        trained_checkpoint = f'checkpoints/sam_spice_kvasir_isic/stage1/{dataset_lower}/stage1_sam_simclr_{dataset_lower}_224.pth'
    
    base_sam = 'weights/sam_vit_h_4b8939.pth'
    
    print("="*80)
    print(f"Testing Trained SAM ROI Extractor - {args.dataset}")
    print("="*80 + "\n")
    
    # Load data based on dataset type
    if is_medmnist:
        # MedMNIST datasets
        print(f"Loading {args.dataset} {args.split} split from medmnist library...")
        from medmnist import INFO
        info = INFO[dataset_lower]
        DataClass = getattr(medmnist, info['python_class'])
        
        # Try as_rgb=True first
        try:
            ds = DataClass(split=args.split, download=True, size=224, as_rgb=True)
        except TypeError:
            # Fallback for older medmnist versions without as_rgb parameter
            print("  Note: as_rgb parameter not supported, will convert manually")
            ds = DataClass(split=args.split, download=True, size=224)
        
        imgs = ds.imgs
        labels = ds.labels
    else:
        # ISIC2019 or Kvasir - load from directory
        print(f"Loading {args.dataset} from filesystem...")
        imgs, labels = load_external_dataset(args.dataset, args.data_dir, args.n_samples)
    
    
    # Ensure uint8 format
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8)
    
    # Debug: Show shape info
    print(f"Loaded {len(imgs)} images")
    print(f"Original images shape: {imgs.shape}")
    print(f"Images dtype: {imgs.dtype}")
    
    # Handle grayscale conversion (simple version that works for most datasets)
    if imgs.ndim == 3:
        # (N, H, W) - grayscale without channel dimension
        print("  Converting grayscale to RGB...")
        imgs = np.stack([imgs, imgs, imgs], axis=-1)
    elif imgs.ndim == 4 and imgs.shape[-1] == 1:
        # (N, H, W, 1) - grayscale with channel dimension  
        print("  Converting grayscale to RGB...")
        imgs = np.repeat(imgs, 3, axis=-1)
    
    print(f"Final images shape: {imgs.shape}")
    print()
    
    # Initialize extractor with dataset-specific parameters
    extractor = TrainedSAMROIExtractor(
        trained_checkpoint_path=trained_checkpoint,
        base_sam_path=base_sam,
        model_type='vit_h',
        device='cuda',
        dataset_name=args.dataset
    )
    
    # Test on images (all if n_samples >= dataset size)
    if args.n_samples >= len(imgs):
        n_test = len(imgs)
        print(f"Extracting ROIs from all {n_test} images...")
    else:
        n_test = args.n_samples
        print(f"Extracting ROIs from {n_test} sample images...")
    
    rois = []
    for i in tqdm(range(n_test)):
        roi = extractor.extract(imgs[i])
        rois.append(roi)
    
    # Compute statistics
    coverage = np.array([roi.sum() / roi.size for roi in rois]) * 100
    
    print("\n" + "="*80)
    print(f"ROI Coverage Statistics - {args.dataset}:")
    print("="*80)
    print(f"  Mean:   {coverage.mean():.2f}%")
    print(f"  Std:    {coverage.std():.2f}%")
    print(f"  Median: {np.median(coverage):.2f}%")
    print(f"  Min:    {coverage.min():.2f}%")
    print(f"  Max:    {coverage.max():.2f}%")
    
    print("\nCoverage distribution:")
    bins = [0, 10, 20, 30, 40, 50, 60, 100]
    hist, _ = np.histogram(coverage, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]:3d}-{bins[i+1]:3d}%: {hist[i]:3d} images ({hist[i]/len(rois)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("Target: 25-40% mean coverage with <15% std")
    print("="*80)
    
    # Evaluate if good
    if 25 <= coverage.mean() <= 40 and coverage.std() < 15:
        print("✅ Excellent! ROI quality looks good for topology features.")
    elif 20 <= coverage.mean() <= 50 and coverage.std() < 20:
        print("✅ Good! Should work well for Stage 2 training.")
    else:
        print("⚠️  May need parameter tuning - see coverage distribution above.")
