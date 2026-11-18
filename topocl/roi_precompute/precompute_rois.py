#!/usr/bin/env python3
"""
Pre-compute and cache ROIs for entire MedMNIST dataset
This eliminates ROI extraction overhead during Stage 2 training!
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import medmnist
import argparse
from trained_sam_roi_extractor import load_trained_sam
from segment_anything import SamAutomaticMaskGenerator
from skimage import measure
from skimage.morphology import binary_dilation, binary_erosion, disk
import time


class FastTrainedSAMROIExtractor:
    """
    Optimized version with batch processing support
    """
    
    def __init__(self, trained_checkpoint_path, base_sam_path, model_type='vit_h', device='cuda'):
        self.device = device
        
        # Load trained SAM
        sam = load_trained_sam(trained_checkpoint_path, base_sam_path, model_type, device)
        
        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,
        )
        
        print("Fast ROI extractor ready!\n")
    
    def _score_mask(self, mask_data, img_area):
        """Score mask for quality"""
        mask = mask_data['segmentation']
        area = mask.sum()
        coverage = area / img_area
        
        if coverage < 0.08 or coverage > 0.65:
            return 0.0
        
        if 0.20 <= coverage <= 0.45:
            coverage_score = 1.0
        elif coverage < 0.20:
            coverage_score = coverage / 0.20
        else:
            coverage_score = max(0, 1.0 - (coverage - 0.45) / 0.20)
        
        iou_score = mask_data['predicted_iou']
        stability_score = mask_data['stability_score']
        
        final_score = (
            0.4 * coverage_score +
            0.3 * iou_score +
            0.3 * stability_score
        )
        
        return final_score
    
    def _refine_roi(self, roi):
        """Post-process ROI"""
        labeled = measure.label(roi)
        regions = measure.regionprops(labeled)
        
        if len(regions) == 0:
            return roi
        
        img_area = roi.size
        min_area = 0.03 * img_area
        
        roi_refined = np.zeros_like(roi)
        for region in regions:
            if region.area >= min_area:
                roi_refined[labeled == region.label] = True
        
        if roi_refined.sum() > 0:
            roi_refined = binary_erosion(roi_refined, disk(1))
            roi_refined = binary_dilation(roi_refined, disk(2))
        
        return roi_refined
    
    def _fallback_roi(self, img_shape):
        """Fallback ROI"""
        roi = np.zeros(img_shape[:2], dtype=bool)
        h, w = img_shape[:2]
        center_h, center_w = h // 2, w // 2
        size_h, size_w = h // 3, w // 3
        roi[center_h - size_h:center_h + size_h,
            center_w - size_w:center_w + size_w] = True
        return roi
    
    def extract(self, img):
        """Extract single ROI"""
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        try:
            masks = self.mask_generator.generate(img)
        except Exception:
            return self._fallback_roi(img.shape)
        
        if len(masks) == 0:
            return self._fallback_roi(img.shape)
        
        img_area = img.shape[0] * img.shape[1]
        scored_masks = []
        for mask_data in masks:
            score = self._score_mask(mask_data, img_area)
            if score > 0.0:
                scored_masks.append((score, mask_data))
        
        if len(scored_masks) == 0:
            return self._fallback_roi(img.shape)
        
        scored_masks.sort(key=lambda x: x[0], reverse=True)
        combined_roi = scored_masks[0][1]['segmentation'].copy()
        
        if len(scored_masks) > 1:
            second_mask = scored_masks[1][1]['segmentation']
            overlap = (combined_roi & second_mask).sum()
            second_area = second_mask.sum()
            if overlap < 0.3 * second_area:
                combined_roi = combined_roi | second_mask
        
        combined_roi = self._refine_roi(combined_roi)
        
        final_coverage = combined_roi.sum() / img_area
        if final_coverage < 0.05 or final_coverage > 0.70:
            return self._fallback_roi(img.shape)
        
        return combined_roi
    
    def extract_batch(self, imgs, show_progress=True):
        """
        Extract ROIs from multiple images
        
        Args:
            imgs: Array of images (N, H, W, C)
            show_progress: Show progress bar
            
        Returns:
            List of ROI masks
        """
        rois = []
        iterator = tqdm(imgs, desc="Extracting ROIs") if show_progress else imgs
        
        for img in iterator:
            roi = self.extract(img)
            rois.append(roi)
        
        return rois


def precompute_rois(dataset_name, splits=['train', 'val', 'test'], 
                    trained_checkpoint=None, base_sam=None,
                    output_dir='./roi_cache', force_recompute=False):
    """
    Pre-compute ROIs for all splits of a dataset and save to disk
    
    Args:
        dataset_name: MedMNIST dataset name
        splits: Which splits to process
        trained_checkpoint: Path to trained SAM checkpoint
        base_sam: Path to base SAM checkpoint
        output_dir: Where to save cached ROIs
        force_recompute: Recompute even if cache exists
    """
    
    dataset_lower = dataset_name.lower()
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print(f"Pre-computing ROIs for {dataset_name}")
    print("="*80)
    
    # Default paths
    if trained_checkpoint is None:
        trained_checkpoint = f'../SAM_SPICE_medmnist_isic2019_kvasir/SAM_SPICE_checkpoints/stage1/{dataset_lower}/stage1_sam_simclr_{dataset_lower}_224.pth'
    if base_sam is None:
        base_sam = '../SAM_SPICE_medmnist_isic2019_kvasir/sam_vit_h_4b8939.pth'
    
    # Check if already cached
    cache_file = os.path.join(output_dir, f'{dataset_lower}_rois.npz')
    if os.path.exists(cache_file) and not force_recompute:
        print(f"\n✅ ROI cache found at: {cache_file}")
        print("Loading to verify...")
        cached = np.load(cache_file, allow_pickle=False)
        print(f"  Cached splits: {list(cached.keys())}")
        for split in cached.keys():
            print(f"    {split}: {len(cached[split])} ROIs")
        print("\nUse --force to recompute")
        return cache_file
    
    # Load extractor
    print(f"\nInitializing ROI extractor...")
    extractor = FastTrainedSAMROIExtractor(trained_checkpoint, base_sam, 'vit_h', 'cuda')
    
    # Process each split
    all_rois = {}
    total_time = 0
    
    for split in splits:
        print(f"\n{'='*80}")
        print(f"Processing {split} split")
        print("="*80)
        
        # Load data
        print(f"Loading {dataset_name} {split} data...")
        DataClass = getattr(medmnist, dataset_name)
        ds = DataClass(split=split, download=True, size=224)
        imgs = ds.imgs
        
        # Prepare images
        if imgs.dtype != np.uint8:
            imgs = (imgs * 255).astype(np.uint8)
        if imgs.ndim == 3:
            imgs = imgs[..., None]
        if imgs.shape[-1] == 1:
            imgs = np.repeat(imgs, 3, axis=-1)
        
        print(f"Found {len(imgs)} images")
        
        # Extract ROIs
        start_time = time.time()
        rois = extractor.extract_batch(imgs, show_progress=True)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Convert to numpy array (save as uint8 for space efficiency)
        rois_array = np.array(rois, dtype=bool)
        all_rois[split] = rois_array
        
        # Statistics
        coverage = np.array([roi.sum() / roi.size for roi in rois]) * 100
        print(f"\nSplit: {split}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(imgs):.2f}s per image)")
        print(f"  Coverage: {coverage.mean():.2f}% ± {coverage.std():.2f}%")
        print(f"  Range: [{coverage.min():.2f}%, {coverage.max():.2f}%]")
    
    # Save to disk
    print(f"\n{'='*80}")
    print("Saving ROI cache...")
    print("="*80)
    
    np.savez_compressed(cache_file, **all_rois)
    
    file_size = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"✅ Saved to: {cache_file}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg: {total_time/sum(len(all_rois[s]) for s in all_rois):.2f}s per image")
    
    print(f"\n{'='*80}")
    print("Summary:")
    print("="*80)
    total_rois = sum(len(all_rois[s]) for s in all_rois)
    print(f"  Dataset: {dataset_name}")
    print(f"  Total ROIs: {total_rois}")
    for split in all_rois.keys():
        print(f"    {split}: {len(all_rois[split])} ROIs")
    print("="*80 + "\n")
    
    return cache_file


def load_cached_rois(dataset_name, split='train', cache_dir='./roi_cache'):
    """
    Load pre-computed ROIs from cache
    
    Args:
        dataset_name: MedMNIST dataset name
        split: Which split to load
        cache_dir: Cache directory
        
    Returns:
        Array of ROI masks (N, H, W)
    """
    dataset_lower = dataset_name.lower()
    cache_file = os.path.join(cache_dir, f'{dataset_lower}_rois.npz')
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"ROI cache not found: {cache_file}\n"
            f"Run: python precompute_rois.py --dataset {dataset_name}"
        )
    
    cached = np.load(cache_file, allow_pickle=False)
    
    if split not in cached:
        raise KeyError(f"Split '{split}' not found in cache. Available: {list(cached.keys())}")
    
    return cached[split]


def verify_cache(dataset_name, cache_dir='./roi_cache', n_samples=5):
    """Verify cached ROIs by visualizing samples"""
    import matplotlib.pyplot as plt
    
    print(f"Verifying cache for {dataset_name}...")
    
    # Load original images
    DataClass = getattr(medmnist, dataset_name)
    ds = DataClass(split='train', download=True, size=224)
    imgs = ds.imgs
    
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8)
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)
    
    # Load cached ROIs
    rois = load_cached_rois(dataset_name, split='train', cache_dir=cache_dir)
    
    print(f"✅ Loaded {len(rois)} cached ROIs")
    print(f"   Shape: {rois.shape}")
    print(f"   Dtype: {rois.dtype}")
    
    # Visualize samples
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.linspace(0, len(imgs)-1, n_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        # Original
        axes[i, 0].imshow(imgs[idx])
        axes[i, 0].set_title(f'Sample {idx}')
        axes[i, 0].axis('off')
        
        # With ROI overlay
        axes[i, 1].imshow(imgs[idx])
        axes[i, 1].contour(rois[idx], colors='red', linewidths=2, levels=[0.5])
        coverage = rois[idx].sum() / rois[idx].size * 100
        axes[i, 1].set_title(f'ROI Overlay ({coverage:.1f}%)')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    output_path = f'verify_cache_{dataset_name.lower()}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Verification plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute ROIs for MedMNIST datasets")
    parser.add_argument('--dataset', type=str, default='DermaMNIST',
                       choices=['DermaMNIST', 'OCTMNIST', 'OrganSMNIST', 'TissueMNIST', 
                               'PathMNIST', 'BloodMNIST', 'BreastMNIST', 'RetinaMNIST',
                               'PneumoniaMNIST', 'OrganAMNIST', 'OrganCMNIST'])
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                       help="Which splits to process")
    parser.add_argument('--output_dir', type=str, default='./roi_cache',
                       help="Output directory for cached ROIs")
    parser.add_argument('--force', action='store_true',
                       help="Force recompute even if cache exists")
    parser.add_argument('--verify', action='store_true',
                       help="Verify cache by visualizing samples")
    parser.add_argument('--batch_datasets', type=str, nargs='+',
                       help="Process multiple datasets")
    
    args = parser.parse_args()
    
    if args.batch_datasets:
        # Batch processing
        for dataset in args.batch_datasets:
            cache_file = precompute_rois(
                dataset, args.splits, 
                output_dir=args.output_dir,
                force_recompute=args.force
            )
            if args.verify:
                verify_cache(dataset, args.output_dir)
    else:
        # Single dataset
        cache_file = precompute_rois(
            args.dataset, args.splits,
            output_dir=args.output_dir,
            force_recompute=args.force
        )
        
        if args.verify:
            verify_cache(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
