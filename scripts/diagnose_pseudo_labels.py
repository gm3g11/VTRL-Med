"""
Diagnostic: Check pseudo-label quality for FixMatch/FlexMatch on Kvasir.

1. Per-class prediction distribution (collapse check)
2. FixMatch/FlexMatch threshold filtering (how many samples survive)
3. Initial pseudo-label bootstrapping (random head → what labels?)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transformsV2
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

from scripts.stage2_unified import (
    SAMFeatureExtractor, SAMRawDataset, SPICEClusteringHead,
    fixmatch_pseudo_labeling, flexmatch_pseudo_labeling, get_dataset_info
)

device = torch.device("cuda")
dataset = "kvasir"
data_dir = os.path.expanduser("data/kvasir-dataset")
num_classes = get_dataset_info(dataset)
img_size = 224
threshold = 0.95

# Load SAM (adapted)
stage1_ckpt = "checkpoints/sam_spice_kvasir_isic/stage1/kvasir/stage1_sam_simclr_kvasir_224.pth"
sam_ckpt = "weights/sam_vit_h_4b8939.pth"

print("Loading SAM (adapted)...")
feature_extractor = SAMFeatureExtractor(
    stage1_ckpt, sam_model_type='vit_h',
    sam_pretrained_path=sam_ckpt, use_stage1=True
).to(device)

norm_params = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
gpu_transform = torch.nn.Sequential(
    transformsV2.ToDtype(torch.float32, scale=True),
    transformsV2.Resize((img_size, img_size), antialias=True),
    transformsV2.Normalize(**norm_params)
).to(device)

# Load train data
train_dataset = SAMRawDataset(dataset, 'train', img_size, data_dir, 42)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)

print("Extracting features...")
all_features = []
with torch.no_grad():
    for raw_img_batch, _ in train_loader:
        img = gpu_transform(raw_img_batch.to(device))
        features = feature_extractor(img)
        all_features.append(features.cpu())
all_features = torch.cat(all_features, dim=0)
print(f"Features: {all_features.shape}")

# Initialize a RANDOM clustering head (epoch 0 state)
print("\n" + "="*70)
print("CHECK 1: Random head prediction distribution (before any training)")
print("="*70)
head = SPICEClusteringHead(feature_extractor.feature_dim, num_classes).to(device)
head.eval()

with torch.no_grad():
    all_logits = []
    for i in range(0, all_features.shape[0], 512):
        batch = all_features[i:i+512].to(device)
        all_logits.append(head(batch).cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = F.softmax(all_logits, dim=1)
    max_probs, preds = torch.max(all_probs, dim=1)

pred_counts = Counter(preds.numpy())
print(f"\nPer-class prediction counts (random head, {len(preds)} total samples):")
for c in range(num_classes):
    count = pred_counts.get(c, 0)
    pct = 100.0 * count / len(preds)
    bar = "#" * int(pct)
    print(f"  Class {c}: {count:5d} ({pct:5.1f}%) {bar}")

# Check confidence distribution
print(f"\nConfidence (max prob) stats:")
print(f"  Mean: {max_probs.mean():.4f}")
print(f"  Std:  {max_probs.std():.4f}")
print(f"  Min:  {max_probs.min():.4f}")
print(f"  Max:  {max_probs.max():.4f}")
print(f"  Median: {max_probs.median():.4f}")

for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    above = (max_probs >= t).sum().item()
    print(f"  Above {t}: {above}/{len(preds)} ({100.0*above/len(preds):.1f}%)")

# CHECK 2: FixMatch filtering
print("\n" + "="*70)
print("CHECK 2: FixMatch pseudo-labels (threshold=0.95)")
print("="*70)
fix_labels = fixmatch_pseudo_labeling(
    all_features[:512].to(device), all_probs[:512].to(device),
    num_classes, confidence_threshold=threshold
)
valid = (fix_labels >= 0).sum().item()
total = len(fix_labels)
print(f"  Valid pseudo-labels: {valid}/{total} ({100.0*valid/total:.1f}%)")
if valid > 0:
    valid_labels = fix_labels[fix_labels >= 0]
    fix_counts = Counter(valid_labels.cpu().numpy())
    print(f"  Distribution of valid labels:")
    for c in range(num_classes):
        count = fix_counts.get(c, 0)
        print(f"    Class {c}: {count}")

# CHECK 3: FlexMatch filtering
print("\n" + "="*70)
print("CHECK 3: FlexMatch pseudo-labels (base threshold=0.95)")
print("="*70)
flex_labels = flexmatch_pseudo_labeling(
    all_features[:512].to(device), all_probs[:512].to(device),
    num_classes, confidence_threshold=threshold
)
valid_flex = (flex_labels >= 0).sum().item()
print(f"  Valid pseudo-labels: {valid_flex}/{total} ({100.0*valid_flex/total:.1f}%)")
if valid_flex > 0:
    valid_flex_labels = flex_labels[flex_labels >= 0]
    flex_counts = Counter(valid_flex_labels.cpu().numpy())
    print(f"  Distribution of valid labels:")
    for c in range(num_classes):
        count = flex_counts.get(c, 0)
        print(f"    Class {c}: {count}")

# Per-class adaptive thresholds
print(f"\n  FlexMatch per-class adaptive thresholds:")
with torch.no_grad():
    probs_dev = all_probs[:512].to(device)
    mp, pl = torch.max(probs_dev, dim=1)
    class_thresholds = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        mask = (pl == c)
        if mask.sum() > 0:
            class_thresholds[c] = (mp[mask] >= threshold).float().mean()
    max_effect = class_thresholds.max().clamp(min=1e-6)
    adaptive = threshold * (class_thresholds / max_effect)
    for c in range(num_classes):
        print(f"    Class {c}: learning_status={class_thresholds[c]:.4f}, "
              f"adaptive_threshold={adaptive[c]:.4f}")

# CHECK 4: Full dataset pseudo-label distribution
print("\n" + "="*70)
print("CHECK 4: Full dataset - FixMatch vs FlexMatch pseudo-label coverage")
print("="*70)
with torch.no_grad():
    all_probs_dev = all_probs.to(device)
    feats_dev = all_features.to(device)

    fix_full = fixmatch_pseudo_labeling(feats_dev, all_probs_dev, num_classes, threshold)
    flex_full = flexmatch_pseudo_labeling(feats_dev, all_probs_dev, num_classes, threshold)

fix_valid = (fix_full >= 0).sum().item()
flex_valid = (flex_full >= 0).sum().item()
N = len(fix_full)
print(f"  FixMatch:  {fix_valid}/{N} valid ({100.0*fix_valid/N:.1f}%)")
print(f"  FlexMatch: {flex_valid}/{N} valid ({100.0*flex_valid/N:.1f}%)")

print(f"\n  FixMatch per-class distribution (full dataset):")
fix_counts_full = Counter(fix_full[fix_full >= 0].cpu().numpy())
for c in range(num_classes):
    count = fix_counts_full.get(c, 0)
    pct = 100.0 * count / max(fix_valid, 1)
    print(f"    Class {c}: {count:5d} ({pct:5.1f}%)")

print(f"\n  FlexMatch per-class distribution (full dataset):")
flex_counts_full = Counter(flex_full[flex_full >= 0].cpu().numpy())
for c in range(num_classes):
    count = flex_counts_full.get(c, 0)
    pct = 100.0 * count / max(flex_valid, 1)
    print(f"    Class {c}: {count:5d} ({pct:5.1f}%)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
top_class_pct = max(pred_counts.values()) / len(preds) * 100
print(f"  Random head collapse? Top class has {top_class_pct:.1f}% of predictions "
      f"({'YES - COLLAPSED' if top_class_pct > 80 else 'No' if top_class_pct < 50 else 'PARTIAL'})")
print(f"  FixMatch coverage:  {100.0*fix_valid/N:.1f}% samples have pseudo-labels")
print(f"  FlexMatch coverage: {100.0*flex_valid/N:.1f}% samples have pseudo-labels")
print(f"  Median confidence:  {max_probs.median():.4f} (threshold: {threshold})")
if max_probs.median() < threshold:
    print(f"  WARNING: Median confidence ({max_probs.median():.4f}) < threshold ({threshold})")
    print(f"  → Most samples filtered out! The head has no confident predictions to learn from.")
    print(f"  → This creates a chicken-and-egg problem: head needs labels to improve,")
    print(f"  → but threshold filters out labels because head isn't confident yet.")
