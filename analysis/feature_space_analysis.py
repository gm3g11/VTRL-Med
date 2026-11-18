#!/usr/bin/env python3
"""
Feature Space Analysis: Discover WHERE and WHEN topology helps

Analyzes:
1. t-SNE/UMAP of visual features colored by rescue status
2. Confidence distribution for rescued vs hurt samples
3. Decision boundary visualization
4. Discovers patterns without assuming thresholds
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from tqdm import tqdm
import os
import argparse
from collections import defaultdict

# Import from your existing code
from stage3_unified import MultimodalModel, load_dataset, SAMDataset, collate_fn_with_topology
from torch.utils.data import DataLoader


def extract_features_and_predictions(model, loader, device, use_topology=False):
    """Extract features, predictions, and confidences"""
    model.eval()
    
    all_features = []
    all_vis_preds = []
    all_vis_probs = []
    all_topo_preds = []
    all_labels = []
    all_indices = []
    
    idx_counter = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            img_weak, img_strong, img_test, topo_weak, topo_strong, labels = batch
            
            topo_input = topo_weak.to(device, non_blocking=True) if use_topology else None
            
            features, vis_logits, topo_logits = model(
                img_test.to(device, non_blocking=True),
                topo_input,
                use_temperature=True
            )
            
            vis_probs = F.softmax(vis_logits, dim=1)
            vis_preds = torch.argmax(vis_logits, dim=1)
            
            all_features.append(features.cpu())
            all_vis_preds.append(vis_preds.cpu())
            all_vis_probs.append(vis_probs.cpu())
            all_labels.append(labels.cpu())
            
            if use_topology and topo_logits is not None:
                topo_preds = torch.argmax(topo_logits, dim=1)
                all_topo_preds.append(topo_preds.cpu())
            
            batch_indices = list(range(idx_counter, idx_counter + len(labels)))
            all_indices.extend(batch_indices)
            idx_counter += len(labels)
    
    return {
        'features': torch.cat(all_features).numpy(),
        'vis_preds': torch.cat(all_vis_preds).numpy(),
        'vis_probs': torch.cat(all_vis_probs).numpy(),
        'topo_preds': torch.cat(all_topo_preds).numpy() if all_topo_preds else None,
        'labels': torch.cat(all_labels).numpy(),
        'indices': np.array(all_indices)
    }


def categorize_samples(visual_only_data, full_model_data):
    """Categorize samples by rescue status"""
    vis_only_correct = (visual_only_data['vis_preds'] == visual_only_data['labels'])
    full_correct = (full_model_data['vis_preds'] == full_model_data['labels'])
    
    # Get max confidence from visual branch
    vis_only_conf = visual_only_data['vis_probs'].max(axis=1)
    full_conf = full_model_data['vis_probs'].max(axis=1)
    
    categories = {
        'rescued': ~vis_only_correct & full_correct,      # Green - topology fixed it
        'hurt': vis_only_correct & ~full_correct,         # Red - topology broke it
        'both_correct': vis_only_correct & full_correct,  # Blue - both right
        'both_wrong': ~vis_only_correct & ~full_correct   # Gray - both wrong
    }
    
    stats = {
        'rescued': {
            'indices': np.where(categories['rescued'])[0],
            'confidences': vis_only_conf[categories['rescued']],
            'labels': visual_only_data['labels'][categories['rescued']],
            'vis_preds': visual_only_data['vis_preds'][categories['rescued']],
            'full_preds': full_model_data['vis_preds'][categories['rescued']]
        },
        'hurt': {
            'indices': np.where(categories['hurt'])[0],
            'confidences': vis_only_conf[categories['hurt']],
            'labels': visual_only_data['labels'][categories['hurt']],
            'vis_preds': visual_only_data['vis_preds'][categories['hurt']],
            'full_preds': full_model_data['vis_preds'][categories['hurt']]
        },
        'both_correct': {
            'indices': np.where(categories['both_correct'])[0],
            'confidences': vis_only_conf[categories['both_correct']]
        },
        'both_wrong': {
            'indices': np.where(categories['both_wrong'])[0],
            'confidences': vis_only_conf[categories['both_wrong']]
        }
    }
    
    return categories, stats


def plot_tsne_by_rescue_status(features, categories, output_path, method='tsne'):
    """t-SNE/UMAP colored by rescue status"""
    
    print(f"\nComputing {method.upper()} projection...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    
    embedding = reducer.fit_transform(features)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: All categories
    ax = axes[0]
    
    colors = {
        'both_correct': ('#3498db', 'Both Correct', 'o', 50, 0.3),
        'both_wrong': ('#95a5a6', 'Both Wrong', 'x', 50, 0.3),
        'hurt': ('#e74c3c', 'Topology Hurt', '^', 80, 0.8),
        'rescued': ('#2ecc71', 'Topology Rescued', '*', 120, 1.0)
    }
    
    # Plot in order: correct -> wrong -> hurt -> rescued (so rescued is on top)
    for category in ['both_correct', 'both_wrong', 'hurt', 'rescued']:
        mask = categories[category]
        if mask.sum() > 0:
            color, label, marker, size, alpha = colors[category]
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, label=f'{label} ({mask.sum()})',
                marker=marker, s=size, alpha=alpha, edgecolors='black', linewidths=0.5
            )
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Feature Space: Colored by Rescue Status', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Focus on rescued vs hurt
    ax = axes[1]
    
    # Plot correct as background
    for category in ['both_correct', 'both_wrong']:
        mask = categories[category]
        if mask.sum() > 0:
            color, label, marker, size, alpha = colors[category]
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, label=label, marker='o', s=20, alpha=0.1, edgecolors='none'
            )
    
    # Highlight rescued and hurt
    for category in ['hurt', 'rescued']:
        mask = categories[category]
        if mask.sum() > 0:
            color, label, marker, size, alpha = colors[category]
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, label=f'{label} ({mask.sum()})',
                marker=marker, s=size * 1.5, alpha=alpha, edgecolors='black', linewidths=1.0
            )
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Focus: Rescued vs Hurt Samples', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confidence_analysis(stats, output_path):
    """Analyze confidence patterns - EXPLORATORY"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Confidence distributions
    ax = axes[0, 0]
    
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    for category, color in [
        ('rescued', '#2ecc71'),
        ('hurt', '#e74c3c'),
        ('both_correct', '#3498db'),
        ('both_wrong', '#95a5a6')
    ]:
        if len(stats[category]['confidences']) > 0:
            data_to_plot.append(stats[category]['confidences'])
            labels_to_plot.append(f"{category.replace('_', ' ').title()}\n(n={len(stats[category]['confidences'])})")
            colors_to_plot.append(color)
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Visual Confidence', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Distribution by Sample Type', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 2: Confidence histograms
    ax = axes[0, 1]
    
    bins = np.linspace(0, 1, 21)
    for category, color, label in [
        ('rescued', '#2ecc71', 'Rescued'),
        ('hurt', '#e74c3c', 'Hurt')
    ]:
        if len(stats[category]['confidences']) > 0:
            ax.hist(stats[category]['confidences'], bins=bins, alpha=0.6,
                   label=f'{label} (n={len(stats[category]["confidences"])})',
                   color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Visual Confidence', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Distribution: Rescued vs Hurt', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Confidence bins analysis
    ax = axes[1, 0]
    
    bin_edges = np.arange(0, 1.1, 0.1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    rescued_bins = np.zeros(len(bin_centers))
    total_errors_bins = np.zeros(len(bin_centers))
    
    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # Rescued samples in this bin
        rescued_in_bin = np.sum(
            (stats['rescued']['confidences'] >= low) & 
            (stats['rescued']['confidences'] < high)
        )
        rescued_bins[i] = rescued_in_bin
        
        # Total errors (rescued + both_wrong) in this bin
        both_wrong_in_bin = np.sum(
            (stats['both_wrong']['confidences'] >= low) & 
            (stats['both_wrong']['confidences'] < high)
        )
        total_errors_bins[i] = rescued_in_bin + both_wrong_in_bin
    
    # Calculate rescue rate
    rescue_rate = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        if total_errors_bins[i] > 0:
            rescue_rate[i] = rescued_bins[i] / total_errors_bins[i] * 100
    
    bars = ax.bar(bin_centers, rescue_rate, width=0.08, alpha=0.7, 
                  color='#2ecc71', edgecolor='black')
    
    # Add value labels
    for bar, rate in zip(bars, rescue_rate):
        if rate > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Visual Confidence Bin', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rescue Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Rescue Rate by Confidence Level', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate key statistics
    rescued_mean = np.mean(stats['rescued']['confidences']) if len(stats['rescued']['confidences']) > 0 else 0
    hurt_mean = np.mean(stats['hurt']['confidences']) if len(stats['hurt']['confidences']) > 0 else 0
    correct_mean = np.mean(stats['both_correct']['confidences']) if len(stats['both_correct']['confidences']) > 0 else 0
    
    summary_text = f"""
    📊 KEY FINDINGS:
    
    Rescued Samples (n={len(stats['rescued']['indices'])}):
      • Mean Confidence: {rescued_mean:.3f}
      • Median: {np.median(stats['rescued']['confidences']):.3f}
      • Std Dev: {np.std(stats['rescued']['confidences']):.3f}
    
    Hurt Samples (n={len(stats['hurt']['indices'])}):
      • Mean Confidence: {hurt_mean:.3f}
      • Median: {np.median(stats['hurt']['confidences']) if len(stats['hurt']['confidences']) > 0 else 0:.3f}
      • Std Dev: {np.std(stats['hurt']['confidences']) if len(stats['hurt']['confidences']) > 0 else 0:.3f}
    
    Both Correct (n={len(stats['both_correct']['indices'])}):
      • Mean Confidence: {correct_mean:.3f}
    
    💡 INSIGHT:
    {"Topology helps when confidence is LOW" if rescued_mean < 0.7 else "Topology helps with OVERCONFIDENT errors" if rescued_mean > 0.8 else "Topology helps across ALL confidence levels"}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Paths
    stage1_path = f"SAM_SPICE_checkpoints/stage1/{args.dataset}/stage1_sam_simclr_{args.dataset}_224.pth"
    stage2_path = f"SAM_SPICE_checkpoints/stage2/{args.dataset}/stage2_sam_simclr_{args.dataset}_224.pth"
    
    visual_only_checkpoint = f"SAM_SPICE_checkpoints/stage3/{args.dataset}/stage3_sam_{args.dataset}_best.pth"
    full_checkpoint = f"SAM_SPICE_checkpoints/stage3_topology/{args.dataset}_more_samples/stage3_sam_{args.dataset}_best.pth"
    
    num_classes_map = {
        'pathmnist': 9, 'bloodmnist': 8, 'dermamnist': 7, 'octmnist': 4,
        'tissuemnist': 8, 'organamnist': 11, 'breastmnist': 2, 'retinamnist': 5
    }
    num_classes = num_classes_map[args.dataset]
    
    print("="*80)
    print(f"Feature Space Analysis: {args.dataset.upper()}")
    print("="*80 + "\n")
    
    # Load visual-only model
    print("Loading visual-only model...")
    visual_only_model = MultimodalModel(
        stage1_path, stage2_path,
        'vit_h', './sam_vit_h_4b8939.pth',
        num_classes, unfrozen_blocks=4, has_stage2=True
    ).to(device)
    visual_only_model.load_state_dict(
        torch.load(visual_only_checkpoint, map_location=device),
        strict=False
    )
    visual_only_model.eval()
    print("✓ Visual-only model loaded\n")
    
    # Load full model
    print("Loading visual+topology model...")
    full_model = MultimodalModel(
        stage1_path, stage2_path,
        'vit_h', './sam_vit_h_4b8939.pth',
        num_classes, unfrozen_blocks=4, has_stage2=True
    ).to(device)
    full_model.load_state_dict(torch.load(full_checkpoint, map_location=device))
    full_model.eval()
    print("✓ Visual+topology model loaded\n")
    
    # Load test data
    print("Loading test data...")
    test_base_dataset = load_dataset(args.dataset, 'test', 224, None, 42)
    test_dataset = SAMDataset(test_base_dataset)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn_with_topology,
        pin_memory=True
    )
    print(f"✓ Test set: {len(test_dataset)} samples\n")
    
    # Extract features and predictions
    print("Extracting features from visual-only model...")
    visual_only_data = extract_features_and_predictions(
        visual_only_model, test_loader, device, use_topology=False
    )
    
    print("Extracting features from visual+topology model...")
    full_model_data = extract_features_and_predictions(
        full_model, test_loader, device, use_topology=True
    )
    
    # Categorize samples
    print("\nCategorizing samples by rescue status...")
    categories, stats = categorize_samples(visual_only_data, full_model_data)
    
    print(f"\n📊 Sample Categories:")
    print(f"  Rescued (V-only wrong, Full correct): {categories['rescued'].sum()}")
    print(f"  Hurt (V-only correct, Full wrong):    {categories['hurt'].sum()}")
    print(f"  Both Correct:                          {categories['both_correct'].sum()}")
    print(f"  Both Wrong:                            {categories['both_wrong'].sum()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    
    # 1. t-SNE
    plot_tsne_by_rescue_status(
        visual_only_data['features'],
        categories,
        os.path.join(args.output_dir, f'feature_space_tsne_{args.dataset}.png'),
        method='tsne'
    )
    
    # 2. UMAP
    plot_tsne_by_rescue_status(
        visual_only_data['features'],
        categories,
        os.path.join(args.output_dir, f'feature_space_umap_{args.dataset}.png'),
        method='umap'
    )
    
    # 3. Confidence analysis
    plot_confidence_analysis(
        stats,
        os.path.join(args.output_dir, f'confidence_analysis_{args.dataset}.png')
    )
    
    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print(f"✓ Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pathmnist', choices=['pathmnist', 'bloodmnist', 'dermamnist', 'octmnist', 'tissuemnist', 'organamnist', 'breastmnist', 'retinamnist'])
    parser.add_argument('--output_dir', default='./paper_figures')
    args = parser.parse_args()
    
    main(args)
