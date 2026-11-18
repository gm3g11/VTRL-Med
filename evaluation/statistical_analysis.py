#!/usr/bin/env python3
"""
Statistical Analysis: When and Why Topology Helps

Generates:
1. Topological feature importance (H0 vs H1 for rescued samples)
2. Class-pair confusion analysis
3. Per-class rescue rates
4. Topological distinctiveness metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
from collections import defaultdict

# Import from existing code
from stage3_unified import MultimodalModel, load_dataset, SAMDataset, collate_fn_with_topology
from torch.utils.data import DataLoader


def extract_all_data(visual_only_model, full_model, test_loader, device):
    """Extract all predictions, features, and topology"""
    visual_only_model.eval()
    full_model.eval()
    
    all_data = {
        'images': [],
        'topo_features': [],
        'labels': [],
        'vis_only_preds': [],
        'vis_only_conf': [],
        'full_preds': [],
        'full_conf': [],
        'topo_preds': [],
        'topo_conf': []
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting data"):
            img_weak, img_strong, img_test, topo_weak, topo_strong, labels = batch
            
            # Visual-only
            _, vis_logits_only, _ = visual_only_model(
                img_test.to(device, non_blocking=True),
                None,
                use_temperature=True
            )
            vis_probs_only = F.softmax(vis_logits_only, dim=1)
            vis_preds_only = torch.argmax(vis_logits_only, dim=1)
            vis_conf_only = vis_probs_only.max(dim=1)[0]
            
            # Full model
            _, vis_logits_full, topo_logits_full = full_model(
                img_test.to(device, non_blocking=True),
                topo_weak.to(device, non_blocking=True),
                use_temperature=True
            )
            vis_probs_full = F.softmax(vis_logits_full, dim=1)
            vis_preds_full = torch.argmax(vis_logits_full, dim=1)
            vis_conf_full = vis_probs_full.max(dim=1)[0]
            
            topo_probs_full = F.softmax(topo_logits_full, dim=1)
            topo_preds_full = torch.argmax(topo_logits_full, dim=1)
            topo_conf_full = topo_probs_full.max(dim=1)[0]
            
            # Store
            all_data['images'].append(img_test.cpu())
            all_data['topo_features'].append(topo_weak.cpu())
            all_data['labels'].append(labels.cpu())
            all_data['vis_only_preds'].append(vis_preds_only.cpu())
            all_data['vis_only_conf'].append(vis_conf_only.cpu())
            all_data['full_preds'].append(vis_preds_full.cpu())
            all_data['full_conf'].append(vis_conf_full.cpu())
            all_data['topo_preds'].append(topo_preds_full.cpu())
            all_data['topo_conf'].append(topo_conf_full.cpu())
    
    # Concatenate
    for key in all_data:
        all_data[key] = torch.cat(all_data[key]).numpy()
    
    return all_data


def analyze_topological_features(data):
    """Analyze H0/H1 features for rescued vs other samples"""
    
    vis_only_correct = (data['vis_only_preds'] == data['labels'])
    full_correct = (data['full_preds'] == data['labels'])
    
    rescued = ~vis_only_correct & full_correct
    hurt = vis_only_correct & ~full_correct
    both_correct = vis_only_correct & full_correct
    both_wrong = ~vis_only_correct & ~full_correct
    
    categories = {
        'rescued': rescued,
        'hurt': hurt,
        'both_correct': both_correct,
        'both_wrong': both_wrong
    }
    
    # Compute topological statistics
    topo_stats = {}
    
    for cat_name, mask in categories.items():
        if mask.sum() == 0:
            continue
        
        topo_features = data['topo_features'][mask]  # (N, 108, 4)
        
        h0_persistence = []
        h1_persistence = []
        h0_counts = []
        h1_counts = []
        
        for i in range(len(topo_features)):
            topo = topo_features[i]
            
            h0 = topo[topo[:, 3] == 0]
            h1 = topo[topo[:, 3] == 1]
            
            h0_valid = h0[h0[:, 2] > 0]
            h1_valid = h1[h1[:, 2] > 0]
            
            if len(h0_valid) > 0:
                h0_persistence.append(h0_valid[:, 2].mean())
                h0_counts.append(len(h0_valid))
            
            if len(h1_valid) > 0:
                h1_persistence.append(h1_valid[:, 2].mean())
                h1_counts.append(len(h1_valid))
        
        topo_stats[cat_name] = {
            'h0_persistence': np.array(h0_persistence),
            'h1_persistence': np.array(h1_persistence),
            'h0_counts': np.array(h0_counts),
            'h1_counts': np.array(h1_counts)
        }
    
    return topo_stats, categories


def plot_topological_feature_importance(topo_stats, output_path):
    """Plot H0/H1 importance for rescued samples"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: H0 Persistence
    ax = axes[0, 0]
    
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    for cat, color in [('rescued', '#2ecc71'), ('hurt', '#e74c3c'), 
                       ('both_correct', '#3498db'), ('both_wrong', '#95a5a6')]:
        if cat in topo_stats and len(topo_stats[cat]['h0_persistence']) > 0:
            data_to_plot.append(topo_stats[cat]['h0_persistence'])
            labels_to_plot.append(f"{cat.replace('_', ' ').title()}\n(n={len(topo_stats[cat]['h0_persistence'])})")
            colors_to_plot.append(color)
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('H0 Persistence (Avg)', fontsize=11, fontweight='bold')
    ax.set_title('H0 Feature Importance', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: H1 Persistence
    ax = axes[0, 1]
    
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    for cat, color in [('rescued', '#2ecc71'), ('hurt', '#e74c3c'),
                       ('both_correct', '#3498db'), ('both_wrong', '#95a5a6')]:
        if cat in topo_stats and len(topo_stats[cat]['h1_persistence']) > 0:
            data_to_plot.append(topo_stats[cat]['h1_persistence'])
            labels_to_plot.append(f"{cat.replace('_', ' ').title()}\n(n={len(topo_stats[cat]['h1_persistence'])})")
            colors_to_plot.append(color)
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('H1 Persistence (Avg)', fontsize=11, fontweight='bold')
    ax.set_title('H1 Feature Importance', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: H0 Counts
    ax = axes[1, 0]
    
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    for cat, color in [('rescued', '#2ecc71'), ('hurt', '#e74c3c')]:
        if cat in topo_stats and len(topo_stats[cat]['h0_counts']) > 0:
            data_to_plot.append(topo_stats[cat]['h0_counts'])
            labels_to_plot.append(f"{cat.replace('_', ' ').title()}")
            colors_to_plot.append(color)
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('H0 Feature Count', fontsize=11, fontweight='bold')
    ax.set_title('Number of H0 Features', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: H1 Counts
    ax = axes[1, 1]
    
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    for cat, color in [('rescued', '#2ecc71'), ('hurt', '#e74c3c')]:
        if cat in topo_stats and len(topo_stats[cat]['h1_counts']) > 0:
            data_to_plot.append(topo_stats[cat]['h1_counts'])
            labels_to_plot.append(f"{cat.replace('_', ' ').title()}")
            colors_to_plot.append(color)
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_to_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('H1 Feature Count', fontsize=11, fontweight='bold')
    ax.set_title('Number of H1 Features (Loops)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Topological Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_class_pair_confusion(data, categories, num_classes, output_path):
    """Analyze which class pairs benefit from topology"""
    
    rescued = categories['rescued']
    
    # Build confusion matrix (true_label x wrong_visual_prediction)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in np.where(rescued)[0]:
        true_label = data['labels'][i]
        wrong_pred = data['vis_only_preds'][i]
        confusion[true_label, wrong_pred] += 1
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Full confusion heatmap
    ax = axes[0]
    
    sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd',
               xticklabels=[f'C{i}' for i in range(num_classes)],
               yticklabels=[f'C{i}' for i in range(num_classes)],
               ax=ax, cbar_kws={'label': 'Rescued Count'})
    
    ax.set_xlabel('Visual-Only Prediction (Wrong)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Which Errors Does Topology Rescue?', fontsize=13, fontweight='bold')
    
    # Plot 2: Top confused pairs
    ax = axes[1]
    ax.axis('off')
    
    # Find top confused pairs
    top_confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                top_confusions.append((i, j, confusion[i, j]))
    
    top_confusions.sort(key=lambda x: x[2], reverse=True)
    
    # Display top-10
    text = "Top-10 Most Common Errors Rescued by Topology:\n\n"
    text += f"{'Rank':<6}{'True→Pred':<15}{'Count':<10}{'%':<10}\n"
    text += "=" * 50 + "\n"
    
    total_rescued = rescued.sum()
    for rank, (true_label, wrong_pred, count) in enumerate(top_confusions[:10], 1):
        pct = count / total_rescued * 100
        text += f"{rank:<6}C{true_label}→C{wrong_pred:<12}{count:<10}{pct:>6.1f}%\n"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_class_rescue_rates(data, categories, num_classes, output_path):
    """Per-class rescue rates"""
    
    vis_only_correct = (data['vis_only_preds'] == data['labels'])
    full_correct = (data['full_preds'] == data['labels'])
    
    rescued = ~vis_only_correct & full_correct
    hurt = vis_only_correct & ~full_correct
    
    # Per-class statistics
    per_class_stats = defaultdict(lambda: {
        'total': 0,
        'vis_only_correct': 0,
        'full_correct': 0,
        'rescued': 0,
        'hurt': 0
    })
    
    for i in range(len(data['labels'])):
        label = data['labels'][i]
        per_class_stats[label]['total'] += 1
        
        if vis_only_correct[i]:
            per_class_stats[label]['vis_only_correct'] += 1
        if full_correct[i]:
            per_class_stats[label]['full_correct'] += 1
        if rescued[i]:
            per_class_stats[label]['rescued'] += 1
        if hurt[i]:
            per_class_stats[label]['hurt'] += 1
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy comparison
    ax = axes[0]
    
    classes = sorted(per_class_stats.keys())
    vis_only_accs = [per_class_stats[c]['vis_only_correct'] / per_class_stats[c]['total'] * 100 
                     for c in classes]
    full_accs = [per_class_stats[c]['full_correct'] / per_class_stats[c]['total'] * 100
                for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax.bar(x - width/2, vis_only_accs, width, label='Visual-Only', 
          color='#3498db', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, full_accs, width, label='Visual+Topology',
          color='#2ecc71', alpha=0.7, edgecolor='black')
    
    # Add improvement arrows
    for i, (v_acc, f_acc) in enumerate(zip(vis_only_accs, full_accs)):
        if f_acc > v_acc:
            ax.annotate('', xy=(i + width/2, f_acc), xytext=(i - width/2, v_acc),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in classes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Plot 2: Rescue and hurt counts
    ax = axes[1]
    
    rescued_counts = [per_class_stats[c]['rescued'] for c in classes]
    hurt_counts = [per_class_stats[c]['hurt'] for c in classes]
    
    x = np.arange(len(classes))
    
    ax.bar(x, rescued_counts, label='Rescued', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.bar(x, [-h for h in hurt_counts], label='Hurt', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Rescued vs Hurt Samples', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in classes])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
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
    print(f"Statistical Analysis: {args.dataset.upper()}")
    print("="*80 + "\n")
    
    # Load models
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
    
    # Extract all data
    print("Extracting all data...")
    data = extract_all_data(visual_only_model, full_model, test_loader, device)
    
    # Analyze topological features
    print("\nAnalyzing topological features...")
    topo_stats, categories = analyze_topological_features(data)
    
    print(f"\n📊 Sample Categories:")
    print(f"  Rescued: {categories['rescued'].sum()}")
    print(f"  Hurt: {categories['hurt'].sum()}")
    print(f"  Both Correct: {categories['both_correct'].sum()}")
    print(f"  Both Wrong: {categories['both_wrong'].sum()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Statistical Visualizations")
    print("="*80)
    
    # 1. Topological feature importance
    plot_topological_feature_importance(
        topo_stats,
        os.path.join(args.output_dir, f'topo_feature_importance_{args.dataset}.png')
    )
    
    # 2. Class-pair confusion
    plot_class_pair_confusion(
        data, categories, num_classes,
        os.path.join(args.output_dir, f'class_pair_confusion_{args.dataset}.png')
    )
    
    # 3. Per-class rescue rates
    plot_per_class_rescue_rates(
        data, categories, num_classes,
        os.path.join(args.output_dir, f'per_class_rescue_rates_{args.dataset}.png')
    )
    
    print("\n" + "="*80)
    print("✓ Statistical Analysis Complete!")
    print(f"✓ Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pathmnist',
                       choices=['pathmnist', 'bloodmnist', 'dermamnist', 'octmnist',
                               'tissuemnist', 'organamnist', 'breastmnist', 'retinamnist'])
    parser.add_argument('--output_dir', default='./paper_figures')
    args = parser.parse_args()
    
    main(args)
