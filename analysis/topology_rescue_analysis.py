#!/usr/bin/env python3
"""
Deep Analysis: Why Does Topology Rescue Visual-Only Failures?

Investigates:
1. Topological feature patterns of rescued vs hurt samples
2. Visual feature confidence/uncertainty
3. Class-pair confusion patterns
4. Topological distinctiveness of rescued samples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from stage3_unified import (
    MultimodalModel, SAMDataset, collate_fn_with_topology,
    load_dataset, extract_ph_from_tensor
)
from torch.utils.data import DataLoader
import os


def analyze_topology_rescue_mechanism(visual_only_model, full_model, test_loader, 
                                      device, num_classes):
    """
    Analyze WHY topology rescues samples
    
    Returns detailed statistics about rescued vs hurt samples
    """
    visual_only_model.eval()
    full_model.eval()
    
    # Storage for analysis
    rescued_samples = []  # Visual-only wrong, Full correct
    hurt_samples = []     # Visual-only correct, Full wrong
    correct_samples = []  # Both correct
    
    with torch.no_grad():
        for batch in test_loader:
            img_weak, img_strong, img_test, topo_weak, topo_strong, labels = batch
            
            # Visual-only predictions
            vis_features_only, vis_logits_only, _ = visual_only_model(
                img_test.to(device, non_blocking=True),
                None,
                use_temperature=True
            )
            vis_probs_only = torch.softmax(vis_logits_only, dim=1)
            vis_preds_only = torch.argmax(vis_logits_only, dim=1)
            
            # Full model predictions
            vis_features_full, vis_logits_full, topo_logits_full = full_model(
                img_test.to(device, non_blocking=True),
                topo_weak.to(device, non_blocking=True),
                use_temperature=True
            )
            vis_probs_full = torch.softmax(vis_logits_full, dim=1)
            vis_preds_full = torch.argmax(vis_logits_full, dim=1)
            topo_probs_full = torch.softmax(topo_logits_full, dim=1)
            topo_preds_full = torch.argmax(topo_logits_full, dim=1)
            
            # Analyze each sample
            for i in range(len(labels)):
                sample_data = {
                    'image': img_test[i].cpu(),
                    'topo_features': topo_weak[i].cpu().numpy(),
                    'true_label': labels[i].item(),
                    'vis_only_pred': vis_preds_only[i].item(),
                    'vis_full_pred': vis_preds_full[i].item(),
                    'topo_pred': topo_preds_full[i].item(),
                    'vis_only_conf': vis_probs_only[i].max().item(),
                    'vis_full_conf': vis_probs_full[i].max().item(),
                    'topo_conf': topo_probs_full[i].max().item(),
                    'vis_only_probs': vis_probs_only[i].cpu().numpy(),
                    'vis_full_probs': vis_probs_full[i].cpu().numpy(),
                    'topo_probs': topo_probs_full[i].cpu().numpy(),
                    'vis_features': vis_features_full[i].cpu().numpy()
                }
                
                vis_only_correct = (sample_data['vis_only_pred'] == sample_data['true_label'])
                full_correct = (sample_data['vis_full_pred'] == sample_data['true_label'])
                
                if not vis_only_correct and full_correct:
                    rescued_samples.append(sample_data)
                elif vis_only_correct and not full_correct:
                    hurt_samples.append(sample_data)
                elif vis_only_correct and full_correct:
                    correct_samples.append(sample_data)
    
    return rescued_samples, hurt_samples, correct_samples


def analyze_topological_features(samples, name="Samples"):
    """Analyze topological feature statistics"""
    if len(samples) == 0:
        return None
    
    h0_persistence_all = []
    h1_persistence_all = []
    h0_counts = []
    h1_counts = []
    
    for sample in samples:
        topo = sample['topo_features']  # (108, 4)
        
        h0 = topo[topo[:, 3] == 0]  # dimension == 0
        h1 = topo[topo[:, 3] == 1]  # dimension == 1
        
        h0_valid = h0[h0[:, 2] > 0]  # persistence > 0
        h1_valid = h1[h1[:, 2] > 0]
        
        if len(h0_valid) > 0:
            h0_persistence_all.append(h0_valid[:, 2].mean())
            h0_counts.append(len(h0_valid))
        
        if len(h1_valid) > 0:
            h1_persistence_all.append(h1_valid[:, 2].mean())
            h1_counts.append(len(h1_valid))
    
    stats = {
        'name': name,
        'count': len(samples),
        'h0_persistence': {
            'mean': np.mean(h0_persistence_all) if h0_persistence_all else 0,
            'std': np.std(h0_persistence_all) if h0_persistence_all else 0
        },
        'h1_persistence': {
            'mean': np.mean(h1_persistence_all) if h1_persistence_all else 0,
            'std': np.std(h1_persistence_all) if h1_persistence_all else 0
        },
        'h0_count': {
            'mean': np.mean(h0_counts) if h0_counts else 0,
            'std': np.std(h0_counts) if h0_counts else 0
        },
        'h1_count': {
            'mean': np.mean(h1_counts) if h1_counts else 0,
            'std': np.std(h1_counts) if h1_counts else 0
        }
    }
    
    return stats


def analyze_confidence_patterns(samples, name="Samples"):
    """Analyze confidence/uncertainty patterns"""
    if len(samples) == 0:
        return None
    
    vis_confs = [s['vis_only_conf'] for s in samples]
    topo_confs = [s['topo_conf'] for s in samples]
    
    # Calculate entropy (uncertainty)
    vis_entropies = [-np.sum(s['vis_only_probs'] * np.log(s['vis_only_probs'] + 1e-8)) 
                     for s in samples]
    topo_entropies = [-np.sum(s['topo_probs'] * np.log(s['topo_probs'] + 1e-8))
                      for s in samples]
    
    stats = {
        'name': name,
        'vis_confidence': {'mean': np.mean(vis_confs), 'std': np.std(vis_confs)},
        'topo_confidence': {'mean': np.mean(topo_confs), 'std': np.std(topo_confs)},
        'vis_entropy': {'mean': np.mean(vis_entropies), 'std': np.std(vis_entropies)},
        'topo_entropy': {'mean': np.mean(topo_entropies), 'std': np.std(topo_entropies)}
    }
    
    return stats


def analyze_confusion_patterns(samples, num_classes):
    """Analyze which class pairs benefit from topology"""
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for sample in samples:
        true_label = sample['true_label']
        wrong_pred = sample['vis_only_pred']
        confusion[true_label, wrong_pred] += 1
    
    return confusion


def visualize_rescue_analysis(rescued_samples, hurt_samples, correct_samples,
                              num_classes, dataset_name, output_dir='./paper_figures'):
    """Create comprehensive visualization of rescue mechanism"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # 1. Topological Features Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    rescued_stats = analyze_topological_features(rescued_samples, "Rescued")
    hurt_stats = analyze_topological_features(hurt_samples, "Hurt")
    correct_stats = analyze_topological_features(correct_samples[:500], "Correct (sample)")
    
    all_stats = [rescued_stats, hurt_stats, correct_stats]
    names = [s['name'] for s in all_stats if s is not None]
    h0_means = [s['h0_persistence']['mean'] for s in all_stats if s is not None]
    h1_means = [s['h1_persistence']['mean'] for s in all_stats if s is not None]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, h0_means, width, label='H0', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, h1_means, width, label='H1', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Avg Persistence', fontweight='bold')
    ax1.set_title('Topological Features by Sample Type', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (h0, h1) in enumerate(zip(h0_means, h1_means)):
        ax1.text(i - width/2, h0, f'{h0:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, h1, f'{h1:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Confidence Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    rescued_conf = analyze_confidence_patterns(rescued_samples, "Rescued")
    hurt_conf = analyze_confidence_patterns(hurt_samples, "Hurt")
    correct_conf = analyze_confidence_patterns(correct_samples[:500], "Correct")
    
    all_conf = [rescued_conf, hurt_conf, correct_conf]
    conf_names = [c['name'] for c in all_conf if c is not None]
    vis_confs = [c['vis_confidence']['mean'] for c in all_conf if c is not None]
    topo_confs = [c['topo_confidence']['mean'] for c in all_conf if c is not None]
    
    x = np.arange(len(conf_names))
    
    ax2.bar(x - width/2, vis_confs, width, label='Visual Conf.', color='#9b59b6', alpha=0.8)
    ax2.bar(x + width/2, topo_confs, width, label='Topo Conf.', color='#2ecc71', alpha=0.8)
    ax2.set_ylabel('Confidence', fontweight='bold')
    ax2.set_title('Prediction Confidence by Sample Type', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conf_names, rotation=15)
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Entropy (Uncertainty) Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    vis_ents = [c['vis_entropy']['mean'] for c in all_conf if c is not None]
    topo_ents = [c['topo_entropy']['mean'] for c in all_conf if c is not None]
    
    ax3.bar(x - width/2, vis_ents, width, label='Visual Entropy', color='#e67e22', alpha=0.8)
    ax3.bar(x + width/2, topo_ents, width, label='Topo Entropy', color='#1abc9c', alpha=0.8)
    ax3.set_ylabel('Entropy (Uncertainty)', fontweight='bold')
    ax3.set_title('Prediction Uncertainty by Sample Type', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(conf_names, rotation=15)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Confusion Matrix for Rescued Samples
    ax4 = fig.add_subplot(gs[1, :])
    
    confusion = analyze_confusion_patterns(rescued_samples, num_classes)
    
    sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=[f'C{i}' for i in range(num_classes)],
                yticklabels=[f'C{i}' for i in range(num_classes)],
                ax=ax4, cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Visual-Only Prediction (Wrong)', fontweight='bold')
    ax4.set_ylabel('True Label', fontweight='bold')
    ax4.set_title('Confusion Pattern: Which Errors Does Topology Rescue?', fontweight='bold')
    
    # 5-8. Example rescued samples with topology visualization
    examples = rescued_samples[:4]
    
    for idx, sample in enumerate(examples):
        ax = fig.add_subplot(gs[2, idx % 3])
        
        # Show image
        img = sample['image'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f"True: C{sample['true_label']}\n"
            f"V-Only: C{sample['vis_only_pred']} ({sample['vis_only_conf']:.2f})\n"
            f"Topo: C{sample['topo_pred']} ({sample['topo_conf']:.2f})",
            fontsize=9, color='green', fontweight='bold'
        )
        
        # Add 4th example in row 4
        if idx == 3:
            ax = fig.add_subplot(gs[3, 0])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(
                f"True: C{sample['true_label']}\n"
                f"V-Only: C{sample['vis_only_pred']} ({sample['vis_only_conf']:.2f})\n"
                f"Topo: C{sample['topo_pred']} ({sample['topo_conf']:.2f})",
                fontsize=9, color='green', fontweight='bold'
            )
    
    # Add explanation text
    explanation = (
        f"📊 Rescued Samples: {len(rescued_samples)}\n"
        f"📊 Hurt Samples: {len(hurt_samples)}\n\n"
        f"Key Findings:\n"
        f"• Rescued samples have {'higher' if h0_means[0] > h0_means[2] else 'lower'} "
        f"H0 persistence than average\n"
        f"• Visual confidence for rescued: {vis_confs[0]:.3f} "
        f"(vs correct: {vis_confs[2]:.3f})\n"
        f"• Topology provides complementary signal when visual is uncertain"
    )
    
    fig.text(0.65, 0.08, explanation, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(
        f'Why Topology Rescues Visual Failures: {dataset_name.upper()}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/topology_rescue_analysis_{dataset_name}.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/topology_rescue_analysis_{dataset_name}.png")
    plt.close()


def print_detailed_analysis(rescued_samples, hurt_samples, correct_samples, num_classes):
    """Print detailed statistical analysis"""
    
    print(f"\n{'='*80}")
    print("DETAILED RESCUE MECHANISM ANALYSIS")
    print('='*80)
    
    # Topological features
    print(f"\n📊 Topological Feature Analysis:")
    print("-" * 80)
    
    for samples, name in [(rescued_samples, "Rescued"), 
                          (hurt_samples, "Hurt"),
                          (correct_samples[:500], "Correct (sample)")]:
        stats = analyze_topological_features(samples, name)
        if stats:
            print(f"\n{name} Samples (n={stats['count']}):")
            print(f"  H0: μ={stats['h0_persistence']['mean']:.4f}, "
                  f"σ={stats['h0_persistence']['std']:.4f}, "
                  f"count={stats['h0_count']['mean']:.1f}")
            print(f"  H1: μ={stats['h1_persistence']['mean']:.4f}, "
                  f"σ={stats['h1_persistence']['std']:.4f}, "
                  f"count={stats['h1_count']['mean']:.1f}")
    
    # Confidence patterns
    print(f"\n📊 Confidence & Uncertainty Analysis:")
    print("-" * 80)
    
    for samples, name in [(rescued_samples, "Rescued"),
                          (hurt_samples, "Hurt"),
                          (correct_samples[:500], "Correct")]:
        conf = analyze_confidence_patterns(samples, name)
        if conf:
            print(f"\n{name} Samples:")
            print(f"  Visual Conf:  {conf['vis_confidence']['mean']:.4f} ± "
                  f"{conf['vis_confidence']['std']:.4f}")
            print(f"  Topo Conf:    {conf['topo_confidence']['mean']:.4f} ± "
                  f"{conf['topo_confidence']['std']:.4f}")
            print(f"  Visual Ent:   {conf['vis_entropy']['mean']:.4f} ± "
                  f"{conf['vis_entropy']['std']:.4f}")
            print(f"  Topo Ent:     {conf['topo_entropy']['mean']:.4f} ± "
                  f"{conf['topo_entropy']['std']:.4f}")
    
    # Most common confusions rescued
    print(f"\n📊 Most Common Errors Rescued by Topology:")
    print("-" * 80)
    
    confusion = analyze_confusion_patterns(rescued_samples, num_classes)
    top_confusions = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                top_confusions.append((i, j, confusion[i, j]))
    
    top_confusions.sort(key=lambda x: x[2], reverse=True)
    
    for true_label, wrong_pred, count in top_confusions[:10]:
        print(f"  C{true_label} → C{wrong_pred}: {count} samples rescued")
    
    print(f"\n{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['pathmnist', 'isic2019', 'kvasir'])
    parser.add_argument('--sam_checkpoint', default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_model_type', default='vit_h')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--output_dir', default='./paper_figures')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    
    num_classes_map = {'pathmnist': 9, 'isic2019': 8, 'kvasir': 8}
    num_classes = num_classes_map[args.dataset]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint paths
    visual_only_checkpoint = f"SAM_SPICE_checkpoints/stage3/{args.dataset}/stage3_sam_{args.dataset}_best.pth"
    
    if args.dataset == 'pathmnist':
        full_checkpoint = f"SAM_SPICE_checkpoints/stage3_topology/{args.dataset}_more_samples/stage3_sam_{args.dataset}_best.pth"
    else:
        full_checkpoint = f"SAM_SPICE_checkpoints/stage3_topology/{args.dataset}/stage3_sam_{args.dataset}_best.pth"
    
    stage1_path = f"SAM_SPICE_checkpoints/stage1/{args.dataset}/stage1_sam_simclr_{args.dataset}_{args.img_size}.pth"
    stage2_path = f"SAM_SPICE_checkpoints/stage2/{args.dataset}/stage2_sam_simclr_{args.dataset}_{args.img_size}.pth"
    
    print(f"\n{'='*80}")
    print(f"Deep Rescue Analysis: {args.dataset.upper()}")
    print('='*80 + '\n')
    
    # Load models
    from stage3_unified import MultimodalModel
    
    print("Loading models...")
    visual_only_model = MultimodalModel(
        stage1_path, stage2_path,
        args.sam_model_type, args.sam_checkpoint,
        num_classes, unfrozen_blocks=4, has_stage2=True
    ).to(device)
    visual_only_model.load_state_dict(
        torch.load(visual_only_checkpoint, map_location=device), strict=False
    )
    visual_only_model.eval()
    
    full_model = MultimodalModel(
        stage1_path, stage2_path,
        args.sam_model_type, args.sam_checkpoint,
        num_classes, unfrozen_blocks=4, has_stage2=True
    ).to(device)
    full_model.load_state_dict(torch.load(full_checkpoint, map_location=device))
    full_model.eval()
    print("✓ Models loaded\n")
    
    # Load data
    test_base_dataset = load_dataset(args.dataset, 'test', args.img_size, args.data_dir, args.seed)
    test_dataset = SAMDataset(test_base_dataset)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_topology,
        pin_memory=True
    )
    print(f"✓ Test set loaded: {len(test_dataset)} samples\n")
    
    # Run analysis
    print("Analyzing rescue mechanism...")
    rescued, hurt, correct = analyze_topology_rescue_mechanism(
        visual_only_model, full_model, test_loader, device, num_classes
    )
    
    print(f"✓ Found {len(rescued)} rescued, {len(hurt)} hurt, {len(correct)} correct\n")
    
    # Visualize
    visualize_rescue_analysis(rescued, hurt, correct, num_classes, args.dataset, args.output_dir)
    
    # Print detailed stats
    print_detailed_analysis(rescued, hurt, correct, num_classes)


if __name__ == '__main__':
    main()
