#!/usr/bin/env python3
"""
Stage 3: Parallel SSL Training (FIXED + DEBUG MODE)

Features:
- Single-GPU training (clean and reliable)
- Debug mode for quick hyperparameter tuning
- Detailed loss tracing and diagnostics
- Automatic ROI fallback (SAM → traditional CV)
- Fixed DINO (clarified as simplified version)

Usage:
  Debug (10 epochs):  python stage3_parallel_train.py --dataset pathmnist --debug
  Full training:      python stage3_parallel_train.py --dataset pathmnist --epochs 150
"""

import os
import json
import argparse
import warnings
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Stage 2 imports
from topocl.stage2.stage2_unified_parallel import (
    HierarchicalTopoEncoder,
    TopologyAwareAugmentation,
    load_dataset_with_rois  # Has ROI fallback built-in!
)

# Stage 3 imports
from topocl.stage3.stage1_encoder_config import load_visual_encoder, load_topo_encoder
from topocl.stage3.encoders import ResNet50Encoder, MoEFusedEncoder
from topocl.stage3.ssl_models import SimCLRModel, BYOLModel, DINOModel, BarlowModel, MoCoV3Model
from topocl.stage3.utils import (
    set_seed, get_num_classes, balance_loss,
    print_expert_statistics, linear_probe_evaluation,
    train_linear_probe_from_encoder,
    get_cosine_schedule_with_warmup
)
from topocl.stage3.stage3_dataset import Stage3TopoImageDataset

warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ============================================================================
# DEBUG MODE - Detailed Loss Analysis
# ============================================================================

def print_loss_breakdown(epoch, losses, all_gates, methods):
    """Print detailed loss analysis for debugging"""
    print(f"\n{'='*80}")
    print(f"🔍 Detailed Loss Analysis - Epoch {epoch}")
    print(f"{'='*80}")
    
    for method in methods:
        loss = losses[method]
        gates = all_gates[method]
        
        # Compute gate statistics
        gate_mean = gates.mean(dim=0).cpu().numpy()
        gate_std = gates.std(dim=0).cpu().numpy()
        gate_entropy = -(gate_mean * np.log(gate_mean + 1e-8)).sum()
        max_entropy = np.log(len(gate_mean))
        
        print(f"\n{method.upper()}:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Gate weights: [{', '.join([f'{g:.3f}' for g in gate_mean])}]")
        print(f"  Gate std:     [{', '.join([f'{s:.3f}' for s in gate_std])}]")
        print(f"  Gate entropy: {gate_entropy:.4f} / {max_entropy:.4f} ({gate_entropy/max_entropy*100:.1f}%)")
        
        # Check for issues
        warnings_found = []
        if loss > 100:
            warnings_found.append("Loss very high (>100)")
        if np.isnan(loss) or np.isinf(loss):
            warnings_found.append("Loss is NaN/Inf!")
        if gate_entropy < max_entropy * 0.3:
            warnings_found.append(f"Gates collapsing (entropy < {max_entropy*0.3:.2f})")
        if (gate_mean > 0.7).any():
            dominant_idx = np.argmax(gate_mean)
            warnings_found.append(f"Gate {dominant_idx} dominates ({gate_mean[dominant_idx]*100:.1f}%)")
        if (gate_mean < 0.01).any():
            ignored = np.where(gate_mean < 0.01)[0]
            warnings_found.append(f"Gates {ignored} ignored (<1%)")
        
        if warnings_found:
            print(f"  ⚠️  WARNINGS:")
            for w in warnings_found:
                print(f"    - {w}")
        else:
            print(f"  ✅ Looks healthy!")
    
    print(f"{'='*80}\n")


def check_loss_stability(losses_history, window=5):
    """Check if training is stable"""
    issues = {}
    
    for method, losses in losses_history.items():
        if len(losses) < window:
            continue
        
        recent = losses[-window:]
        
        # Check for NaN/Inf
        if any(np.isnan(l) or np.isinf(l) for l in recent):
            issues[method] = "NaN/Inf detected"
            continue
        
        # Check for explosion
        if any(l > 1000 for l in recent):
            issues[method] = "Loss exploding (>1000)"
            continue
        
        # Check for no decrease
        if len(losses) >= 10:
            early = np.mean(losses[:5])
            late = np.mean(losses[-5:])
            if late >= early * 0.95:  # Less than 5% improvement
                issues[method] = "Loss not decreasing"
    
    return issues


def save_debug_results(args, losses_history, best_accs, methods):
    """Save debug results for hyperparameter tuning"""
    debug_dir = os.path.join(args.save_dir, args.dataset, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON
    losses_history_json = {}
    for method in methods:
        losses_history_json[method] = [float(l) for l in losses_history[method]]
    
    debug_summary = {
        'args': vars(args),
        'losses_history': losses_history_json,
        'final_losses': {m: float(losses_history[m][-1]) for m in methods},
        'best_accs': {m: float(best_accs[m]) for m in methods},
        'training_stable': all(
            not (np.isnan(losses_history[m][-1]) or np.isinf(losses_history[m][-1]))
            for m in methods
        )
    }
    
    with open(os.path.join(debug_dir, 'debug_summary.json'), 'w') as f:
        json.dump(debug_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🐛 Debug Summary")
    print(f"{'='*80}")
    print(f"Results saved to: {debug_dir}/debug_summary.json\n")
    
    # Print recommendations
    stability_issues = check_loss_stability(losses_history)
    
    print(f"📊 Final Results:")
    for method in methods:
        status = "✅" if method not in stability_issues else "⚠️"
        print(f"  {status} {method.upper():8s}: Loss={losses_history[method][-1]:.4f}, "
              f"Acc={best_accs[method]:.2f}%")
    
    print(f"\n💡 Recommendations:")
    if stability_issues:
        print(f"  Issues found in: {', '.join(stability_issues.keys())}")
        for method, issue in stability_issues.items():
            print(f"    - {method}: {issue}")
        print(f"\n  Suggested fixes:")
        print(f"    - Reduce learning rates (try 0.5× current)")
        print(f"    - Check data loading (ROI coverage)")
        print(f"    - Increase warmup steps")
    elif all(best_accs[m] < 40 for m in methods):
        print(f"  All methods < 40% accuracy - learning rates might be too low")
        print(f"  Try: --lr_topo 3e-4 --lr_fusion 7e-4")
    elif any(losses_history[m][-1] > 50 for m in methods):
        print(f"  Some losses still high - try:")
        print(f"    - Reduce --balance_coef (current: {args.balance_coef})")
        print(f"    - Increase --lr_topo (current: {args.lr_topo})")
    else:
        print(f"  ✅ Training looks stable! Ready for full run.")
        print(f"  Run full training with: --epochs 150")
    
    print(f"{'='*80}\n")


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_method_model(method: str, args, device):
    """Create SSL model with proper architecture"""
    from stage1_encoder_config import get_encoder_variant
    from encoders import ResNetEncoder
    
    print(f"  Creating {method.upper()} model...")
    
    # Get architecture
    variant = get_encoder_variant(method, args.dataset)
    
    # Create encoders
    visual_encoder = ResNetEncoder(variant=variant)
    topo_encoder = HierarchicalTopoEncoder(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        n_h0=args.n_h0 * 3,
        n_h1=args.n_h1 * 3
    )
    
    # Load pre-trained weights
    print(f"    Loading pre-trained weights...")
    load_visual_encoder(method, args.dataset, visual_encoder)
    load_topo_encoder(args.dataset, topo_encoder)
    
    # Create MoE encoder
    moe_encoder = MoEFusedEncoder(
        visual_encoder,
        topo_encoder,
        embed_dim=args.embed_dim,
        out_dim=args.fusion_embed_dim,
        dropout=args.dropout
    ).to(device)
    
    # Create SSL model
    if method == 'simclr':
        model = SimCLRModel(moe_encoder, proj_dim=args.proj_dim)
    elif method == 'byol':
        model = BYOLModel(moe_encoder, proj_dim=args.proj_dim, m=args.moco_m)
    elif method == 'dino':
        # Note: Simplified DINO without multi-crop
        model = DINOModel(moe_encoder, proj_dim=args.proj_dim, m=args.moco_m)
        if args.debug:
            print(f"    ⚠️  Note: Using simplified DINO (no multi-crop)")
    elif method == 'barlow':
        model = BarlowModel(moe_encoder, proj_dim=args.proj_dim)
    elif method == 'mocov3':
        model = MoCoV3Model(moe_encoder, proj_dim=args.proj_dim, m=args.moco_m)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    ✓ {variant}, {total_params:.2f}M params")
    
    return model, moe_encoder


def create_optimizer(model, method, args):
    """Create optimizer with differential learning rates"""
    if method == 'simclr':
        encoder = model.encoder
    elif method == 'byol':
        encoder = model.online_encoder
    elif method == 'dino':
        encoder = model.student_encoder
    elif method == 'barlow':
        encoder = model.encoder
    elif method == 'mocov3':
        encoder = model.encoder_q
    else:
        raise ValueError(f"Unknown method: {method}")
    
    param_groups = [
        {'params': encoder.visual_encoder.parameters(), 'lr': args.lr_img},
        {'params': encoder.topo_encoder.parameters(), 'lr': args.lr_topo},
        {'params': [p for n, p in encoder.named_parameters()
                    if 'visual_encoder' not in n and 'topo_encoder' not in n],
         'lr': args.lr_fusion},
        {'params': [p for n, p in model.named_parameters()
                    if not any(x in n for x in ['encoder', 'online_encoder', 'student_encoder'])],
         'lr': args.lr_fusion}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(models, optimizers, schedulers, scalers, train_loader,
                    args, device, epoch):
    """Train for one epoch"""
    methods = list(models.keys())
    
    for model in models.values():
        model.train()
    
    losses = {method: 0.0 for method in methods}
    all_gates = {method: [] for method in methods}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        img1, img2, h0_1, h1_1, h0_2, h1_2, labels = batch
        
        # Move to device
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        h0_1 = h0_1.to(device, non_blocking=True)
        h1_1 = h1_1.to(device, non_blocking=True)
        h0_2 = h0_2.to(device, non_blocking=True)
        h1_2 = h1_2.to(device, non_blocking=True)
        
        batch_losses = {}
        
        # Train each method
        for method in methods:
            model = models[method]
            optimizer = optimizers[method]
            scaler = scalers[method]
            
            optimizer.zero_grad()
            
            with autocast('cuda', dtype=torch.float16, enabled=args.use_amp):
                loss, gates = model(img1, img2, h0_1, h1_1, h0_2, h1_2,
                                   temperature=args.temperature)
                
                bal_loss = balance_loss(gates, args.balance_low, args.balance_high)
                total_loss = loss + args.balance_coef * bal_loss
            
            scaler.scale(total_loss).backward()
            
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            schedulers[method].step()
            
            losses[method] += loss.item()
            all_gates[method].append(gates.detach().cpu())
            batch_losses[method] = loss.item()
        
        # Update progress bar
        loss_str = " | ".join([f"{m[:3]}:{batch_losses[m]:.3f}" for m in methods])
        pbar.set_postfix_str(loss_str)
    
    # Average losses
    for method in losses.keys():
        losses[method] /= len(train_loader)
        all_gates[method] = torch.cat(all_gates[method], dim=0)
    
    return losses, all_gates


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 3: Parallel SSL Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['kvasir', 'pathmnist', 'octmnist', 'organsmnist', 'isic2019'])
    parser.add_argument('--roi_dir', type=str, default='../TopoSSL',
                       help='Directory containing ROI files (will fallback if not found)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Quick debug mode: 10 epochs, frequent eval, detailed tracing')
    parser.add_argument('--debug_epochs', type=int, default=10,
                       help='Number of epochs in debug mode')
    
    # Learning rates
    parser.add_argument('--lr_img', type=float, default=1e-5)
    parser.add_argument('--lr_topo', type=float, default=2e-4)
    parser.add_argument('--lr_fusion', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--n_h0', type=int, default=96)
    parser.add_argument('--n_h1', type=int, default=192)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--fusion_embed_dim', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=256,
                       help='Projection dimension (keep ≤ batch_size for Barlow)')
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (default: save_dir/dataset)')
    # SSL
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--moco_m', type=float, default=0.996)

    
    # Expert balance
    parser.add_argument('--balance_coef', type=float, default=0.015)
    parser.add_argument('--balance_low', type=float, default=0.05)
    parser.add_argument('--balance_high', type=float, default=0.5)
    
    # Optimization
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    # Evaluation
    parser.add_argument('--probe_freq', type=int, default=30)
    parser.add_argument('--monitor_freq', type=int, default=20)
    parser.add_argument('--early_stop_patience', type=int, default=40)
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./stage3_results')
    
    args = parser.parse_args()
    
    # Debug mode adjustments
    if args.debug:
        print("\n" + "="*80)
        print("🐛 DEBUG MODE ENABLED")
        print("="*80)
        print(f"  - Epochs: {args.debug_epochs} (quick check)")
        print(f"  - Evaluation: Every 2 epochs")
        print(f"  - Detailed loss tracing")
        print(f"  - Results saved to: {args.save_dir}/debug_runs/")
        print("="*80 + "\n")
        
        args.epochs = args.debug_epochs
        args.probe_freq = 2
        args.monitor_freq = 2
        args.save_dir = os.path.join(args.save_dir, 'debug_runs')
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    methods = ['simclr', 'byol', 'dino', 'barlow', 'mocov3']
    
    print(f"\n{'='*80}")
    print(f"Stage 3: Parallel SSL Training")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rates: img={args.lr_img}, topo={args.lr_topo}, fusion={args.lr_fusion}")
    if args.debug:
        print(f"Mode: 🐛 DEBUG")
    print(f"{'='*80}\n")
    
    # Load data with automatic ROI fallback
    print("Loading dataset...")
    print(f"  ROI directory: {args.roi_dir}")
    print(f"  Will automatically fallback to traditional CV if SAM ROIs not found\n")
    
    imgs_train, labels_train, rois_train, using_sam_train = load_dataset_with_rois(
        args.dataset, 'train', roi_base_dir=args.roi_dir, seed=args.seed
    )
    imgs_val, labels_val, rois_val, using_sam_val = load_dataset_with_rois(
        args.dataset, 'val', roi_base_dir=args.roi_dir, seed=args.seed
    )
    imgs_test, labels_test, rois_test, using_sam_test = load_dataset_with_rois(
        args.dataset, 'test', roi_base_dir=args.roi_dir, seed=args.seed
    )
    
    # Print ROI info
    print(f"\n📊 ROI Information:")
    for split, rois, using_sam in [('train', rois_train, using_sam_train),
                                     ('val', rois_val, using_sam_val),
                                     ('test', rois_test, using_sam_test)]:
        roi_method = "SAM" if using_sam else "Traditional CV"
        coverages = [roi.sum() / roi.size * 100 for roi in rois[:100]]  # Sample
        print(f"  {split:5s}: {roi_method:15s} - Coverage {np.mean(coverages):.1f}±{np.std(coverages):.1f}%")
    print()
    
    # Merge train+val
    imgs_train = np.concatenate([imgs_train, imgs_val], axis=0)
    labels_train = np.concatenate([labels_train, labels_val], axis=0)
    rois_train = rois_train + rois_val
    
    # Keep raw copies for evaluation
    imgs_train_raw = imgs_train.copy()
    labels_train_raw = labels_train.copy()
    rois_train_raw = rois_train.copy()
    imgs_test_raw = imgs_test.copy()
    labels_test_raw = labels_test.copy()
    rois_test_raw = rois_test.copy()
    
    print(f"Training samples: {len(imgs_train)}")
    print(f"Test samples: {len(imgs_test)}\n")
    
    # Create datasets
    augmenter = TopologyAwareAugmentation()
    
    train_dataset = Stage3TopoImageDataset(
        imgs_train, labels_train, rois_train, augmenter,
        n_h0=args.n_h0, n_h1=args.n_h1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True
    )
    
    num_classes = get_num_classes(args.dataset)
    
    # Create models
    print("="*80)
    print("Creating 5 independent SSL models...")
    print("="*80 + "\n")
    
    models = {}
    encoders = {}
    
    for method in methods:
        model, encoder = create_method_model(method, args, device)
        models[method] = model
        encoders[method] = encoder
    
    print(f"\n✓ All 5 models created successfully!\n")
    # === ADD THIS CODE TO LOAD CHECKPOINTS ===
    if args.resume:
        checkpoint_dir = args.checkpoint_dir or os.path.join(args.save_dir, args.dataset)
        print(f"\n{'=' * 80}")
        print(f"Loading checkpoints from: {checkpoint_dir}")
        print(f"{'=' * 80}\n")

        start_epoch = 0
        for method in methods:
            checkpoint_path = os.path.join(checkpoint_dir, method, 'best_model.pt')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Load model weights
                if 'model_state_dict' in checkpoint:
                    models[method].load_state_dict(checkpoint['model_state_dict'])
                if 'encoder_state_dict' in checkpoint:
                    encoders[method].load_state_dict(checkpoint['encoder_state_dict'])

                epoch = checkpoint.get('epoch', 0)
                acc = checkpoint.get('probe_acc', 0)
                start_epoch = max(start_epoch, epoch)

                print(f"  ✓ {method.upper():8s}: Loaded from epoch {epoch} (acc: {acc:.2f}%)")
            else:
                print(f"  ✗ {method.upper():8s}: No checkpoint found, training from scratch")

        print(f"\n  Resuming from epoch {start_epoch}\n")
    else:
        start_epoch = 0
    
    # Create optimizers
    print("Creating optimizers and schedulers...")
    optimizers = {method: create_optimizer(models[method], method, args)
                 for method in methods}
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = min(20 * len(train_loader), num_training_steps // 10)
    
    schedulers = {
        method: get_cosine_schedule_with_warmup(
            optimizers[method], num_warmup_steps, num_training_steps
        )
        for method in methods
    }
    # === ADD THIS: Fast-forward schedulers if resuming ===
    if args.resume and start_epoch > 0:
        print(f"\n⏩ Fast-forwarding schedulers from epoch {start_epoch}...")
        steps_to_skip = start_epoch * len(train_loader)
        for method in methods:
            for _ in range(steps_to_skip):
                schedulers[method].step()
        print(f"  ✓ Skipped {steps_to_skip} scheduler steps\n")
    # === END OF FIX ===
    
    scalers = {method: GradScaler('cuda', enabled=args.use_amp) for method in methods}
    
    print(f"✓ Training setup complete!")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}\n")
    
    # Setup save directory
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # Tracking
    best_accs = {method: 0.0 for method in methods}
    results_history = {method: [] for method in methods}
    losses_history = {method: [] for method in methods}
    early_stop_counters = {method: 0 for method in methods}
    
    # Training loop
    print("="*80)
    print(f"Starting training for {args.epochs} epochs...")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        losses, all_gates = train_one_epoch(
            models, optimizers, schedulers, scalers,
            train_loader, args, device, epoch
        )
        
        # Track losses
        for method in methods:
            losses_history[method].append(losses[method])
        
        # Print losses
        loss_str = " | ".join([f"{m.upper()}:{losses[m]:.4f}" for m in methods])
        print(f"Epoch {epoch:3d}/{args.epochs}: {loss_str}")
        
        # Debug mode: detailed analysis every 2 epochs
        if args.debug and epoch % 2 == 0:
            print_loss_breakdown(epoch, losses, all_gates, methods)
        
        # Monitor experts
        if epoch % args.monitor_freq == 0:
            print(f"\n{'='*80}")
            print(f"Expert Gate Statistics - Epoch {epoch}")
            print('='*80)
            for method in methods:
                print_expert_statistics(epoch, all_gates[method], method)
            print()
        
        # Linear probe evaluation
        if epoch % args.probe_freq == 0 or epoch == args.epochs:
            print(f"\n{'='*80}")
            print(f"Linear Probe Evaluation - Epoch {epoch}")
            print('='*80)
            
            for method in methods:
                probe_acc = linear_probe_evaluation(
                    encoders[method],
                    imgs_train_raw, labels_train_raw, rois_train_raw,
                    imgs_test_raw, labels_test_raw, rois_test_raw,
                    n_h0=args.n_h0 * 3, n_h1=args.n_h1 * 3,
                    num_classes=num_classes,
                    device=device, epochs=10
                )
                print(f"  {method.upper():8s}: {probe_acc:.2f}%")
                
                results_history[method].append({
                    'epoch': epoch,
                    'train_loss': losses[method],
                    'probe_acc': probe_acc
                })
                
                # Save best
                if probe_acc > best_accs[method]:
                    best_accs[method] = probe_acc
                    early_stop_counters[method] = 0
                    
                    method_dir = os.path.join(save_dir, method)
                    os.makedirs(method_dir, exist_ok=True)
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': models[method].state_dict(),
                        'encoder_state_dict': encoders[method].state_dict(),
                        'probe_acc': probe_acc
                    }, os.path.join(method_dir, 'best_model.pt'))
                else:
                    early_stop_counters[method] += args.probe_freq
            
            print()
        
        # Check for early stopping (only in non-debug mode)
        if not args.debug and all(counter >= args.early_stop_patience 
                                   for counter in early_stop_counters.values()):
            print(f"\n⏹️  Early stopping at epoch {epoch}\n")
            break
    
    # Final results
    print(f"\n{'='*80}")
    print("Training Complete!")
    print('='*80)
    print("\nBest Linear Probe Accuracies:")
    for method in sorted(methods):
        print(f"  {method.upper():8s}: {best_accs[method]:.2f}%")
    print()
    
    # Save results
    for method in methods:
        method_dir = os.path.join(save_dir, method)
        with open(os.path.join(method_dir, 'results.json'), 'w') as f:
            json.dump(results_history[method], f, indent=2)
    
    # Save summary
    summary = {
        'dataset': args.dataset,
        'epochs_trained': epoch,
        'best_accuracies': best_accs,
        'args': vars(args)
    }
    
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Debug mode: save analysis
    if args.debug:
        save_debug_results(args, losses_history, best_accs, methods)
    
    print(f"✓ Results saved to: {save_dir}\n")
    print('='*80 + "\n")


if __name__ == '__main__':
    main()
