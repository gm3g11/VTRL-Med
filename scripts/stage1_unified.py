#!/usr/bin/env python3
"""
Unified SAM + SimCLR Stage 1
Works with: MedMNIST, Kvasir, ISIC2019
All datasets return: (weak_view, strong_view, test_view, label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse
import warnings

warnings.filterwarnings("ignore")
from segment_anything import sam_model_registry


# ============================================================================
#  Dataset Loading (Unified Format)
# ============================================================================

def is_medmnist(dataset_name):
    """Check if dataset is MedMNIST."""
    medmnist_datasets = [
        'bloodmnist', 'pathmnist', 'dermamnist', 'octmnist',
        'pneumoniamnist', 'retinamnist', 'breastmnist', 'tissuemnist',
        'organamnist', 'organcmnist', 'organsmnist'
    ]
    return dataset_name in medmnist_datasets


def load_dataset(dataset_name, split, img_size, data_dir=None, seed=42):
    """
    Load dataset - all return (weak, strong, test, label) format.
    """
    
    if is_medmnist(dataset_name):
        # MedMNIST: use MedMNIST_SPICE class
        from vtrlmed.datasets.medmnist_dataset import MedMNIST_SPICE
        dataset = MedMNIST_SPICE(dataset_name, split, img_size, download=True)
        print(f"✓ Loaded MedMNIST: {dataset_name}")
        
    else:
        # Kvasir or ISIC2019
        from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE
        
        if data_dir is None:
            raise ValueError(f"--data_dir required for {dataset_name}")
        
        if dataset_name == 'kvasir':
            dataset = Kvasir_SPICE(data_dir, split, img_size, seed=seed)
        elif dataset_name == 'isic2019':
            dataset = ISIC2019_SPICE(data_dir, split, img_size, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"✓ Loaded custom dataset: {dataset_name} from {data_dir}")
    
    return dataset


def simclr_collate_fn(batch):
    """
    Collate function for SimCLR training.
    Input: batch of (weak, strong, test, label)
    Output: (weak_views, strong_views) for contrastive learning
    """
    weak, strong, _, _ = zip(*batch)
    return torch.stack(weak), torch.stack(strong)


# ============================================================================
#  Model Components (Unchanged)
# ============================================================================

class SAMBackbone(nn.Module):
    """SAM ViT backbone for feature extraction."""
    
    def __init__(self, sam_model_type='vit_h', sam_checkpoint='./sam_vit_h_4b8939.pth', 
                 unfrozen_blocks=4):
        super().__init__()
        
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder
        
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if unfrozen_blocks > 0:
            if hasattr(self.backbone, 'neck'):
                for param in self.backbone.neck.parameters():
                    param.requires_grad = True
            
            for block in self.backbone.blocks[-unfrozen_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"SAM {sam_model_type}: feature_dim={self.feature_dim}")
        print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    def forward(self, x):
        features = self.backbone.patch_embed(x)
        
        if features.shape[1:3] != self.backbone.pos_embed.shape[1:3]:
            pos_embed = F.interpolate(
                self.backbone.pos_embed.permute(0, 3, 1, 2),
                size=features.shape[1:3], mode='bicubic', align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            pos_embed = self.backbone.pos_embed
        
        features = features + pos_embed
        
        for blk in self.backbone.blocks:
            features = blk(features)
        
        pooled_features = F.adaptive_avg_pool2d(
            features.permute(0, 3, 1, 2), 1
        ).flatten(1)
        
        return pooled_features


class SimCLRProjectionHead(nn.Module):
    """SimCLR projection head."""
    
    def __init__(self, input_dim, hidden_dim=2048, output_dim=256):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.projector(x)


class SAMSimCLR(nn.Module):
    """SAM + SimCLR model."""
    
    def __init__(self, sam_model_type='vit_h', sam_checkpoint='./sam_vit_h_4b8939.pth',
                 unfrozen_blocks=4, proj_hidden_dim=2048, proj_output_dim=256):
        super().__init__()
        
        self.backbone = SAMBackbone(sam_model_type, sam_checkpoint, unfrozen_blocks)
        self.projector = SimCLRProjectionHead(
            self.backbone.feature_dim, proj_hidden_dim, proj_output_dim
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def nt_xent_loss(z_i, z_j, temperature=0.1):
    """NT-Xent loss for SimCLR."""
    batch_size = z_i.shape[0]
    
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    z = torch.cat([z_i, z_j], dim=0)
    
    sim_matrix = torch.mm(z, z.t()) / temperature
    labels = torch.cat([torch.arange(batch_size) + batch_size, 
                       torch.arange(batch_size)]).to(z.device)
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix.masked_fill_(mask, float('-inf'))
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# ============================================================================
#  Training
# ============================================================================

def train_sam_simclr(args):
    """Train SAM + SimCLR model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("="*80)
    train_dataset = load_dataset(
        args.dataset, 'train', args.img_size, args.data_dir, args.seed
    )
    print("="*80 + "\n")
    
    # Build sampler
    sampler = None
    shuffle = True
    if args.balanced_sampling and hasattr(train_dataset, 'samples'):
        labels = [s[1] for s in train_dataset.samples]
        class_counts = {}
        for l in labels:
            class_counts[l] = class_counts.get(l, 0) + 1
        weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
        shuffle = False
        print(f"Balanced sampling enabled: {len(class_counts)} classes, "
              f"min={min(class_counts.values())}, max={max(class_counts.values())}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=simclr_collate_fn
    )
    
    # Build model
    print("Building model...")
    model = SAMSimCLR(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        unfrozen_blocks=args.unfrozen_blocks,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim
    ).to(device)
    print()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    
    # Mixed precision
    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("Mixed precision training enabled")
    else:
        scaler = None
    
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"Temperature: {args.temperature}")
    print("="*80 + "\n")
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for weak_view, strong_view in progress_bar:
            weak_view = weak_view.to(device, non_blocking=True)
            strong_view = strong_view.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if args.use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
                    _, proj1 = model(weak_view)
                    _, proj2 = model(strong_view)
                    loss = nt_xent_loss(proj1, proj2, temperature=args.temperature)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, proj1 = model(weak_view)
                _, proj2 = model(strong_view)
                loss = nt_xent_loss(proj1, proj2, temperature=args.temperature)
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    # Save checkpoint (backbone only)
    output_path = os.path.join(
        args.output_dir, 
        f"stage1_sam_simclr_{args.dataset}_{args.img_size}.pth"
    )
    
    backbone_state_dict = model.backbone.backbone.state_dict()
    torch.save(backbone_state_dict, output_path)
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"Saved: {output_path}")
    print(f"Feature dim: {model.backbone.feature_dim}")
    print("Ready for Stage 2/3")
    print(f"{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser('Unified SAM+SimCLR Stage 1')
    
    # Dataset
    parser.add_argument('--dataset', required=True, type=str,
                        help='Dataset name (e.g., pathmnist, kvasir, isic2019)')
    parser.add_argument('--data_dir', default=None, type=str,
                        help='Data directory (required for kvasir/isic2019)')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--output_dir', default='./checkpoints', type=str)
    parser.add_argument('--seed', default=42, type=int)
    
    # Model
    parser.add_argument('--sam_model_type', default='vit_h', type=str, 
                        choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--sam_checkpoint', default='./sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--unfrozen_blocks', default=4, type=int)
    parser.add_argument('--proj_hidden_dim', default=2048, type=int)
    parser.add_argument('--proj_output_dim', default=256, type=int)
    
    # Training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--balanced_sampling', action='store_true',
                        help='Use class-balanced sampling (for imbalanced datasets)')
    parser.add_argument('--num_workers', default=16, type=int)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_sam_simclr(args)
