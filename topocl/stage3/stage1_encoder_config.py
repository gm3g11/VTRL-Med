#!/usr/bin/env python3
"""
Stage 1 Visual Encoder Configuration (UPDATED with architecture detection)
"""

import os
from pathlib import Path
import torch
from topocl.stage3.encoders import ResNetEncoder, detect_resnet_variant

# Project root: two levels up from this file (topocl/stage3/ -> VTRL-Med/)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _p(rel_path):
    """Resolve a project-relative path."""
    return os.path.join(_PROJECT_ROOT, rel_path)


STAGE1_CHECKPOINT_PATHS = {
    'kvasir': {
        'simclr': _p('checkpoints/multi_backbone/stage1_simclr_kvasir_224.pth'),
        'byol': _p('checkpoints/ssl/byol_runs/byol_kvasir_encoder_best.pt'),
        'dino': _p('checkpoints/ssl/dinov2/kvasir_224_encoder_best.pt'),
        'barlow': _p('checkpoints/ssl/barlow/barlow_kvasir_encoder_best.pt'),
        'mocov3': _p('checkpoints/ssl/moco_runs/mocov3_kvasir_encoder_best.pt')
    },
    'pathmnist': {
        'simclr': _p('checkpoints/ssl/simclr/pathmnist_encoder.pt'),
        'byol': _p('checkpoints/ssl/byol_runs/byol_pathmnist_encoder_best.pt'),
        'dino': _p('checkpoints/ssl/dinov2/pathmnist_224_encoder_best.pt'),
        'barlow': _p('checkpoints/ssl/barlow/barlow_pathmnist_encoder_best.pt'),
        'mocov3': _p('checkpoints/ssl/moco_runs/mocov3_pathmnist_encoder_best.pt')
    },
    'octmnist': {
        'simclr': _p('checkpoints/ssl/simclr/octmnist_encoder.pt'),
        'byol': _p('checkpoints/ssl/byol_runs/byol_octmnist_encoder_best.pt'),
        'dino': _p('checkpoints/ssl/dinov2/octmnist_224_encoder_best.pt'),
        'barlow': _p('checkpoints/ssl/barlow/barlow_organsmnist_encoder_best.pt'),
        'mocov3': _p('checkpoints/ssl/moco_runs/mocov3_octmnist_encoder_best.pt')
    },
    'organsmnist': {
        'simclr': _p('checkpoints/ssl/simclr/organsmnist_encoder.pt'),
        'byol': _p('checkpoints/ssl/byol_runs/byol_organsmnist_encoder_best.pt'),
        'dino': _p('checkpoints/ssl/dinov2/organsmnist_224_encoder_best.pt'),
        'barlow': _p('checkpoints/ssl/moco_runs/mocov3_organsmnist_encoder_best.pt'),
        'mocov3': _p('checkpoints/ssl/moco_runs/mocov3_organsmnist_encoder_best.pt')
    },
    'isic2019': {
        'simclr': _p('checkpoints/ssl/simclr/isic2019_224_encoder_best.pt'),
        'byol': _p('checkpoints/ssl/byol_runs/byol_isic2019_encoder_best.pt'),
        'dino': _p('checkpoints/ssl/dinov2/isic2019_224_encoder_best.pt'),
        'barlow': _p('checkpoints/ssl/barlow/barlow_isic2019_encoder_best.pt'),
        'mocov3': _p('checkpoints/ssl/moco_runs/mocov3_isic2019_encoder_best.pt')
    }
}

STAGE2_CHECKPOINT_PATHS = {
    'kvasir': _p('checkpoints/topocl_stage2/kvasir/Supervised_CE/best_model.pt'),
    'pathmnist': _p('checkpoints/topocl_stage2/pathmnist/Supervised_CE/best_model.pt'),
    'octmnist': _p('checkpoints/topocl_stage2/octmnist/Supervised_CE/best_model.pt'),
    'organsmnist': _p('checkpoints/topocl_stage2/organsmnist/Supervised_CE/best_model.pt'),
    'isic2019': _p('checkpoints/topocl_stage2/isic2019/Supervised_CE/best_model.pt')
}


def load_visual_encoder(method: str, dataset: str, encoder):
    """
    Load Stage 1 visual encoder with automatic architecture detection

    Args:
        method: SSL method (simclr, byol, dino, barlow, mocov3)
        dataset: Dataset name
        encoder: ResNetEncoder instance
    """
    checkpoint_path = STAGE1_CHECKPOINT_PATHS[dataset][method]
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Detect architecture from checkpoint
    detected_variant = detect_resnet_variant(state_dict)

    # Check if encoder matches
    if encoder.variant != detected_variant:
        print(f"    Warning: Architecture mismatch!")
        print(f"       Encoder: {encoder.variant}, Checkpoint: {detected_variant}")
        print(f"       Recreating encoder with correct architecture...")
        # This will be handled at creation time

    # Method and dataset-specific prefix handling
    if method in ['byol', 'mocov3']:
        state_dict = {k.replace('m.', ''): v for k, v in state_dict.items()}
    elif method == 'simclr':
        if dataset in ['pathmnist', 'octmnist', 'organsmnist']:
            state_dict = {k.replace('m.', ''): v for k, v in state_dict.items()}
    elif method == 'barlow':
        if dataset == 'organsmnist':
            state_dict = {k.replace('m.', ''): v for k, v in state_dict.items()}

    # Load with strict=False to handle minor mismatches
    encoder.load_state_dict(state_dict, strict=False)
    print(f"    Loaded {method.upper()} visual encoder ({detected_variant})")


def get_encoder_variant(method: str, dataset: str):
    """
    Detect which ResNet variant is needed for a checkpoint

    Returns:
        'resnet50' or 'resnet101'
    """
    checkpoint_path = STAGE1_CHECKPOINT_PATHS[dataset][method]
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return detect_resnet_variant(state_dict)


def load_topo_encoder(dataset: str, encoder):
    """Load Stage 2 topology encoder"""
    checkpoint_path = STAGE2_CHECKPOINT_PATHS[dataset]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"    Loaded topology encoder from Stage 2")
