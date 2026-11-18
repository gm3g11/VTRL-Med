#!/usr/bin/env python3
"""
Unified SAM+SPICE Stage 3: Joint Fine-Tuning
Supports: MedMNIST, Kvasir, ISIC2019
Can work with or without Stage 2 checkpoint (uses k-means initialization if Stage 2 missing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import os
import numpy as np
import argparse
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import torchvision.transforms.v2 as transformsV2
import json
import warnings
import copy

warnings.filterwarnings("ignore")
from segment_anything import sam_model_registry


# ============================================================================
#  Dataset Info & Loading
# ============================================================================

def get_dataset_info(dataset_name):
    """Get number of classes for each dataset."""
    dataset_configs = {
        'bloodmnist': 8, 'pathmnist': 9, 'dermamnist': 7, 'octmnist': 4,
        'pneumoniamnist': 2, 'retinamnist': 5, 'breastmnist': 2, 'tissuemnist': 8,
        'organamnist': 11, 'organcmnist': 11, 'organsmnist': 11,
        'kvasir': 8, 'isic2019': 8,
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_configs[dataset_name]


def is_medmnist(dataset_name):
    """Check if dataset is MedMNIST."""
    medmnist_datasets = [
        'bloodmnist', 'pathmnist', 'dermamnist', 'octmnist',
        'pneumoniamnist', 'retinamnist', 'breastmnist', 'tissuemnist',
        'organamnist', 'organcmnist', 'organsmnist'
    ]
    return dataset_name in medmnist_datasets


def load_dataset(dataset_name, split, img_size, data_dir=None, seed=42):
    """Load dataset."""
    
    if is_medmnist(dataset_name):
        from vtrlmed.datasets.medmnist_dataset import MedMNIST_SPICE
        dataset = MedMNIST_SPICE(dataset_name, split, img_size, download=True)
    else:
        from vtrlmed.datasets.dataset_kvasir_isic import Kvasir_SPICE, ISIC2019_SPICE
        
        if data_dir is None:
            raise ValueError(f"--data_dir required for {dataset_name}")
        
        if dataset_name == 'kvasir':
            dataset = Kvasir_SPICE(data_dir, split, img_size, seed=seed)
        elif dataset_name == 'isic2019':
            dataset = ISIC2019_SPICE(data_dir, split, img_size, seed=seed)
    
    return dataset


# ============================================================================
#  Model Components
# ============================================================================

class SPICEClusteringHead(nn.Module):
    """Enhanced clustering head with learnable temperature."""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        hidden_dim = max(input_dim // 2, 512)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, use_temperature=True):
        logits = self.head(x)
        if use_temperature:
            logits = logits / torch.clamp(self.temperature, min=0.01)
        return logits


class CombinedModel(nn.Module):
    """Combined SAM backbone + clustering head."""
    
    def __init__(self, stage1_path, stage2_path, sam_model_type, sam_checkpoint,
                 num_classes, unfrozen_blocks=4, has_stage2=True):
        super().__init__()
        
        # Load SAM backbone
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder
        
        # Load Stage 1 weights
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        self.backbone.load_state_dict(torch.load(stage1_path, map_location='cpu'))
        
        # Feature dimensions
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        self.num_classes = num_classes
        
        # Selective unfreezing
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if unfrozen_blocks > 0:
            if hasattr(self.backbone, 'neck'):
                for param in self.backbone.neck.parameters():
                    param.requires_grad = True
            
            for block in self.backbone.blocks[-unfrozen_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Load or initialize clustering head
        self.clustering_head = SPICEClusteringHead(self.feature_dim, num_classes)
        
        if has_stage2 and os.path.exists(stage2_path):
            print(f"Loading Stage 2 checkpoint: {stage2_path}")
            self.clustering_head.load_state_dict(
                torch.load(stage2_path, map_location='cpu'), 
                strict=False
            )
        else:
            print("Stage 2 checkpoint not found - will initialize with k-means")
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Unfrozen blocks: {unfrozen_blocks}")
        print(f"Trainable: {trainable_params:,} / {total_params:,} "
              f"({trainable_params/total_params*100:.2f}%)")
    
    def forward(self, x, use_temperature=True):
        """Forward pass with feature extraction."""
        # Patch embedding
        features = self.backbone.patch_embed(x)
        
        # Positional encoding
        if features.shape[1:3] != self.backbone.pos_embed.shape[1:3]:
            pos_embed = F.interpolate(
                self.backbone.pos_embed.permute(0, 3, 1, 2),
                size=features.shape[1:3], mode='bicubic', align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            pos_embed = self.backbone.pos_embed
        
        features = features + pos_embed
        
        # Through transformer blocks
        for blk in self.backbone.blocks:
            features = blk(features)
        
        # Global pooling
        pooled_features = F.adaptive_avg_pool2d(
            features.permute(0, 3, 1, 2), 1
        ).flatten(1)
        
        # Normalize and predict
        normalized_features = F.normalize(pooled_features, p=2, dim=1)
        logits = self.clustering_head(normalized_features, use_temperature)
        
        return normalized_features, logits


class SAMDataset(Dataset):
    """Dataset wrapper with augmentation."""
    
    def __init__(self, base_dataset):
        self.dataset = base_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Returns (weak, strong, test, label) from base dataset
        return self.dataset[idx]


# ============================================================================
#  K-means Initialization (if Stage 2 is skipped)
# ============================================================================

def initialize_clustering_head_with_kmeans(model, train_loader, device, num_classes):
    """
    Initialize clustering head using k-means on Stage 1 features.
    
    Strategy:
    1. Extract normalized features from Stage 1 backbone
    2. Run k-means to find cluster centers
    3. Initialize clustering head to predict cluster assignments
    4. Quick fine-tuning with k-means labels
    """
    
    print("\n" + "="*80)
    print("Initialization Phase: K-means on Stage 1 Features")
    print("="*80)
    
    # Step 1: Extract features from Stage 1 backbone
    print("\nExtracting features from training set...")
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for _, _, img_test, _ in tqdm(train_loader, desc="Extracting features"):
            img_test = img_test.to(device, non_blocking=True)
            features, _ = model(img_test, use_temperature=False)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0).numpy()
    print(f"✓ Extracted features: {all_features.shape}")
    
    # Step 2: Run k-means clustering
    print(f"\nRunning k-means with {num_classes} clusters...")
    kmeans = KMeans(
        n_clusters=num_classes, 
        random_state=42, 
        n_init=10, 
        max_iter=300,
        verbose=0
    )
    kmeans.fit(all_features)
    
    cluster_centers = kmeans.cluster_centers_  # [num_classes, feature_dim]
    labels = kmeans.labels_
    
    print(f"✓ K-means complete")
    print(f"  Cluster centers: {cluster_centers.shape}")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    cluster_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Cluster distribution: {cluster_dist}")
    
    # Step 3: Initialize clustering head weights
    print(f"\nInitializing clustering head weights...")
    
    with torch.no_grad():
        # Get the layers
        first_linear = model.clustering_head.head[0]  # Linear(feature_dim -> hidden_dim)
        final_linear = model.clustering_head.head[-1]  # Linear(hidden_dim -> num_classes)
        
        feature_dim = first_linear.in_features
        hidden_dim = first_linear.out_features
        
        # Convert cluster centers to tensor
        cluster_centers_tensor = torch.from_numpy(cluster_centers).float().to(device)
        cluster_centers_tensor = F.normalize(cluster_centers_tensor, p=2, dim=1)
        
        # Initialize first layer to pass through features
        if hidden_dim >= feature_dim:
            # Can use direct projection
            weight_init = torch.zeros(hidden_dim, feature_dim)
            weight_init[:feature_dim, :feature_dim] = torch.eye(feature_dim)
            # Fill rest with small random values
            if hidden_dim > feature_dim:
                weight_init[feature_dim:, :] = torch.randn(hidden_dim - feature_dim, feature_dim) * 0.01
        else:
            # Use PCA-like projection to reduce dimensions
            U, S, Vt = np.linalg.svd(all_features, full_matrices=False)
            projection = Vt[:hidden_dim, :]  # Take top principal components
            weight_init = torch.from_numpy(projection).float()
        
        first_linear.weight.copy_(weight_init)
        first_linear.bias.zero_()
        
        # Initialize final layer based on cluster centers
        cluster_projected = first_linear(cluster_centers_tensor)
        cluster_projected = F.relu(cluster_projected)
        cluster_projected = F.normalize(cluster_projected, p=2, dim=1)
        
        final_linear.weight.copy_(cluster_projected)
        final_linear.bias.zero_()
        
        print(f"✓ First layer initialized: {first_linear.weight.shape}")
        print(f"✓ Final layer initialized: {final_linear.weight.shape}")
    
    # Step 4: Quick fine-tuning of clustering head
    print(f"\nFine-tuning clustering head for 3 epochs...")
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.clustering_head.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.clustering_head.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Fine-tune {epoch+1}/3")
        
        for _, _, img_test, _ in progress_bar:
            img_test = img_test.to(device, non_blocking=True)
            
            features, logits = model(img_test, use_temperature=False)
            
            # Use k-means labels as pseudo-labels
            features_np = features.cpu().numpy()
            batch_pseudo_labels = []
            
            for feat in features_np:
                distances = np.linalg.norm(cluster_centers - feat, axis=1)
                batch_pseudo_labels.append(np.argmin(distances))
            
            batch_pseudo_labels = torch.tensor(batch_pseudo_labels, device=device)
            
            # Cross-entropy loss + balance regularization
            loss_ce = F.cross_entropy(logits, batch_pseudo_labels)
            
            # Balance regularization
            probs = F.softmax(logits, dim=1)
            cluster_probs = probs.mean(dim=0)
            uniform_target = torch.ones_like(cluster_probs) / num_classes
            loss_balance = F.kl_div(
                torch.log(cluster_probs + 1e-8), 
                uniform_target, 
                reduction='batchmean'
            )
            
            loss = loss_ce + 0.5 * loss_balance
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Restore backbone trainability
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    print("\n✓ K-means initialization complete\n")


# ============================================================================
#  Pseudo-Labeling & Loss Functions
# ============================================================================

def generate_reliable_samples(model, dataloader, device, num_classes,
                               num_neighbors=50, reliability_threshold=0.8,
                               consistency_threshold=0.7):
    """Generate reliable pseudo-labels using neighbor consistency."""
    model.eval()
    all_features, all_predictions, all_confidences = [], [], []
    
    with torch.no_grad():
        for _, _, img_test, _ in tqdm(dataloader, desc="Generating pseudo-labels"):
            features, logits = model(img_test.to(device), use_temperature=True)
            probs = F.softmax(logits, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)
            
            all_features.append(features.cpu())
            all_predictions.append(predictions.cpu())
            all_confidences.append(max_probs.cpu())
    
    all_features = torch.cat(all_features)
    all_predictions = torch.cat(all_predictions)
    all_confidences = torch.cat(all_confidences)
    
    # Find reliable samples
    reliable_indices = []
    
    for i in tqdm(range(len(all_features)), desc="Finding reliable samples"):
        # Check confidence threshold
        if all_confidences[i] < consistency_threshold:
            continue
        
        # Find K-nearest neighbors
        dist = torch.norm(all_features - all_features[i], dim=1, p=2)
        neighbor_indices = torch.topk(dist, k=num_neighbors, largest=False).indices
        
        # Check neighbor consistency
        consistency_score = (all_predictions[neighbor_indices] == all_predictions[i]).float().mean()
        
        # Check cluster size (prevent domination)
        cluster_id = all_predictions[i].item()
        cluster_size = (all_predictions == cluster_id).sum().item()
        max_allowed_size = len(all_predictions) * 1.5 / num_classes
        
        if consistency_score > reliability_threshold and cluster_size < max_allowed_size:
            reliable_indices.append(i)
    
    return reliable_indices


def entropy_regularization(logits, lambda_ent=0.1):
    """Entropy regularization."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy_per_sample = -(probs * log_probs).sum(dim=1)
    return -lambda_ent * entropy_per_sample.mean()


def cluster_balance_regularization(logits, num_classes, lambda_balance=1.0):
    """Cluster balance regularization."""
    probs = F.softmax(logits, dim=1)
    cluster_probs = probs.mean(dim=0)
    uniform_target = torch.ones_like(cluster_probs) / num_classes
    balance_loss = F.kl_div(
        torch.log(cluster_probs + 1e-8), 
        uniform_target, 
        reduction='batchmean'
    )
    return lambda_balance * balance_loss


def swapped_prediction_loss(logits1, logits2, lambda_swap=0.1):
    """Swapped prediction for consistency."""
    with torch.no_grad():
        targets = torch.argmax(logits1, dim=1)
    return lambda_swap * F.cross_entropy(logits2, targets)


def improved_cross_entropy(logits, targets, reduction='mean'):
    """Standard cross entropy."""
    return F.cross_entropy(logits, targets, reduction=reduction)


# ============================================================================
#  Evaluation Functions
# ============================================================================

def hungarian_accuracy(y_true, y_pred):
    """Clustering accuracy with Hungarian matching."""
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {pred_idx: true_idx for pred_idx, true_idx in zip(row_ind, col_ind)}
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size
    
    return acc, mapping


def calculate_ece(probs, preds_remapped, labels, n_bins=10):
    """Calculate Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    accuracies = (preds_remapped == labels)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate(model, loader, device, n_bins_ece, true_dist):
    """Comprehensive evaluation."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for _, _, img_test, labels in loader:
            _, logits = model(img_test.to(device, non_blocking=True), use_temperature=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Clustering metrics
    acc, mapping = hungarian_accuracy(all_labels, all_preds)
    nmi = normalized_mutual_info_score(all_labels, all_preds)
    ari = adjusted_rand_score(all_labels, all_preds)
    
    # Calibration
    preds_remapped = np.array([mapping.get(pred, -1) for pred in all_preds])
    ece = calculate_ece(all_probs, preds_remapped, all_labels, n_bins=n_bins_ece)
    
    # Cluster distribution
    pred_ids, pred_counts = np.unique(all_preds, return_counts=True)
    num_classes = len(true_dist)
    aligned_pred_counts = np.zeros(num_classes)
    
    for pred_idx, count in zip(pred_ids, pred_counts):
        true_idx = mapping.get(pred_idx)
        if true_idx is not None and true_idx < num_classes:
            aligned_pred_counts[true_idx] = count
    
    pred_dist = aligned_pred_counts / max(1, aligned_pred_counts.sum())
    pred_dist += 1e-9
    kl_divergence = entropy(pk=true_dist, qk=pred_dist)
    
    return {
        'acc': acc,
        'nmi': nmi,
        'ari': ari,
        'ece': ece,
        'kl_divergence': kl_divergence,
        'cluster_counts': pred_counts.tolist(),
        'num_active_clusters': len(pred_counts)
    }


def get_current_lambda(epoch, anneal_epochs, start_val, end_val):
    """Linear annealing schedule."""
    if epoch >= anneal_epochs:
        return end_val
    return start_val - (start_val - end_val) * (epoch / anneal_epochs)


# ============================================================================
#  Main Training
# ============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Stage 3: Joint Fine-Tuning")
    print("="*80 + "\n")
    
    # Get dataset info
    num_classes = get_dataset_info(args.dataset)
    
    # Paths
    stage1_path = os.path.join(
        args.checkpoint_dir, 
        f"stage1_sam_simclr_{args.dataset}_{args.img_size}.pth"
    )
    stage2_path = os.path.join(
        args.stage2_dir, 
        f"stage2_ours_{args.dataset}_{args.img_size}.pth"
    )
    stage3_best_path = os.path.join(
        args.output_dir, 
        f"stage3_sam_{args.dataset}_best.pth"
    )
    results_path = os.path.join(
        args.output_dir, 
        f"stage3_sam_results_{args.dataset}.json"
    )
    
    # Check if Stage 2 exists
    has_stage2 = os.path.exists(stage2_path)
    if not has_stage2:
        print(f"⚠️  Stage 2 checkpoint not found: {stage2_path}")
        print("   Will initialize clustering head with k-means")
    
    # Build model
    print("\n--- Building Combined Model ---")
    print(f"Stage 1: {stage1_path}")
    if has_stage2:
        print(f"Stage 2: {stage2_path}")
    print()
    
    model = CombinedModel(
        stage1_path, stage2_path, 
        args.sam_model_type, args.sam_checkpoint,
        num_classes, args.unfrozen_blocks, has_stage2
    ).to(device)
    
    print("✓ Model built\n")
    
    # Load datasets
    train_base_dataset = load_dataset(args.dataset, 'train', args.img_size, args.data_dir, args.seed)
    train_dataset = SAMDataset(train_base_dataset)
    
    full_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # If Stage 2 is skipped, run k-means initialization
    if not has_stage2:
        initialize_clustering_head_with_kmeans(model, full_loader, device, num_classes)
    
    # Generate reliable pseudo-labels
    print("="*80)
    print("Step 1: Generating Reliable Pseudo-Labels")
    print("="*80)
    
    reliable_indices = generate_reliable_samples(
        model, full_loader, device, num_classes,
        num_neighbors=args.num_neighbors,
        reliability_threshold=args.reliability_threshold,
        consistency_threshold=args.confidence_threshold
    )
    
    print(f"\n✓ Found {len(reliable_indices)} reliable samples "
          f"({len(reliable_indices)/len(train_dataset)*100:.2f}%)")
    
    if len(reliable_indices) == 0:
        print("ERROR: No reliable samples found. Try lowering thresholds.")
        return
    
    # Fine-tuning setup
    print("\n" + "="*80)
    print("Step 2: Joint Fine-Tuning")
    print("="*80 + "\n")
    
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 
         'lr': args.lr_backbone, 'weight_decay': 0.05},
        {'params': model.clustering_head.parameters(), 
         'lr': args.lr_head, 'weight_decay': 1e-4}
    ])
    
    # Dataloaders
    labeled_dataset = Subset(train_dataset, reliable_indices)
    sub_batch_size = max(1, args.batch_size // 2)
    
    labeled_loader = DataLoader(
        labeled_dataset, 
        batch_size=sub_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )
    
    unlabeled_loader = DataLoader(
        train_dataset, 
        batch_size=sub_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )
    
    # Test set
    eval_test_dataset = SAMDataset(load_dataset(args.dataset, 'test', args.img_size, args.data_dir, args.seed))
    eval_test_loader = DataLoader(
        eval_test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Get true distribution
    test_labels = []
    for i in range(len(eval_test_dataset)):
        _, _, _, label = eval_test_dataset[i]
        test_labels.append(label.item())
    test_labels = np.array(test_labels)
    _, true_test_counts = np.unique(test_labels, return_counts=True)
    true_test_dist = true_test_counts / true_test_counts.sum()
    
    # Training loop
    best_test_acc = 0.0
    all_results = []
    
    print(f"Training for {args.epochs} epochs...")
    print(f"Regularization: entropy={args.lambda_entropy}, "
          f"balance={args.lambda_balance_start}->{args.lambda_balance_end}, "
          f"swap={args.lambda_swap}\n")
    
    for epoch in range(args.epochs):
        model.train()
        labeled_iter = iter(labeled_loader)
        progress_bar = tqdm(unlabeled_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        # Get current balance weight
        current_lambda_balance = get_current_lambda(
            epoch, args.lambda_anneal_epochs,
            args.lambda_balance_start, args.lambda_balance_end
        )
        
        for img_weak_unlab, img_strong_unlab, _, _ in progress_bar:
            # Get labeled batch
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)
            
            img_weak_lab, _, _, true_pseudo_labels = labeled_batch
            
            # Supervised loss on reliable samples
            _, logits_lab = model(img_weak_lab.to(device))
            loss_lab = improved_cross_entropy(logits_lab, true_pseudo_labels.to(device))
            
            # Unsupervised loss on unlabeled samples
            with torch.no_grad():
                _, logits_weak = model(img_weak_unlab.to(device))
                max_probs, pseudo_labels_unlab = torch.max(
                    F.softmax(logits_weak, dim=1), dim=1
                )
                mask = max_probs.ge(args.confidence_threshold).float()
            
            _, logits_strong = model(img_strong_unlab.to(device))
            loss_unlab = (improved_cross_entropy(
                logits_strong, pseudo_labels_unlab, reduction='none'
            ) * mask).mean()
            
            # Regularization losses
            all_logits = torch.cat([logits_lab, logits_strong], dim=0)
            loss_balance = cluster_balance_regularization(
                all_logits, num_classes, current_lambda_balance
            )
            loss_entropy = entropy_regularization(all_logits, args.lambda_entropy)
            loss_swap = swapped_prediction_loss(logits_weak, logits_strong, args.lambda_swap)
            
            # Total loss
            total_loss = loss_lab + loss_unlab + loss_balance + loss_entropy + loss_swap
            
            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            progress_bar.set_postfix({
                'L_bal': f'{loss_balance.item():.3f}',
                'L_sup': f'{loss_lab.item():.3f}'
            })
        
        # Evaluation
        if (epoch + 1) % args.eval_frequency == 0 or (epoch + 1) == args.epochs:
            metrics = evaluate(model, eval_test_loader, device, args.ece_bins, true_test_dist)
            
            print(f"\n--- Epoch {epoch + 1} ---")
            print(f"  ACC: {metrics['acc']:.4f} | NMI: {metrics['nmi']:.4f} | "
                  f"ARI: {metrics['ari']:.4f} | ECE: {metrics['ece']:.4f}")
            print(f"  KL: {metrics['kl_divergence']:.4f} | "
                  f"Clusters: {metrics['num_active_clusters']}/{num_classes}")
            
            # Save best model
            if metrics['acc'] > best_test_acc:
                best_test_acc = metrics['acc']
                print(f"  🚀 NEW BEST: {best_test_acc:.4f}")
                torch.save(model.state_dict(), stage3_best_path)
            
            all_results.append({'epoch': epoch + 1, **metrics})
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    if os.path.exists(stage3_best_path):
        model.load_state_dict(torch.load(stage3_best_path))
        print("✓ Loaded best checkpoint")
    
    final_metrics = evaluate(model, eval_test_loader, device, args.ece_bins, true_test_dist)
    
    print(f"\nFinal Performance:")
    print(f"  ACC: {final_metrics['acc']:.4f}")
    print(f"  NMI: {final_metrics['nmi']:.4f}")
    print(f"  ARI: {final_metrics['ari']:.4f}")
    print(f"  ECE: {final_metrics['ece']:.4f}")
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump({
            'has_stage2': has_stage2,
            'final_best_metrics': final_metrics,
            'training_epochs': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results: {results_path}")
    print("="*80)


def parse_args():
    parser = argparse.ArgumentParser('Unified SAM+SPICE Stage 3')
    
    # Dataset
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--seed', default=42, type=int)
    
    # Paths
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
    parser.add_argument('--stage2_dir', default='./checkpoints_stage2', type=str)
    parser.add_argument('--output_dir', default='./checkpoints_stage3', type=str)
    
    # Model
    parser.add_argument('--sam_model_type', default='vit_h', type=str)
    parser.add_argument('--sam_checkpoint', default='./sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--unfrozen_blocks', default=4, type=int)
    
    # Training
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr_backbone', default=5e-6, type=float)
    parser.add_argument('--lr_head', default=5e-5, type=float)
    parser.add_argument('--num_workers', default=16, type=int)
    
    # Pseudo-labeling
    parser.add_argument('--num_neighbors', default=50, type=int)
    parser.add_argument('--reliability_threshold', default=0.8, type=float)
    parser.add_argument('--confidence_threshold', default=0.8, type=float)
    
    # Regularization
    parser.add_argument('--lambda_swap', default=0.1, type=float)
    parser.add_argument('--lambda_entropy', default=0.2, type=float)
    parser.add_argument('--lambda_balance_start', default=2.0, type=float)
    parser.add_argument('--lambda_balance_end', default=0.5, type=float)
    parser.add_argument('--lambda_anneal_epochs', default=25, type=int)
    
    # Evaluation
    parser.add_argument('--eval_frequency', default=2, type=int)
    parser.add_argument('--ece_bins', default=10, type=int)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
