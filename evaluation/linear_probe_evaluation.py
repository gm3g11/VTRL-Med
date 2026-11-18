#!/usr/bin/env python3
"""
Linear Probe Evaluation for SAM+SimCLR Stage 1
Evaluates quality of learned features by training a linear classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import torchvision.transforms.v2 as transformsV2
import json
import warnings

warnings.filterwarnings("ignore")
from segment_anything import sam_model_registry


class FrozenSAMFeatureExtractor(nn.Module):
    """Frozen SAM backbone for feature extraction"""
    
    def __init__(self, stage1_checkpoint_path, sam_model_type='vit_h', 
                 sam_checkpoint='./sam_vit_h_4b8939.pth'):
        super().__init__()
        
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.backbone = sam_model.image_encoder
        
        print(f"Loading Stage 1 weights from: {stage1_checkpoint_path}")
        self.backbone.load_state_dict(torch.load(stage1_checkpoint_path, map_location='cpu'))
        
        feature_dims = {'vit_b': 768, 'vit_l': 1024, 'vit_h': 1280}
        self.feature_dim = feature_dims[sam_model_type]
        
        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        print(f"SAM backbone frozen, feature dim: {self.feature_dim}")
    
    def forward(self, x):
        with torch.no_grad():
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
            
            return F.normalize(pooled_features, p=2, dim=1)


class LinearClassifier(nn.Module):
    """Simple linear classifier for probing"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        return self.fc(x)


class MedMNISTDataset:
    """Dataset wrapper for MedMNIST"""
    
    def __init__(self, dataset_name, split='train', img_size=224):
        import medmnist
        from medmnist import INFO
        
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(split=split, download=True, transform=None, size=224)
        
        # Standard ImageNet normalization
        self.transform = transformsV2.Compose([
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.float32, scale=True),
            transformsV2.Resize((img_size, img_size), antialias=True),
            transformsV2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self.transform(image), torch.tensor(label).long().squeeze()


def extract_features(model, dataloader, device):
    """Extract features for entire dataset"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            features = model(images.to(device, non_blocking=True))
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels


def train_linear_classifier(train_features, train_labels, val_features, val_labels, 
                            num_classes, device, args):
    """Train linear classifier on frozen features"""
    
    # Create dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.probe_batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.probe_batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize classifier
    classifier = LinearClassifier(train_features.shape[1], num_classes).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.probe_lr, 
                                  weight_decay=args.probe_weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.probe_epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_accs = []
    
    for epoch in range(args.probe_epochs):
        classifier.train()
        epoch_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = classifier(features)
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = classifier.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.probe_epochs} - Loss: {avg_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")
    
    # Restore best model
    classifier.load_state_dict(best_model_state)
    
    return classifier, train_losses, val_accs, best_val_acc


def evaluate_classifier(classifier, features, labels, device, num_classes):
    """Evaluate classifier and compute metrics"""
    classifier.eval()
    
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for feats, labs in loader:
            feats = feats.to(device)
            logits = classifier(feats)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labs.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    class_accuracies = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    return accuracy, class_accuracies, all_preds, all_labels


def compute_feature_quality_metrics(features, labels):
    """Compute feature quality metrics"""
    features_np = features.numpy()
    labels_np = labels.numpy()
    
    # Silhouette score (higher is better, range [-1, 1])
    if len(np.unique(labels_np)) > 1:
        silhouette = silhouette_score(features_np, labels_np, metric='cosine', sample_size=10000)
    else:
        silhouette = 0.0
    
    # k-NN accuracy
    knn = KNeighborsClassifier(n_neighbors=10, metric='cosine', n_jobs=-1)
    
    # Use 80-20 split for k-NN
    n_train = int(0.8 * len(features_np))
    train_feats, train_labs = features_np[:n_train], labels_np[:n_train]
    test_feats, test_labs = features_np[n_train:], labels_np[n_train:]
    
    knn.fit(train_feats, train_labs)
    knn_preds = knn.predict(test_feats)
    knn_acc = (knn_preds == test_labs).mean()
    
    # Intra-class vs inter-class distance ratio (lower is better)
    num_classes = len(np.unique(labels_np))
    intra_distances = []
    inter_distances = []
    
    for c in range(num_classes):
        class_mask = labels_np == c
        class_feats = features_np[class_mask]
        
        if len(class_feats) > 1:
            # Intra-class distances
            class_center = class_feats.mean(axis=0, keepdims=True)
            intra_dist = np.linalg.norm(class_feats - class_center, axis=1).mean()
            intra_distances.append(intra_dist)
            
            # Inter-class distances
            for c2 in range(num_classes):
                if c2 != c:
                    other_mask = labels_np == c2
                    other_feats = features_np[other_mask]
                    if len(other_feats) > 0:
                        other_center = other_feats.mean(axis=0, keepdims=True)
                        inter_dist = np.linalg.norm(class_center - other_center)
                        inter_distances.append(inter_dist)
    
    avg_intra = np.mean(intra_distances) if intra_distances else 0.0
    avg_inter = np.mean(inter_distances) if inter_distances else 1.0
    separation_ratio = avg_intra / (avg_inter + 1e-8)
    
    return {
        'silhouette_score': silhouette,
        'knn_accuracy': knn_acc,
        'avg_intra_dist': avg_intra,
        'avg_inter_dist': avg_inter,
        'separation_ratio': separation_ratio
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*80)
    
    # Determine number of classes
    num_classes = args.dataset_num_classes[args.dataset]
    
    # Load Stage 1 checkpoint
    stage1_checkpoint = os.path.join(
        args.checkpoint_dir, 
        f"stage1_sam_simclr_{args.dataset}_{args.img_size}_blocks{args.unfrozen_blocks}.pth"
    )
    
    if not os.path.exists(stage1_checkpoint):
        print(f"ERROR: Checkpoint not found: {stage1_checkpoint}")
        return
    
    print(f"Evaluating: {args.dataset} with {args.unfrozen_blocks} unfrozen blocks")
    print(f"Checkpoint: {stage1_checkpoint}")
    print("="*80)
    
    # Build feature extractor
    feature_extractor = FrozenSAMFeatureExtractor(
        stage1_checkpoint, 
        args.sam_model_type,
        args.sam_checkpoint
    ).to(device)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MedMNISTDataset(args.dataset, 'train', args.img_size)
    val_dataset = MedMNISTDataset(args.dataset, 'val', args.img_size)
    test_dataset = MedMNISTDataset(args.dataset, 'test', args.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    # Extract features
    print("\nExtracting features from frozen backbone...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    val_features, val_labels = extract_features(feature_extractor, val_loader, device)
    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    
    print(f"Train features: {train_features.shape}")
    print(f"Val features: {val_features.shape}")
    print(f"Test features: {test_features.shape}")
    
    # Compute feature quality metrics
    print("\nComputing feature quality metrics...")
    feature_metrics = compute_feature_quality_metrics(
        torch.cat([train_features, val_features, test_features]), 
        torch.cat([train_labels, val_labels, test_labels])
    )
    
    print(f"Silhouette Score: {feature_metrics['silhouette_score']:.4f}")
    print(f"k-NN Accuracy: {feature_metrics['knn_accuracy']:.4f}")
    print(f"Separation Ratio: {feature_metrics['separation_ratio']:.4f}")
    
    # Train linear classifier
    print("\nTraining linear classifier...")
    classifier, train_losses, val_accs, best_val_acc = train_linear_classifier(
        train_features, train_labels, val_features, val_labels, 
        num_classes, device, args
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_acc, class_accs, test_preds, test_labels_np = evaluate_classifier(
        classifier, test_features, test_labels, device, num_classes
    )
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Per-class accuracies: {[f'{acc:.3f}' for acc in class_accs]}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels_np, test_preds, digits=4))
    
    # Save results
    results = {
        'dataset': args.dataset,
        'unfrozen_blocks': args.unfrozen_blocks,
        'num_classes': num_classes,
        'test_accuracy': float(test_acc),
        'best_val_accuracy': float(best_val_acc),
        'class_accuracies': [float(acc) for acc in class_accs],
        'feature_quality': {k: float(v) for k, v in feature_metrics.items()},
        'training_history': {
            'train_losses': [float(loss) for loss in train_losses],
            'val_accuracies': [float(acc) for acc in val_accs]
        }
    }
    
    output_file = os.path.join(
        args.output_dir, 
        f"linear_probe_{args.dataset}_blocks{args.unfrozen_blocks}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)


def parse_args():
    parser = argparse.ArgumentParser('Linear Probe Evaluation')
    
    # Dataset
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
    parser.add_argument('--output_dir', default='./linear_probe_results', type=str)
    
    # Model
    parser.add_argument('--sam_model_type', default='vit_h', type=str)
    parser.add_argument('--sam_checkpoint', default='./sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--unfrozen_blocks', required=True, type=int)
    
    # Training
    parser.add_argument('--probe_epochs', default=100, type=int)
    parser.add_argument('--probe_batch_size', default=256, type=int)
    parser.add_argument('--probe_lr', default=1e-3, type=float)
    parser.add_argument('--probe_weight_decay', default=0.0, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    
    # Dataset class counts
    parser.set_defaults(dataset_num_classes={
        'bloodmnist': 8, 'pathmnist': 9, 'dermamnist': 7, 'octmnist': 4,
        'pneumoniamnist': 2, 'retinamnist': 5, 'breastmnist': 2, 'tissuemnist': 8,
        'organamnist': 11, 'organcmnist': 11, 'organsmnist': 11,
    })
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
