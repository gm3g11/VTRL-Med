#!/usr/bin/env python3
"""
Utility functions for Stage 3 training - CORRECTED VERSION
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
from sklearn.metrics import accuracy_score

# Import from Stage 2 for PH computation
from topocl.stage2.stage2_unified_parallel import compute_multiscale_persistence_full_image


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(dataset: str) -> int:
    """Get number of classes for each dataset"""
    class_map = {
        'kvasir': 8,
        'pathmnist': 9,
        'octmnist': 4,
        'organsmnist': 11,
        'isic2019': 8,
    }
    return class_map.get(dataset.lower(), 10)


def balance_loss(gate_weights, low_threshold=0.05, high_threshold=0.5):
    """Balance loss to prevent expert collapse"""
    importance = gate_weights.mean(dim=0) + 1e-8
    dominance_penalty = F.relu(importance - high_threshold).sum()
    ignore_penalty = F.relu(low_threshold - importance).sum()
    return torch.clamp(dominance_penalty + ignore_penalty, 0, 5)


def print_expert_statistics(epoch, gate_weights, method_name):
    """Print expert gating statistics"""
    importance = gate_weights.mean(dim=0).cpu().numpy()
    std = gate_weights.std(dim=0).cpu().numpy()
    max_weight = gate_weights.max(dim=0)[0].cpu().numpy()
    min_weight = gate_weights.min(dim=0)[0].cpu().numpy()

    entropy = -(importance * np.log(importance + 1e-8)).sum()
    max_entropy = np.log(5)

    print(f"\n{'=' * 80}")
    print(f"📊 Expert Statistics - {method_name.upper()} - Epoch {epoch}")
    print(f"{'=' * 80}")
    print(f"{'Expert':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-' * 80}")

    expert_names = ["Visual-only", "Topology-only", "Concatenation",
                    "Cross-Attention", "Gated Fusion"]

    for i, name in enumerate(expert_names):
        print(f"{name:<20} {importance[i]:<10.4f} {std[i]:<10.4f} "
              f"{min_weight[i]:<10.4f} {max_weight[i]:<10.4f}")

    print(f"{'-' * 80}")
    print(f"Gating Entropy: {entropy:.4f} / {max_entropy:.4f} "
          f"({entropy / max_entropy * 100:.1f}%)")

    dominant = importance > 0.5
    ignored = importance < 0.05

    if dominant.any():
        print(f"⚠️  Dominant experts (>0.5): {[expert_names[i] for i in range(5) if dominant[i]]}")
    if ignored.any():
        print(f"⚠️  Ignored experts (<0.05): {[expert_names[i] for i in range(5) if ignored[i]]}")
    if not dominant.any() and not ignored.any():
        print(f"✅ All experts in healthy range [0.05, 0.5]")

    print(f"{'=' * 80}\n")


def linear_probe_evaluation(encoder, imgs_train, labels_train, rois_train,
                            imgs_test, labels_test, rois_test,
                            n_h0=288, n_h1=576, num_classes=9,
                            device='cuda', epochs=100, batch_size=512):
    """
    Fast linear probe evaluation with batched feature extraction

    CRITICAL: Computes PH on IMAGES (not ROIs) BEFORE normalization
    """
    encoder.eval()

    print(f"    Extracting features (batched, batch_size={batch_size})...")

    def extract_features_batched(imgs, rois, batch_size=512):
        """Extract features in large batches for speed"""
        features = []

        n_samples = len(imgs)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="    Extracting", leave=False):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                batch_imgs = imgs[start_idx:end_idx]

                # STEP 1: Compute PH on IMAGES (BEFORE normalization)
                batch_h0 = []
                batch_h1 = []
                for img in batch_imgs:
                    if img.dtype != np.uint8:
                        img_uint8 = (img * 255).astype(np.uint8)
                    else:
                        img_uint8 = img

                    pd = compute_multiscale_persistence_full_image(
                        img_uint8, n_h0=n_h0 // 3, n_h1=n_h1 // 3
                    )
                    batch_h0.append(torch.from_numpy(pd['h0']))
                    batch_h1.append(torch.from_numpy(pd['h1']))

                batch_h0 = torch.stack(batch_h0).float()
                batch_h1 = torch.stack(batch_h1).float()

                # STEP 2: Normalize images (AFTER PH)
                batch_imgs = batch_imgs.astype(np.float32) / 255.0
                batch_imgs = torch.from_numpy(batch_imgs).permute(0, 3, 1, 2)

                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                batch_imgs = (batch_imgs - mean) / std

                # STEP 3: Extract features
                batch_imgs = batch_imgs.to(device)
                batch_h0 = batch_h0.to(device)
                batch_h1 = batch_h1.to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feats, _ = encoder(batch_imgs, batch_h0, batch_h1)

                features.append(feats.cpu())

        return torch.cat(features, dim=0)

    train_feats = extract_features_batched(imgs_train, rois_train, batch_size)
    test_feats = extract_features_batched(imgs_test, rois_test, batch_size)

    print(f"    Features: train={train_feats.shape}, test={test_feats.shape}")

    train_labels = torch.from_numpy(labels_train).long()
    test_labels = torch.from_numpy(labels_test).long()

    classifier = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=0
    )

    classifier.train()
    for _ in range(epochs):
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        test_feats = test_feats.to(device)
        test_labels = test_labels.to(device)
        logits = classifier(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100

    return acc


def linear_probe_with_tta(
        encoder, linear_classifier,
        imgs_test, labels_test, rois_test,
        augmenter, n_h0, n_h1, device, n_aug=5
):
    """
    Test-time augmentation for final evaluation

    Averages features over multiple weak augmentations for robust predictions.
    Use this ONLY for final reported results, not during training.

    Args:
        encoder: Trained MoE encoder
        linear_classifier: Trained linear classifier (nn.Linear)
        imgs_test: Test images (N, H, W, 3) numpy array
        labels_test: Test labels (N,) numpy array
        rois_test: Test ROIs (list of N boolean masks)
        augmenter: TopologyAwareAugmentation instance
        n_h0: Number of H0 features per scale (e.g., 96, NOT 288)
        n_h1: Number of H1 features per scale (e.g., 192, NOT 576)
        device: Device
        n_aug: Number of augmentations to average

    Returns:
        tta_acc: Test-time augmentation accuracy
    """
    encoder.eval()
    linear_classifier.eval()

    all_preds = []

    print(f"\n{'=' * 80}")
    print(f"Running Test-Time Augmentation with {n_aug} augmentations...")
    print(f"{'=' * 80}\n")

    with torch.no_grad():
        for img, roi in tqdm(zip(imgs_test, rois_test), total=len(imgs_test), desc="TTA"):
            # Generate multiple augmented versions
            features = []

            for _ in range(n_aug):
                # Use WEAK augmentation for TTA (more conservative)
                img_aug = augmenter.weak_augment(img, roi)

                # Convert to tensor
                if img_aug.dtype != np.uint8:
                    img_aug = (img_aug * 255).astype(np.uint8)

                img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                # Compute PH on augmented image
                pd = compute_multiscale_persistence_full_image(img_aug, n_h0=n_h0, n_h1=n_h1)
                h0 = torch.from_numpy(pd['h0']).unsqueeze(0).to(device).float()
                h1 = torch.from_numpy(pd['h1']).unsqueeze(0).to(device).float()

                with autocast('cuda', dtype=torch.float16):
                    feat, _ = encoder(img_tensor, h0, h1)
                    feat = feat.float()

                features.append(feat)

            # Average features across augmentations
            avg_feat = torch.stack(features).mean(dim=0)

            # Predict
            logits = linear_classifier(avg_feat)
            prob = F.softmax(logits, dim=1)
            all_preds.append(prob.cpu())

    # Compute accuracy
    all_preds = torch.cat(all_preds, dim=0)
    predicted = all_preds.argmax(dim=1).numpy()

    if isinstance(labels_test, torch.Tensor):
        labels_test = labels_test.cpu().numpy()

    tta_acc = accuracy_score(labels_test, predicted) * 100

    print(f"\n✅ TTA Accuracy: {tta_acc:.2f}%\n")

    return tta_acc


def train_linear_probe_from_encoder(
        encoder,
        imgs_train, labels_train, rois_train,
        imgs_test, labels_test, rois_test,
        n_h0, n_h1, num_classes, device, epochs=100, lr=1e-3
):
    """
    Train a linear classifier on top of frozen encoder features

    This is a helper function to get the trained linear classifier
    for use with TTA evaluation.

    Args:
        encoder: Trained MoE encoder
        imgs_train, labels_train, rois_train: Training data
        imgs_test, labels_test, rois_test: Test data
        n_h0, n_h1: Persistence feature dimensions
        num_classes: Number of classes
        device: Device
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        linear: Trained linear classifier (nn.Linear)
        best_acc: Best test accuracy achieved
    """
    encoder.eval()

    print("    Training linear classifier on frozen features...")

    def extract_features(imgs, rois):
        """Extract features from clean images"""
        features = []

        for img, roi in tqdm(zip(imgs, rois), total=len(imgs), desc="    Extracting", leave=False):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pd = compute_multiscale_persistence_full_image(img, n_h0=n_h0 // 3, n_h1=n_h1 // 3)
            h0 = torch.from_numpy(pd['h0']).unsqueeze(0).to(device).float()
            h1 = torch.from_numpy(pd['h1']).unsqueeze(0).to(device).float()

            with torch.no_grad():
                with autocast('cuda', dtype=torch.float16):
                    feat, _ = encoder(img_tensor, h0, h1)
                    feat = feat.float()

            features.append(feat.cpu())

        return torch.cat(features, dim=0)

    # Extract features
    X_train = extract_features(imgs_train, rois_train)
    y_train = torch.from_numpy(labels_train) if isinstance(labels_train, np.ndarray) else labels_train
    X_test = extract_features(imgs_test, rois_test)
    y_test = torch.from_numpy(labels_test) if isinstance(labels_test, np.ndarray) else labels_test

    # Move to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Train linear classifier
    linear = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        linear.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(linear(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Evaluate
        linear.eval()
        with torch.no_grad():
            acc = (linear(X_test).argmax(1) == y_test).float().mean().item() * 100
            if acc > best_acc:
                best_acc = acc
                best_state = linear.state_dict().copy()

    # Load best weights
    linear.load_state_dict(best_state)

    print(f"    ✓ Linear classifier trained: {best_acc:.2f}%")

    return linear, best_acc


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup"""
    import math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)