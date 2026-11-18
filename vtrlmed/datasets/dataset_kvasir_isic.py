import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


class Kvasir_SPICE(Dataset):
    """
    Kvasir Dataset for SPICE pipeline.
    Creates 85/15 train/test split directly from class folders.
    
    Returns: (weak_view, strong_view, test_view, label)
    """
    
    def __init__(self, data_dir, split='train', img_size=224, download=False, seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        
        # Get all class folders
        classes = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
        
        # Collect all samples
        all_samples = []
        for class_name in classes:
            class_path = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                all_samples.append((img_path, class_idx))
        
        # Create train/test split (85/15)
        train_samples, test_samples = train_test_split(
            all_samples, test_size=0.15, random_state=seed, 
            stratify=[s[1] for s in all_samples]
        )
        
        if split == 'train':
            self.samples = train_samples
        else:  # test
            self.samples = test_samples
        
        print(f"Kvasir {split}: {len(self.samples)} images, {len(classes)} classes")
        
        # ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        # Weak augmentation
        self.transform_weak = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomCrop(size=self.img_size, padding=int(self.img_size * 0.0625)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
        # Strong augmentation
        self.transform_strong = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        
        # Test transform
        self.transform_test = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        
        label = torch.as_tensor(label).long()
        
        weak_view = self.transform_weak(img)
        strong_view = self.transform_strong(img)
        test_view = self.transform_test(img)
        
        return weak_view, strong_view, test_view, label


class ISIC2019_SPICE(Dataset):
    """
    ISIC2019 Dataset for SPICE pipeline.
    Combines official train+test, then creates 85/15 train/test split.
    
    Returns: (weak_view, strong_view, test_view, label)
    """
    
    def __init__(self, data_dir, split='train', img_size=224, download=False, seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        # Get classes from train directory
        classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
        
        # Collect all samples from both train and test
        all_samples = []
        
        for class_name in classes:
            class_idx = self.class_to_idx[class_name]
            
            # From train folder
            train_class_path = os.path.join(train_dir, class_name)
            if os.path.exists(train_class_path):
                images = [f for f in os.listdir(train_class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in images:
                    img_path = os.path.join(train_class_path, img_name)
                    all_samples.append((img_path, class_idx))
            
            # From test folder
            test_class_path = os.path.join(test_dir, class_name)
            if os.path.exists(test_class_path):
                images = [f for f in os.listdir(test_class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in images:
                    img_path = os.path.join(test_class_path, img_name)
                    all_samples.append((img_path, class_idx))
        
        # Create train/test split (85/15)
        train_samples, test_samples = train_test_split(
            all_samples, test_size=0.15, random_state=seed, 
            stratify=[s[1] for s in all_samples]
        )
        
        if split == 'train':
            self.samples = train_samples
        else:  # test
            self.samples = test_samples
        
        print(f"ISIC2019 {split}: {len(self.samples)} images, {len(classes)} classes")
        
        # ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        # Weak augmentation
        self.transform_weak = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomCrop(size=self.img_size, padding=int(self.img_size * 0.0625)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
        
        # Strong augmentation
        self.transform_strong = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        
        # Test transform
        self.transform_test = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        
        label = torch.as_tensor(label).long()
        
        weak_view = self.transform_weak(img)
        strong_view = self.transform_strong(img)
        test_view = self.transform_test(img)
        
        return weak_view, strong_view, test_view, label
