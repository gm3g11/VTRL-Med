import torch
from torch.utils.data import Dataset
from torchvision import transforms
import medmnist
from medmnist import INFO
from PIL import Image


class MedMNIST_SPICE(Dataset):
    """
    Unified Dataset class for the entire SPICE pipeline.

    This class prepares MedMNIST data by creating multiple augmented views
    as required by all three stages of the SPICE training process.

    For each item, __getitem__ returns a 4-element tuple:
    (weakly_augmented_view, strongly_augmented_view, test_view, label)
    """

    def __init__(self, data_flag, split='train', img_size=64, download=True):
        self.data_flag = data_flag
        self.info = INFO[data_flag]
        self.img_size = img_size

        # Initialize the appropriate MedMNIST dataset
        MedMNIST_Class = getattr(medmnist, self.info['python_class'])
        self.dataset = MedMNIST_Class(split=split, download=download, size=self.img_size)

        # Use standard ImageNet normalization as is common for ResNet backbones
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Weak augmentation for generating pseudo-labels (Stages 2 & 3)
        self.transform_weak = transforms.Compose([
            transforms.RandomCrop(size=self.img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])

        # Strong augmentation for contrastive/consistency learning (Stages 1 & 3)
        self.transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])

        # Test/Inference augmentation (Stages 2 & 3)
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Load the PIL Image and label from the base MedMNIST dataset
        img, label = self.dataset[index]

        # Ensure image is in RGB format for consistency with transforms
        img = img.convert('RGB')

        # Squeeze label to be a scalar tensor
        label = torch.as_tensor(label).squeeze()

        # Generate all required views
        weak_view = self.transform_weak(img)
        strong_view = self.transform_strong(img)
        test_view = self.transform_test(img)

        # This return order is crucial for compatibility with all training scripts
        return weak_view, strong_view, test_view, label