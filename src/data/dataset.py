"""
PyTorch Dataset classes for lung segmentation.

Handles data loading, preprocessing, and augmentation for chest X-ray images.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .preprocessing import enhance_image, normalize_image


class LungSegmentationDataset(Dataset):
    """
    Dataset class for lung segmentation from chest X-rays.
    
    Args:
        image_paths: List of paths to X-ray images
        mask_paths: List of paths to segmentation masks
        image_size: Target image size (height, width)
        transform: Albumentations transform pipeline
        preprocessing_config: Configuration for image enhancement
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        preprocessing_config: Optional[dict] = None,
        augment: bool = False
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform
        self.preprocessing_config = preprocessing_config or {}
        self.augment = augment
        
        # Verify that we have matching images and masks
        assert len(image_paths) == len(mask_paths), \
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        
        print(f"Dataset initialized with {len(self.image_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single image-mask pair.
        
        Returns:
            image: Preprocessed image tensor (1, H, W)
            mask: Binary mask tensor (1, H, W)
        """
        # Load image and mask
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_paths[idx]}")
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {self.mask_paths[idx]}")
        
        # Apply preprocessing to image (enhancement)
        if self.preprocessing_config.get('clahe', {}).get('enabled', True):
            image = enhance_image(
                image,
                apply_clahe_flag=True,
                apply_unsharp_flag=self.preprocessing_config.get('unsharp_masking', {}).get('enabled', True),
                apply_gamma_flag=self.preprocessing_config.get('gamma_correction', {}).get('enabled', True),
                clahe_params=self.preprocessing_config.get('clahe', {}),
                unsharp_params=self.preprocessing_config.get('unsharp_masking', {}),
                gamma=self.preprocessing_config.get('gamma_correction', {}).get('gamma', 1.2)
            )
            image = image.numpy()
        
        # Ensure images are uint8 for albumentations
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Apply transformations (resize + augmentation)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default: just resize and convert to tensor
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size)
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Normalize image
        if isinstance(image, torch.Tensor):
            if image.dim() == 2:
                image = image.unsqueeze(0)
            norm_mean = self.preprocessing_config.get('normalization', {}).get('mean', [0.485])
            norm_std = self.preprocessing_config.get('normalization', {}).get('std', [0.229])
            image = normalize_image(image, mean=norm_mean, std=norm_std)
        
        # Normalize mask to [0, 1]
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.max() > 1.0:
                mask = mask / 255.0
        
        return image, mask


def get_transforms(image_size: Tuple[int, int], augment: bool = False) -> A.Compose:
    """
    Create transformation pipeline using Albumentations.
    
    Args:
        image_size: Target size (height, width)
        augment: Whether to include augmentation transforms
        
    Returns:
        Albumentations Compose transform
    """
    if augment:
        # Training transforms with augmentation
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ToTensorV2()
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2()
        ])
    
    return transform


def prepare_data_paths(
    data_dir: Path,
    image_subdir: str = "CXR_png",
    mask_subdir: str = "masks"
) -> Tuple[List[Path], List[Path]]:
    """
    Prepare matched lists of image and mask paths.
    
    Args:
        data_dir: Root directory containing image and mask subdirectories
        image_subdir: Subdirectory name for images
        mask_subdir: Subdirectory name for masks
        
    Returns:
        Tuple of (image_paths, mask_paths)
    """
    image_dir = Path(data_dir) / image_subdir
    mask_dir = Path(data_dir) / mask_subdir
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    # Get all image files
    image_paths = sorted(list(image_dir.glob("*.png")))
    
    # Find matching masks
    mask_paths = []
    matched_image_paths = []
    
    for img_path in image_paths:
        # Look for mask with matching stem
        potential_masks = list(mask_dir.glob(f"*{img_path.stem}*.png"))
        
        if potential_masks:
            mask_paths.append(potential_masks[0])
            matched_image_paths.append(img_path)
    
    print(f"Found {len(matched_image_paths)} matched image-mask pairs")
    print(f"  Images: {image_dir}")
    print(f"  Masks: {mask_dir}")
    
    return matched_image_paths, mask_paths


def create_data_loaders(
    config: dict,
    data_dir: Path,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary with data settings
        data_dir: Root directory of dataset
        batch_size: Override config batch size
        num_workers: Override config num_workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get data paths
    image_paths, mask_paths = prepare_data_paths(data_dir)
    
    # Get config values
    image_size = tuple(config['data']['image_size'])
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    batch_size = batch_size or config['training']['batch_size']
    num_workers = num_workers or config['data']['num_workers']
    pin_memory = config['data']['pin_memory']
    
    # Calculate split sizes
    total_size = len(image_paths)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nData splits:")
    print(f"  Train: {train_size} ({train_split*100:.1f}%)")
    print(f"  Val:   {val_size} ({val_split*100:.1f}%)")
    print(f"  Test:  {test_size} ({test_split*100:.1f}%)")
    
    # Split the data
    indices = list(range(total_size))
    np.random.seed(config['experiment']['seed'])
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_image_paths = [image_paths[i] for i in train_indices]
    train_mask_paths = [mask_paths[i] for i in train_indices]
    val_image_paths = [image_paths[i] for i in val_indices]
    val_mask_paths = [mask_paths[i] for i in val_indices]
    test_image_paths = [image_paths[i] for i in test_indices]
    test_mask_paths = [mask_paths[i] for i in test_indices]
    
    # Create datasets
    train_dataset = LungSegmentationDataset(
        train_image_paths,
        train_mask_paths,
        image_size=image_size,
        transform=get_transforms(image_size, augment=True),
        preprocessing_config=config['data'].get('preprocessing', {}),
        augment=True
    )
    
    val_dataset = LungSegmentationDataset(
        val_image_paths,
        val_mask_paths,
        image_size=image_size,
        transform=get_transforms(image_size, augment=False),
        preprocessing_config=config['data'].get('preprocessing', {}),
        augment=False
    )
    
    test_dataset = LungSegmentationDataset(
        test_image_paths,
        test_mask_paths,
        image_size=image_size,
        transform=get_transforms(image_size, augment=False),
        preprocessing_config=config['data'].get('preprocessing', {}),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing dataset module...")
    # This would require actual data to test
    print("✓ Dataset module created successfully")
