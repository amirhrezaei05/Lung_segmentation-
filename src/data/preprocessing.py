"""
Image preprocessing functions for chest X-ray enhancement.

Implements CLAHE, unsharp masking, and gamma correction for improved
lung boundary visibility.
"""

import cv2
import numpy as np
import torch
from typing import Union


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE enhances local contrast which is particularly useful for
    medical images where subtle details matter.
    
    Args:
        image: Input grayscale image (H, W) or (H, W, 1)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image with same shape as input
    """
    if image.ndim == 3:
        image = image.squeeze(-1)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    
    return enhanced


def apply_unsharp_masking(
    image: np.ndarray,
    kernel_size: int = 9,
    sigma: float = 10.0,
    alpha: float = 1.5,
    beta: float = -0.5
) -> np.ndarray:
    """
    Apply unsharp masking to enhance edges and details.
    
    Unsharp masking = Original + α*Original - β*Blurred
    
    Args:
        image: Input grayscale image
        kernel_size: Size of Gaussian kernel (should be odd)
        sigma: Standard deviation for Gaussian blur
        alpha: Weight for original image
        beta: Weight for blurred image (negative to subtract)
        
    Returns:
        Sharpened image
    """
    if image.ndim == 3:
        image = image.squeeze(-1)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Unsharp mask formula
    enhanced = cv2.addWeighted(image, alpha, blurred, beta, 0)
    
    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float = 1.2
) -> np.ndarray:
    """
    Apply gamma correction to adjust image brightness.
    
    Output = 255 * (Input / 255) ^ gamma
    
    Args:
        image: Input grayscale image (0-255 range)
        gamma: Gamma value (>1 brightens, <1 darkens)
        
    Returns:
        Gamma-corrected image
    """
    if image.ndim == 3:
        image = image.squeeze(-1)
    
    # Normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to [0, 255]
    enhanced = (corrected * 255).astype(np.uint8)
    
    return enhanced


def enhance_image(
    image: Union[np.ndarray, torch.Tensor],
    apply_clahe_flag: bool = True,
    apply_unsharp_flag: bool = True,
    apply_gamma_flag: bool = True,
    clahe_params: dict = None,
    unsharp_params: dict = None,
    gamma: float = 1.2
) -> torch.Tensor:
    """
    Apply full enhancement pipeline to chest X-ray image.
    
    Pipeline: CLAHE → Unsharp Masking → Gamma Correction
    
    Args:
        image: Input image (numpy array or torch tensor)
        apply_clahe_flag: Whether to apply CLAHE
        apply_unsharp_flag: Whether to apply unsharp masking
        apply_gamma_flag: Whether to apply gamma correction
        clahe_params: Parameters for CLAHE
        unsharp_params: Parameters for unsharp masking
        gamma: Gamma value for correction
        
    Returns:
        Enhanced image as torch tensor
    """
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Remove channel dimension if present
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    # Apply CLAHE
    if apply_clahe_flag:
        params = clahe_params or {}
        image = apply_clahe(
            image,
            clip_limit=params.get('clip_limit', 2.0),
            tile_grid_size=params.get('tile_grid_size', (8, 8))
        )
    
    # Apply unsharp masking
    if apply_unsharp_flag:
        params = unsharp_params or {}
        image = apply_unsharp_masking(
            image,
            kernel_size=params.get('kernel_size', 9),
            sigma=params.get('sigma', 10.0),
            alpha=params.get('alpha', 1.5),
            beta=params.get('beta', -0.5)
        )
    
    # Apply gamma correction
    if apply_gamma_flag:
        image = apply_gamma_correction(image, gamma)
    
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).float()
    
    return image_tensor


def normalize_image(
    image: torch.Tensor,
    mean: list = [0.485],
    std: list = [0.229]
) -> torch.Tensor:
    """
    Normalize image using mean and std.
    
    Uses ImageNet statistics by default, adapted for grayscale.
    
    Args:
        image: Input tensor (H, W) or (C, H, W)
        mean: Mean values for normalization
        std: Standard deviation values
        
    Returns:
        Normalized tensor
    """
    if image.ndim == 2:
        image = image.unsqueeze(0)
    
    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply normalization
    for c in range(image.shape[0]):
        image[c] = (image[c] - mean[c]) / std[c]
    
    return image


if __name__ == "__main__":
    # Test the preprocessing functions
    print("Testing preprocessing functions...")
    
    # Create a test image
    test_image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    
    # Test CLAHE
    clahe_result = apply_clahe(test_image)
    print(f"✓ CLAHE: {test_image.shape} -> {clahe_result.shape}")
    
    # Test unsharp masking
    unsharp_result = apply_unsharp_masking(test_image)
    print(f"✓ Unsharp masking: {test_image.shape} -> {unsharp_result.shape}")
    
    # Test gamma correction
    gamma_result = apply_gamma_correction(test_image)
    print(f"✓ Gamma correction: {test_image.shape} -> {gamma_result.shape}")
    
    # Test full pipeline
    enhanced = enhance_image(test_image)
    print(f"✓ Full enhancement: {test_image.shape} -> {enhanced.shape}")
    
    # Test normalization
    normalized = normalize_image(enhanced)
    print(f"✓ Normalization: {enhanced.shape} -> {normalized.shape}")
    print(f"  Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    print("\n✓ All preprocessing tests passed!")
