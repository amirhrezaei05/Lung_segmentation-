"""
Inference script for lung segmentation.

Make predictions on new chest X-ray images.
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.models import get_model
from src.utils.config import load_config
from src.data.preprocessing import enhance_image, normalize_image
from torchvision import transforms


class LungSegmentationInference:
    """
    Inference class for lung segmentation predictions.
    """
    
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        self.config = load_config(config_path)
        
        # Load model
        print("Loading model...")
        self.model = get_model(self.config['model'])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get preprocessing config
        self.image_size = tuple(self.config['data']['image_size'])
        self.preprocessing_config = self.config['data'].get('preprocessing', {})
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Image size: {self.image_size}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Apply enhancement
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
        else:
            image = torch.from_numpy(image).float()
        
        # Resize
        transform = transforms.Compose([
            transforms.Resize(self.image_size)
        ])
        
        if image.dim() == 2:
            image = image.unsqueeze(0)
        image = transform(image)
        
        # Normalize
        norm_mean = self.preprocessing_config.get('normalization', {}).get('mean', [0.485])
        norm_std = self.preprocessing_config.get('normalization', {}).get('std', [0.229])
        image = normalize_image(image, mean=norm_mean, std=norm_std)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    
    @torch.no_grad()
    def predict(self, image_path, threshold=0.5):
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to image file
            threshold: Probability threshold for binary mask
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        logits = self.model(image_tensor)
        probs = torch.sigmoid(logits)
        
        # Apply threshold
        mask = (probs > threshold).float()
        
        # Convert to numpy
        probs_np = probs.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        return {
            'probabilities': probs_np,
            'binary_mask': mask_np,
            'threshold': threshold
        }
    
    def predict_batch(self, image_dir, output_dir=None, threshold=0.5):
        """
        Make predictions on a directory of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save predictions
            threshold: Probability threshold
        """
        image_dir = Path(image_dir)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        
        print(f"Processing {len(image_files)} images...")
        
        for image_path in image_files:
            print(f"  {image_path.name}...", end=' ')
            
            # Predict
            result = self.predict(image_path, threshold)
            
            # Save if output directory specified
            if output_dir is not None:
                output_path = output_dir / f"{image_path.stem}_mask.png"
                mask_uint8 = (result['binary_mask'] * 255).astype(np.uint8)
                cv2.imwrite(str(output_path), mask_uint8)
            
            print("✓")
        
        print(f"\n✓ Processed {len(image_files)} images")
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with original image.
        
        Args:
            image_path: Path to image file
            save_path: Path to save visualization
        """
        # Load original image
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Get prediction
        result = self.predict(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Probability map
        axes[1].imshow(result['probabilities'], cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        
        # Binary mask
        axes[2].imshow(result['binary_mask'], cmap='gray')
        axes[2].set_title('Segmentation Mask')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Lung segmentation inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, help='Directory of images')
    parser.add_argument('--output_dir', type=str, help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = LungSegmentationInference(
        args.checkpoint,
        args.config,
        args.device
    )
    
    # Single image prediction
    if args.image:
        print(f"\nProcessing: {args.image}")
        
        if args.visualize:
            save_path = f"{Path(args.image).stem}_prediction.png"
            inference.visualize_prediction(args.image, save_path)
        else:
            result = inference.predict(args.image, args.threshold)
            print(f"✓ Prediction complete")
            print(f"  Mean probability: {result['probabilities'].mean():.3f}")
            print(f"  Lung coverage: {result['binary_mask'].mean()*100:.1f}%")
    
    # Batch prediction
    elif args.input_dir:
        inference.predict_batch(
            args.input_dir,
            args.output_dir,
            args.threshold
        )
    
    else:
        print("Error: Specify --image or --input_dir")
        return
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
