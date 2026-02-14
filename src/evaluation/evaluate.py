"""
Model evaluation script with comprehensive metrics and statistical testing.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.models import get_model
from src.data.dataset import create_data_loaders
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics, aggregate_metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation with statistical analysis.
    """
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("Loading model...")
        self.model = get_model(config['model'])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Device: {self.device}")
    
    @torch.no_grad()
    def evaluate(self, data_loader, save_dir=None):
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary of aggregated metrics and per-sample results
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        all_metrics = []
        predictions = []
        
        for batch_idx, (images, masks) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            probs = torch.sigmoid(outputs)
            
            # Calculate metrics for each sample in batch
            for i in range(images.shape[0]):
                sample_metrics = calculate_metrics(probs[i:i+1], masks[i:i+1])
                all_metrics.append(sample_metrics)
                
                # Store prediction for visualization
                predictions.append({
                    'image': images[i].cpu(),
                    'mask': masks[i].cpu(),
                    'prediction': probs[i].cpu(),
                    'metrics': sample_metrics
                })
            
            del images, masks, outputs, probs
            torch.cuda.empty_cache()
        
        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics)
        
        # Print results
        self._print_results(aggregated)
        
        # Save visualizations if requested
        if save_dir is not None:
            self._save_visualizations(predictions, aggregated, Path(save_dir))
        
        return aggregated, all_metrics, predictions
    
    def _print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Dice Coefficient:     {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
        print(f"IoU Score:            {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
        print(f"Precision:            {metrics['precision']:.4f} ± {metrics['precision_std']:.4f}")
        print(f"Recall:               {metrics['recall']:.4f} ± {metrics['recall_std']:.4f}")
        print(f"Pixel Accuracy:       {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
        if not np.isinf(metrics.get('hausdorff_95', float('inf'))):
            print(f"Hausdorff Distance:   {metrics['hausdorff_95']:.2f} ± {metrics['hausdorff_95_std']:.2f}")
        print("="*60)
    
    def _save_visualizations(self, predictions, metrics, save_dir):
        """Save visualization plots"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot sample predictions (best, worst, median)
        self._plot_sample_predictions(predictions, save_dir)
        
        # Plot metric distributions
        self._plot_metric_distributions(predictions, save_dir)
        
        print(f"\n✓ Visualizations saved to {save_dir}")
    
    def _plot_sample_predictions(self, predictions, save_dir, num_samples=9):
        """Plot sample predictions"""
        # Sort by dice score
        sorted_preds = sorted(predictions, key=lambda x: x['metrics']['dice'])
        
        # Get best, worst, and median samples
        indices = [0, len(sorted_preds)//2, -1]  # worst, median, best
        samples = [sorted_preds[i] for i in indices] * 3  # Repeat to get 9 samples
        
        fig, axes = plt.subplots(3, 9, figsize=(20, 7))
        
        for i, sample in enumerate(samples[:9]):
            col = i
            
            # Image
            img = sample['image'].squeeze().numpy()
            axes[0, col].imshow(img, cmap='gray')
            axes[0, col].set_title(f"Dice: {sample['metrics']['dice']:.3f}")
            axes[0, col].axis('off')
            
            # Ground truth
            mask = sample['mask'].squeeze().numpy()
            axes[1, col].imshow(mask, cmap='gray')
            axes[1, col].axis('off')
            
            # Prediction
            pred = (sample['prediction'].squeeze().numpy() > 0.5).astype(float)
            axes[2, col].imshow(pred, cmap='gray')
            axes[2, col].axis('off')
        
        axes[0, 0].set_ylabel('Input', fontsize=12)
        axes[1, 0].set_ylabel('Ground Truth', fontsize=12)
        axes[2, 0].set_ylabel('Prediction', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_distributions(self, predictions, save_dir):
        """Plot distribution of metrics"""
        metrics_to_plot = ['dice', 'iou', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metrics_to_plot):
            values = [p['metrics'][metric_name] for p in predictions]
            
            axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
            axes[i].axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
            axes[i].set_xlabel(metric_name.capitalize(), fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].set_title(f'{metric_name.capitalize()} Distribution', fontsize=14)
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()


def compare_models(model_results, output_path='model_comparison.png'):
    """
    Compare multiple models statistically.
    
    Args:
        model_results: Dictionary mapping model names to their metric lists
        output_path: Path to save comparison plot
    """
    print("\n" + "="*60)
    print("STATISTICAL MODEL COMPARISON")
    print("="*60)
    
    # Paired t-tests between models
    model_names = list(model_results.keys())
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            
            # Get dice scores
            dice1 = [m['dice'] for m in model_results[model1]]
            dice2 = [m['dice'] for m in model_results[model2]]
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(dice1, dice2)
            
            print(f"\n{model1} vs {model2}:")
            print(f"  Mean Dice: {np.mean(dice1):.4f} vs {np.mean(dice2):.4f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                better = model1 if np.mean(dice1) > np.mean(dice2) else model2
                print(f"  ✓ {better} is significantly better (p < 0.05)")
            else:
                print(f"  No significant difference")


def main():
    parser = argparse.ArgumentParser(description='Evaluate lung segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Data split to evaluate')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        Path(args.data_dir)
    )
    
    # Select appropriate loader
    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    data_loader = loaders[args.split]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config, args.checkpoint, args.device)
    
    # Evaluate
    save_dir = Path(args.save_dir) / Path(args.checkpoint).stem
    aggregated, all_metrics, predictions = evaluator.evaluate(
        data_loader,
        save_dir=save_dir
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
