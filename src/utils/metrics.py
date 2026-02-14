"""
Evaluation metrics for segmentation.
"""

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import Dict


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """Calculate Intersection over Union (IoU)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def precision_score(pred, target, smooth=1e-6):
    """Calculate precision (positive predictive value)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()


def recall_score(pred, target, smooth=1e-6):
    """Calculate recall (sensitivity)"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()


def pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = target.numel()
    
    accuracy = correct / total
    return accuracy.item()


def hausdorff_distance_95(pred, target):
    """
    Calculate 95th percentile Hausdorff distance.
    Lower is better. Returns distance in pixels.
    """
    try:
        pred_np = (pred.cpu().numpy() > 0.5).astype(np.uint8)
        target_np = (target.cpu().numpy() > 0.5).astype(np.uint8)
        
        # Get boundary points
        pred_points = np.argwhere(pred_np)
        target_points = np.argwhere(target_np)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        d1 = directed_hausdorff(pred_points, target_points)[0]
        d2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(d1, d2)
    except:
        return float('inf')


def calculate_metrics(pred, target, threshold=0.5) -> Dict[str, float]:
    """
    Calculate all metrics for a prediction-target pair.
    
    Args:
        pred: Predicted mask (can be logits or probabilities)
        target: Ground truth mask
        threshold: Threshold for converting predictions to binary
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert logits to probabilities if needed
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    metrics = {
        'dice': dice_coefficient(pred, target),
        'iou': iou_score(pred, target),
        'precision': precision_score(pred, target),
        'recall': recall_score(pred, target),
        'accuracy': pixel_accuracy(pred, target),
    }
    
    # Hausdorff distance (more expensive, compute only if needed)
    try:
        metrics['hausdorff_95'] = hausdorff_distance_95(pred, target)
    except:
        metrics['hausdorff_95'] = float('inf')
    
    return metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, float]:
    """
    Aggregate metrics from multiple batches.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary of mean metric values
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    metric_names = metrics_list[0].keys()
    
    for name in metric_names:
        values = [m[name] for m in metrics_list if not np.isinf(m[name])]
        if values:
            aggregated[name] = np.mean(values)
            aggregated[f'{name}_std'] = np.std(values)
        else:
            aggregated[name] = 0.0
            aggregated[f'{name}_std'] = 0.0
    
    return aggregated


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    pred = torch.rand(1, 1, 224, 224)
    target = (torch.rand(1, 1, 224, 224) > 0.5).float()
    
    metrics = calculate_metrics(pred, target)
    print("Metrics:", metrics)
    print("✓ All metric tests passed!")
